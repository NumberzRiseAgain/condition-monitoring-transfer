#!/usr/bin/env python3
"""Does the method survive a speed transient? Glenn Shevach's question, answered.

On the 24 August call Mr Shevach made the observation that decides whether any of
this is useful on an arresting engine:

    "you might not get certain fault modes excited... during a retract versus an
     arrestment"
    "it's just a ton more energy during an arrestment that is getting absorbed"

The interesting measurement is the arrestment — a large, fast speed transient —
and every result cbmx holds so far is from a machine turning at a constant speed.
A method that only works at constant speed has not been tested for the case that
matters.

WHAT THIS RUN IS, AND WHAT IT IS NOT.

    It is: a real fault signature — Paderborn KA04, outer-race damage produced by
    running a bearing to failure — replayed through a KNOWN, SYNTHETIC speed
    profile, so that the effect of the transient can be isolated and the cure
    measured against it.

    It is NOT: a measurement on genuinely variable-speed data. The Ottawa
    variable-speed set is the real test and has not been run. Nothing here may be
    described as "we validated on variable-speed data".

Why the synthetic profile is still worth running. The failure mode being
demonstrated is not subtle and does not depend on where the speed profile came
from: a fault line sits at a fixed multiple of shaft speed, so when the speed
sweeps, the line sweeps with it and its energy spreads across many spectral bins
until it sits below the floor. The cure — resample against shaft angle instead of
time — either recovers it or does not. Using a real damage signature and a known
profile isolates exactly that, and the known profile is what allows the third
experiment below, which is the one that actually matters for AAG.

THREE EXPERIMENTS.

    1. Constant speed          the baseline. What we have already reported.
    2. Swept speed, no cure    conventional envelope analysis at nominal speed.
    3. Swept speed, tracked    angle-domain resampling with the speed profile.

    4. And the engineering question behind all of it: how accurately does the
       speed have to be known? An arresting engine may not have a tachometer on
       the shaft we care about. Experiment 4 degrades the speed estimate — gain
       error and noise — and finds where order tracking stops working.

THE POINT OF EXPERIMENT 4, FOR THE PROPOSAL.

    Marko confirmed the deployed system has electrical current on the motor. Our
    own MCSA work measured the supply fundamental at 59.99 Hz at 900 rpm and
    99.97 Hz at 1500 rpm against a four-pole-pair drive — that is shaft speed
    recovered from the current channel to better than 0.1%, with no tachometer.

    So if experiment 4 shows order tracking tolerates a speed error of a fraction
    of a percent, the arresting engine already carries its own speed reference,
    and the transient case needs no new instrumentation beyond what the topic
    already assumes. That is a much stronger answer to Mr Shevach than "we would
    add an encoder".

Usage:
    PYTHONPATH=src python3 tools/eval_transient.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.io.paderborn import _channels                              # noqa: E402
from cbmx.physics.bearing import get as get_bearing                  # noqa: E402
from cbmx.physics.envelope import (Band, envelope_spectrum,          # noqa: E402
                                   envelope_spectrum_order_tracked)

FS = 64000.0
# The band was commissioned in the fixed-speed run and is reused unchanged
# here: it is a property of the structure, not of the speed.
BAND = Band(500.0, 2000.0, 0.0, 0.0)


def load(path):
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = next(k for k in m if not k.startswith("__"))
    ch = _channels(m[key])
    return ch, float(np.mean(ch["speed"])) / 60.0


def sweep(x: np.ndarray, fs: float, f0: float, profile: np.ndarray):
    """Replay a constant-speed record as if the shaft had followed `profile`.

    The mapping is exact rather than approximate, and that matters: every
    angular feature of the original — the impacts, their spacing, the sidebands,
    the resonance they excite — is preserved, and only the time base changes.

        theta(tau) = integral of profile  ==  f0 * t(tau)
        so  t(tau) = (1/f0) * integral of profile

    Sampling the original at t(tau) therefore produces the record the same
    bearing would have produced had the shaft actually swept that way. Nothing
    about the fault is synthesised; only the speed history is.
    """
    n = x.shape[0]
    tau = np.arange(n) / fs
    if profile.shape[0] != n:
        profile = np.interp(tau, np.linspace(0, tau[-1], profile.shape[0]), profile)
    theta = np.concatenate([[0.0], np.cumsum(np.diff(tau) * profile[:-1])])
    t_src = theta / f0
    keep = t_src <= tau[-1]
    t_src, prof = t_src[keep], profile[keep]
    return np.interp(t_src, tau, x), prof


def prominence(spec, order: float, tol: float = 0.02) -> float:
    _, db = spec.prominence_db(order, tol=tol)
    return float(db)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/paderborn")
    ap.add_argument("--faulty", default="KA04")
    ap.add_argument("--healthy", default="K001")
    ap.add_argument("--cond", default="N15_M07_F10")
    ap.add_argument("--drop", type=float, default=0.45,
                    help="fraction of speed lost across the record")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    b = get_bearing("PU6203")
    order = b.orders()["outer_race"]
    print(f"bearing {b.designation}   outer race {order:.4f} x shaft")
    print(f"band {BAND.lo:.0f}-{BAND.hi:.0f} Hz, reused unchanged from the "
          f"fixed-speed run\n")

    files = {}
    for code in (a.faulty, a.healthy):
        g = sorted(glob.glob(str(Path(a.data) / code / f"{a.cond}_{code}_*.mat")))
        if not g:
            print(f"no records for {code} at {a.cond}")
            return 1
        files[code] = g

    results = {}
    for code in (a.faulty, a.healthy):
        rows = []
        for p in files[code][:6]:
            ch, f0 = load(p)
            x = ch["vibration_1"]
            n = x.shape[0]

            # ── a decelerating sweep, the shape of a recovery ──────────────
            # Linear from f0 down to (1-drop)*f0. An arrestment is faster and
            # harder than this; a linear ramp is the conservative case for
            # showing the effect, and is easy for a reader to reason about.
            prof_full = np.linspace(f0, f0 * (1.0 - a.drop), n)
            y, prof = sweep(x, FS, f0, prof_full)

            # 1. constant speed, the baseline we already reported
            s1 = envelope_spectrum(x, FS, f0, band=BAND)
            p1 = prominence(s1, order)

            # 2. swept, analysed as if constant at the MEAN speed — what a
            #    conventional fixed-speed monitor would actually do
            mean_hz = float(np.mean(prof))
            s2 = envelope_spectrum(y, FS, mean_hz, band=BAND)
            p2 = prominence(s2, order)

            # 3. swept, order-tracked with the true speed
            s3 = envelope_spectrum_order_tracked(y, FS, prof, band=BAND)
            p3 = prominence(s3, order)

            rows.append((p1, p2, p3, f0, mean_hz))

        arr = np.array([[r[0], r[1], r[2]] for r in rows])
        med = np.nanmedian(arr, axis=0)
        results[code] = {"constant": med[0], "swept_untracked": med[1],
                         "swept_tracked": med[2], "n": len(rows)}
        print(f"── {code}  ({len(rows)} records, {a.drop:.0%} speed drop)")
        print(f"   constant speed, as reported        {med[0]:7.1f} dB")
        print(f"   swept, conventional analysis       {med[1]:7.1f} dB")
        print(f"   swept, angle-domain resampling     {med[2]:7.1f} dB")
        print()

    f, h = results[a.faulty], results[a.healthy]
    print("=" * 70)
    print("separation, faulty minus healthy")
    for k, lab in (("constant", "constant speed          "),
                   ("swept_untracked", "swept, untracked        "),
                   ("swept_tracked", "swept, order-tracked    ")):
        print(f"   {lab}{f[k] - h[k]:7.1f} dB")

    # ── 4. how well must the speed be known? ──────────────────────────────
    print("\n" + "=" * 70)
    print("speed-estimate sensitivity — order tracking fed a WRONG speed")
    print(f"   {'gain error':>12}{'noise':>9}{'faulty dB':>12}{'healthy dB':>12}"
          f"{'separation':>12}")
    sens = {}
    rng = np.random.default_rng(20260825)
    # Widened until it breaks, because the break point IS the requirement.
    for gain, noise in ((1.000, 0.00), (1.005, 0.00), (1.010, 0.00),
                        (1.020, 0.00), (1.030, 0.00), (1.050, 0.00),
                        (1.080, 0.00), (1.120, 0.00), (1.200, 0.00),
                        (1.000, 0.01), (1.000, 0.03)):
        got = {}
        for code in (a.faulty, a.healthy):
            vals = []
            for p in files[code][:4]:
                ch, f0 = load(p)
                x = ch["vibration_1"]
                n = x.shape[0]
                prof_full = np.linspace(f0, f0 * (1.0 - a.drop), n)
                y, prof = sweep(x, FS, f0, prof_full)
                est = prof * gain
                if noise:
                    est = est * (1.0 + noise * rng.standard_normal(est.shape[0]))
                s = envelope_spectrum_order_tracked(y, FS, est, band=BAND)
                vals.append(prominence(s, order))
            got[code] = float(np.nanmedian(vals))
        sep = got[a.faulty] - got[a.healthy]
        sens[f"gain{gain}_noise{noise}"] = {"faulty": got[a.faulty],
                                            "healthy": got[a.healthy],
                                            "separation": sep}
        print(f"   {(gain-1)*100:>11.1f}%{noise*100:>8.0f}%"
              f"{got[a.faulty]:>12.1f}{got[a.healthy]:>12.1f}{sep:>12.1f}")

    # ── 5. does it hold under harsher, more arrestment-like profiles? ─────
    print("\n" + "=" * 70)
    print("profile shape — speed known exactly")
    print(f"   {'profile':<22}{'untracked sep':>15}{'tracked sep':>13}")
    shapes = {
        "45% linear":        lambda f0, n: np.linspace(f0, f0 * 0.55, n),
        "70% linear":        lambda f0, n: np.linspace(f0, f0 * 0.30, n),
        "70% exponential":   lambda f0, n: f0 * np.exp(np.linspace(0, np.log(0.30), n)),
        # A V: decelerate then accelerate. This is the closest shape here to an
        # arrestment followed by a retract, and it is the hardest of the four.
        "down then up":      lambda f0, n: f0 * (0.35 + 0.65 * np.abs(np.linspace(-1, 1, n))),
    }
    shape_out = {}
    for lab, fn in shapes.items():
        got = {}
        for code in (a.faulty, a.healthy):
            u, t = [], []
            for p in files[code][:4]:
                ch, f0 = load(p)
                x = ch["vibration_1"]
                y, pr = sweep(x, FS, f0, fn(f0, x.shape[0]))
                u.append(prominence(envelope_spectrum(
                    y, FS, float(np.mean(pr)), band=BAND), order))
                t.append(prominence(envelope_spectrum_order_tracked(
                    y, FS, pr, band=BAND), order))
            got[code] = (float(np.nanmedian(u)), float(np.nanmedian(t)))
        su = got[a.faulty][0] - got[a.healthy][0]
        st = got[a.faulty][1] - got[a.healthy][1]
        shape_out[lab] = {"untracked_sep": su, "tracked_sep": st}
        print(f"   {lab:<22}{su:>15.1f}{st:>13.1f}")
    print("   A NEGATIVE untracked separation means the damaged bearing reads")
    print("   QUIETER than the healthy one — the monitor does not merely miss")
    print("   the fault, it ranks the broken machine as the healthier of the two.")

    print("\nNOTE: synthetic speed profile over a real fault signature.")
    print("The real test is genuinely variable-speed data, and it HAS now been run:")
    print("see RUNLOG_2026-08-29_kaist.md and tools/eval_kaist.py. This synthetic")
    print("profile is retained because it sweeps harder than the KAIST rig does.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"records": results, "sensitivity": sens, "shapes": shape_out,
             "drop": a.drop, "band_hz": [BAND.lo, BAND.hi],
             "caveat": "synthetic speed profile over real Paderborn damage; "
                       "Ottawa variable-speed data NOT run"},
            indent=2, default=str))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
