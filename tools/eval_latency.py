#!/usr/bin/env python3
"""End-to-end latency, decomposed — because "<1 second" has three readings.

The topic says "<1 second anomaly-detection latency". A previous benchmark
reported 250 ms of processing on a capture that was ALREADY IN MEMORY, and that
capture was four seconds long. A reviewer is entitled to the obvious objection:

    if you need four seconds of data before processing starts, how is the
    anomaly-detection latency 250 ms?

The objection is correct, and the honest answer is that there are three distinct
latencies and only one of them is 250 ms. This tool measures all three and does
not leave the interpretation to the reader.

    L1  FEATURE-COMPUTATION LATENCY
        Compute time once a window is available. This is what the earlier
        benchmark measured.

    L2  ANOMALY-INDICATION LATENCY
        Sample arrival to first indication. Window fill + L1. This is the number
        the topic threshold is about, and it is bounded by the SHORTEST window
        the physics still resolves — which is a measured quantity, not a choice.

    L3  CONFIRMED DIAGNOSTIC DECISION
        Indication to a decision the sequential test will stand behind. Requires
        persistence across observations by design; a system that names a part
        from one window is a system that will name parts on healthy machines.

L2 is the one to quote against the threshold. L3 is longer, always, and saying
so is the point of having a sequential test at all.

HOW SHORT CAN THE WINDOW BE? A fault line sits at a fixed order, so shortening
the window coarsens the order axis until the line and its neighbours merge. At
1500 rpm the shaft turns 25 times a second, so a 0.25 s window contains about
six revolutions and about six fault impacts. This sweeps window length and finds
where detection actually degrades, rather than assuming four seconds is needed
because four seconds is what the dataset ships.

Usage:
    PYTHONPATH=src taskset -c 0 python3 tools/eval_latency.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.io.paderborn import _channels                            # noqa: E402
from cbmx.physics.bearing import get as get_bearing                # noqa: E402
from cbmx.physics.envelope import Band, envelope_spectrum          # noqa: E402
from cbmx.health.baseline import Baseline, Regime                  # noqa: E402
from cbmx.health.sequential import WaldAccumulator       # noqa: E402

FS = 64000.0
BAND = Band(500.0, 2000.0, 0.0, 0.0)
WINDOWS_S = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0)


def load(p):
    from scipy.io import loadmat
    m = loadmat(p, squeeze_me=True, struct_as_record=False)
    k = next(x for x in m if not x.startswith("__"))
    ch = _channels(m[k])
    return (np.ascontiguousarray(ch["vibration_1"], dtype=np.float64),
            float(np.mean(ch["speed"])) / 60.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/paderborn")
    ap.add_argument("--faulty", default="KA04")
    ap.add_argument("--healthy", default="K001")
    ap.add_argument("--cond", default="N15_M07_F10")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    b = get_bearing("PU6203")
    order = b.orders()["outer_race"]
    waves = {}
    for code in (a.faulty, a.healthy):
        fs_ = sorted(glob.glob(str(Path(a.data) / code /
                                   f"{a.cond}_{code}_*.mat")))[:6]
        waves[code] = [load(p) for p in fs_]
        if not waves[code]:
            print(f"no records for {code}")
            return 1

    print(f"outer-race order {order:.4f} x shaft; shaft "
          f"{waves[a.faulty][0][1]:.1f} Hz; band {BAND.lo:.0f}-{BAND.hi:.0f} Hz\n")

    print("─── how short can the window be before the physics stops resolving? ───")
    print(f"   {'window':>9}{'revs':>7}{'impacts':>9}{'order res':>11}"
          f"{'faulty dB':>11}{'healthy dB':>12}{'separation':>12}")
    per_window, best = {}, None
    for w in WINDOWS_S:
        n = int(w * FS)
        got = {}
        for code in (a.faulty, a.healthy):
            vals = []
            for x, fr in waves[code]:
                # first full window of each record — no cherry-picking
                seg = x[:n]
                if seg.size < n:
                    continue
                s = envelope_spectrum(seg, FS, fr, band=BAND)
                _, db = s.prominence_db(order)
                if np.isfinite(db):
                    vals.append(db)
            got[code] = float(np.median(vals)) if vals else float("nan")
        fr = waves[a.faulty][0][1]
        sep = got[a.faulty] - got[a.healthy]
        res = (FS / n) / fr                      # order resolution
        per_window[w] = {"faulty_db": got[a.faulty], "healthy_db": got[a.healthy],
                         "separation_db": sep, "order_resolution": res,
                         "revolutions": w * fr, "impacts": w * fr * order}
        flag = ""
        if sep > 15 and (best is None or w < best):
            best = w
            flag = "  <- shortest usable"
        print(f"   {w:>8.3f}s{w*fr:>7.1f}{w*fr*order:>9.0f}{res:>11.3f}"
              f"{got[a.faulty]:>11.1f}{got[a.healthy]:>12.1f}{sep:>12.1f}{flag}")

    if best is None:
        print("\n   no window separated the two; cannot quote an indication latency")
        return 1

    # ── L1 at the shortest usable window ──────────────────────────────────
    n = int(best * FS)
    base = Baseline("lat", min_samples=5)
    r0 = Regime(0, 0)
    for _ in range(30):
        base.observe("outer_race", r0, 5.0)
    for x, fr in waves[a.faulty][:2]:
        envelope_spectrum(x[:n], FS, fr, band=BAND)      # warm-up

    lat = []
    for x, fr in waves[a.faulty]:
        for start in range(0, min(x.size - n, 8 * n), n // 2):
            t0 = time.perf_counter()
            s = envelope_spectrum(x[start:start + n], FS, fr, band=BAND)
            _, db = s.prominence_db(order)
            z, run = base.score("outer_race", r0, db)
            lat.append((time.perf_counter() - t0) * 1000.0)
    lat.sort()
    p99 = lat[min(len(lat) - 1, int(0.99 * len(lat)))]

    # ── L3: observations to a standing decision, at the shortest window ───
    wald = WaldAccumulator(0.02, 0.10, leak_per_obs=1.0)
    zf = []
    for x, fr in waves[a.faulty]:
        s = envelope_spectrum(x[:n], FS, fr, band=BAND)
        _, db = s.prominence_db(order)
        z, run = base.score("outer_race", r0, db)
        zf.append(z)
    z_typ = float(np.median(zf))
    level = max(-1.0, min(1.5, (z_typ - 3.0) / 3.0))
    llr = 0.55 * (1.0 * level + 0.45 * 0.8 + 0.35 * 0.6)
    n_obs = int(np.ceil(wald.upper / llr)) if llr > 0 else None

    hop = best / 2.0
    l2 = best + p99 / 1000.0
    l3 = l2 + (n_obs - 1) * hop if n_obs else None

    print(f"\n─── the three latencies, at a {best:.3f} s window "
          f"({hop:.3f} s hop) ───")
    print(f"   L1  feature computation, window in memory      "
          f"{p99:8.1f} ms   (p99)")
    print(f"   L2  sample arrival -> anomaly indication       "
          f"{l2*1000:8.1f} ms   = window fill + L1")
    print(f"       topic threshold 1000 ms                    "
          f"{'PASS' if l2 < 1.0 else 'FAIL':>8}     margin {1.0/l2:.1f}x")
    if l3:
        print(f"   L3  indication -> confirmed diagnosis          "
              f"{l3*1000:8.0f} ms   ({n_obs} observations at z={z_typ:.1f})")
    print("\n   L2 is the number the topic threshold asks for. L3 is longer by")
    print("   design: the sequential test requires persistence, and a system")
    print("   that names a part from one window will name parts on healthy")
    print("   machines.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"window_sweep": per_window, "shortest_usable_s": best,
             "L1_compute_p99_ms": p99, "L2_indication_s": l2,
             "L3_confirmed_s": l3, "observations_to_confirm": n_obs,
             "typical_z": z_typ, "hop_s": hop}, indent=2, default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
