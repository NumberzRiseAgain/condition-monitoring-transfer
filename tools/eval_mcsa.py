#!/usr/bin/env python3
"""Vibration against motor current, on the same bearings, with the same physics.

This is the experiment behind the ladder in the DV087 proposal. Matthew Marko's
answer of 24 August was that the deployed Advanced Arresting Gear has no
vibration sensors, that motor current, pressure and temperature are what it
does have, and that new vibration sensors are a reasonable thing to propose.
The obvious question is then: how much does the vibration sensor actually buy?

That question has an answer, on real damage, in public data. Paderborn recorded
motor current and vibration *synchronously*, at 64 kHz, on the same bearings,
under the same four operating conditions — twelve bearings damaged with
machining tools and fourteen damaged by running them to failure. It is the only
public set that lets the two sensor types be compared without changing anything
else.

What is held constant across both channels here:

    the geometry            FAG 6203, from the dataset's own paper
    the predicted lines     BPFO, BPFI, cage — computed, never searched for
    the baseline            robust median/MAD per operating condition, healthy only
    the evidence test       the same Wald accumulator, the same alpha and beta
    the attribution rule    the named part must be the loudest, and the others quiet

What differs is only where the number is read from. In vibration the fault
frequency appears directly, as impacts. In current it appears as sidebands about
the supply fundamental, because the rotor displacement modulates the air-gap
permeance — the same orders, one modulation removed.

Two guards, one per channel, and both matter more than they look:

    current     lines that fall within a guard band of a supply harmonic k*f1
                are dropped and reported unobservable. On this rig the
                ball-defect order is 3.9932 and the drive has four pole pairs,
                so the ball sideband sits on the second supply harmonic at every
                speed the rig offers. Measured naively it reads 29 dB of ball
                evidence on a certified-healthy bearing.

    vibration   lines that fall within a guard band of a shaft harmonic k*fr are
                dropped for the same reason. Unbalance and misalignment live
                there, they are large, and they are not bearing faults.

Usage:
    PYTHONPATH=src python3 tools/eval_mcsa.py [--data data/paderborn]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.io.paderborn import REAL_DAMAGE, _channels, _fault_of   # noqa: E402
from cbmx.health.baseline import Baseline, Regime                 # noqa: E402
from cbmx.health.sequential import Channels, WaldAccumulator      # noqa: E402
from cbmx.physics.bearing import get as get_bearing               # noqa: E402
from cbmx.physics import mcsa                                     # noqa: E402

FS = 64000.0
PARTS = ("outer_race", "inner_race", "cage")     # no ball: see below
ALPHA, BETA = 0.02, 0.10
HARMONICS = (1, 2, 3)
GUARD_HZ = 1.5
Z_OPERATING = 3.0

# The rolling element is excluded from the whole run, and the reason is in the
# dataset paper rather than in our results: Lessmeier et al. report that damage
# at the rolling elements was never observed in the accelerated lifetime tests.
# There is no ball-defect record anywhere in this dataset. Scoring the line
# anyway would produce a false-alarm rate for a fault class that is not present,
# which is a number that means nothing and reads like coverage.


def shaft_harmonics(fr: float, f_max: float, n: int = 24) -> np.ndarray:
    k = np.arange(1, n + 1) * fr
    return k[k < f_max]


def measure_vibration(x: np.ndarray, bearing, shaft_hz: float) -> dict:
    """Prominence at each fault's first three harmonics, in the raw spectrum.

    Mean over the harmonics rather than maximum, for the same reason as in the
    current case: a maximum over three candidate lines is a three-fold search,
    and a three-fold search finds something on a healthy bearing too.
    """
    sp = mcsa.spectrum(x, FS)
    shafts = shaft_harmonics(shaft_hz, FS / 2.0)
    out = {}
    for part, order in bearing.orders().items():
        if part not in PARTS:
            continue
        vals, blocked = [], 0
        for k in HARMONICS:
            f = k * order * shaft_hz
            if shafts.size and float(np.min(np.abs(shafts - f))) < GUARD_HZ:
                blocked += 1
                continue
            db, _ = sp.prominence_db(f, tol_hz=max(1.0, 4 * sp.df))
            if np.isfinite(db):
                vals.append(db)
        out[part] = {"prominence_db": float(np.mean(vals)) if vals else np.nan,
                     "n_lines": len(vals), "n_blocked": blocked,
                     "observable": bool(vals)}
    return out


def measure_current(x: np.ndarray, bearing, shaft_hz: float) -> dict:
    r = mcsa.measure(x, FS, bearing, shaft_hz, k_max=2)
    return {p: r[p] for p in PARTS}


def attribution_channels(z: dict, part: str) -> Channels | None:
    """Level, specificity, quietness-elsewhere — the same three-channel shape.

    `direction` here asks whether the part being tested is the loudest of the
    three. A real outer-race spall should show on the outer-race line and not on
    the others; if everything rose together, something changed about the machine
    or the sensor, and naming a part would be a guess dressed as a diagnosis.
    """
    have = {k: v for k, v in z.items() if np.isfinite(v)}
    if part not in have:
        return None
    e = have[part]
    others = [v for k, v in have.items() if k != part]

    level = max(-1.0, min(1.5, (e - Z_OPERATING) / Z_OPERATING))
    is_lead = all(e >= o for o in others) if others else True
    direction = 0.8 if is_lead else -0.7
    if not others:
        corroboration = 0.0
    elif max(others) < 1.5:
        corroboration = 0.6              # the others are quiet: specific
    elif max(others) > Z_OPERATING:
        corroboration = -0.4             # everything is loud: not a part fault
    else:
        corroboration = 0.0
    return Channels(level, direction, corroboration, {"z": round(e, 2)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/paderborn")
    ap.add_argument("--healthy", default="K001")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    bearing = get_bearing("PU6203")
    root = Path(a.data)
    codes = sorted(d.name for d in root.iterdir() if d.is_dir())
    print(f"bearing: {bearing.designation}")
    print("  orders: " + "  ".join(f"{k} {v:.4f}x" for k, v in
                                   bearing.orders().items()))
    print(f"  source: {bearing.source}")
    print(f"  healthy reference: {a.healthy}")
    print(f"  codes present: {', '.join(codes)}")
    print()

    # ── measure everything once ────────────────────────────────────────────
    # rows: (code, condition, channel, part) -> prominence dB
    rows = []
    for code in codes:
        for p in sorted(glob.glob(str(root / code / "*.mat"))):
            stem = Path(p).stem
            cond = "_".join(stem.split("_")[:3])
            try:
                from scipy.io import loadmat
                m = loadmat(p, squeeze_me=True, struct_as_record=False)
                key = next(k for k in m if not k.startswith("__"))
                ch = _channels(m[key])
                shaft = float(np.mean(ch["speed"])) / 60.0
                cur = measure_current(ch["phase_current_1"], bearing, shaft)
                vib = measure_vibration(ch["vibration_1"], bearing, shaft)
            except Exception as e:                       # noqa: BLE001
                print(f"  SKIP {stem}: {type(e).__name__}: {e}")
                continue
            for chan, res in (("current", cur), ("vibration", vib)):
                for part in PARTS:
                    rows.append((code, cond, chan, part, shaft,
                                 res[part]["prominence_db"]
                                 if res[part]["observable"] else np.nan))
    print(f"measured {len({(r[0], r[1]) for r in rows})} (bearing, condition) "
          f"pairs from {len(rows)//(2*len(PARTS))} files\n")

    # ── baselines from the healthy bearing, per condition ──────────────────
    results = {}
    for chan in ("current", "vibration"):
        base = Baseline(f"paderborn/{chan}", min_samples=8)
        conds = sorted({r[1] for r in rows})
        cmap = {c: i for i, c in enumerate(conds)}
        for code, cond, ch_, part, shaft, db in rows:
            if ch_ != chan or code != a.healthy or not np.isfinite(db):
                continue
            base.observe(part, Regime(cmap[cond], 0), db)

        print(f"══ {chan} " + "═" * (68 - len(chan)))
        hdr = f"{'bearing':<8}{'condition':<14}{'n':>4}"
        for part in PARTS:
            hdr += f"{part[:9]:>11}{'z':>7}"
        print(hdr + f"{'  verdict':<24}")

        per = defaultdict(list)
        for code, cond, ch_, part, shaft, db in rows:
            if ch_ == chan:
                per[(code, cond)].append((part, db))

        out = {}
        for (code, cond) in sorted(per):
            vals = defaultdict(list)
            for part, db in per[(code, cond)]:
                if np.isfinite(db):
                    vals[part].append(db)
            n = max((len(v) for v in vals.values()), default=0)
            if not n:
                continue
            r = Regime(cmap[cond], 0)
            zs, meds = {}, {}
            for part in PARTS:
                if part not in vals:
                    zs[part] = np.nan
                    continue
                med = float(np.median(vals[part]))
                meds[part] = med
                z, run = base.score(part, r, med)
                zs[part] = z if run.ready else np.nan

            # sequential test, one entity per (bearing, part), fed the per-file
            # observations in order so the accumulator does real work
            wald = WaldAccumulator(ALPHA, BETA, leak_per_obs=1.0)
            verdicts = {}
            for part in PARTS:
                if part not in vals:
                    continue
                for db in vals[part]:
                    z_one = {}
                    for q in PARTS:
                        if q in vals:
                            zq, runq = base.score(
                                q, r, db if q == part else float(np.median(vals[q])))
                            z_one[q] = zq if runq.ready else np.nan
                    chs = attribution_channels(z_one, part)
                    j = wald.observe(f"{code}/{part}", chs)
                verdicts[part] = j.decision if vals[part] else "no_baseline"

            named = [p for p, d in verdicts.items() if d == "name_it"]
            truth = _fault_of(code)
            real = " REAL" if code in REAL_DAMAGE else ""
            line = f"{code:<8}{cond:<14}{n:>4}"
            for part in PARTS:
                line += (f"{meds.get(part, float('nan')):>11.1f}"
                         f"{zs.get(part, float('nan')):>7.1f}")
            verdict = ("+".join(named) if named else "—")
            ok = ("OK " if (named == [truth]) else
                  "   " if (not named and truth == "normal") else "!! ")
            print(line + f"  {ok}{verdict:<20}(truth {truth}{real})")
            out[f"{code}/{cond}"] = {"n": n, "median_db": meds, "z": zs,
                                     "named": named, "truth": truth}
        results[chan] = out
        print()

    print("═" * 74)
    print("OK  = named exactly the part the dataset says is damaged")
    print("!!  = named nothing on a damaged bearing, or named the wrong part")
    print(f"alpha={ALPHA} beta={BETA}; leak disabled (records are independent "
          f"4 s captures, not a time series)")
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
