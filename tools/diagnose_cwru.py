#!/usr/bin/env python3
"""Why did it get those wrong? Measure, do not guess.

Run after eval_cwru.py. This changes nothing and decides nothing — it prints the
numbers behind the confusion matrix so the failures can be attributed to a cause
instead of a hunch.

Four questions, in the order they matter:

  1. Which lines never got a baseline, and from which operating point?
     A record scored against no baseline is an abstention, not a miss.

  2. For every record, what did all four candidate lines actually measure?
     A wrong call with a 0.2 dB margin over the runner-up is a different defect
     from a wrong call with a 12 dB margin. The first says the decision rule
     needs a margin requirement; the second says the physics search is landing
     on the wrong line.

  3. Do the search windows of different symptoms overlap?
     A rolling-element defect throws sidebands at the cage rate around its own
     order. If one of those lands inside the inner-race window, ball faults will
     read as inner-race faults for a purely geometric reason — and that is a
     bug in the search, not a limit of the data.

  4. How much healthy data does each regime actually have?
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.graph.asset import Asset                                  # noqa: E402
from cbmx.health.baseline import Baseline                           # noqa: E402
from cbmx.io.cwru import Record, align_rate, load_dir, windows      # noqa: E402
from cbmx.monitor import Monitor                                    # noqa: E402
from cbmx.physics.envelope import commission_band, envelope_spectrum  # noqa: E402

PARTS = ["outer_race", "inner_race", "rolling_element", "cage"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cwru")
    ap.add_argument("--asset", default="configs/asset_motor_bench.yaml")
    ap.add_argument("--sensor", default="acc_de")
    ap.add_argument("--window", type=float, default=0.75)
    ap.add_argument("--target-fs", type=float, default=12000.0)
    ap.add_argument("--min-samples", type=int, default=30)
    a = ap.parse_args()

    recs = [align_rate(r, a.target_fs) for r in load_dir(a.data)]
    if not recs:
        print(f"no records in {a.data}")
        return 1
    asset = Asset.load(a.asset)
    sensor = asset.sensors[a.sensor]
    bearing = asset.bearings[sensor.watches[0]]
    orders = bearing.orders()

    healthy = [r for r in recs if r.is_healthy]
    commission = [r for r in recs if not r.is_healthy
                  and abs(r.defect_in - 0.007) < 1e-6 and r.load_hp == 0]

    # ── 3. window overlap — geometry only, no data needed ───────────────────
    from cbmx.physics.attribute import collisions
    print("=" * 78)
    print("SEARCH LINE COLLISIONS  (pure geometry — this needs no data at all)")
    print("=" * 78)
    print(f"\n   {'line':<20}{'order':>9}{'window':>18}")
    for p, o in orders.items():
        w = max(0.02 * o, 0.03)
        print(f"   {p:<20}{o:9.4f}   {o-w:7.3f} – {o+w:<7.3f}")

    cols = collisions(bearing)
    print(f"\n   {len(cols)} collisions across the full predicted families")
    print("   (fundamentals, harmonics and sidebands — not just the fundamentals)\n")
    for c in cols:
        star = "  <<<" if ("h1" in c["a"].split(":")[1] or "h1" in c["b"].split(":")[1]) else ""
        print(f"   {c['a']:<26}{c['a_order']:8.3f}   vs {c['b']:<26}"
              f"{c['b_order']:8.3f}   sep {c['separation']:.3f}{star}")
    print("\n   Rows marked <<< involve a fundamental: those are the ones that turn")
    print("   one defect into another in the confusion matrix. --attribution family")
    print("   drops every contested line from both hypotheses.\n")

    # ── 4. baseline coverage ────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("HEALTHY DATA PER OPERATING POINT")
    print("=" * 78 + "\n")
    for r in sorted(healthy, key=lambda x: x.load_hp):
        dur = len(r.signal) / r.fs
        print(f"   normal @ {r.load_hp}hp   {dur:5.2f}s total, "
              f"{0.6*dur:5.2f}s to the baseline, "
              f"{0.6*dur/a.window:4.1f} independent windows at {a.window}s")
    shortest = min(0.6 * len(r.signal) / r.fs for r in healthy)
    print(f"\n   shortest is {shortest:.2f}s. The estimator needs "
          f"{a.min_samples} samples; overlapping windows can reach that count "
          f"but\n   cannot create information that is not there.")

    band = commission_band([r.signal for r in commission], a.target_fs,
                           orders, float(np.mean([r.shaft_hz for r in commission])))
    sensor.band_lo_hz, sensor.band_hi_hz = band.lo, band.hi

    bl = Baseline(asset.id, min_samples=a.min_samples)
    learner = Monitor(asset, a.sensor, bl, window_s=a.window)
    t = 0.0
    target = a.min_samples + 12
    hop = max(a.window / 8.0, min(a.window, shortest / target))
    for r in healthy:
        cut = int(0.6 * len(r.signal))
        head = Record(r.file_no, r.fault, r.defect_in, r.load_hp, r.rpm_nominal,
                      r.fs, r.signal[:cut], r.rpm_measured, r.channel, r.path)
        for wt, w in windows(head, a.window, hop):
            learner.step(t + wt, w, r.shaft_hz, load=r.load_hp / 3.0, learning=True)
        t += len(head.signal) / head.fs

    print("\n   baseline lines, by regime  (s = speed bin, l = load bin)\n")
    by_regime = defaultdict(list)
    for k, run in bl.stats.items():
        sym, reg = k.split("@")
        by_regime[reg].append((sym.split("::")[-1], run))
    for reg in sorted(by_regime):
        ready = sum(1 for _, r in by_regime[reg] if r.ready)
        n = max(r.n for _, r in by_regime[reg])
        flag = "" if ready == len(by_regime[reg]) else "   <-- INCOMPLETE"
        print(f"   {reg:<8} {ready}/{len(by_regime[reg])} ready, "
              f"n={n:<4}{flag}")

    # ── 2. the margins ──────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("WHAT EVERY LINE MEASURED, PER RECORD  (dB over local floor)")
    print("=" * 78)
    print("\n   A correct call with a small margin is luck. A wrong call with a")
    print("   small margin wants a margin rule. A wrong call with a large margin")
    print("   means the search is landing on the wrong line.\n")
    hdr = (f"   {'record':<32}"
           + "".join(f"{p[:9]:>11}" for p in PARTS)
           + f"{'winner':>17}{'margin':>8}")
    print(hdr)
    print("   " + "-" * (len(hdr) - 3))

    margins = defaultdict(list)
    for r in sorted([x for x in recs if not x.is_healthy],
                    key=lambda x: (x.fault, x.defect_in, x.load_hp)):
        # One representative window from the middle of the record.
        mid = len(r.signal) // 2
        n = int(a.window * r.fs)
        seg = r.signal[mid:mid + n]
        if len(seg) < n:
            continue
        es = envelope_spectrum(seg, r.fs, r.shaft_hz, band=band)
        sc = {p: es.prominence_db(o)[1] for p, o in orders.items()}
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        win, second = ranked[0], ranked[1]
        margin = win[1] - second[1]
        ok = win[0] == r.fault
        margins[(r.fault, ok)].append(margin)
        mark = "" if ok else "  X"
        print(f"   {r.label:<32}" + "".join(f"{sc[p]:11.1f}" for p in PARTS)
              + f"{win[0]:>17}{margin:8.1f}{mark}")

    print("\n   margin over the runner-up, by outcome")
    for (fault, ok), v in sorted(margins.items()):
        print(f"     {fault:<18}{'correct' if ok else 'WRONG  ':<9}"
              f"n={len(v):<3} median {np.median(v):5.1f} dB  "
              f"range {min(v):5.1f}–{max(v):5.1f}")

    print("\n" + "=" * 78)
    print("Read the collisions section first. If a sideband lands in another")
    print("line's window, that is a search bug and fixing it is not tuning.")
    print("If the margins are small and the winners are near-ties, the decision")
    print("rule needs to require a margin — which IS a design change prompted by")
    print("this dataset, and any number measured after it is no longer held out.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
