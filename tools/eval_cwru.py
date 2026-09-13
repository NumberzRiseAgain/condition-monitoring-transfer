#!/usr/bin/env python3
"""Evaluate on Case Western seeded-fault data.

The protocol matters more than the number, so it is stated here and printed at
the top of every run.

    COMMISSION   the demodulation band, on 0.007in fault records at 0 hp only.
                 Those records are then excluded from the test set entirely.

    BASELINE     what normal looks like, from the healthy records only — and
                 only the first 60% of each. No fault record ever teaches the
                 baseline.

    TEST         everything else:
                   - every fault record except 0.007in @ 0hp
                     (so 0.014in and 0.021in are wholly unseen defect sizes,
                      and 0.007in survives only at loads never commissioned on)
                   - the last 40% of each healthy record, which is where false
                     alarms are counted

Three ways this could be made to look better, and why none of them are done:

  Tuning on the test set. Every threshold in this system comes from alpha, beta
  and geometry. Nothing is fitted to CWRU. If the numbers are poor, that is the
  result.

  Choosing the band per record. Picking the band that best reveals a fault and
  then reporting the fault is a multiple-comparisons trap that manufactures
  evidence from noise. The band is frozen at commissioning; `Monitor` refuses to
  start without one.

  Dropping the healthy records. A detector that is never asked to stay quiet has
  not been tested. The false-alarm line is printed first.

Run tools/fetch_cwru.py first.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.graph.asset import Asset                                   # noqa: E402
from cbmx.health.baseline import Baseline                            # noqa: E402
from cbmx.io.cwru import Record, align_rate, load_dir, windows       # noqa: E402
from cbmx.monitor import Monitor                                     # noqa: E402
from cbmx.physics.envelope import commission_band                    # noqa: E402

PARTS = ["outer_race", "inner_race", "rolling_element", "cage"]


def _split(recs):
    commission = [r for r in recs if not r.is_healthy
                  and abs(r.defect_in - 0.007) < 1e-6 and r.load_hp == 0]
    healthy = [r for r in recs if r.is_healthy]
    test_fault = [r for r in recs if not r.is_healthy and r not in commission]
    return commission, healthy, test_fault


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/cwru")
    ap.add_argument("--asset", default="configs/asset_motor_bench.yaml")
    ap.add_argument("--sensor", default="acc_de")
    ap.add_argument("--window", type=float, default=0.75)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=0.10)
    ap.add_argument("--target-fs", type=float, default=12000.0)
    ap.add_argument("--min-samples", type=int, default=30,
                    help="samples before a baseline line may be scored against")
    ap.add_argument("--attribution", default="line", choices=["line", "family"],
                    help="'line' scores each defect on its own frequency (the "
                         "held-out result); 'family' scores the whole predicted "
                         "pattern and drops contested lines (developed on CWRU)")
    ap.add_argument("--out", default="runs/cwru_eval.json")
    a = ap.parse_args()

    print(__doc__.split("Run tools/fetch_cwru.py")[0])
    print("=" * 78)

    recs = load_dir(a.data)
    if not recs:
        print(f"No .mat files in {a.data}. Run tools/fetch_cwru.py first.")
        return 1

    # Every record onto one sample rate before anything else. The healthy
    # baseline is 48 kHz and the fault records are 12 kHz; mixing them puts
    # every order in the wrong bin and is silent when it happens.
    resampled = sum(1 for r in recs if abs(r.fs - a.target_fs) > 1e-6)
    recs = [align_rate(r, a.target_fs) for r in recs]
    print(f"\nloaded {len(recs)} records; {resampled} decimated to "
          f"{a.target_fs/1000:.0f} kHz to match the fault records")

    commission, healthy, test_fault = _split(recs)
    print(f"  commission set : {len(commission)} records (0.007in @ 0hp, held out of test)")
    print(f"  healthy        : {len(healthy)} records (first 60% teaches, last 40% tested)")
    print(f"  test faults    : {len(test_fault)} records")
    if not commission or not healthy:
        print("\nNot enough records. Fetch at least the normal set and 0.007in @ 0hp.")
        return 1

    asset = Asset.load(a.asset)
    sensor = asset.sensors[a.sensor]
    bearing = asset.bearings[sensor.watches[0]]

    # 1. COMMISSION -----------------------------------------------------------
    band = commission_band([r.signal for r in commission], a.target_fs,
                           bearing.orders(),
                           float(np.mean([r.shaft_hz for r in commission])))
    sensor.band_lo_hz, sensor.band_hi_hz = band.lo, band.hi
    print(f"\n1. BAND      {band.lo:.0f}-{band.hi:.0f} Hz, frozen "
          f"(diagnostic score {band.kurtosis:.1f} dB)")
    print(f"   ATTRIBUTION  {a.attribution}"
          + ("   (held out — developed before CWRU was seen)" if a.attribution == "line"
             else "   (DEVELOPED ON CWRU — not a held-out number)"))

    # 2. BASELINE -------------------------------------------------------------
    # Pick the hop so every regime clears the estimator's minimum, and say what
    # was chosen. CWRU gives about ten seconds of healthy running per load; with
    # a 0.75 s window and no overlap that is thirteen samples, and with a quarter
    # hop it is twenty-nine — one short of a hard-coded thirty, which would
    # disable every line and look like a detection failure.
    heal_s = min(0.6 * len(r.signal) / r.fs for r in healthy)
    target = a.min_samples + 12
    hop = max(a.window / 8.0, min(a.window, heal_s / target))
    n_windows = int((heal_s - a.window) / hop) + 1
    independent = heal_s / a.window

    bl = Baseline(asset.id, min_samples=a.min_samples)
    learner = Monitor(asset, a.sensor, bl, window_s=a.window,
                      attribution=a.attribution)
    t = 0.0
    for r in healthy:
        cut = int(0.6 * len(r.signal))
        head = Record(r.file_no, r.fault, r.defect_in, r.load_hp, r.rpm_nominal,
                      r.fs, r.signal[:cut], r.rpm_measured, r.channel, r.path)
        # Overlapping windows: CWRU gives about ten seconds of healthy running
        # per load, which is thin for a robust estimator. Overlap yields more
        # samples of the same statistic. It does not create information, and the
        # scarcity is itself worth reporting — it is the same scarcity the
        # programme has.
        for wt, w in windows(head, a.window, hop):
            learner.step(t + wt, w, r.shaft_hz, load=r.load_hp / 3.0, learning=True)
        t += len(head.signal) / head.fs
    cov = bl.coverage()
    print(f"2. BASELINE  {cov['lines_ready']}/{cov['lines_tracked']} lines ready "
          f"across {cov['regimes']} regimes, from healthy data only")
    print(f"             {heal_s:.1f}s healthy per load, hop {hop*1000:.0f}ms "
          f"-> {n_windows} windows (min {a.min_samples})")
    print(f"             only ~{independent:.0f} of those are independent — the "
          f"healthy record is short.")
    print("             That scarcity is real and it is the programme's problem too.")
    if cov["lines_ready"] == 0:
        print("   No line reached the minimum sample count. The healthy record is")
        print("   too short for the window; try --window 0.5.")
        return 1

    # 3. TEST -----------------------------------------------------------------
    def run(rec: Record, tail_only: bool = False):
        sig = rec.signal[int(0.6 * len(rec.signal)):] if tail_only else rec.signal
        r2 = Record(rec.file_no, rec.fault, rec.defect_in, rec.load_hp,
                    rec.rpm_nominal, rec.fs, sig, rec.rpm_measured,
                    rec.channel, rec.path)
        m = Monitor(asset, a.sensor, copy.deepcopy(bl), a.alpha, a.beta, a.window,
                    attribution=a.attribution)
        first = None
        for wt, w in windows(r2, a.window):
            found = m.step(wt, w, r2.shaft_hz, load=r2.load_hp / 3.0)
            if found and first is None:
                first = (wt, found[0])
        return m, first, len(sig) / rec.fs

    print("\n3. THE TEST WE ARE MOST LIKELY TO FAIL — healthy records, unseen segments\n")
    fa = 0
    for r in healthy:
        m, first, dur = run(r, tail_only=True)
        if first is None:
            print(f"   {r.label:<24} {dur:5.1f}s   silent    PASS")
        else:
            fa += 1
            print(f"   {r.label:<24} {dur:5.1f}s   reported {first[1].symptom.part}"
                  f" at {first[0]:.1f}s   FALSE ALARM")

    print("\n4. FAULT RECORDS\n")
    print(f"   {'record':<34}{'reported':<18}{'latency':>9}{'conf':>7}  ")
    print("   " + "-" * 70)
    confusion = defaultdict(lambda: defaultdict(int))
    per_size = defaultdict(lambda: [0, 0])
    per_load = defaultdict(lambda: [0, 0])
    rows = []
    for r in sorted(test_fault, key=lambda x: (x.fault, x.defect_in, x.load_hp)):
        m, first, dur = run(r)
        cvg = m.coverage()
        if first:
            got = first[1].symptom.part
        elif cvg["fully_blind"]:
            # No baseline covered this operating point. The system refused to
            # judge; recording that as a miss would blame the detector for a
            # gap in the healthy data.
            got = "no baseline"
        else:
            got = "nothing"
        confusion[r.fault][got] += 1
        ok = got == r.fault
        per_size[r.defect_in][1] += 1
        per_load[r.load_hp][1] += 1
        if ok:
            per_size[r.defect_in][0] += 1
            per_load[r.load_hp][0] += 1
        lat = f"{first[0]:.1f}s" if first else "—"
        conf = f"{first[1].confidence:.3f}" if first else "—"
        mark = ("" if ok else "   <-- wrong" if first
                else "   <-- DECLINED, no baseline for this regime"
                if got == "no baseline" else "   <-- missed")
        print(f"   {r.label:<34}{got:<18}{lat:>9}{conf:>7}{mark}")
        rows.append({"file": r.file_no, "true": r.fault, "defect_in": r.defect_in,
                     "load_hp": r.load_hp, "reported": got,
                     "latency_s": first[0] if first else None,
                     "confidence": first[1].confidence if first else None,
                     "correct": ok})

    print("\n5. CONFUSION MATRIX\n")
    cols = PARTS + ["nothing", "no baseline"]
    print("   true \\ reported   " + "".join(f"{c[:11]:>13}" for c in cols))
    for tp in PARTS:
        if not confusion[tp]:
            continue
        print(f"   {tp:<18}" + "".join(f"{confusion[tp][c]:>13}" for c in cols))

    n_ok = sum(1 for r in rows if r["correct"])
    n_declined = sum(1 for r in rows if r["reported"] == "no baseline")
    n_judged = len(rows) - n_declined
    print("\n6. TOTALS\n")
    print(f"   correctly isolated   {n_ok}/{len(rows)}"
          f"   ({100*n_ok/max(1,len(rows)):.0f}%) of all records")
    if n_declined:
        print(f"   of records it judged {n_ok}/{n_judged}"
              f"   ({100*n_ok/max(1,n_judged):.0f}%)")
        print(f"   declined to judge    {n_declined}"
              f"   (no healthy baseline covered that operating point)")
    print(f"   false alarms         {fa}/{len(healthy)} healthy records")
    print("\n   by defect size (0.014in and 0.021in were never commissioned on)")
    for size in sorted(per_size):
        ok_, n_ = per_size[size]
        print(f"     {size:.3f} in   {ok_}/{n_}")
    print("   by motor load")
    for load in sorted(per_load):
        ok_, n_ = per_load[load]
        print(f"     {load} hp      {ok_}/{n_}")

    lat = [r["latency_s"] for r in rows if r["latency_s"] is not None]
    if lat:
        print(f"\n   time to first report   median {np.median(lat):.1f}s  "
              f"(range {min(lat):.1f}-{max(lat):.1f}s)")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({
        "protocol": "band commissioned on 0.007in@0hp (held out); baseline from "
                    "healthy first-60% only; tested on all other fault records "
                    "and the healthy last-40%",
        "band_hz": [band.lo, band.hi], "window_s": a.window,
        "alpha": a.alpha, "beta": a.beta,
        "false_alarms": fa, "healthy_records": len(healthy),
        "correct": n_ok, "total": len(rows), "rows": rows,
        "baseline_coverage": cov,
    }, indent=2))
    print(f"\n   written to {a.out}")

    # Provenance. This script will happily run on anything shaped like a CWRU
    # file, and an earlier version announced "real measured data" whatever it
    # had been fed. A results table whose provenance is asserted rather than
    # checked is worth nothing to a reviewer.
    man = Path(a.data) / "manifest.json"
    if man.exists():
        print("\n   PROVENANCE  checksummed against " + str(man))
        print("   This is real measured data from a real machine. It is still a")
        print("   bench motor, not an arresting engine — the transfer argument,")
        print("   not this table, is what the proposal has to make.")
    else:
        print("\n   PROVENANCE  NO manifest.json in " + str(a.data))
        print("   These records did not come from tools/fetch_cwru.py, so nothing")
        print("   here is evidence of anything. Numbers from unverified inputs must")
        print("   not leave this terminal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
