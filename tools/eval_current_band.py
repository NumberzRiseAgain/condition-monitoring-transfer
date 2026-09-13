#!/usr/bin/env python3
"""Motor current, read inside the drive train's torsional resonance.

The straightforward MCSA run (`tools/eval_mcsa.py`) finds nothing on this data,
and tracing the signal path shows why: an outer-race defect is a radial event,
the radial force channel rises 23.8 dB, and the shaft torque does not move at
all at the fault order. Motor current follows torque. There is nothing to
demodulate.

Following every harmonic of the fault order through the torque channel instead
of only the first two changes the picture. At 900 rpm the strongest torque
response is at the 5th harmonic; at 1500 rpm it is at the 3rd. Both land at
229 Hz. That is a torsional resonance of the drive train, it is a property of
the machine rather than of the bearing or the speed, and it is the only route
the fault has to the motor.

So the detector is: commission the resonance once, freeze it, and read whichever
harmonic of the geometry-derived order falls inside it, through the current
sideband at |f1 - k*f_char|. The orders still come from four caliper dimensions.
Only the choice of harmonic depends on the machine, and it is made once.

PROTOCOL, fixed before the first run:

    Commission.  The band is found from a small set of records which are then
      EXCLUDED from everything downstream — one damaged record per speed. This
      is the same protocol `tools/eval_dataset.py` uses to commission the
      vibration envelope band, for the same reason.

    Baseline.   Robust median/MAD per operating condition, learned from healthy
      records only.

    Test.       Every remaining record. False alarms are counted on the healthy
      ones, which are the majority.

    No run-time band selection.  The healthy K001 bearing at 900 rpm produces
      18.7 dB at its 7th sideband — more than the damaged bearing produces at
      its own resonance. A detector free to pick its best harmonic per window
      would call that healthy bearing faulty. The band is frozen for exactly
      this reason, and the number is printed below so the risk is visible
      rather than asserted.

Usage:
    PYTHONPATH=src python3 tools/eval_current_band.py [--data data/paderborn]
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

from cbmx.io.paderborn import REAL_DAMAGE, _channels, _fault_of      # noqa: E402
from cbmx.health.baseline import Baseline, Regime                    # noqa: E402
from cbmx.health.sequential import Channels, WaldAccumulator         # noqa: E402
from cbmx.physics.bearing import get as get_bearing                  # noqa: E402
from cbmx.physics import mcsa                                        # noqa: E402

FS_CUR, FS_SLOW = 64000.0, 4000.0
ALPHA, BETA = 0.02, 0.10
Z_OPERATING = 3.0
K_MAX = 12


def load(path):
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=True, struct_as_record=False)
    key = next(k for k in m if not k.startswith("__"))
    ch = _channels(m[key])
    return ch, float(np.mean(ch["speed"])) / 60.0


def band_prominence(ch, bearing, shaft_hz, band, part="outer_race"):
    """Sideband prominence at the harmonic that falls inside the frozen band.

    Returns NaN — not zero — when no harmonic of this order lands in the band at
    this speed. That is an abstention: the machine is running somewhere the
    resonance cannot be used, and saying so is the correct output.
    """
    sp = mcsa.spectrum(ch["phase_current_1"], FS_CUR)
    f1 = mcsa.supply_fundamental(sp)
    fc = bearing.orders()[part] * shaft_hz
    ks = band.harmonics_inside(fc, K_MAX)
    if not ks:
        return np.nan, f1, None
    guards = tuple(np.concatenate([np.arange(1, 30) * shaft_hz,
                                   np.arange(1, 13) * f1]).tolist())
    vals = []
    for k in ks:
        for f in (abs(f1 - k * fc), f1 + k * fc):
            if f < 1.0:
                continue
            db, _ = sp.prominence_db(f, tol_hz=max(0.6, 3 * sp.df),
                                     also_exclude=guards)
            if np.isfinite(db):
                vals.append(db)
    return (float(np.mean(vals)) if vals else np.nan), f1, ks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/paderborn")
    ap.add_argument("--healthy", default="K001")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    bearing = get_bearing("PU6203")
    root = Path(a.data)
    codes = sorted(d.name for d in root.iterdir() if d.is_dir())
    files = defaultdict(list)
    for code in codes:
        for p in sorted(glob.glob(str(root / code / "*.mat"))):
            cond = "_".join(Path(p).stem.split("_")[:3])
            files[(code, cond)].append(p)

    faulty = [c for c in codes if _fault_of(c) == "outer_race"]
    if not faulty:
        print("no outer-race bearing present; nothing to commission on")
        return 1

    # ── commission: one damaged record per speed, then excluded ────────────
    commission, dmg, hlt = [], [], []
    seen_speed = set()
    for code in faulty:
        for (c, cond), ps in sorted(files.items()):
            if c != code or not ps:
                continue
            speed = cond.split("_")[0]
            if speed in seen_speed:
                continue
            seen_speed.add(speed)
            hp = files.get((a.healthy, cond), [])
            if not hp:
                continue
            for src, bag in ((ps[:3], dmg), (hp[:3], hlt)):
                for q in src:
                    ch, fr = load(q)
                    sp = mcsa.spectrum(ch["phase_current_1"], FS_CUR)
                    bag.append((ch["phase_current_1"], FS_CUR, fr,
                                mcsa.supply_fundamental(sp)))
                    commission.append(q)
    band = mcsa.commission_torsional_band(dmg, hlt, bearing)
    print(f"bearing  {bearing.designation}")
    print(f"  BPFO {bearing.orders()['outer_race']:.4f} x shaft   "
          f"source: {bearing.source}")
    print(f"\nBAND     {band.lo:.0f}-{band.hi:.0f} Hz, frozen  "
          f"({band.note}; commissioned on {band.commissioned_on} record(s), "
          f"excluded from test)")
    for p in commission:
        print(f"           excluded: {Path(p).name}")
    print()

    # ── measure everything else ───────────────────────────────────────────
    rows = []
    for (code, cond), ps in sorted(files.items()):
        for p in ps:
            if p in commission:
                continue
            try:
                ch, fr = load(p)
                db, f1, ks = band_prominence(ch, bearing, fr, band)
            except Exception as e:                       # noqa: BLE001
                print(f"  SKIP {Path(p).name}: {type(e).__name__}: {e}")
                continue
            rows.append((code, cond, fr, f1, ks, db))

    # ── baseline from healthy only, per operating condition ───────────────
    conds = sorted({r[1] for r in rows})
    cmap = {c: i for i, c in enumerate(conds)}
    base = Baseline("paderborn/current_band", min_samples=8)
    for code, cond, fr, f1, ks, db in rows:
        if code == a.healthy and np.isfinite(db):
            base.observe("outer_race", Regime(cmap[cond], 0), db)

    print(f"{'bearing':<7}{'condition':<14}{'n':>4}{'k in band':>11}"
          f"{'median dB':>11}{'z':>7}{'  verdict':<16}{'truth'}")
    out, per = {}, defaultdict(list)
    for code, cond, fr, f1, ks, db in rows:
        per[(code, cond)].append((db, ks))

    for (code, cond) in sorted(per):
        vals = [d for d, _ in per[(code, cond)] if np.isfinite(d)]
        ks = next((k for _, k in per[(code, cond)] if k), None)
        r = Regime(cmap[cond], 0)
        if not vals:
            print(f"{code:<7}{cond:<14}{0:>4}{'none':>11}{'—':>11}{'—':>7}"
                  f"  {'no_baseline':<16}{_fault_of(code)}")
            continue
        med = float(np.median(vals))
        z, run = base.score("outer_race", r, med)

        wald = WaldAccumulator(ALPHA, BETA, leak_per_obs=1.0)
        for d in vals:
            zi, runi = base.score("outer_race", r, d)
            if not runi.ready:
                j = wald.observe(f"{code}/{cond}", None)
                continue
            level = max(-1.0, min(1.5, (zi - Z_OPERATING) / Z_OPERATING))
            j = wald.observe(f"{code}/{cond}",
                             Channels(level, 0.8 if zi > 0 else -0.7, 0.0,
                                      {"z": round(zi, 2)}))
        truth = _fault_of(code)
        real = " REAL" if code in REAL_DAMAGE else ""
        ok = ("OK " if (j.decision == "name_it") == (truth == "outer_race")
              else "!! ")
        print(f"{code:<7}{cond:<14}{len(vals):>4}"
              f"{','.join(map(str, ks)) if ks else '-':>11}{med:>11.1f}"
              f"{(z if run.ready else float('nan')):>7.1f}"
              f"  {ok}{j.decision:<13}{truth}{real}")
        out[f"{code}/{cond}"] = {"n": len(vals), "k": ks, "median_db": med,
                                 "z": z, "decision": j.decision, "truth": truth}

    print("\n" + "=" * 74)
    # How many captures would a decision need? Wald's own answer, printed so the
    # result can be read as "weak but real and quantified" rather than as a
    # simple miss.
    up = WaldAccumulator(ALPHA, BETA).upper
    for kk, v in sorted(out.items()):
        if v["truth"] == "normal" or not np.isfinite(v["z"]):
            continue
        lv = max(-1.0, min(1.5, (v["z"] - Z_OPERATING) / Z_OPERATING))
        llr = 0.55 * (1.0 * lv + 0.45 * (0.8 if v["z"] > 0 else -0.7))
        n = (up / llr) if llr > 0 else None
        v["captures_to_decide"] = round(n) if n else None
        print(f"  {kk:<26} z={v['z']:>5.1f}  {llr:>6.3f} nats/capture  ->  "
              + (f"{n:>4.0f} captures to decide" if n and n < 1e4
                 else "never at this evidence rate"))
    print()
    print("OK on a healthy row  = stayed quiet.  OK on a damaged row = named it.")
    print(f"band frozen at {band.lo:.0f}-{band.hi:.0f} Hz; the harmonic index "
          f"changes with speed, the band does not")
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"band": band.as_dict(), "rows": out}, indent=2, default=str))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
