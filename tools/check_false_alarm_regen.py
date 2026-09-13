#!/usr/bin/env python3
"""Validate the false-alarm regeneration without needing the raw data.

Two checks, both offline. Neither touches ZeMA or Paderborn records, so this
runs anywhere the repository is checked out and takes a couple of seconds.

  1. THE BOUND. Every Clopper-Pearson upper bound the paper quotes is
     recomputed and compared against the published value. This exists because
     the scipy-free fallback that used to back this function was wrong on some
     inputs and silently so: at 0/180 it returned 93.25% where the answer is
     1.65%. If scipy is installed, eval_prognosis also computes both routes and
     stops on disagreement, so this check exercises that too.

  2. THE HYDRAULIC TABLE. Read straight out of the committed run artifact,
     which is a computed file, and printed beside the literals that were
     withdrawn on 13 September 2026.

Usage, from 08_Analysis/code:

    PYTHONPATH=src .venv/bin/python tools/check_false_alarm_regen.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ARTIFACT = HERE.parent.parent / "results" / "fresh_2026-09-09" / "hydraulic_freeze.json"

# (k, n, the percentage printed in the paper)
PUBLISHED = [(0, 80, 3.68), (0, 74, 3.97), (0, 180, 1.65),
             (0, 244, 1.22), (2, 244, 2.56), (0, 668, 0.45)]

# The table withdrawn from the manuscript, for the side-by-side.
WITHDRAWN = {"cooler": (0, 244), "pump": (2, 244), "accumulator": (0, 180)}


def load_module():
    spec = importlib.util.spec_from_file_location("ep", HERE / "eval_prognosis.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main() -> int:
    art = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ARTIFACT
    m = load_module()

    try:
        import scipy                                      # noqa: F401
        scipy_note = "scipy present, so both routes are computed and must agree"
    except Exception:                                     # noqa: BLE001
        scipy_note = "scipy ABSENT, so the binomial identity is the only route"

    print("\n1. THE BOUND  (" + scipy_note + ")\n")
    bad = 0
    for k, n, want in PUBLISHED:
        got = m.clopper_pearson_upper(k, n) * 100.0
        ok = abs(got - want) < 0.02
        bad += not ok
        print(f"   upper({k:>3}, {n:>4}) = {got:6.2f}%   paper says {want:5.2f}%   "
              f"{'ok' if ok else 'MISMATCH'}")
    print(f"\n   {'all six reproduce' if not bad else f'{bad} MISMATCHED'}")

    print(f"\n2. THE HYDRAULIC TABLE  (read from {art})\n")
    if not art.exists():
        print(f"   artifact not found: {art}")
        return 2
    blob = json.loads(art.read_text())
    print(f"   {'component':<14}{'k/n':>10}{'rate':>9}{'95% up':>9}"
          f"{'no-baseline':>13}   withdrawn")
    pk = pn = pnb = 0
    for c in sorted(blob):
        b = blob[c].get("buckets", {}).get("healthy_heldout")
        if b is None or "n" not in b:
            print(f"   {c:<14}  no buckets.healthy_heldout.n in the artifact")
            return 2
        k, n = int(b.get("name_it", 0)), int(b["n"])
        nb = int(b.get("no_baseline", 0))
        pk += k; pn += n; pnb += nb
        w = WITHDRAWN.get(c)
        wtxt = f"{w[0]}/{w[1]}" + ("  same" if w == (k, n) else "  CHANGED") if w else "(absent from it)"
        print(f"   {c:<14}{f'{k}/{n}':>10}{k/n:>9.2%}"
              f"{m.clopper_pearson_upper(k, n):>9.2%}{nb:>13}   {wtxt}")
    print(f"   {'POOLED, all ' + str(len(blob)):<14}{f'{pk}/{pn}':>10}{pk/pn:>9.2%}"
          f"{m.clopper_pearson_upper(pk, pn):>9.2%}{pnb:>13}   0/668  CHANGED")
    print("\n   The withdrawn table said 0 false alarms in 668 held-out healthy")
    print("   cycles across 'all four components', a one-sided 95% upper bound")
    print("   of 0.45%. It listed three components. The one it left out is the")
    print("   valve, and the valve is where the alarms are.")
    print("\n   A false alarm is a NAME IT verdict on healthy material. WATCHING")
    print("   and NO BASELINE are not alarms; the no-baseline column is printed")
    print("   so the denominator question stays visible.\n")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
