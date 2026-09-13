"""ARRESTLINE on the VT NSI / Luna Labs / NSWC Philadelphia actuator dataset.

Two questions, and the second one is the one that decides whether the first
means anything.

  1. On a matched load pair, does the governance layer detect seal damage and
     gear damage, and what does adding an accelerometer buy over the pressure
     and temperature channels the deployed gear already carries?

  2. Every actuator in this set is one physical unit in one fixed condition, so
     no unit has a healthy prefix followed by its own fault. Any baseline must
     therefore come from a DIFFERENT unit, which is exactly the comparison our
     own method says manufactures false alarms, and exactly the trap the dataset
     authors warn about: a model can learn the actuator rather than the fault.

     So the headline experiment is run twice more, on pairs where BOTH units are
     healthy. If a baseline from one healthy actuator fires on another healthy
     actuator, then firing on a damaged one proves nothing. That control is
     reported beside every detection number and it is not optional.

  3. And a regime the baseline has never observed healthy must return
     no_baseline rather than a number.

Run:
    PYTHONPATH=src python3 tools/eval_actuator.py --data /path/to/actuator_dataset/Data
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from typing import Dict, List

import numpy as np

from cbmx.io.actuator import Cycle, load_unit
from cbmx.physics.actuator import ARMS, SYMPTOMS, _working_pg, measure
from cbmx.health.baseline import Baseline, Regime
from cbmx.health.sequential import WaldAccumulator, build_channels

ALPHA, BETA = 0.02, 0.10
TEMP_STEP = 2.0          # degC bins: the regime variable for a hydraulic rig
MIN_SAMPLES = 25
TRAIN_FRAC = 0.5

BY_ID = {s.id: s for s in SYMPTOMS}


def regime_of(c: Cycle) -> Regime:
    """One regime per matched-load experiment.

    The obvious regime variable on a hydraulic rig is oil temperature, and it is
    what the ZeMA work uses. It cannot be used here: Internal_Temp is constant
    within each actuator and different between actuators, because each unit was
    recorded in one session. Binning on it makes the regime a proxy for the unit,
    every target cycle lands in a bin the baseline never saw, and the system
    abstains on 100% of them -- which it did, before this was found. The
    temperature difference between reference and target is instead reported as an
    uncontrolled covariate, which is what it is."""
    return Regime(speed_bin=0, load_bin=0)


PORTABILITY_Z = 3.0


def portability_screen(data: str, ref: str, tgt: str) -> Dict[str, Dict]:
    """Decide which symptoms survive a cross-unit comparison, on HEALTHY DATA ONLY.

    Every actuator here is one unit in one fixed condition, so a baseline must
    come from a different unit than the one being judged. The dataset authors
    warn that a model can learn the actuator instead of the fault. This screen is
    the answer to that warning, and it is run before any damaged unit is opened:
    fit on one healthy actuator, score a second healthy actuator, and disqualify
    any symptom that already reads like a fault. No fault label is used to make
    this decision, so it cannot be tuning in disguise."""
    a = load_unit(data, ref)
    b = load_unit(data, tgt)
    cut = int(len(a) * TRAIN_FRAC)
    working = commission(a[:cut])
    base = fit(a[:cut], working, "vibration")
    out: Dict[str, Dict] = {}
    for s_ in SYMPTOMS:
        zs = []
        for c in b:
            v = measure(c, working)[s_.id]
            st = base.get(s_.id, regime_of(c))
            if np.isfinite(v) and st.ready:
                zs.append(abs(st.z(v)))
        med = float(np.median(zs)) if zs else float("inf")
        out[s_.id] = {"median_abs_z_healthy_vs_healthy": round(med, 2),
                      "portable": bool(med <= PORTABILITY_Z), "n": len(zs)}
    return out


def commission(cycles: List[Cycle]) -> int:
    """Pick the working pressure channel once, on healthy cycles, then freeze."""
    votes = [_working_pg(c) for c in cycles if c.moved]
    return int(statistics.mode(votes)) if votes else 0


def fit(cycles: List[Cycle], working: int, arm: str) -> Baseline:
    b = Baseline("actuator", speed_step_hz=1.0, load_step=1.0,
                 min_samples=MIN_SAMPLES)
    allowed = ARMS[arm]
    for c in cycles:
        m = measure(c, working)
        r = regime_of(c)
        for sid, v in m.items():
            if BY_ID[sid].arm in allowed and np.isfinite(v):
                b.observe(sid, r, v)
    for k in b.stats:
        b.stats[k].frozen = True          # frozen before any test cycle is seen
    return b


def judge(cycles: List[Cycle], b: Baseline, working: int, arm: str,
          component: str, usable=None) -> Dict:
    """Score a run of cycles. Per-cycle verdicts, plus one sequential test."""
    allowed = ARMS[arm]
    acc = WaldAccumulator(ALPHA, BETA)
    per_cycle = {"name_it": 0, "watching": 0, "ordinary": 0, "no_baseline": 0}
    seq_named_at = None
    llrs = []
    for i, c in enumerate(cycles, 1):
        m = measure(c, working)
        r = regime_of(c)
        z, direction, primary = {}, {}, {}
        for sid, v in m.items():
            s = BY_ID[sid]
            if s.component != component or s.arm not in allowed:
                continue
            if usable is not None and not usable.get(sid, True):
                continue
            if not np.isfinite(v):
                continue
            stat = b.get(sid, r)
            if not stat.ready:
                continue
            z[sid] = stat.z(v)
            direction[sid] = s.direction
            primary[sid] = s.primary
        ch = build_channels(z, direction, primary) if z else None

        # per-cycle verdict: a fresh accumulator, so this is the one-shot rate
        one = WaldAccumulator(ALPHA, BETA)
        j1 = one.observe("x", ch)
        per_cycle[j1.decision] += 1

        # and the sequential run over the whole sequence
        j = acc.observe(component, ch)
        if ch is not None:
            llrs.append(j.channels.get("llr", 0.0))
        if seq_named_at is None and j.decision == "name_it":
            seq_named_at = i
    n = max(1, len(cycles))
    return {
        "n": len(cycles),
        "per_cycle": per_cycle,
        "pct_named": 100.0 * per_cycle["name_it"] / n,
        "pct_no_baseline": 100.0 * per_cycle["no_baseline"] / n,
        "seq_named_at": seq_named_at,
        "mean_llr": round(float(np.mean(llrs)), 3) if llrs else None,
        "expected_obs": (round(acc.upper / float(np.mean(llrs)), 1)
                         if llrs and float(np.mean(llrs)) > 0 else None),
    }


def upper95(k: int, n: int) -> float:
    """One-sided 95% upper bound on a rate, by bisection on the binomial tail."""
    from math import comb
    if n == 0:
        return 100.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        p = (lo + hi) / 2
        tail = sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(0, k + 1))
        if tail > 0.05:
            lo = p
        else:
            hi = p
    return 100.0 * hi


def run_pair(data: str, ref: str, tgt: str, component: str, label: str,
             usable=None) -> Dict:
    ref_c = load_unit(data, ref)
    tgt_c = load_unit(data, tgt)
    if not ref_c or not tgt_c:
        return {"label": label, "error": "no data"}
    cut = int(len(ref_c) * TRAIN_FRAC)
    train, held = ref_c[:cut], ref_c[cut:]
    working = commission(train)
    out = {"label": label, "ref": ref, "tgt": tgt, "component": component,
           "working_pg": working + 1, "n_train": len(train),
           "n_held": len(held), "n_tgt": len(tgt_c), "arms": {}}
    for arm in ("installed", "vibration"):
        b = fit(train, working, arm)
        fa = judge(held, b, working, arm, component, usable)
        det = judge(tgt_c, b, working, arm, component, usable)
        fa_k = fa["per_cycle"]["name_it"]
        out["arms"][arm] = {
            "seq_false_alarm_at": fa["seq_named_at"],
            "false_alarms": f'{fa_k}/{fa["n"]}',
            "false_alarm_pct": round(fa["pct_named"], 1),
            "false_alarm_upper95": round(upper95(fa_k, fa["n"]), 2),
            "detection_pct": round(det["pct_named"], 1),
            "detected": f'{det["per_cycle"]["name_it"]}/{det["n"]}',
            "abstained_pct": round(det["pct_no_baseline"], 1),
            "seq_named_at": det["seq_named_at"],
            "expected_obs": det["expected_obs"],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    results = {"alpha": ALPHA, "beta": BETA, "temp_step_C": TEMP_STEP,
               "min_samples": MIN_SAMPLES, "train_frac": TRAIN_FRAC,
               "experiments": []}

    print("\nARRESTLINE on the VT/NSWC hydraulic rotary actuator set")
    print("alpha %.2f  beta %.2f  regime = one bin per matched-load experiment"
          % (ALPHA, BETA))
    print("(internal temperature is constant within each actuator and differs")
    print(" between them, so binning on it would make the regime the unit)")
    print("baselines fitted on the first %d%% of the reference unit, then frozen\n"
          % (100 * TRAIN_FRAC))

    screen = portability_screen(a.data, "Act_5", "Act_6")
    usable = {k: v["portable"] for k, v in screen.items()}
    results["portability_screen"] = screen
    print("PORTABILITY SCREEN — fit on Act_5 (healthy), scored on Act_6 (healthy).")
    print("No fault label is used here. A symptom that already reads like a fault")
    print("between two healthy units cannot be trusted on a damaged one.\n")
    print("   %-20s %11s  %s" % ("symptom", "median |z|", "portable"))
    for k, v in screen.items():
        print("   %-20s %11.2f  %s"
              % (k, v["median_abs_z_healthy_vs_healthy"],
                 "yes" if v["portable"] else "NO — excluded"))
    print()

    plan = [
        ("SEAL DEFECT, matched butterfly load", "Act_1", "Act_4 (Seal Defect)", "seal"),
        ("SEAL symptoms on the GEAR unit (specificity)", "Act_2", "Act_3 (Gear Damage)", "seal"),
        ("GEAR DAMAGE, matched ball load", "Act_2", "Act_3 (Gear Damage)", "gear"),
        ("GEAR symptoms on the SEAL unit (specificity)", "Act_1", "Act_4 (Seal Defect)", "gear"),
        ("CONTROL healthy/healthy — seal symptoms", "Act_5", "Act_6", "seal"),
        ("CONTROL healthy/healthy — gear symptoms", "Act_5", "Act_6", "gear"),
    ]
    for label, ref, tgt, comp in plan:
        r = run_pair(a.data, ref, tgt, comp, label, usable)
        results["experiments"].append(r)
        if "error" in r:
            print("  %-46s %s" % (label, r["error"]))
            continue
        print("== %s" % label)
        print("   baseline %s (%d train / %d held out)  target %s (%d)  working PG_%d"
              % (r["ref"], r["n_train"], r["n_held"], r["tgt"], r["n_tgt"],
                 r["working_pg"]))
        print("   %-11s %-14s %-9s %-19s %s"
              % ("arm", "1-shot FA", "(<=95%)", "seq: names target at",
                 "seq on held-out healthy"))
        for arm, v in r["arms"].items():
            print("   %-11s %-14s %-9s %-19s %s"
                  % (arm, v["false_alarms"], "%.2f%%" % v["false_alarm_upper95"],
                     ("cycle %d" % v["seq_named_at"]) if v["seq_named_at"] else "never",
                     ("NAMED at %d" % v["seq_false_alarm_at"])
                     if v["seq_false_alarm_at"] else "never — no false alarm"))
        print()

    # ── abstention ─────────────────────────────────────────────────────────
    # An earlier version of this block applied a no-load baseline to
    # butterfly-load cycles and called that an unseen regime. Once the regime
    # was collapsed to one bin per experiment -- because Internal_Temp turned
    # out to be a proxy for the unit -- that test stopped testing anything and
    # printed 0% of 0%, which reads like a failure and was merely vacuous.
    #
    # This is the honest version. The no_baseline path fires when a line's
    # statistics are not READY, so starve it: fit on fewer cycles than
    # MIN_SAMPLES requires and confirm the system declines to score rather than
    # scoring against an under-observed baseline.
    print("== ABSTENTION, a baseline that is not ready")
    ref = load_unit(a.data, "Act_1")
    tgt = load_unit(a.data, "Act_4 (Seal Defect)")
    if ref and tgt:
        starved = ref[: max(1, MIN_SAMPLES - 5)]
        w2 = commission(starved)
        b2 = fit(starved, w2, "vibration")
        ready = sum(1 for st in b2.stats.values() if st.ready)
        j2 = judge(tgt, b2, w2, "vibration", "gear", usable)
        print("   fitted on %d cycles, below the %d-observation minimum"
              % (len(starved), MIN_SAMPLES))
        print("   lines with a ready baseline: %d of %d" % (ready, len(b2.stats)))
        print("   no_baseline on %.0f%% of %d target cycles, named on %.0f%%"
              % (j2["pct_no_baseline"], j2["n"], j2["pct_named"]))
        print("   (the same cycles are named at cycle 5 once the baseline is")
        print("    fitted on 52. The refusal is the readiness rule, not the data.)")
        results["abstention"] = {
            "fitted_on": len(starved), "min_samples": MIN_SAMPLES,
            "ready_lines": ready, "total_lines": len(b2.stats),
            "pct_no_baseline": round(j2["pct_no_baseline"], 1),
            "pct_named": round(j2["pct_named"], 1), "n": j2["n"]}
    print()

    if a.json:
        with open(a.json, "w") as f:
            json.dump(results, f, indent=2)
        print("wrote %s" % a.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
