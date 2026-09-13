#!/usr/bin/env python3
"""Run cbmx's governance layer over the ZeMA hydraulic rig and report honestly.

Protocol, fixed before the first run:

  Commissioning.  Channel roles (which numbered sensor is the working line, the
    switched line, the hot and cold temperature probes) are resolved once from
    the first 200 cycles, using unlabelled statistics only. No profile row is
    read at commissioning. This mirrors the field: an installer knows there are
    six pressure transducers and not which is which.

  Baseline.  Per component, the healthy pool is the cycles where THAT component
    is at its optimal condition and the rig reported the cycle settled. Even
    indices fit the baseline; odd indices are held out. Splitting on parity
    rather than on time matches the regime coverage of the two halves, which is
    what makes the false-alarm number comparable — and it means adjacent, nearly
    identical cycles land on both sides. That is a real optimism and it is
    reported as such: the time-split run below is the pessimistic bound.

  Scoring.  Every cycle not used for fitting is streamed in time order. Each
    component's sequential test accumulates across cycles with a leak, exactly
    as it would across arrestments. Nothing tells the test when a condition
    changed.

  Freezing.  A baseline keeps learning online while its component reads
    'ordinary', and stops the moment the test moves to 'watching'. Without that
    a slow degradation is absorbed as the new normal. The --no-freeze run
    measures how much that matters here rather than asserting it.

What is deliberately NOT done, because it would make the numbers better and the
result meaningless: no symptom is selected by how well it separates the labels,
no threshold is tuned against the labels, and the virtual CE/CP/SE channels —
which the rig computes from the temperatures the cooler fault moves — are never
read.

Usage:
    PYTHONPATH=src python3 tools/eval_hydraulic.py [--no-freeze] [--time-split]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.io.hydraulic import (COMPONENTS, N_CYCLES, HydraulicRig)   # noqa: E402
from cbmx.physics.hydraulic import (BY_COMPONENT, SYMPTOMS, ChannelMap,  # noqa: E402
                                    compute_matrix, regime_variables)
from cbmx.health.baseline import Baseline, Regime                     # noqa: E402
from cbmx.health.sequential import WaldAccumulator, build_channels    # noqa: E402

COMMISSIONING_CYCLES = 200
TEMP_BIN_C = 3.0
MIN_SAMPLES = 25
ALPHA = 0.02          # tolerated false-alarm rate
BETA = 0.10           # tolerated miss rate


def robust_center_scale(x: np.ndarray) -> tuple:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0, 1.0
    c = float(np.median(x))
    s = float(np.median(np.abs(x - c)) * 1.4826)
    return c, (s if s > 1e-12 else 1.0)


def regime_for(component: str, temp: float) -> Regime:
    """Oil temperature is the nuisance variable for three of the four.

    For the cooler it is downstream of the fault, so binning on it would file
    every cooler fault into an unseen regime and the system would abstain on
    exactly the cases it is meant to catch. The cooler therefore runs in a
    single regime and carries the temperature nuisance the others do not.
    """
    if component == "cooler":
        return Regime(0, 0)
    return Regime(int(temp // TEMP_BIN_C), 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hydraulic")
    ap.add_argument("--no-freeze", action="store_true",
                    help="let baselines keep learning through a developing fault")
    ap.add_argument("--time-split", action="store_true",
                    help="fit on the first 60%% of each healthy pool by cycle "
                         "index instead of on even indices — pessimistic, no "
                         "adjacent-cycle leakage")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    rig = HydraulicRig(a.root)
    prof = rig.profile
    print(rig.summary())
    print()

    # ── commissioning ───────────────────────────────────────────────────────
    commissioning = np.arange(COMMISSIONING_CYCLES)
    ch_map = ChannelMap.resolve(rig, commissioning)
    print(f"commissioning on cycles 0-{COMMISSIONING_CYCLES-1}, labels unread")
    print(f"  {ch_map.describe()}")
    dead = rig.dead_channels()
    print(f"  dead-channel check: {dead if dead else 'none'}")
    print()

    # ── symptoms ────────────────────────────────────────────────────────────
    print("computing symptoms over all cycles ...")
    M = compute_matrix(rig, ch_map, N_CYCLES, progress_every=500)
    temps = regime_variables(rig, N_CYCLES, ch_map)
    sym_idx = {s.id: j for j, s in enumerate(SYMPTOMS)}
    direction = {s.id: s.direction for s in SYMPTOMS}
    primary = {s.id: s.primary for s in SYMPTOMS}
    print(f"  {M.shape[1]} symptoms x {M.shape[0]} cycles; "
          f"{int(np.isnan(M).sum())} undefined values")
    print(f"  oil temperature {temps.min():.1f}-{temps.max():.1f} C -> "
          f"{len(set((temps // TEMP_BIN_C).astype(int)))} regimes at "
          f"{TEMP_BIN_C:.0f} C bins")
    print()

    stable = prof.stable
    results = {}

    for comp in COMPONENTS:
        syms = BY_COMPONENT[comp]
        pool = np.where(prof.healthy_mask(comp) & stable)[0]
        if a.time_split:
            cut = int(0.6 * pool.size)
            fit_idx, held_healthy = pool[:cut], pool[cut:]
        else:
            fit_idx, held_healthy = pool[0::2], pool[1::2]
        fit_set = set(fit_idx.tolist())

        # standardise each symptom on the fit pool only
        cs = {}
        for s in syms:
            c, sc = robust_center_scale(M[fit_idx, sym_idx[s.id]])
            cs[s.id] = (c, sc)

        base = Baseline(f"hydraulic/{comp}", min_samples=MIN_SAMPLES)
        for i in fit_idx:
            r = regime_for(comp, temps[i])
            for s in syms:
                v = M[i, sym_idx[s.id]]
                if np.isfinite(v):
                    c, sc = cs[s.id]
                    base.observe(s.id, r, (v - c) / sc)

        # ── streaming test pass ────────────────────────────────────────────
        wald = WaldAccumulator(ALPHA, BETA)
        sev = prof.severity_series(comp)
        per_sev = defaultdict(lambda: defaultdict(int))
        held_set = set(held_healthy.tolist())
        decisions = np.empty(N_CYCLES, dtype=object)
        decisions[:] = ""
        llr_by_sev = defaultdict(list)

        for i in range(N_CYCLES):
            if i in fit_set:
                continue
            r = regime_for(comp, temps[i])
            zs = {}
            for s in syms:
                v = M[i, sym_idx[s.id]]
                if not np.isfinite(v):
                    continue
                c, sc = cs[s.id]
                z, run = base.score(s.id, r, (v - c) / sc)
                if run.ready:
                    zs[s.id] = z

            ch = build_channels(zs, direction, primary)
            j = wald.observe(comp, ch)
            decisions[i] = j.decision

            # online learning, and the freeze that stops a fault teaching the
            # system that it is normal
            if ch is not None:
                learning = a.no_freeze or j.decision == "ordinary"
                if learning:
                    for s in syms:
                        v = M[i, sym_idx[s.id]]
                        if np.isfinite(v):
                            c, sc = cs[s.id]
                            base.observe(s.id, r, (v - c) / sc)
                elif not a.no_freeze:
                    for s in syms:
                        base.freeze(s.id, r)

            bucket = "healthy_heldout" if (sev[i] == 0 and i in held_set) else (
                "healthy_other" if sev[i] == 0 else f"sev{sev[i]}")
            per_sev[bucket][j.decision] += 1
            per_sev[bucket]["n"] += 1
            if ch is not None:
                llr_by_sev[int(sev[i])].append(j.channels.get("llr", 0.0))

        cov = base.coverage()      # after the pass, so 'frozen' means something

        # ── episodes ───────────────────────────────────────────────────────
        # The rig switches the valve every 10 cycles and the pump every 41.
        # A sequential test is built to require persistent evidence, so scoring
        # it per cycle against a condition that flips faster than its own
        # decision time measures the mismatch, not the method. The episode is
        # the unit the operational question is asked in: during this stretch of
        # degraded valve, did the system ever name it, and during this stretch
        # of healthy valve, did it stay quiet.
        eps = []
        i = 0
        while i < N_CYCLES:
            j0 = i
            while i < N_CYCLES and sev[i] == sev[j0]:
                i += 1
            hit = [k for k in range(j0, i) if decisions[k] == "name_it"]
            eps.append({"sev": int(sev[j0]), "start": j0, "len": i - j0,
                        "named": bool(hit),
                        "latency": (hit[0] - j0) if hit else None})
        fault_eps = [e for e in eps if e["sev"] > 0]
        healthy_eps = [e for e in eps if e["sev"] == 0]
        caught = [e["latency"] for e in fault_eps if e["named"]]

        # ── evidence rate: how fast would this decide, per severity ────────
        wu = WaldAccumulator(ALPHA, BETA).upper
        rate = {}
        for s_, vals in sorted(llr_by_sev.items()):
            m = float(np.mean(vals)) if vals else 0.0
            rate[s_] = {"mean_llr": m,
                        "cycles_to_decide": (wu / m) if m > 1e-6 else None,
                        "n": len(vals)}

        results[comp] = {
            "fit_cycles": int(fit_idx.size),
            "heldout_healthy": int(held_healthy.size),
            "coverage": cov,
            "buckets": {k: dict(v) for k, v in per_sev.items()},
            "evidence_rate": rate,
            "episodes_fault": len(fault_eps),
            "episodes_fault_named": sum(e["named"] for e in fault_eps),
            "episodes_healthy": len(healthy_eps),
            "episodes_healthy_with_alarm": sum(e["named"] for e in healthy_eps),
            "median_episode_len": int(np.median([e["len"] for e in eps])),
            "median_latency_cycles": (int(np.median(caught)) if caught else None),
        }

        # ── report ─────────────────────────────────────────────────────────
        print(f"── {comp} " + "─" * (66 - len(comp)))
        print("   symptoms: " + ", ".join(
            f"{s.id}({'+' if s.direction > 0 else '-'}"
            f"{'' if s.primary else ',aux'})" for s in syms))
        print(f"   baseline fit on {fit_idx.size} healthy settled cycles; "
              f"{held_healthy.size} held out")
        print(f"   lines {cov['lines_ready']}/{cov['lines_tracked']} ready "
              f"across {cov['regimes']} regimes, {cov['lines_frozen']} frozen")
        order = ["healthy_heldout", "healthy_other"] + [
            f"sev{k}" for k in sorted({v for v in COMPONENTS[comp]['severity'].values() if v})]
        print(f"   {'condition':<18}{'n':>6}{'name_it':>10}{'watching':>10}"
              f"{'ordinary':>10}{'no_base':>10}{'llr/cyc':>9}{'->decide':>10}")
        for b in order:
            d = per_sev.get(b)
            if not d:
                continue
            n = d["n"]
            sv = 0 if b.startswith("healthy") else int(b[3:])
            r = rate.get(sv, {})
            ctd = r.get("cycles_to_decide")
            print(f"   {b:<18}{n:>6}"
                  f"{d.get('name_it',0)/n:>9.1%}"
                  f"{d.get('watching',0)/n:>10.1%}"
                  f"{d.get('ordinary',0)/n:>10.1%}"
                  f"{d.get('no_baseline',0)/n:>10.1%}"
                  f"{r.get('mean_llr',0.0):>9.2f}"
                  + (f"{ctd:>9.0f}c" if ctd and ctd < 1e4 else f"{'never':>10}"))
        print(f"   episodes: fault {results[comp]['episodes_fault_named']}"
              f"/{results[comp]['episodes_fault']} named · healthy "
              f"{results[comp]['episodes_healthy_with_alarm']}"
              f"/{results[comp]['episodes_healthy']} raised an alarm · "
              f"median episode {results[comp]['median_episode_len']} cycles")
        print(f"   median latency where named: "
              f"{results[comp]['median_latency_cycles']} cycles")
        print()

    print("=" * 74)
    print(f"alpha={ALPHA} beta={BETA} -> Wald boundaries "
          f"+{WaldAccumulator(ALPHA, BETA).upper:.2f} / "
          f"{WaldAccumulator(ALPHA, BETA).lower:.2f} nats")
    print(f"freeze: {'OFF (ablation)' if a.no_freeze else 'on'}   "
          f"split: {'time (pessimistic)' if a.time_split else 'parity'}")
    print("false-alarm rate to read is the name_it column on healthy_heldout.")

    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=2, default=str))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
