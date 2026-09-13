#!/usr/bin/env python3
"""From detection toward prognosis — and an honest false-alarm bound.

Two things a technical reviewer will probe, addressed together because both are
about what the numbers are allowed to mean.

═══ PART A — does the health indicator carry severity information? ═══

Detection answers "is something wrong". Prognosis needs more: the indicator has
to move MONOTONICALLY with how wrong, or there is nothing to extrapolate along.
The ZeMA rig grades every component, so this is directly measurable rather than
assumed: cooler at three grades, valve at four, pump at three, accumulator at
four.

If the indicator is monotone in true severity, then a horizon estimate is a
matter of estimating a rate and projecting to a boundary — ordinary engineering.
If it is flat or non-monotone, no amount of curve fitting will produce a
trustworthy remaining-useful-life number, and saying so is worth more than a
plotted line.

What is NOT claimed here: a validated remaining-useful-life model. The ZeMA rig
sets conditions in blocks rather than degrading continuously in time, so there
is no ground-truth failure trajectory to score a horizon against. What can be
established from this data is the precondition for prognosis and the arithmetic
that follows from it, with the trajectory assumption stated in the open.

═══ PART B — what the false-alarm rate is actually entitled to say ═══

The evidence increment combined into the sequential test is a bounded score over
three channels, scaled to nats. It is NOT a calibrated log-likelihood ratio
estimated from fitted healthy and faulty densities. That distinction matters and
has been stated loosely elsewhere:

    Wald's boundaries, log((1-beta)/alpha) and log(beta/(1-alpha)), deliver the
    nominal error rates alpha and beta only when the accumulated quantity really
    is a log-likelihood ratio under the two hypotheses.

Ours is a monotone evidence score with the same sign convention and bounded
increments. So alpha and beta here set a defensible, statable OPERATING POINT —
they are the two knobs a maintainer argues about, and everything about when the
system speaks follows from them — but they do not by themselves guarantee a 2%
operational false-alarm probability.

The honest thing is to report the MEASURED false-alarm rate with a proper
interval. Observing zero false alarms in 80 records does not establish a rate
below 2%; by the rule of three the 95% upper bound on 0/80 is about 3.7%. This
part computes Clopper-Pearson bounds for every healthy set we have measured, so
the volume can quote an interval rather than a point.

Usage:
    PYTHONPATH=src python3 tools/eval_prognosis.py
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.io.hydraulic import COMPONENTS, N_CYCLES, HydraulicRig     # noqa: E402
from cbmx.physics.hydraulic import (BY_COMPONENT, SYMPTOMS,          # noqa: E402
                                    ChannelMap, compute_matrix)
from cbmx.health.sequential import WaldAccumulator                   # noqa: E402


# ── Part B helpers ──────────────────────────────────────────────────────────

def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """Exact ONE-SIDED upper bound on a binomial rate.

    One-sided deliberately. The claim being made is "the false-alarm rate is
    below X", which is a one-sided question; quoting the upper end of a
    two-sided 95% interval answers a different question and inflates the number
    by about a fifth. At k=0 this reduces to the rule of three, 3/n, which is
    where the familiar 3.7% for 0/80 comes from.

    Exact rather than normal-approximated, because at k=0 the normal
    approximation is not merely inaccurate — it returns zero width, which is the
    overclaim this function exists to prevent.
    """
    if n == 0:
        return 1.0
    if k == n:
        return 1.0
    return _beta_ppf(conf, k + 1, n - k)


def _binom_cdf_le(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), summed in log space.

    Exact and stable for the small k this module sees. It exists because the
    series expansion of the incomplete beta that used to back the fallback
    diverges for the (a, b) this module actually asks for: at a=1, b=180 it
    returned a clamped garbage value for mid-range x, which sent the bisection
    the wrong way and produced a 93% upper bound where the answer is 1.65%.
    That was silent, and it was wrong only on some inputs, which is worse.
    """
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    from math import lgamma, log, log1p, exp
    lnC = lgamma(n + 1)
    tot = 0.0
    for i in range(k + 1):
        tot += exp(lnC - lgamma(i + 1) - lgamma(n - i + 1)
                   + i * log(p) + (n - i) * log1p(-p))
    return min(1.0, tot)


def _cp_upper_exact(k: int, n: int, conf: float) -> float:
    """Solve P(X <= k | n, p) = 1 - conf for p. Monotone decreasing in p."""
    target = 1.0 - conf
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if _binom_cdf_le(k, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _beta_ppf(p: float, a: float, b: float) -> float:
    """Upper bound via the binomial identity, cross-checked against scipy.

    k = a - 1 and n = a + b - 1 recovers the Clopper-Pearson parameters.
    When scipy is present both routes are computed and must agree; a
    disagreement stops the run rather than picking one, because this function
    produces numbers that are quoted as bounds in a proposal.
    """
    k, n, conf = int(round(a - 1)), int(round(a + b - 1)), p
    ours = _cp_upper_exact(k, n, conf)
    try:
        from scipy.stats import beta as _sb
        theirs = float(_sb.ppf(p, a, b))
    except Exception:                                    # noqa: BLE001
        return ours
    if abs(ours - theirs) > 1e-6:
        sys.exit(f"BOUND DISAGREEMENT for k={k} n={n} conf={conf}: "
                 f"binomial identity {ours!r} vs scipy {theirs!r}. "
                 f"Stopping rather than quoting either.")
    return theirs


def robust(v: np.ndarray):
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan"), float("nan")
    c = float(np.median(v))
    s = float(np.median(np.abs(v - c)) * 1.4826)
    return c, (s if s > 1e-12 else 1.0)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _need(path_str: str, what: str) -> Path:
    """A missing artifact stops the run and says which flag would supply it."""
    if not path_str:
        sys.exit(f"REFUSING TO GUESS: {what} was not given.\n"
                 f"Part B reads measured counts from run artifacts. It has no "
                 f"literals left and no defaults.\nSupply the file and re-run.")
    p = Path(path_str)
    if not p.exists():
        sys.exit(f"REFUSING TO GUESS: {what} not found at {p}\n"
                 f"Run the experiment that writes it, then re-run this.")
    return p


def read_false_alarms(a):
    """Every false-alarm count, read from the run that produced it.

    Replaces a table of hardcoded literals found on 13 September 2026. Those
    literals reached the paper and the volume through a hashed result file and
    were quoted as measurements; one was internally inconsistent with its own
    component rows and two matched no run on the tree. See
    08_Analysis/paper/BLOCKER_false_alarm_table_2026-09-13.md.

    Definition, applied identically everywhere: a false alarm is a NAME IT
    verdict on healthy material. WATCHING and NO BASELINE are not alarms. The
    NO BASELINE count is carried alongside because whether a refusal belongs in
    the denominator is a real question and the reader should be able to see it.
    """
    rows, sources = [], {}

    hyd = _need(a.hydraulic, "the hydraulic run artifact (--hydraulic)")
    blob = json.loads(hyd.read_text())
    sources["hydraulic"] = {"path": str(hyd), "sha256_16": _sha(hyd)}
    pooled_k = pooled_n = pooled_nb = 0
    for comp in sorted(blob):
        b = blob[comp].get("buckets", {}).get("healthy_heldout")
        if b is None or "n" not in b:
            sys.exit(f"REFUSING TO GUESS: {hyd} has no "
                     f"buckets.healthy_heldout.n for component {comp!r}. "
                     f"Re-run eval_hydraulic.py; do not hand-fill this.")
        k, n = int(b.get("name_it", 0)), int(b["n"])
        nb = int(b.get("no_baseline", 0))
        pooled_k += k; pooled_n += n; pooled_nb += nb
        rows.append((f"Hydraulic {comp}, held-out healthy cycles", k, n,
                     f"hydraulic:{comp}"))
    # Every component, and the count of components, both come from the file.
    rows.append((f"Hydraulic, all {len(blob)} components pooled",
                 pooled_k, pooled_n, "hydraulic:pooled"))
    sources["hydraulic"]["components"] = sorted(blob)
    sources["hydraulic"]["no_baseline_cycles_excluded_from_k"] = pooled_nb

    print(f"   [hydraulic] {hyd}")
    print(f"               sha256:{sources['hydraulic']['sha256_16']}  "
          f"components: {', '.join(sorted(blob))}\n")

    pad_pairs = [] if a.skip_paderborn else [
        (a.paderborn_vib, "Paderborn vibration, healthy records", "paderborn_vib"),
        (a.paderborn_cur, "Paderborn motor current, banded, healthy", "paderborn_cur"),
    ]
    if a.skip_paderborn:
        sources["paderborn"] = {
            "status": "NOT RUN ON THIS HOST",
            "note": "no fleet artifact was available; no count is reported for "
                    "either Paderborn channel. This is an absence, not a zero.",
        }
        print("   " + "=" * 68)
        print("   PADERBORN: NOT RUN ON THIS HOST")
        print("   " + "-" * 68)
        print("   No fleet artifact was available, so NO COUNT is reported for")
        print("   either Paderborn channel, vibration or motor current.")
        print("   The evidence file records this as an ABSENCE, not a zero.")
        print("   The two withdrawn literals, 0/80 vibration and 0/74 current,")
        print("   are NOT reinstated and NOT replaced. They stay unreported")
        print("   until the fleet runs on a host that has the data.")
        print("   " + "=" * 68 + "\n")

    for flag, label, key in pad_pairs:
        f = _need(flag, f"the {key} fleet artifact (--{key.replace('_','-')})")
        d = json.loads(f.read_text())
        for need in ("false_alarm_records", "healthy_records"):
            if need not in d:
                sys.exit(f"REFUSING TO GUESS: {f} has no {need!r}. "
                         f"Top-level keys present: {sorted(d)}")
        k, n = int(d["false_alarm_records"]), int(d["healthy_records"])
        band = d.get("band_hz")
        rows.append((label, k, n, key))
        sources[key] = {"path": str(f), "sha256_16": _sha(f), "band_hz": band,
                        "folds": len(d.get("folds", []))}
        print(f"   [{key}] band {band}, {len(d.get('folds', []))} held-out bearings")

    # What the withdrawn table claimed, printed beside what was measured, so the
    # run log itself records the correction rather than a later note asserting it.
    WITHDRAWN = {"Paderborn vibration, healthy records": (0, 80),
                 "Paderborn motor current, banded, healthy": (0, 74),
                 "Hydraulic cooler, held-out healthy cycles": (0, 244),
                 "Hydraulic pump, held-out healthy cycles": (2, 244),
                 "Hydraulic accumulator, held-out healthy cycles": (0, 180)}
    print("\n   withdrawn literal -> regenerated, for the run log:")
    for lab, k, n, _ in rows:
        if lab in WITHDRAWN:
            ok, on = WITHDRAWN[lab]
            flag = "same" if (ok, on) == (k, n) else "CHANGED"
            print(f"   {lab:<46}{f'{ok}/{on}':>10} -> {f'{k}/{n}':<10}{flag}")
        else:
            print(f"   {lab:<46}{'(new)':>10} -> {f'{k}/{n}':<10}")
    print()
    return rows, sources


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hydraulic")
    ap.add_argument("--json", default="")
    # Part B reads run artifacts. It has no default and no fallback on purpose:
    # the defect this replaced was a table of literals that looked like a
    # measurement all the way through the evidence pack. A missing file must
    # stop the run, not quietly produce a number.
    ap.add_argument("--hydraulic", default="",
                    help="hydraulic_freeze.json from eval_hydraulic.py")
    ap.add_argument("--paderborn-vib", default="",
                    help="Paderborn vibration fleet JSON from eval_dataset.py "
                         "(--holdout bearing)")
    ap.add_argument("--paderborn-cur", default="",
                    help="Paderborn motor-current fleet JSON, same shape")
    # Absence recorded, never inferred. Without this flag a missing Paderborn
    # artifact stops the run, which is the default and the right default. With
    # it the run proceeds and evidence.json says in writing which datasets were
    # not on this host, so a reader can tell "not measured" from "measured zero".
    ap.add_argument("--skip-paderborn", action="store_true",
                    help="emit the hydraulic half alone and record in the "
                         "evidence file that the Paderborn fleet did not run here")
    ap.add_argument("--evidence", default="",
                    help="write the canonical evidence JSON here. The paper, "
                         "the volume and the certification pack all read this "
                         "one file, so they cannot drift apart again.")
    a = ap.parse_args()

    rig = HydraulicRig(a.root)
    prof = rig.profile
    ch = ChannelMap.resolve(rig, np.arange(200))
    M = compute_matrix(rig, ch, N_CYCLES)
    idx = {s.id: j for j, s in enumerate(SYMPTOMS)}
    stable = prof.stable

    wald = WaldAccumulator(0.02, 0.10)
    print("═══ PART A — health indicator against true severity ═══\n")
    print("Indicator = the component's primary symptom, direction-corrected and")
    print("standardised on that component's own healthy, settled cycles.")
    print("Positive means 'further from healthy, in the direction physics says'.\n")

    out = {}
    for comp in COMPONENTS:
        prim = [s for s in BY_COMPONENT[comp] if s.primary][0]
        col = M[:, idx[prim.id]]
        healthy = prof.healthy_mask(comp) & stable
        c, s = robust(col[healthy])

        sev = prof.severity_series(comp)
        grades = sorted(set(sev.tolist()))
        row, spread = {}, {}
        for g in grades:
            m = (sev == g) & stable
            v = (col[m] - c) / s * prim.direction
            row[g] = float(np.nanmedian(v))
            spread[g] = float(np.nanmedian(np.abs(v - np.nanmedian(v))) * 1.4826)

        mono = all(row[grades[i + 1]] >= row[grades[i]] for i in range(len(grades) - 1))
        print(f"── {comp}   primary symptom {prim.id}   "
              f"{'MONOTONE' if mono else 'NOT MONOTONE'}")
        print(f"   {'severity grade':<18}" + "".join(f"{g:>10}" for g in grades))
        print(f"   {'indicator (sigma)':<18}" + "".join(f"{row[g]:>10.2f}" for g in grades))
        print(f"   {'spread (sigma)':<18}" + "".join(f"{spread[g]:>10.2f}" for g in grades))

        # ── the horizon arithmetic, with the assumption stated ─────────────
        # Take the per-grade indicator as the degradation path. If a machine
        # walks that path at a stated rate, when does the evidence accumulator
        # reach its boundary? The rate is the assumption; everything else is
        # measured.
        step = (row[grades[-1]] - row[grades[0]]) / max(1, len(grades) - 1)
        # evidence per cycle at a given indicator level, from the same channel
        # weights the detector uses

        def llr_at(z):
            level = max(-1.0, min(1.5, (z - 3.0) / 3.0))
            return 0.55 * (1.0 * level + 0.45 * (0.8 if z > 0 else -0.7))
        horizon = None
        if step > 0:
            S, cyc, z = 0.0, 0, row[grades[0]]
            per_grade_cycles = 200          # ASSUMPTION, stated
            while cyc < 20000 and S < wald.upper:
                z += step / per_grade_cycles
                S = max(wald.lower - 2, S * 0.97 + llr_at(z))
                cyc += 1
            horizon = cyc if S >= wald.upper else None
        noise_cycles = None
        if step > 0:
            # uncertainty: how many cycles of indicator drift are swamped by the
            # within-grade spread? This is the resolution floor of any horizon.
            noise_cycles = float(np.median(list(spread.values())) /
                                 (step / 200))
        print(f"   indicator gain per severity grade   {step:>8.2f} sigma")
        if horizon:
            print(f"   cycles to evidence boundary          {horizon:>8,d}   "
                  f"(assumes 200 cycles per grade)")
        else:
            print(f"   cycles to evidence boundary          {'never':>8}   "
                  f"— indicator does not rise enough to decide")
        if noise_cycles and np.isfinite(noise_cycles):
            print(f"   resolution floor from within-grade spread "
                  f"{noise_cycles:>6.0f} cycles")
        print()
        out[comp] = {"symptom": prim.id, "monotone": bool(mono),
                     "indicator_by_grade": row, "spread_by_grade": spread,
                     "gain_per_grade_sigma": step,
                     "cycles_to_boundary": horizon,
                     "resolution_floor_cycles": noise_cycles,
                     "assumption": "200 cycles per severity grade; ZeMA sets "
                                   "conditions in blocks, so no ground-truth "
                                   "failure trajectory exists to score against"}

    print("═══ PART B — what the false-alarm numbers are entitled to say ═══\n")
    print("Wald boundaries set an OPERATING POINT. The accumulated quantity is a")
    print("bounded three-channel evidence score, not a calibrated log-likelihood")
    print("ratio from fitted densities, so alpha does not by itself guarantee an")
    print("operational false-alarm probability. Measured rates, with exact")
    print("binomial intervals, are what may be quoted.\n")
    observed, sources = read_false_alarms(a)
    print(f"   {'set':<46}{'k/n':>10}{'observed':>10}{'95% upper':>11}")
    fa = {}
    for lab, k, n, src in observed:
        hi = clopper_pearson_upper(k, n)
        print(f"   {lab:<46}{f'{k}/{n}':>10}{k/n:>10.3%}{hi:>11.2%}")
        fa[lab] = {"k": k, "n": n, "observed": k / n,
                   "upper95_onesided": hi, "source": src}
    print("\n   Zero observed false alarms in 80 records is consistent with a true")
    print("   rate anywhere below about 3.7%. That is evidence of a low rate, not")
    print("   proof of one below the 2% design point. Only the hydraulic sets,")
    print("   with hundreds of held-out healthy cycles, bound the rate below the")
    print("   design point at all — and the time-split set is the strongest of")
    print("   them. The volume quotes the bound, never the point estimate alone.")

    if a.evidence:
        ev = {
            "generated_by": "tools/eval_prognosis.py Part B",
            "generated_utc": _dt.datetime.now(_dt.timezone.utc)
                              .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "definition": "a false alarm is a NAME IT verdict on healthy "
                          "material; WATCHING and NO BASELINE are not alarms",
            "sources": sources,
            "false_alarm": fa,
        }
        Path(a.evidence).parent.mkdir(parents=True, exist_ok=True)
        Path(a.evidence).write_text(json.dumps(ev, indent=2))
        print(f"\n   canonical evidence written to {a.evidence}")
        print("   The paper, Volume 2 and the certification pack read this file.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"prognosis": out, "false_alarm": fa}, indent=2, default=str))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
