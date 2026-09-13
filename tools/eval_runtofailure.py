#!/usr/bin/env python3
"""Run-to-failure: does the indicator move, and how much warning does it give?

    PYTHONPATH=src python3 tools/eval_runtofailure.py --dataset ims --data data/ims --probe
    PYTHONPATH=src python3 tools/eval_runtofailure.py --dataset ims --data data/ims

Every other evaluation in this repository asks "is this bearing damaged?" This
one asks the question the topic actually poses second — *predict possible
failures* — and it is a different question with different ways of being wrong.

**What this data gives, and what it does not.** The headline is a lead time —
how much life remained when the system first spoke — reported beside a
monotonicity, which is whether the indicator climbs as life is consumed or
merely jumps at the end.

It also gives something the earlier draft of this file said it could not: a
control. On NASA/IMS, four bearings share one shaft and one test, and the
readme names which one failed. Twelve bearings across three tests, four
documented failures, **eight that survived**. So a call on a surviving bearing
is scored and reported, and the count appears next to the lead time exactly as
false alarms appear next to detection everywhere else in this repository.

That control is weaker than Paderborn's and is labelled as such wherever it is
reported. A surviving bearing sits on the same shaft as a failing one, so late
in a test it is exposed to structural transmission from a defect metres of steel
away. A late call on a control may be cross-talk rather than a false alarm; a
call *before* the failing bearing's own call cannot be. Both are printed, with
their timing, rather than collapsed into one number that hides the difference.

**The trap this file is built to avoid.** An indicator that rises steeply in the
last two captures is worthless for maintenance planning and will still score
beautifully on any "did it detect the failure" metric. So detection is never
reported without the lead time beside it, exactly as detection is never reported
without false alarms elsewhere in this repository.

**Two halves, and the second one may not run.**

*Geometry-free.* RMS, kurtosis and spectral crest per capture. These need no
bearing dimensions and no fault labels. They answer: does anything move, when
does it start moving, and how much warning does that give. This is the standard
prognostic health-indicator construction and it always runs.

*Declared physics.* Prominence at the orders the bearing geometry predicts, fed
to the same Wald accumulator every other evaluation here uses. This is the half
that makes it *our* method rather than generic condition indicators — and on
this dataset it does not run. The IMS readme never publishes the Rexnord
ZA-2115's dimensions, and `PROGNOSIS_STOPPING_RULE.md`, written before the data
was downloaded, forbids inventing them. Without `--bearing` the half is skipped
and said to be skipped. Part naming is Paderborn's claim, not this one's.

**The baseline is the bearing's own early life, frozen.** That is the deployment
story and not a convenience: a monitor is commissioned on an asset believed
healthy, learns what that asset's normal looks like, and is frozen so that slow
degradation cannot teach it that degradation is normal. The freeze matters more
here than anywhere else in this repository, because run-to-failure data is
precisely where an unfrozen baseline would follow the machine down.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.io import ims                                             # noqa: E402
try:                                                                # noqa: E402
    from cbmx.io import xjtu
except ImportError:                     # the PRC-hosted alternative, not used
    xjtu = None
from cbmx.physics.bearing import get as get_bearing      # noqa: E402
from cbmx.health.baseline import Baseline, Regime                   # noqa: E402

PARTS = ("outer_race", "inner_race", "rolling_element", "cage")

TITLE = {"ims": "NASA/IMS run-to-failure", "xjtu": "XJTU-SY run-to-failure"}


# ── geometry-free health indicators ─────────────────────────────────────────
def indicators(x: np.ndarray, fs: float = ims.FS) -> Dict[str, float]:
    """Three numbers per capture, none of which knows what a bearing is.

    `rms` is energy and rises late. `kurtosis` is impulsiveness and classically
    rises *early* then falls back as spalling spreads and the signal becomes
    more Gaussian again — a well-known non-monotonicity, and a good reason never
    to build a prognostic on kurtosis alone. `crest_db` is the 99th percentile of
    the spectrum over its own median: discrete lines emerging out of broadband
    noise, which is what a developing defect looks like before it is loud.
    """
    v = np.asarray(x, dtype=np.float64)
    v = v - v.mean()
    rms = float(np.sqrt(np.mean(v * v)))
    s = v.std()
    kurt = float(np.mean((v / s) ** 4) - 3.0) if s > 1e-12 else 0.0
    w = np.hanning(v.shape[0])
    S = np.abs(np.fft.rfft(v * w))
    f = np.fft.rfftfreq(v.shape[0], d=1.0 / fs)
    m = (f > 200.0) & (f < fs / 2 * 0.95)
    band = S[m]
    crest = 20.0 * math.log10(max(float(np.percentile(band, 99)), 1e-15)
                              / max(float(np.median(band)), 1e-15))
    return {"rms": rms, "kurtosis": kurt, "crest_db": crest}


def spearman(a: List[float], b: List[float]) -> float:
    """Rank correlation, written out rather than imported, because the only
    thing wanted here is monotonicity and scipy's version brings a p-value that
    would be meaningless on a series this autocorrelated."""
    def rank(v):
        order = np.argsort(np.asarray(v, dtype=float))
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    d = math.sqrt(float(np.dot(ra, ra) * np.dot(rb, rb)))
    return float(np.dot(ra, rb) / d) if d > 0 else 0.0


def first_sustained(flags: List[bool], run: int = 3) -> Optional[int]:
    """Index of the first of `run` consecutive True values.

    A single crossing is noise; the question a maintainer asks is when the
    machine started saying it persistently. Requiring persistence is also what
    stops a lead time being measured from a one-capture spike that the system
    itself would not have acted on.
    """
    for i in range(len(flags) - run + 1):
        if all(flags[i:i + run]):
            return i
    return None


# ── the run ─────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", default="ims", choices=["ims", "xjtu"],
                    help="ims is NASA/University of Cincinnati, four bearings on "
                         "one shaft run to natural failure over days. xjtu is "
                         "the PRC-hosted alternative and is not used.")
    ap.add_argument("--probe", action="store_true",
                    help="print what is on disk and stop")
    ap.add_argument("--bearing", default=None,
                    help="catalogue key for the Rexnord ZA-2115; without it the "
                         "declared-physics half is skipped. The IMS readme does "
                         "not publish that geometry, so this is normally absent.")
    ap.add_argument("--band", nargs=2, type=float, default=[2000.0, 10000.0],
                    metavar=("LO", "HI"))
    ap.add_argument("--channel", default="horizontal")
    ap.add_argument("--baseline-frac", type=float, default=0.15,
                    help="fraction of each bearing's life used to learn normal")
    ap.add_argument("--scope", default="readme", choices=["readme", "full"],
                    help="readme (default): score only the recording window the "
                         "dataset's own documentation describes. full: score "
                         "every capture on disk. These differ only on set 3, "
                         "which ships 6,324 captures for an experiment the "
                         "readme documents as 4,448. See ims.clip_to_published.")
    ap.add_argument("--every", type=int, default=1,
                    help="score every Nth capture; 1 reads them all")
    ap.add_argument("--z-alarm", type=float, default=6.0,
                    help="robust sigmas defining a sustained rise, geometry-free half")
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=0.10)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    P = print
    M = ims if a.dataset == "ims" else xjtu
    if M is None:
        P("The xjtu loader is not installed in this checkout. That dataset is")
        P("hosted outside the United States and is deliberately not used; see")
        P("PROGNOSIS_STOPPING_RULE.md. Run with --dataset ims.")
        return 1
    found = M.scan(a.data)
    if a.probe or not found:
        P(M.describe(a.data))
        return 0 if found else 1

    bg = None
    if a.bearing:
        bg = get_bearing(a.bearing)
    P("=" * 78)
    P(f"{TITLE[a.dataset]} — lead time, monotonicity, and the controls")
    P("=" * 78)
    P(f"\n  {sum(len(v) for v in found.values())} bearings, "
      f"{sum(l.n_captures for v in found.values() for l in v.values())} captures, "
      f"{M.FS/1000:g} kHz")
    P(f"  channel {a.channel}   baseline = first {100*a.baseline_frac:.0f}% of each "
      f"bearing's own life, then FROZEN")
    if bg:
        P(f"\n  declared geometry: {bg.designation}")
        for p, o in bg.orders().items():
            P(f"     {p:<18}{o:8.4f} x shaft")
        if "UNVERIFIED" in bg.source:
            P("     ! geometry UNVERIFIED against the dataset's own documentation")
    else:
        P("\n  No --bearing given. The declared-physics half is SKIPPED, not")
        P("  approximated. The IMS readme describes the rig, the channels and the")
        P("  failures and never gives the ZA-2115's dimensions, so under the rule")
        P("  declared in PROGNOSIS_STOPPING_RULE.md before this data was")
        P("  downloaded, no geometry is invented and no part is named. Inventing")
        P("  it would make every order meaningless while producing perfectly")
        P("  confident output.")
    if not M.FAULT_LABELS:
        P("\n  FAULT_LABELS is empty. Lead time and monotonicity are reported;")
        P("  part-naming accuracy is NOT, because there is nothing to score it")
        P(f"  against. See src/cbmx/io/{a.dataset}.py.")
    else:
        P(f"\n  {len(M.FAULT_LABELS)} bearings have a failure the documentation")
        P("  states; the rest ran to the end of their test without one and are")
        P("  scored as CONTROLS. Part-naming accuracy is still not reported —")
        P("  that needs geometry, which the labels do not supply.")

    # ── normalise the two layouts into one iteration ────────────────────────
    # IMS is <test>/<bearing 1-4>, four bearings sharing one shaft and one
    # failure; XJTU is <condition>/<bearing folder>, one bearing per life. Both
    # reduce to: a name, an object that yields captures in time order, and a way
    # to ask how much wall-clock time had elapsed at capture i. Keeping that
    # difference here rather than in the loaders means the physics below is
    # written once.
    units = []
    for grp in sorted(found):
        if a.dataset == "ims":
            lives = {b: found[grp][b] for b in sorted(found[grp])}
            if a.scope == "readme":
                lives = {b: ims.clip_to_published(l) for b, l in lives.items()}
            any_life = lives[min(lives)]
            # One read per capture, sliced four ways. See ims.iter_test: the
            # four bearings shared a shaft and therefore share a file, and
            # reading it once per bearing quadruples the I/O for nothing.
            per_b: Dict[int, List[Dict[str, float]]] = {b: [] for b in lives}
            for _, _, arr in ims.iter_test(any_life, every=a.every):
                for b in lives:
                    col = ims.columns_for(grp, b)
                    if arr.shape[1] < col.stop:
                        continue
                    per_b[b].append(indicators(arr[:, col][:, 0], M.FS))
            for b, life in lives.items():
                units.append((f"{grp}/b{b}", life, per_b[b], life.hours_at,
                              life.label))
        else:
            for key in sorted(found[grp], key=lambda k: xjtu._natural(str(k))):
                life = found[grp][key]
                series = [indicators(c.channel(a.channel), M.FS)
                          for c in life.captures(every=a.every)]
                units.append((f"{key}", life, series,
                              (lambda i, L=life:
                               i * getattr(M, "CAPTURE_INTERVAL_S", 60.0) / 3600.0),
                              getattr(life, "label", "") or ""))

    rows: List[Dict] = []
    P(f"\n{'-'*78}")
    P(f"  {'unit':<14}{'what':<17}{'caps':>6}{'life h':>8}{'rho(rms)':>10}"
      f"{'rho(cr)':>9}{'call':>7}{'lead':>9}{'% left':>8}")
    P(f"{'-'*78}")

    for name, life, series, elapsed_h, label in units:
        n_base = max(int(a.baseline_frac * len(series)), 8)
        if len(series) < n_base + 5:
            P(f"  {name:<14}{label or 'control':<17}{life.n_captures:>6}"
              f"   too few captures to score")
            continue

        base = Baseline(f"rtf/{name}", min_samples=8)
        reg = Regime(0, 0)
        for s_ in series[:n_base]:
            for k in ("rms", "crest_db"):
                base.observe(k, reg, s_[k])
        for st in base.stats.values():          # FREEZE
            st.frozen = True

        z_rms = [base.score("rms", reg, s_["rms"])[0] for s_ in series]
        z_cr = [base.score("crest_db", reg, s_["crest_db"])[0] for s_ in series]
        zmax = [max(p_, q_) for p_, q_ in zip(z_rms, z_cr)]

        t = list(range(len(series)))
        rho_rms = spearman(t, [s_["rms"] for s_ in series])
        rho_cr = spearman(t, [s_["crest_db"] for s_ in series])
        total_h = elapsed_h(life.n_captures - 1) or 0.0

        hit = first_sustained([z > a.z_alarm for z in zmax])
        row = {"unit": name, "captures": life.n_captures, "life_h": total_h,
               "rho_rms": rho_rms, "rho_crest": rho_cr,
               "label": label, "is_failure": bool(label)}
        if hit is None:
            P(f"  {name:<14}{label or 'control':<17}{life.n_captures:>6}"
              f"{total_h:>8.1f}{rho_rms:>10.2f}{rho_cr:>9.2f}"
              f"{'never':>7}{'-':>9}{'-':>8}")
            row["first_call"] = None
        else:
            call_i = hit * a.every
            lead_h = total_h - elapsed_h(call_i)
            frac_left = 100.0 * lead_h / total_h if total_h > 0 else float("nan")
            P(f"  {name:<14}{label or 'control':<17}{life.n_captures:>6}"
              f"{total_h:>8.1f}{rho_rms:>10.2f}{rho_cr:>9.2f}"
              f"{call_i:>7}{lead_h:>8.1f}h{frac_left:>7.0f}%")
            row.update({"first_call": call_i, "lead_hours": lead_h,
                        "pct_life_left": frac_left})
        rows.append(row)

    fails = [r for r in rows if r["is_failure"]]
    ctrls = [r for r in rows if not r["is_failure"]]
    called_f = [r for r in fails if r.get("first_call") is not None]
    called_c = [r for r in ctrls if r.get("first_call") is not None]
    P(f"{'-'*78}")

    if called_f:
        leads = [r["lead_hours"] for r in called_f]
        fracs = [r["pct_life_left"] for r in called_f]
        P(f"\n  SPOKE ON {len(called_f)} of {len(fails)} documented failures")
        P(f"  lead time    median {np.median(leads):.1f} h   "
          f"range {min(leads):.1f}-{max(leads):.1f} h")
        P(f"  life remaining at first call   median {np.median(fracs):.0f}%   "
          f"range {min(fracs):.0f}-{max(fracs):.0f}%")
    else:
        P(f"\n  SPOKE ON 0 of {len(fails)} documented failures")

    # The controls, reported beside the failures and never instead of them.
    P(f"\n  CONTROLS: spoke on {len(called_c)} of {len(ctrls)} bearings that "
      f"survived their test")
    if called_c:
        # A control that speaks BEFORE the failing bearing on its own shaft
        # cannot be explained by transmission from that failure. One that
        # speaks after it might be. The distinction is the whole value of the
        # column, so it is computed rather than asserted.
        first_by_test = {}
        for r in called_f:
            t = r["unit"].split("/")[0]
            h = r["life_h"] - r["lead_hours"]
            first_by_test[t] = min(first_by_test.get(t, 1e9), h)
        early = late = 0
        for r in called_c:
            t = r["unit"].split("/")[0]
            h = r["life_h"] - r["lead_hours"]
            if t in first_by_test and h >= first_by_test[t]:
                late += 1
            else:
                early += 1
            P(f"     {r['unit']:<12} called at {h:.1f} h"
              + (f", after {t}'s failure spoke at {first_by_test[t]:.1f} h"
                 " — may be shaft transmission"
                 if t in first_by_test and h >= first_by_test[t]
                 else " — before any failure on its shaft spoke, so this is a"
                      " false alarm"))
        P(f"\n     {early} cannot be explained by cross-talk and count as false")
        P(f"     alarms; {late} follow a failure on the same shaft and are")
        P("     reported as ambiguous rather than counted either way.")
    else:
        P("     No surviving bearing ever spoke. Note this is a weaker control")
        P("     than Paderborn's: these bearings share a shaft with a failing")
        P("     one, so silence here is a stronger result than it looks, and a")
        P("     call would have been a weaker fault than it looks.")

    if rows:
        P(f"\n  monotonicity   median rho(rms) "
          f"{np.median([r['rho_rms'] for r in rows]):.2f}"
          f"   median rho(crest) {np.median([r['rho_crest'] for r in rows]):.2f}")

    P("")
    P("  Read the lead time and the controls together. A high 'spoke on' count")
    P("  with a lead time of a few captures is a failure detector, not a")
    P("  prognostic, and is worth nothing to a maintenance planner. The claim")
    P("  worth making is a lead time long enough to schedule against, on most")
    P("  of the fleet, without the surviving bearings speaking too.")

    if called_f:
        med = float(np.median([r["pct_life_left"] for r in called_f]))
        if med < 10:
            P("")
            P("  MEDIAN LIFE REMAINING IS BELOW 10%. Under the rule declared in")
            P("  PROGNOSIS_STOPPING_RULE.md before this data was downloaded,")
            P("  PROGNOSIS IS NOT CLAIMED. The volume says the indicator is a")
            P("  failure detector, not a prognostic, on this data, and")
            P("  remaining-useful-life stays a Phase II task. The threshold is")
            P("  not to be swept and the persistence requirement is not to be")
            P("  relaxed to move this number.")
    elif not fails:
        P("\n  No labelled failures in scope — nothing to measure a lead time on.")
    else:
        P("")
        P("  The indicator never rose persistently on any documented failure.")
        P("  Check the monotonicity column before changing anything: rho near")
        P("  zero means there is nothing to detect in this indicator, and no")
        P("  threshold will fix that.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(
            {"data": str(a.data), "channel": a.channel,
             "baseline_frac": a.baseline_frac, "z_alarm": a.z_alarm,
             "bearing_key": a.bearing, "labels_present": bool(M.FAULT_LABELS),
             "scope": a.scope, "every": a.every,
             "declared_physics_ran": bool(a.bearing),
             "units": rows}, indent=2, default=float))
        P(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
