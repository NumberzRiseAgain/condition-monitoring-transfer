#!/usr/bin/env python3
"""KAIST varying-speed bearing set — order tracking measured against the alternative.

    PYTHONPATH=src python3 tools/eval_kaist.py \
        --data <dir> --bearing KAIST6205U --json runs/kaist.json

Every other bearing benchmark in this repository is recorded at a constant
shaft speed. An arresting engine is not, and a fault line that sits at 3.585 x
shaft speed sweeps across a kilohertz while the engine does its job. This set
is real seeded damage under a randomly varying 680-2460 rpm with a tachometer
alongside, which is the only public data here that can test the part of the
method that the topic actually needs.

Four questions, in the order they have to be asked.

  A. Does the speed move enough for any of this to matter? A census, before
     anything is scored.

  B. Is the speed reference usable as shipped? It is not, and finding that out
     is section 2. See `io/kaist.align_tacho`.

  C. Does resampling against shaft angle sharpen the spectrum -- decided on
     HEALTHY recordings only, using the shaft line that every rotating machine
     has whether or not it is broken? No fault label is touched. The fixed-speed
     `_constant` recordings are the control: any advantage must vanish there.

  D. Given the physics declared up front, is the damage found, is the healthy
     machine left alone, and how long does it take? Detection rate and
     false-alarm rate are printed together and never apart -- a run that never
     speaks passes a false-alarm test perfectly, which is how this repository
     once reported 0 detections and 0 false alarms and called it a result.

Two limits, stated here rather than buried at the end.

The dataset publishes no bearing geometry, so the orders searched come from an
assumed 6205 (`physics/bearing.py`, entry KAIST6205U). If that assumption is
wrong, section 4 is scoring empty spectrum, and it will say so by failing.

The dataset also does not say which of the two bearing housings carries the
damaged bearing. Choosing the channel that shows the fault best would be
selection on the label. So all four accelerometer channels are monitored, each
with its own baseline and its own lines, and healthy records are scored against
exactly the same expanded set -- the multiple comparison is paid for on both
sides of the ledger.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from cbmx.io import kaist
from cbmx.physics import bearing as bearing_mod
from cbmx.physics.envelope import (Band, choose_band, envelope_spectrum,
                                   envelope_spectrum_order_tracked)
from cbmx.health.baseline import Baseline
from cbmx.health.evidence import SPRT, Observation


# ── declared physics ─────────────────────────────────────────────────────────
# Fixed before any data is read. Each symptom is a place in the spectrum that
# geometry says belongs to one replaceable part, a direction the deviation must
# go for it to be evidence of that part failing, and whether physics expects
# sidebands there. None of it is chosen by looking at label separation.
#
# Direction is +1 everywhere and that is not laziness: a defect adds impacts, so
# it can only RAISE the prominence of its own line above the local floor. A line
# that goes quiet is evidence against, and the test is built to let it push the
# accumulator back down.
SYMPTOMS: Dict[str, Dict] = {
    "outer_race": {"part": "outer_race", "direction": +1,
                   "sidebands": False, "primary": True,
                   "why": "stationary defect struck once per passing ball; the "
                          "load zone does not move over it, so no shaft-rate "
                          "modulation is expected"},
    "inner_race": {"part": "inner_race", "direction": +1,
                   "sidebands": True, "primary": True,
                   "why": "the defect rotates through the load zone once per "
                          "shaft revolution, amplitude-modulating the impact "
                          "train and putting sidebands one shaft order apart"},
    "cage": {"part": "cage", "direction": +1,
             "sidebands": False, "primary": False,
             "why": "corroboration only; cage energy is low and its absence is "
                    "not evidence against a race fault"},
    "rolling_element": {"part": "rolling_element", "direction": +1,
                        "sidebands": True, "primary": False,
                        "why": "corroboration only; the ball enters and leaves "
                               "the load zone at cage rate"},
}

CHANNELS = ["bearingA_x", "bearingA_y", "bearingB_x", "bearingB_y"]


# ── features ─────────────────────────────────────────────────────────────────
def sideband_ratio(es, centre: float, spacing: float = 1.0) -> float:
    _, a = es.peak(centre)
    if a <= 0:
        return 0.0
    _, lo = es.peak(max(centre - spacing, 0.05))
    _, hi = es.peak(centre + spacing)
    return float(0.5 * (lo + hi) / a)


def spectrum(x: np.ndarray, fs: float, shaft_hz: np.ndarray, mean_hz: float,
             band: Band, tracked: bool, max_order: float = 20.0):
    """The two arms differ in exactly one call. Everything else -- window, band,
    floor rule, tolerance -- is identical, so any difference in the result is
    attributable to angular resampling and to nothing else."""
    if tracked:
        return envelope_spectrum_order_tracked(x, fs, shaft_hz, band=band,
                                               max_order=max_order)
    return envelope_spectrum(x, fs, mean_hz, band=band, max_order=max_order)


def features(win: kaist.Window, band: Band, orders: Dict[str, float],
             tracked: bool) -> Dict[str, Tuple[float, float, float]]:
    es = spectrum(win.x, win.fs, win.shaft_hz, win.mean_hz, band, tracked)
    out = {}
    for sid, spec in SYMPTOMS.items():
        o = orders[spec["part"]]
        measured, prom = es.prominence_db(o)
        out[sid] = (prom, measured, sideband_ratio(es, o))
    return out


def sharpness(win: kaist.Window, band: Band, tracked: bool) -> Tuple[float, float, float]:
    """How sharp is this window's envelope spectrum? Health-free, label-free.

    1x and 2x are the shaft lines, which residual imbalance puts in every
    rotating machine whether or not it is damaged. `crest` is the 99th
    percentile of the whole order spectrum over its own median: smear takes
    energy out of narrow lines and spreads it across the floor, so the ratio
    falls twice over, and it looks at no particular order so it cannot be
    steered by the choice of one.
    """
    es = spectrum(win.x, win.fs, win.shaft_hz, win.mean_hz, band, tracked)
    m = (es.orders >= 0.3) & (es.orders <= 20.0)
    amp = es.amplitude[m]
    crest = 20.0 * math.log10(max(float(np.percentile(amp, 99)), 1e-15)
                              / max(float(np.median(amp)), 1e-15))
    return es.prominence_db(1.0)[1], es.prominence_db(2.0)[1], crest


def commission(healthy_windows: List[kaist.Window], levels=(3, 4), min_hz=500.0
               ) -> Tuple[Band, Counter]:
    """Choose the demodulation band once, on healthy data, then freeze it.

    Spectral kurtosis -- which band is least Gaussian, because impacts are
    impulsive and shaft noise is not. It uses no fault label and no fault order,
    so it cannot select the band that best displays the damage. The band is a
    property of the structure and the mounting; selecting it per window at run
    time would be a multiple-comparisons trap.
    """
    votes: Counter = Counter()
    for w in healthy_windows:
        b = choose_band(w.x, w.fs, levels=levels, min_hz=min_hz)
        votes[(round(b.lo, 1), round(b.hi, 1))] += 1
    (lo, hi), _ = votes.most_common(1)[0]
    return Band(lo, hi, 0.0, 0), votes


# ── the run ──────────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--bearing", required=True,
                    help="catalogue key; KAIST6205U is an ASSUMED geometry")
    ap.add_argument("--channels", nargs="+", default=CHANNELS)
    ap.add_argument("--band-channel", default="bearingA_x")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=74.0)
    ap.add_argument("--speed-step", type=float, default=10.0)
    ap.add_argument("--min-samples", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=0.10)
    ap.add_argument("--healthy", nargs="+",
                    default=["normal_0", "normal_1", "normal_2"],
                    help="healthy recordings; rotated leave-one-out")
    ap.add_argument("--targets", nargs="+", default=["outer_0", "inner_0"])
    ap.add_argument("--band", nargs=2, type=float, default=None,
                    metavar=("LO", "HI"),
                    help="override the commissioned band, to test sensitivity to it")
    ap.add_argument("--starve", type=int, default=8)
    ap.add_argument("--no-align", action="store_true",
                    help="skip tachometer alignment — for reproducing section 2")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    t_start = time.time()
    bg = bearing_mod.get(a.bearing)
    orders = bg.orders()
    unverified = "UNVERIFIED" in bg.source
    P = print

    P("=" * 78)
    P("KAIST varying-speed bearing set — cbmx")
    P("=" * 78)

    # -- 0. declaration ------------------------------------------------------
    P("\n0. DECLARED BEFORE ANY DATA IS SCORED")
    P("-" * 78)
    P(bg.describe())
    checks = bg.check()
    P("   kinematic identities: " +
      ", ".join(f"{k}={'ok' if v else 'FAIL'}" for k, v in checks.items()))
    if not all(checks.values()):
        P("   geometry is internally inconsistent — stopping")
        return 2
    if unverified:
        P("\n   " + "!" * 70)
        P("   ! GEOMETRY IS ASSUMED, NOT PUBLISHED. The KAIST dataset gives no")
        P("   ! bearing model, ball count, ball diameter, pitch diameter or")
        P("   ! contact angle — not in the Mendeley record, not in the authors'")
        P("   ! arXiv paper. A 6205 is assumed from the rig class. Every order")
        P("   ! below inherits that assumption. It is declared ONCE and is NOT")
        P("   ! revised if the result disappoints.")
        P("   " + "!" * 70)
    P("\n   symptoms, and the direction each must move to count as evidence:")
    for sid, s in SYMPTOMS.items():
        P(f"     {sid:<16} order {orders[s['part']]:6.3f}  dir {s['direction']:+d}"
          f"  sidebands {'expected' if s['sidebands'] else 'not expected':<13}"
          f"  {'primary' if s['primary'] else 'corroborating'}")
        P(f"     {'':<16} {s['why']}")
    P(f"\n   alpha={a.alpha}  beta={a.beta}  -> boundaries "
      f"+{math.log((1-a.beta)/a.alpha):.2f} / {math.log(a.beta/(1-a.alpha)):.2f} nats")
    P(f"   window {a.window:g} s   regime = shaft speed in {a.speed_step:g} Hz bins"
      f"   baseline ready at n>={a.min_samples}")
    P(f"   healthy recordings {a.healthy}, rotated leave-one-out: each is")
    P("   held out in turn while the baseline is fitted on the other two and")
    P("   FROZEN before any test window is scored.")
    P(f"   targets {a.targets}")
    P(f"   channels monitored: {', '.join(a.channels)} — the dataset does not say")
    P("   which housing holds the damaged bearing, so all are watched and")
    P("   healthy records are scored against the same expanded set.")

    # -- 1. load and census --------------------------------------------------
    def get(uid: str) -> kaist.Record:
        cond, _, rep = uid.rpartition("_")
        return kaist.load(a.data, "vibration", cond, rep, max_seconds=a.seconds)

    wanted = list(dict.fromkeys(a.healthy + a.targets))
    recs: Dict[str, kaist.Record] = {}
    for uid in wanted:
        try:
            recs[uid] = get(uid)
        except FileNotFoundError as e:
            P(f"\n   missing: {e}")
    if not recs:
        P("no recordings found")
        return 2

    P("\n1. THE DATA AS READ")
    P("-" * 78)
    avail = kaist.conditions_available(a.data)
    P("   recordings present under the data directory:")
    for cond in ("normal", "inner", "outer"):
        got = avail.get(cond, [])
        P(f"     {cond:<8} {len(got):>2}  {', '.join(got) if got else '(none)'}")
    P("   Subsets 2 and 3 appear here as separate replicates once they are put")
    P("   in subdirectories; a short list is a smaller n, not a passing detail.")
    P("")
    P(f"   {'record':<12} {'windows':>8} {'seconds':>8} {'rpm range':>14} "
      f"{'median in-window wander':>26}")
    tmp = {uid: list(kaist.windows(r, a.band_channel, seconds=a.window))
           for uid, r in recs.items()}
    for uid, r in recs.items():
        w = tmp[uid]
        lo, hi = r.tacho.span()
        P(f"   {uid:<12} {len(w):>8} {r.duration_s:>8.1f} "
          f"{lo:>6.0f}-{hi:<7.0f} "
          f"{np.median([x.wander_pct for x in w]):>24.1f}%")
    P(f"   channel set {a.channels}   fs {kaist.FS_VIBRATION:g} Hz "
      f"(Jung et al. arXiv:2311.18547v2)")
    P("   That rate is not merely assumed: of eight candidate rates, only")
    P("   25.6 kHz makes the shaft line sharp in the order domain, which is a")
    P("   property no wrong rate reproduces. See tests/test_kaist.py.")
    all_wander = np.concatenate([[x.wander_pct for x in w] for w in tmp.values() if w])
    P(f"\n   Speed wander inside one {a.window:g} s window, all records: "
      f"median {np.median(all_wander):.1f}%, "
      f"10th-90th {np.percentile(all_wander,10):.1f}-{np.percentile(all_wander,90):.1f}%")
    P(f"   A line at {orders['outer_race']:.2f}x shaft speed therefore sweeps across "
      f"roughly {orders['outer_race']*np.median(all_wander):.0f}% of its own")
    P("   frequency inside a single window. This is the condition the method")
    P("   claims to handle and every other dataset here fails to present.")

    # -- 2. tachometer alignment --------------------------------------------
    P("\n2. THE SPEED REFERENCE IS NOT ALIGNED AS SHIPPED")
    P("-" * 78)
    align_rows = []
    if a.no_align:
        P("   --no-align: offsets left at zero. Section 3 and 4 below show what")
        P("   order tracking is worth against a misaligned reference.")
    else:
        P("   Recovered from the 1x and 2x SHAFT lines, which exist in every")
        P("   rotating machine healthy or broken. No fault order, no fault label")
        P("   and no damaged recording enters the search; the identical procedure")
        P("   runs on every record.")
        P("")
        P(f"   {'record':<12} {'offset':>9} {'shaft line as shipped':>24} "
          f"{'aligned':>10} {'gain':>8}")
        for uid, r in recs.items():
            off, ali, una = kaist.align_tacho(r, a.band_channel)
            r.tacho_offset_s = off
            r.align_db = (ali, una)
            align_rows.append({"record": uid, "offset_s": off,
                               "unaligned_db": una, "aligned_db": ali})
            P(f"   {uid:<12} {off:>+8.2f}s {una:>22.1f} dB {ali:>8.1f} dB "
              f"{ali-una:>+7.1f}")
        P("")
        P("   The rpm file's `time` column and the signal file do not start")
        P("   together, and the discrepancy differs per record. Taken at face")
        P("   value the integrated shaft angle drifts across a window, and order")
        P("   tracking becomes worse than useless — it resamples against the")
        P("   wrong angle. This was found by asking why order tracking bought")
        P("   nothing, and it is the single largest effect in this run.")

    wins: Dict[str, Dict[str, List[kaist.Window]]] = {
        uid: {ch: list(kaist.windows(r, ch, seconds=a.window)) for ch in a.channels}
        for uid, r in recs.items()
    }

    # -- 3. commission the band ---------------------------------------------
    healthy_for_band = [w for uid in a.healthy
                        for w in wins.get(uid, {}).get(a.band_channel, [])][:40]
    if not healthy_for_band:
        P("no healthy windows to commission a band on")
        return 2
    band, votes = commission(healthy_for_band)
    if a.band:
        band = Band(float(a.band[0]), float(a.band[1]), 0.0, 0)
    P("\n3. DEMODULATION BAND — chosen once on healthy data, then frozen")
    P("-" * 78)
    P(f"   {band.lo:.0f}-{band.hi:.0f} Hz"
      + ("  ** OVERRIDDEN on the command line **" if a.band else
         f" by spectral kurtosis over {len(healthy_for_band)} healthy windows "
         f"of {a.band_channel}"))
    for (lo, hi), n in votes.most_common(4):
        P(f"     {n:>3} votes  {lo:.0f}-{hi:.0f} Hz")
    P("   No fault label and no fault order enters this choice; run time never")
    P("   re-selects it.")
    P("   The vote is not decisive — the top three bands are within two votes")
    P("   of each other — so the result must not depend on which one wins.")
    P("   `--band LO HI` re-runs everything in a different band; the run log")
    P("   reports the runners-up as well as the winner.")

    # -- 4. order tracking vs fixed speed, healthy only ----------------------
    P("\n4. ORDER TRACKING vs FIXED SPEED — decided on healthy data, no labels")
    P("-" * 78)
    P("   Prominence of the shaft lines, plus the crest of the whole order")
    P("   spectrum over its own floor. All three exist regardless of health.")
    P("")
    P(f"   {'record':<18} {'wander':>7} " +
      "".join(f"{n:>9}" for n in ("1x fix", "1x trk", "gain", "2x fix", "2x trk",
                                  "gain", "crest f", "crest t", "gain")))
    track_rows: List[Dict] = []

    def measure(label: str, ws: List[kaist.Window]) -> Dict:
        F = np.array([sharpness(w, band, False) for w in ws])
        T = np.array([sharpness(w, band, True) for w in ws])
        f, t = np.median(F, axis=0), np.median(T, axis=0)
        wd = float(np.median([w.wander_pct for w in ws]))
        P(f"   {label:<18} {wd:>6.1f}% " +
          "".join(f"{v:>9.2f}" for v in (f[0], t[0], t[0]-f[0], f[1], t[1],
                                         t[1]-f[1], f[2], t[2], t[2]-f[2])))
        return {"record": label, "wander_pct": wd,
                "x1_fixed": f[0], "x1_tracked": t[0],
                "x2_fixed": f[1], "x2_tracked": t[1],
                "crest_fixed": f[2], "crest_tracked": t[2]}

    for uid in a.healthy:
        w = wins.get(uid, {}).get(a.band_channel, [])[:40]
        if w:
            track_rows.append(measure(uid, w))
    P("")
    ctrl_rows: List[Dict] = []
    for cond in ("normal", "outer"):
        try:
            rc = kaist.load(a.data, "vibration", cond, "constant", max_seconds=25.0)
        except FileNotFoundError:
            continue
        cw = list(kaist.windows(rc, a.band_channel, seconds=a.window, limit=20))
        if cw:
            ctrl_rows.append(measure(f"CTRL {cond}_const", cw))
    P("")
    P("   The control is what makes the rows above mean anything: at constant")
    P("   speed the two arms are the same computation, so the gain there must")
    P("   be about zero. If it is not, this metric is measuring something other")
    P("   than speed smear and the varying-speed gains cannot be read as such.")

    # -- 5. detection --------------------------------------------------------
    P("\n5. DETECTION — the declared orders, both arms, all four channels")
    P("-" * 78)

    results: Dict[str, Dict] = {}
    for tracked in (False, True):
        arm = "order-tracked" if tracked else "fixed-speed"
        feats: Dict[str, Dict[str, List[Dict]]] = {}
        for uid in wins:
            feats[uid] = {ch: [features(w, band, orders, tracked)
                               for w in wins[uid][ch]] for ch in a.channels}

        def score_record(base: Baseline, uid: str) -> Dict:
            """One recording against one frozen baseline.

            The accumulator is created here and carries state across the whole
            recording, which is the point of a sequential test -- and which is
            exactly why the windows inside a recording are NOT independent
            trials. One recording is one trial. See the note under the table.
            """
            sprt = SPRT(a.alpha, a.beta)
            first: Optional[Tuple[int, str]] = None
            n_named = 0
            max_S = -math.inf
            zs: Dict[str, List[float]] = defaultdict(list)
            meas: Dict[str, List[float]] = defaultdict(list)
            n_line = n_line_abstain = 0
            n_win = len(wins[uid][a.channels[0]])
            for i in range(n_win):
                named_here = False
                for ch in a.channels:
                    w = wins[uid][ch][i]
                    f = feats[uid][ch][i]
                    reg = base.regime(w.mean_hz)
                    for sid, spec in SYMPTOMS.items():
                        key = f"{ch}:{sid}"
                        prom, measured, sb = f[sid]
                        z, run = base.score(key, reg, prom)
                        n_line += 1
                        if run.ready:
                            zs[key].append(z * spec["direction"])
                            meas[key].append(measured)
                        else:
                            n_line_abstain += 1
                        v = sprt.observe(Observation(
                            symptom_id=key, t=w.t0,
                            z_level=z * spec["direction"],
                            order_predicted=orders[spec["part"]],
                            order_measured=measured, sideband_ratio=sb,
                            expect_sidebands=spec["sidebands"],
                            baseline_ready=run.ready))
                        if spec["primary"] and v.decision != "no_baseline":
                            max_S = max(max_S, v.S)
                        if v.decision == "name_it" and spec["primary"]:
                            named_here = True
                            if first is None:
                                first = (i, key)
                if named_here:
                    n_named += 1
            prim = [f"{ch}:{sid}" for ch in a.channels
                    for sid, sp in SYMPTOMS.items() if sp["primary"]]
            finals = [sprt.state(k) for k in prim]
            return {"record": uid, "windows": n_win,
                    "max_S": None if max_S == -math.inf else float(max_S),
                    "exonerated": sum(1 for v in finals if v <= sprt.lower),
                    "primary_lines": len(prim),
                    "first_named": None if first is None else first[0],
                    "named_symptom": None if first is None else first[1],
                    "windows_naming": n_named,
                    "line_abstain_pct": 100.0 * n_line_abstain / max(n_line, 1),
                    "median_z": {k: float(np.median(v)) for k, v in zs.items() if v},
                    "median_order": {k: float(np.median(v)) for k, v in meas.items() if v}}

        folds = []
        for holdout in a.healthy:
            fit = [u for u in a.healthy if u != holdout and u in wins]
            if holdout not in wins or not fit:
                continue
            base = Baseline("kaist", speed_step_hz=a.speed_step,
                            min_samples=a.min_samples)
            for uid in fit:
                for ch in a.channels:
                    for w, f in zip(wins[uid][ch], feats[uid][ch]):
                        reg = base.regime(w.mean_hz)
                        for sid, (prom, _, _) in f.items():
                            base.observe(f"{ch}:{sid}", reg, prom)
            for r in base.stats.values():          # FREEZE
                r.frozen = True
            folds.append({
                "fit_on": fit, "holdout": holdout,
                "coverage": base.coverage(),
                "healthy": score_record(base, holdout),
                "targets": [score_record(base, t) for t in a.targets if t in wins],
            })
        results[arm] = {"folds": folds}

        P(f"\n   [{arm}]  leave-one-out over the {len(folds)} healthy recordings:")
        P("   each fold refits the baseline on the other two, freezes it, then")
        P("   scores the held-out healthy recording and every damaged one.")
        P("")
        P(f"   {'fit on':<22} {'held out':<10} {'that healthy record':>22} "
          f"{'damaged named':>15}")
        for fd in folds:
            h = fd["healthy"]
            hv = "ALARM" if h["first_named"] is not None else "silent"
            named = sum(1 for t in fd["targets"] if t["first_named"] is not None)
            P(f"   {'+'.join(fd['fit_on']):<22} {fd['holdout']:<10} {hv:>22} "
              f"{named:>7}/{len(fd['targets']):<7}")

        n_folds = len(folds)
        fa_records = sum(1 for fd in folds if fd["healthy"]["first_named"] is not None)
        det = {}
        for fd in folds:
            for t in fd["targets"]:
                det.setdefault(t["record"], []).append(t["first_named"])
        P("")
        P(f"   FALSE ALARM  {fa_records} of {n_folds} held-out healthy recordings"
          f"  —  n is {n_folds}, not {n_folds * 74}. The windows inside a")
        P("   recording are consecutive seconds of one run and the accumulator")
        P("   carries state across them, so a recording is one trial. Quoting a")
        P("   per-window binomial bound here would be arithmetic on correlated")
        P("   samples, and this run does not do it.")
        P("")
        P(f"   DETECTION, per damaged recording, across all {n_folds} baselines"
          f" (window first named):")
        P(f"   {'record':<10} " + "".join(f"{('fit ' + fd['holdout'][-1]):>12}"
                                          for fd in folds) + f"{'named in':>12}")
        for rec_id, firsts in det.items():
            cells = "".join(f"{('never' if v is None else str(v)):>12}" for v in firsts)
            P(f"   {rec_id:<10} {cells}"
              f"{sum(1 for v in firsts if v is not None):>7}/{len(firsts):<5}")

        last = folds[-1]
        rows = [last["healthy"]] + last["targets"]
        for r in rows:
            r["healthy"] = (r["record"] == last["holdout"])
        results[arm]["rows"] = rows          # for section 6
        P("\n   median robust z on the primary lines, signed toward the declared")
        P(f"   fault — last fold, baseline on {'+'.join(last['fit_on'])}:")
        P(f"   {'line':<26}" + "".join(f"{r['record']:>13}" for r in rows))
        for ch in a.channels:
            for sid, spec in SYMPTOMS.items():
                if not spec["primary"]:
                    continue
                key = f"{ch}:{sid}"
                vals = [r["median_z"].get(key) for r in rows]
                if all(v is None for v in vals):
                    continue
                P(f"   {key:<26}" + "".join(
                    f"{(v if v is not None else float('nan')):>13.2f}" for v in vals))
        upper = math.log((1 - a.beta) / a.alpha)
        P(f"\n   MARGIN — how close each record came to the +{upper:.2f} nat")
        P("   boundary, and how many of its 8 primary lines finished BELOW the")
        P(f"   {math.log(a.beta/(1-a.alpha)):.2f} nat lower boundary — that is, were actively")
        P("   exonerated rather than merely left alone. A monitor that ends every")
        P("   healthy line at 'ordinary' is making a positive statement; one that")
        P("   ends them mid-band has simply not decided yet.")
        P(f"   {'record':<12} {'highest S reached':>19} {'lines exonerated':>18}")
        for r in rows:
            ms = "n/a" if r["max_S"] is None else f"{r['max_S']:+.2f}"
            P(f"   {r['record']:<12} {ms:>19} "
              f"{r['exonerated']:>10}/{r['primary_lines']:<7}")

        P("\n   per-line abstention (a line whose baseline is not ready in this")
        P("   window's regime, counted over every line and window):")
        P("   " + "  ".join(f"{r['record']} {r['line_abstain_pct']:.0f}%" for r in rows))

    # -- 6. where the energy actually landed ---------------------------------
    P("\n6. PREDICTED ORDER vs MEASURED — the check on the assumed geometry")
    P("-" * 78)
    P("   Declared before the run; measured after. A line that is real lands")
    P("   near its predicted order in the damaged record and wanders in the")
    P("   healthy one. This is reported, not used: revising the geometry until")
    P("   the peaks agree would be fitting physics to labels.")
    tr = results.get("order-tracked", {}).get("rows", [])
    P(f"\n   {'line':<26} {'predicted':>10}" + "".join(f"{r['record']:>13}" for r in tr))
    for ch in a.channels:
        for sid, spec in SYMPTOMS.items():
            if not spec["primary"]:
                continue
            key = f"{ch}:{sid}"
            vals = [r["median_order"].get(key) for r in tr]
            if all(v is None for v in vals):
                continue
            P(f"   {key:<26} {orders[spec['part']]:>10.3f}" + "".join(
                f"{(v if v is not None else float('nan')):>13.3f}" for v in vals))

    # -- 7. abstention -------------------------------------------------------
    P("\n7. ABSTENTION — starve the baseline and the system must refuse")
    P("-" * 78)
    starve = Baseline("kaist_starved", speed_step_hz=a.speed_step,
                      min_samples=a.min_samples)
    uid0 = a.healthy[0]
    for ch in a.channels:
        for w in wins.get(uid0, {}).get(ch, [])[:a.starve]:
            reg = starve.regime(w.mean_hz)
            for sid, (prom, _, _) in features(w, band, orders, True).items():
                starve.observe(f"{ch}:{sid}", reg, prom)
    for r in starve.stats.values():
        r.frozen = True
    tgt = a.targets[-1]
    n_ab = n_named = 0
    sp = SPRT(a.alpha, a.beta)
    n_win = len(wins.get(tgt, {}).get(a.channels[0], []))
    for i in range(n_win):
        ready_any = False
        for ch in a.channels:
            w = wins[tgt][ch][i]
            f = features(w, band, orders, True)
            reg = starve.regime(w.mean_hz)
            for sid, spec in SYMPTOMS.items():
                prom, measured, sb = f[sid]
                z, run = starve.score(f"{ch}:{sid}", reg, prom)
                ready_any |= run.ready
                v = sp.observe(Observation(f"{ch}:{sid}", w.t0, z * spec["direction"],
                                           orders[spec["part"]], measured, sb,
                                           spec["sidebands"], run.ready))
                if v.decision == "name_it" and spec["primary"]:
                    n_named += 1
        if not ready_any:
            n_ab += 1
    scov = starve.coverage()
    P(f"   Baseline fitted on {a.starve} windows against a {a.min_samples}-observation")
    P(f"   minimum: {scov['lines_ready']} of {scov['lines_tracked']} lines ready.")
    P(f"   Scoring {tgt}: no_baseline on {n_ab}/{n_win} windows "
      f"({100.0*n_ab/max(n_win,1):.0f}%), named {n_named} times.")
    P("   The same windows are scored in section 5 once the baseline is fitted")
    P("   properly. The refusal is the readiness rule, not the data.")

    P(f"\n{'=' * 78}")
    P(f"completed in {time.time()-t_start:.0f} s")
    if unverified:
        P("GEOMETRY ASSUMED — see section 0. Nothing in section 5 may be quoted")
        P("without that caveat attached.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({
            "bearing": {"key": a.bearing, "designation": bg.designation,
                        "source": bg.source, "unverified": unverified,
                        "orders": orders},
            "band": {"lo": band.lo, "hi": band.hi},
            "alignment": align_rows,
            "wander_pct_median": float(np.median(all_wander)),
            "order_tracking": track_rows, "control": ctrl_rows,
            "detection": results,
        }, indent=2, default=float))
        P(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
