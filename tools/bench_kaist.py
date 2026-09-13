#!/usr/bin/env python3
"""Per-window compute cost of the order-tracked path, under varying speed.

    PYTHONPATH=src taskset -c 0 python3 tools/bench_kaist.py --data data/kaist

`bench_edge.py` measures the same budget on Paderborn, at a constant shaft
speed. That is the easy case and it is not the case the topic describes. This
one measures the path an arresting engine would actually run: the shaft speed
moves 13% inside a one-second window, so the envelope has to be resampled
against shaft angle before it is transformed, and that resampling is real work
that the fixed-speed number does not contain.

The comparison is the point. Reporting only the order-tracked figure invites
"and what did that cost you"; reporting both answers it before it is asked.

HONESTY ABOUT THE PLATFORM. This runs on a general-purpose x86 container pinned
to one core to approximate a small edge box. It is NOT ruggedized hardware and
no claim about MIL-SPEC hardware may rest on it. What it establishes is the
shape of the number — orders of magnitude inside the budget, or outside it.
Single-core pinning is the conservative direction; a fielded box gets the whole
device.

WHAT IS TIMED. Everything from a captured window sitting in memory to a decision
existing: demodulation in the frozen band, angular resampling where the arm
calls for it, the envelope spectrum, prominence and sideband ratio at every
predicted line on every channel, the robust-baseline comparison and the
sequential test. NOT timed: reading CSV off disk, which is an artefact of how
this dataset ships and would be a ring buffer on the asset.

A DISTRIBUTION, NOT A MEAN. A latency budget is a promise about the worst case,
so p50, p95, p99 and max are what get printed.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.io import kaist                                          # noqa: E402
from cbmx.physics.bearing import get as get_bearing                # noqa: E402
from cbmx.physics.envelope import Band                             # noqa: E402
from cbmx.health.baseline import Baseline                          # noqa: E402
from cbmx.health.evidence import SPRT, Observation                 # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_kaist import SYMPTOMS, CHANNELS, features                # noqa: E402


def pct(v: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(v), q))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--bearing", default="KAIST6205U")
    ap.add_argument("--record", default="inner_0")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--band", nargs=2, type=float, default=[11200.0, 12800.0])
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    bg = get_bearing(a.bearing)
    orders = bg.orders()
    band = Band(float(a.band[0]), float(a.band[1]), 0.0, 0)

    cond, _, rep = a.record.rpartition("_")
    rec = kaist.load(a.data, "vibration", cond, rep, max_seconds=a.seconds)
    off, ali, una = kaist.align_tacho(rec, CHANNELS[0])
    rec.tacho_offset_s = off
    wins = {ch: list(kaist.windows(rec, ch, seconds=a.window)) for ch in CHANNELS}
    n_win = len(wins[CHANNELS[0]])

    print("=" * 74)
    print("cbmx — per-window compute cost, varying speed, order-tracked path")
    print("=" * 74)
    print(f"  platform      {platform.platform()}")
    print(f"  python        {platform.python_version()}   numpy {np.__version__}")
    print(f"  record        {a.record}, {n_win} windows of {a.window:g} s at "
          f"{kaist.FS_VIBRATION:g} Hz")
    print(f"  channels      {len(CHANNELS)} accelerometers, "
          f"{len(SYMPTOMS)} declared lines each = "
          f"{len(CHANNELS)*len(SYMPTOMS)} lines per window")
    print(f"  band          {band.lo:.0f}-{band.hi:.0f} Hz, frozen")
    print(f"  speed wander  {np.median([w.wander_pct for w in wins[CHANNELS[0]]]):.1f}% "
          f"median inside a window")
    print("\n  Not timed: CSV parsing. Timed: demodulation, angular resampling where")
    print("  the arm uses it, envelope spectrum, prominence and sidebands on every")
    print("  line, baseline comparison, sequential test.\n")

    # A fitted, frozen baseline, so the scoring half of the work is real rather
    # than short-circuited by an unready line.
    base = Baseline("bench", speed_step_hz=10.0, min_samples=5)
    for ch in CHANNELS:
        for w in wins[ch][:20]:
            reg = base.regime(w.mean_hz)
            for sid, (prom, _, _) in features(w, band, orders, True).items():
                base.observe(f"{ch}:{sid}", reg, prom)
    for r in base.stats.values():
        r.frozen = True

    out: Dict[str, Dict] = {}
    for tracked in (False, True):
        arm = "order-tracked" if tracked else "fixed-speed"
        per_window: List[float] = []
        sprt = SPRT(0.02, 0.10)
        for i in range(n_win):
            t0 = time.perf_counter()
            for ch in CHANNELS:
                w = wins[ch][i]
                f = features(w, band, orders, tracked)
                reg = base.regime(w.mean_hz)
                for sid, spec in SYMPTOMS.items():
                    prom, measured, sb = f[sid]
                    z, run = base.score(f"{ch}:{sid}", reg, prom)
                    sprt.observe(Observation(
                        f"{ch}:{sid}", w.t0, z * spec["direction"],
                        orders[spec["part"]], measured, sb,
                        spec["sidebands"], run.ready))
            per_window.append((time.perf_counter() - t0) * 1000.0)

        s = {"p50_ms": pct(per_window, 50), "p95_ms": pct(per_window, 95),
             "p99_ms": pct(per_window, 99), "max_ms": max(per_window),
             "mean_ms": statistics.mean(per_window), "n": len(per_window)}
        out[arm] = s
        print(f"  [{arm}]  all {len(CHANNELS)} channels, "
              f"{len(CHANNELS)*len(SYMPTOMS)} lines, per {a.window:g} s window")
        print(f"     p50 {s['p50_ms']:7.1f} ms    p95 {s['p95_ms']:7.1f} ms    "
              f"p99 {s['p99_ms']:7.1f} ms    max {s['max_ms']:7.1f} ms")
        print(f"     duty cycle at p95: "
              f"{100.0*s['p95_ms']/(a.window*1000.0):.1f}% of real time\n")

    f_, t_ = out["fixed-speed"], out["order-tracked"]
    print("  THE COST OF ORDER TRACKING")
    print(f"     p95 {f_['p95_ms']:.1f} ms -> {t_['p95_ms']:.1f} ms, "
          f"a factor of {t_['p95_ms']/max(f_['p95_ms'],1e-9):.2f}")
    print("     Angular resampling is an interpolation onto a 256-samples-per-")
    print("     revolution grid; it is linear in the window and it does not")
    print("     change the order of the cost.")
    print("\n  AGAINST THE TOPIC'S THRESHOLD")
    print("     The stated budget is 1 s from sample arrival to anomaly")
    print(f"     indication. Worst observed window here is {t_['max_ms']:.1f} ms on one")
    print(f"     pinned core while scoring {len(CHANNELS)*len(SYMPTOMS)} lines, which is "
          f"{1000.0/max(t_['max_ms'],1e-9):.0f}x inside it.")
    print("     Read that as the shape of the number, not as a hardware claim.")

    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({
            "platform": platform.platform(),
            "record": a.record, "windows": n_win,
            "window_s": a.window, "channels": len(CHANNELS),
            "lines_per_window": len(CHANNELS) * len(SYMPTOMS),
            "band": [band.lo, band.hi], "arms": out,
        }, indent=2))
        print(f"\n  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
