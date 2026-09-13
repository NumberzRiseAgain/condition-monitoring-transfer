#!/usr/bin/env python3
"""The two thresholds the topic states as numbers, measured rather than asserted.

    "<1 second anomaly-detection latency"
    "<10 MB/hour data-transfer footprints to shore when intermittent links
     reconnect"

Both are checkable by an evaluator in seconds against our text, and an unmeasured
threshold reads as an unread one. This measures both end to end.

HONESTY ABOUT THE PLATFORM. This runs on a general-purpose host, pinned
to a single core to approximate the compute budget of a small edge box. It is
NOT ruggedized hardware and no claim about MIL-SPEC hardware may rest on it.
What it does establish is the shape of the number: whether the pipeline is three
orders of magnitude inside the budget or three orders outside it. Single-core
pinning is the conservative direction — a fielded edge box would be given the
whole device.

WHAT IS TIMED. Everything between a captured waveform arriving in memory and a
decision existing: demodulation in the frozen band, the envelope spectrum,
prominence at every predicted fault line, the robust-baseline comparison, the
sequential test, and serialisation of the evidence record. Not timed: file I/O
from the .mat container, which is an artefact of how the public dataset ships
and would be a ring buffer on the asset.

A DISTRIBUTION, NOT A MEAN. A latency budget is a promise about the worst case a
maintainer will meet, so the mean is the least interesting statistic here. p50,
p95, p99 and max are reported.

Usage:
    PYTHONPATH=src taskset -c 0 python3 tools/bench_edge.py
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.io.paderborn import _channels                            # noqa: E402
from cbmx.physics.bearing import get as get_bearing                # noqa: E402
from cbmx.physics.envelope import Band, envelope_spectrum          # noqa: E402
from cbmx.health.baseline import Baseline, Regime                  # noqa: E402
from cbmx.health.sequential import Channels, WaldAccumulator       # noqa: E402
from cbmx.report import Alternative, Report                        # noqa: E402

FS = 64000.0
BAND = Band(500.0, 2000.0, 0.0, 0.0)
PARTS = ("outer_race", "inner_race", "rolling_element", "cage")


def cpu_name() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:                                    # noqa: BLE001
        pass
    return platform.processor() or "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/paderborn/KA04")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--arrestments-per-hour", type=float, default=60.0,
                    help="reports/hour assumed for the bandwidth projection")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    b = get_bearing("PU6203")
    files = sorted(glob.glob(str(Path(a.data) / "*.mat")))[:a.n]
    if not files:
        print(f"no records under {a.data}")
        return 1

    # Pre-load waveforms and pre-warm the baseline, so the timed section is the
    # decision path and not dataset unpacking or first-call import cost.
    from scipy.io import loadmat
    waves = []
    for p in files:
        m = loadmat(p, squeeze_me=True, struct_as_record=False)
        k = next(x for x in m if not x.startswith("__"))
        ch = _channels(m[k])
        waves.append((np.ascontiguousarray(ch["vibration_1"], dtype=np.float64),
                      float(np.mean(ch["speed"])) / 60.0, Path(p).stem))

    base = Baseline("bench", min_samples=5)
    r0 = Regime(0, 0)
    for part in PARTS:
        for _ in range(30):
            base.observe(part, r0, 5.0)

    # warm-up, excluded from the statistics
    for x, fr, _ in waves[:2]:
        envelope_spectrum(x, FS, fr, band=BAND)

    lat_ms, sizes_compact, sizes_full = [], [], []
    wald = WaldAccumulator(0.02, 0.10, leak_per_obs=1.0)

    for x, fr, stem in waves:
        t0 = time.perf_counter()

        spec = envelope_spectrum(x, FS, fr, band=BAND)
        zs, meas = {}, {}
        for part in PARTS:
            order = b.orders()[part]
            where, db = spec.prominence_db(order)
            meas[part] = (order, where, db)
            z, run = base.score(part, r0, db)
            zs[part] = z if run.ready else float("nan")

        lead = max(PARTS, key=lambda p: (zs[p] if np.isfinite(zs[p]) else -9e9))
        e = zs[lead]
        level = max(-1.0, min(1.5, (e - 3.0) / 3.0))
        others = [zs[p] for p in PARTS if p != lead and np.isfinite(zs[p])]
        ch = Channels(level, 0.8 if e > 0 else -0.7,
                      0.6 if others and max(others) < 1.5 else 0.0,
                      {"z": round(float(e), 2)})
        j = wald.observe(lead, ch)

        order, where, db = meas[lead]
        rep = Report(
            asset="AAG/HPU/pump_motor", lru="drive-end bearing",
            part=lead, stock_number="6203-2RS",
            order_predicted=order, freq_predicted_hz=order * fr,
            freq_measured_hz=where * fr, shaft_hz=fr,
            z=float(e), regime=r0.key, baseline_n=base.get(lead, r0).n,
            baseline_frozen=base.get(lead, r0).frozen,
            decision=j.decision, evidence_nats=j.S, boundary_nats=j.upper,
            observations=j.n_obs, false_alarm_rate=0.02, miss_rate=0.10,
            alternatives=[Alternative(p, float(zs[p]), "below operating point")
                          for p in PARTS if p != lead],
            sensor="vibration_1", band_hz=[BAND.lo, BAND.hi],
            capture_id=stem, method="envelope/frozen-band",
            prev_hash="")
        payload = rep.serialise(compact=True)

        lat_ms.append((time.perf_counter() - t0) * 1000.0)
        sizes_compact.append(len(payload))
        sizes_full.append(len(rep.serialise(compact=False)))

    lat_ms.sort()

    def pct(p):
        return lat_ms[min(len(lat_ms) - 1, int(round(p / 100 * len(lat_ms))))]

    med_bytes = int(statistics.median(sizes_compact))
    per_hour_mb = med_bytes * a.arrestments_per_hour / 1e6
    dur_s = waves[0][0].size / FS

    print(f"platform      {cpu_name()}")
    print(f"              single core (affinity {sorted(os.sched_getaffinity(0))}), "
          f"Python {platform.python_version()}, NumPy {np.__version__}")
    print("              NOT ruggedized hardware — see module docstring")
    print(f"capture       {dur_s:.2f} s at {FS/1000:.0f} kHz "
          f"= {waves[0][0].size:,} samples, {waves[0][0].nbytes/1e6:.1f} MB in memory")
    print(f"records       {len(waves)}\n")

    print("LATENCY  capture in memory -> decision + serialised record")
    print(f"   p50 {pct(50):8.1f} ms      p95 {pct(95):8.1f} ms")
    print(f"   p99 {pct(99):8.1f} ms      max {lat_ms[-1]:8.1f} ms")
    print(f"   topic threshold 1000 ms -> margin {1000/pct(99):.0f}x at p99\n")

    print("EVIDENCE RECORD  what actually crosses the link")
    print(f"   compact (fields only)      {med_bytes:,} bytes")
    print(f"   with rendered sentence     {int(statistics.median(sizes_full)):,} bytes")
    print(f"   raw capture, for contrast  {waves[0][0].nbytes:,} bytes "
          f"({waves[0][0].nbytes/med_bytes:,.0f}x larger)\n")

    print(f"BANDWIDTH  at {a.arrestments_per_hour:.0f} reports/hour")
    print(f"   {per_hour_mb:.3f} MB/hour against a 10 MB/hour budget "
          f"-> {10/per_hour_mb:.0f}x margin")
    print(f"   budget would be reached at "
          f"{10e6/med_bytes:,.0f} reports/hour")

    out = {"cpu": cpu_name(), "cores_used": 1, "n_records": len(waves),
           "capture_seconds": dur_s, "fs_hz": FS,
           "latency_ms": {"p50": pct(50), "p95": pct(95), "p99": pct(99),
                          "max": lat_ms[-1]},
           "record_bytes_compact": med_bytes,
           "record_bytes_with_sentence": int(statistics.median(sizes_full)),
           "reports_per_hour_assumed": a.arrestments_per_hour,
           "mb_per_hour": per_hour_mb,
           # 12 Sep 2026: this string used to say "x86" whatever the host was, while
           # the cpu field above was read from the machine. A run on an aarch64 host
           # therefore shipped a caveat that contradicted its own provenance. The
           # architecture is now read from the same place as everything else.
           "host_arch": platform.machine(),
           "caveat": f"general-purpose {platform.machine()} host pinned to one core; "
                     "NOT ruggedized hardware; file I/O excluded"}
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
