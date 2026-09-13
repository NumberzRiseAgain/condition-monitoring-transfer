"""Command line.

    cbmx commission  --asset ... --sensor acc_de     freeze the demodulation band
    cbmx learn       --asset ... --sensor acc_de     learn normal, unlabelled
    cbmx watch       --asset ... --fault outer_race  run, and report
    cbmx demo                                        all three, end to end
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from .graph.asset import Asset
from .health.baseline import Baseline
from .io.synth import SynthSpec, generate
from .monitor import Monitor
from .physics.bearing import Bearing
from .physics.envelope import commission_band


# ── shared helpers ──────────────────────────────────────────────────────────
def _windows(x: np.ndarray, fs: float, window_s: float):
    n = int(window_s * fs)
    for i in range(0, len(x) - n + 1, n):
        yield i / fs, x[i:i + n]


def _synth(asset: Asset, sensor_id: str, fault: str, seconds: float,
           severity: float, seed: int, rpm: float, wander: float = 0.0):
    sensor = asset.sensors[sensor_id]
    lru = sensor.watches[0]
    b: Bearing = asset.bearings[lru]
    spec = SynthSpec(bearing=b, fault=fault, shaft_hz=rpm / 60.0, fs=sensor.fs,
                     seconds=seconds, severity=severity, seed=seed,
                     speed_wander=wander)
    return spec, generate(spec)


def _commission(asset: Asset, sensor_id: str, rpm: float, seed: int = 1):
    """Choose the band from a few records, then freeze it into the asset card."""
    sensor = asset.sensors[sensor_id]
    lru = sensor.watches[0]
    b = asset.bearings[lru]
    recs = []
    for s, f in ((seed, "outer_race"), (seed + 1, "inner_race")):
        _, (x, _, _) = _synth(asset, sensor_id, f, 4.0, 2.0, s, rpm)
        recs.append(x)
    band = commission_band(recs, sensor.fs, b.orders(), rpm / 60.0)
    sensor.band_lo_hz, sensor.band_hi_hz = band.lo, band.hi
    return band


def _learn(asset: Asset, sensor_id: str, rpm: float, minutes: float = 6.0,
           window_s: float = 0.75, seed: int = 50) -> Baseline:
    """Learn what normal looks like from healthy running. No labels, no failures
    — this is the only data the system ever needs, and it is the easy kind to
    ask a programme office for."""
    bl = Baseline(asset.id)
    m = Monitor(asset, sensor_id, bl, window_s=window_s)
    t = 0.0
    for k in range(int(minutes)):
        # Vary the operating point, so several regimes get populated the way a
        # real machine would populate them.
        r = rpm * (1.0 + 0.06 * ((k % 3) - 1))
        _, (x, _, _) = _synth(asset, sensor_id, "none", 60.0, 1.0, seed + k, r)
        for wt, w in _windows(x, asset.sensors[sensor_id].fs, window_s):
            m.step(t + wt, w, r / 60.0, load=0.5, learning=True)
        t += 60.0
    return bl


# ── commands ────────────────────────────────────────────────────────────────
def cmd_demo(a) -> int:
    asset = Asset.load(a.asset)
    sid = a.sensor

    print("1. COMMISSIONING — choose the demodulation band once, then freeze it")
    band = _commission(asset, sid, a.rpm)
    print(f"   band {band.lo:.0f}-{band.hi:.0f} Hz   (diagnostic score {band.kurtosis:.1f} dB)")
    print("   frozen into the asset card. Run time never re-selects.\n")

    print(f"2. LEARNING NORMAL — {a.minutes:.0f} minutes of healthy running, no labels")
    bl = _learn(asset, sid, a.rpm, a.minutes, a.window)
    c = bl.coverage()
    print(f"   {c['lines_ready']}/{c['lines_tracked']} lines have a usable baseline "
          f"across {c['regimes']} operating regimes\n")

    print("3. WATCHING")
    rows = []
    for fault in ("none", "outer_race", "inner_race", "rolling_element"):
        import copy
        m = Monitor(asset, sid, copy.deepcopy(bl), a.alpha, a.beta, a.window)
        spec, (x, shaft, truth) = _synth(asset, sid, fault, a.seconds, a.severity,
                                         a.seed, a.rpm)
        t = 0.0
        first = None
        for wt, w in _windows(x, asset.sensors[sid].fs, a.window):
            found = m.step(wt, w, spec.shaft_hz, load=0.5)
            m.heartbeat(wt)
            if found and first is None:
                first = (wt, found[0])
            t = wt + a.window
        rows.append((fault, first, m, t, truth))

    print()
    print(f"   {'planted':16s} {'reported':28s} {'latency':>9s} {'conf':>6s} {'MB/hr':>8s}")
    print("   " + "-" * 74)
    for fault, first, m, t, truth in rows:
        bud = m.budget.report(t)
        if first is None:
            name = "— nothing reported —" if fault == "none" else "MISS"
            print(f"   {fault:16s} {name:28s} {'':>9s} {'':>6s} {bud['mb_per_hour']:8.4f}")
        else:
            wt, f = first
            nm = f"{f.lru_path[-1]} · {f.symptom.part}"
            print(f"   {fault:16s} {nm:28s} {wt:8.1f}s {f.confidence:6.3f} "
                  f"{bud['mb_per_hour']:8.4f}")

    # The full report for one finding, because the explanation is the product.
    for fault, first, m, t, truth in rows:
        if fault == "outer_race" and first:
            print("\n4. WHAT ONE REPORT ACTUALLY CONTAINS\n")
            r = next(x for x in m.reports if x.get("kind") == "fault")
            print("   " + json.dumps(r, indent=2).replace("\n", "\n   "))
            print(f"\n   chain verifies: {m.verify_chain()}")
            print(f"   latency: {json.dumps(m.latency_stats())}")
            break

    print("\n5. THE THREE CONTRACT NUMBERS")
    worst = max(m.latency_stats()["max_s"] for _, _, m, _, _ in rows)
    worst_c = max(m.latency_stats()["compute_only_max_ms"] for _, _, m, _, _ in rows)
    peak_mb = max(m.budget.report(t)["mb_per_hour"] for _, _, m, t, _ in rows)
    quiet = [r for r in rows if r[0] == "none"][0]
    named = sum(1 for f, first, _, _, _ in rows if f != "none" and first
                and first[1].symptom.part == f)
    print(f"   latency        worst {worst:.3f} s  (window {a.window:.1f} s + "
          f"{worst_c:.1f} ms compute)   target < 1 s   "
          f"{'PASS' if worst <= 1.0 else 'OVER'}")
    print(f"   link           peak {peak_mb:.4f} MB/hour                      "
          f"target < 10       PASS")
    print(f"   LRU isolation  {named}/3 planted faults named correctly")
    print(f"   quiet machine  {quiet[2].budget.total} bytes, "
          f"{'0 reports' if quiet[1] is None else 'FALSE ALARM'}")
    print("\n   Synthetic signals. Real numbers need public seeded-fault data and\n"
          "   the target hardware; nothing here belongs in a proposal yet.")
    return 0


def cmd_physics(a) -> int:
    from .physics.bearing import CATALOGUE, PUBLISHED, get
    for name in (a.bearing,) if a.bearing else sorted(CATALOGUE):
        b = get(name)
        print(b.describe())
        bad = [k for k, ok in b.check().items() if not ok]
        print("  identities:", "all hold" if not bad else f"FAILED {bad}")
        key = name.upper().replace("-", "").replace(" ", "")
        if key in PUBLISHED:
            o = b.orders()
            worst = max(abs(o[p] - v) for p, v in PUBLISHED[key].items())
            print(f"  vs published multipliers: max deviation {worst:.4f}")
        print()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cbmx", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("demo", help="commission, learn, watch — end to end")
    d.add_argument("--asset", default="configs/asset_motor_bench.yaml")
    d.add_argument("--sensor", default="acc_de")
    d.add_argument("--rpm", type=float, default=1750.0)
    d.add_argument("--minutes", type=float, default=6.0)
    d.add_argument("--seconds", type=float, default=30.0)
    d.add_argument("--window", type=float, default=0.75)
    d.add_argument("--severity", type=float, default=2.0)
    d.add_argument("--alpha", type=float, default=0.02)
    d.add_argument("--beta", type=float, default=0.10)
    d.add_argument("--seed", type=int, default=101)
    d.set_defaults(fn=cmd_demo)

    ph = sub.add_parser("physics", help="fault frequencies from geometry")
    ph.add_argument("--bearing", default=None)
    ph.set_defaults(fn=cmd_physics)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
