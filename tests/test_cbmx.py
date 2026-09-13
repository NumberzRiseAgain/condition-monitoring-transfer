"""Tests for the places where being wrong would be silent.

The physics tests are regressions against published multipliers: if our
arithmetic ever drifts from the numbers the dataset's own documentation states,
that must be a failure and not a quiet disagreement a reviewer discovers first.

The rest test the traps — the factor of two, the geometry that does not
self-consistently describe a real bearing, the self-tuning baseline that learns
to accept a developing fault, and the window length that cannot meet its own
latency requirement.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cbmx.graph.asset import Asset                                  # noqa: E402
from cbmx.health.baseline import Baseline          # noqa: E402
from cbmx.health.evidence import SPRT, Observation                  # noqa: E402
from cbmx.io.synth import SynthSpec, generate                       # noqa: E402
from cbmx.physics.bearing import CATALOGUE, PUBLISHED, Bearing, get  # noqa: E402
from cbmx.physics.envelope import commission_band, envelope_spectrum  # noqa: E402


# ── physics ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_matches_published_multipliers(name):
    """Only bearings whose multipliers the dataset itself publishes can be
    regression-tested this way. The rest are covered by the identities below —
    and MFPT's is verified separately, because its published figures are in Hz
    at a stated shaft speed rather than as multipliers."""
    b, pub = get(name), PUBLISHED[name]
    o = b.orders()
    for part, expected in pub.items():
        assert o[part] == pytest.approx(expected, abs=0.001), (
            f"{name} {part}: ours {o[part]:.4f} vs published {expected:.4f}. "
            "The geometry in the catalogue is wrong, not the formulae."
        )


@pytest.mark.parametrize("name", sorted(CATALOGUE))
def test_kinematic_identities(name):
    for k, ok in get(name).check().items():
        assert ok, f"{name}: identity {k} violated"


def test_ball_defect_is_twice_the_spin_rate():
    """The factor-of-two trap. Published tables quote the doubled figure under
    the name BSF; searching at the spin rate finds nothing and a failing bearing
    reads as healthy."""
    b = get("SKF6205")
    assert b.ball_defect == pytest.approx(2 * b.bsf)
    assert b.orders()["rolling_element"] == pytest.approx(b.ball_defect)
    assert b.bsf == pytest.approx(2.3567, abs=0.001)


def test_6203_has_eight_balls_not_nine():
    """Its neighbour in the same rig has nine, which makes this an easy and
    entirely silent mistake: with nine, none of the published multipliers can be
    reconciled with each other."""
    assert get("SKF6203").n_rolling == 8
    wrong = Bearing("wrong", 9, 0.2656, 1.122)
    assert abs(wrong.bpfo - PUBLISHED["SKF6203"]["outer_race"]) > 0.3


def test_mfpt_geometry_reproduces_its_published_frequencies():
    """MFPT publishes 81.12 Hz outer and 118.875 Hz inner at a 25 Hz shaft.
    A transcribed geometry you have not checked against something the dataset
    states is a guess wearing a lab coat."""
    b = get("MFPTNICE")
    assert b.bpfo * 25.0 == pytest.approx(81.12, abs=0.02)
    assert b.bpfi * 25.0 == pytest.approx(118.875, abs=0.02)


def test_every_catalogue_bearing_is_self_consistent():
    for name in CATALOGUE:
        bad = [k for k, ok in get(name).check().items() if not ok]
        assert not bad, f"{name}: {bad}"


def test_geometry_is_validated():
    with pytest.raises(ValueError):
        Bearing("silly", 9, 2.0, 1.0)          # ball larger than pitch circle
    with pytest.raises(ValueError):
        Bearing("silly", 2, 0.3, 1.5)          # not enough rolling elements


def test_inner_always_faster_than_outer():
    for name in CATALOGUE:
        b = get(name)
        assert b.bpfi > b.bpfo
        assert b.bpfo == pytest.approx(b.n_rolling * b.ftf)


def test_sidebands_only_where_the_defect_moves():
    from cbmx.physics.bearing import harmonics_for
    b = get("SKF6205")
    assert harmonics_for(b, "outer_race").sidebands == 0, (
        "a stationary defect does not modulate; expecting sidebands there "
        "invites false confidence from neighbouring noise"
    )
    assert harmonics_for(b, "inner_race").sidebands > 0


# ── detection ──────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def bench():
    asset = Asset.load(str(ROOT / "configs" / "asset_motor_bench.yaml"))
    b = asset.bearings["de_bearing"]
    recs = [generate(SynthSpec(bearing=b, fault=f, severity=2.0, seed=s))[0]
            for s, f in ((1, "outer_race"), (2, "inner_race"))]
    band = commission_band(recs, 12000.0, b.orders(), 29.17)
    asset.sensors["acc_de"].band_lo_hz = band.lo
    asset.sensors["acc_de"].band_hi_hz = band.hi
    return asset, b, band


@pytest.mark.parametrize("fault", ["outer_race", "inner_race", "rolling_element", "cage"])
def test_planted_fault_is_the_strongest_line(bench, fault):
    asset, b, band = bench
    spec = SynthSpec(bearing=b, fault=fault, severity=2.0, seed=17)
    x, _, _ = generate(spec)
    es = envelope_spectrum(x, spec.fs, spec.shaft_hz, band=band)
    scores = {p: es.prominence_db(o)[1] for p, o in b.orders().items()}
    assert max(scores, key=scores.get) == fault, scores


def test_healthy_machine_produces_no_report(bench):
    """The test most likely to embarrass us, so it is in the suite.

    Note what is NOT asserted: that a healthy record has no standout line. It
    does — noise routinely puts one candidate 4-5 dB above the others, and an
    earlier version of this test failed for exactly that reason. That failure
    was the test being wrong, not the system. Nothing decides on raw dB. The
    decision is made on how far a line sits above *its own* learned baseline,
    which is why a few dB of noise spread is ordinary and reported as nothing.
    """
    from cbmx.monitor import Monitor
    asset, b, band = bench
    bl = Baseline(asset.id)
    learn = Monitor(asset, "acc_de", bl, window_s=0.75)
    t = 0.0
    for seed in range(6):
        x, _, _ = generate(SynthSpec(bearing=b, fault="none", seconds=20.0, seed=200 + seed))
        for i in range(0, len(x) - 9000 + 1, 9000):
            learn.step(t, x[i:i + 9000], 29.17, load=0.5, learning=True)
            t += 0.75

    watch = Monitor(asset, "acc_de", bl, window_s=0.75)
    x, _, _ = generate(SynthSpec(bearing=b, fault="none", seconds=40.0, seed=999))
    for i in range(0, len(x) - 9000 + 1, 9000):
        watch.step(t, x[i:i + 9000], 29.17, load=0.5)
        t += 0.75
    assert watch.findings == [], (
        f"false alarm on a healthy machine: "
        f"{[f.symptom.id for f in watch.findings]}"
    )


def test_measured_order_lands_on_the_predicted_one(bench):
    asset, b, band = bench
    spec = SynthSpec(bearing=b, fault="outer_race", severity=4.0, seed=5)
    x, _, _ = generate(spec)
    es = envelope_spectrum(x, spec.fs, spec.shaft_hz, band=band)
    order, _ = es.peak(b.bpfo)
    assert abs(order - b.bpfo) / b.bpfo < 0.02


# ── baseline ───────────────────────────────────────────────────────────────
def test_baseline_refuses_to_score_until_it_has_seen_enough():
    bl = Baseline("a")
    r = bl.regime(29.0)
    for _ in range(5):
        bl.observe("s", r, 3.0)
    z, run = bl.score("s", r, 40.0)
    assert not run.ready and z == 0.0, "must not emit a confident z from 5 samples"


def test_baseline_is_robust_to_a_few_large_readings():
    """Mean and standard deviation would be dragged up by the very readings we
    are looking for, raising the threshold that was meant to catch them."""
    bl = Baseline("a")
    r = bl.regime(29.0)
    rng = np.random.default_rng(0)
    for _ in range(200):
        bl.observe("s", r, float(rng.normal(3.0, 1.0)))
    clean = bl.get("s", r).median
    for _ in range(8):
        bl.observe("s", r, 60.0)
    assert bl.get("s", r).median - clean < 3.0


def test_frozen_baseline_stops_learning():
    """Without this a slow degradation is absorbed as the new normal and nothing
    is ever reported — silently."""
    bl = Baseline("a")
    r = bl.regime(29.0)
    for _ in range(60):
        bl.observe("s", r, 3.0)
    bl.freeze("s", r)
    before = bl.get("s", r).median
    for _ in range(200):
        bl.observe("s", r, 45.0)
    assert bl.get("s", r).median == pytest.approx(before)


def test_regimes_are_kept_apart():
    bl = Baseline("a")
    lo, hi = bl.regime(15.0), bl.regime(45.0)
    assert lo.key != hi.key
    for _ in range(60):
        bl.observe("s", lo, 3.0)
        bl.observe("s", hi, 20.0)
    assert bl.score("s", lo, 20.0)[0] > 3.0     # loud for the slow regime
    assert bl.score("s", hi, 20.0)[0] < 1.0     # ordinary for the fast one


# ── evidence ───────────────────────────────────────────────────────────────
def _obs(**kw):
    d = dict(symptom_id="s", t=0.0, z_level=6.0, order_predicted=3.585,
             order_measured=3.585, sideband_ratio=0.05, expect_sidebands=False,
             baseline_ready=True)
    d.update(kw)
    return Observation(**d)


def test_thresholds_come_from_alpha_and_beta():
    import math
    s = SPRT(false_alarm_rate=0.02, miss_rate=0.10)
    assert s.upper == pytest.approx(math.log(0.9 / 0.02))
    assert s.lower == pytest.approx(math.log(0.10 / 0.98))
    assert SPRT(false_alarm_rate=0.001).upper > s.upper


def test_never_fires_on_a_single_observation():
    s = SPRT()
    assert s.observe(_obs(z_level=40.0)).decision == "watching"


def test_sustained_evidence_eventually_names_the_part():
    s = SPRT()
    d = [s.observe(_obs(t=i * 0.75)).decision for i in range(40)]
    assert "name_it" in d


def test_a_quiet_line_is_ruled_out_rather_than_left_undecided():
    """Reaching the lower boundary is what lets the system stop paying attention
    to a line. Without it everything stays 'undecided' forever."""
    s = SPRT()
    last = None
    for i in range(40):
        last = s.observe(_obs(t=i * 0.75, z_level=0.0, sideband_ratio=0.5))
    assert last.decision == "ordinary"


def test_wrong_sidebands_argue_against_the_diagnosis():
    # Four observations, deliberately: past the boundary both accumulators
    # saturate against the clamp and compare equal, which says nothing.
    s1, s2 = SPRT(), SPRT()
    for i in range(4):
        a = s1.observe(_obs(t=i, expect_sidebands=True, sideband_ratio=0.6))
        b = s2.observe(_obs(t=i, expect_sidebands=True, sideband_ratio=0.02))
    assert a.S > b.S
    assert a.channels["sideband"] > 0 > b.channels["sideband"]


def test_a_peak_in_the_wrong_place_argues_against_it():
    s1, s2 = SPRT(), SPRT()
    for i in range(4):
        a = s1.observe(_obs(t=i, order_measured=3.585))
        b = s2.observe(_obs(t=i, order_measured=3.9))     # 9% off — another line
    assert a.S > b.S


def test_no_baseline_means_no_verdict():
    assert SPRT().observe(_obs(baseline_ready=False)).decision == "no_baseline"


# ── the contract numbers ───────────────────────────────────────────────────
def test_monitor_refuses_an_uncommissioned_sensor():
    """Selecting the demodulation band at run time is a multiple-comparisons
    trap: pick the band that best shows a fault, then report the fault."""
    from cbmx.monitor import Monitor
    asset = Asset.load(str(ROOT / "configs" / "asset_motor_bench.yaml"))
    with pytest.raises(ValueError, match="commission"):
        Monitor(asset, "acc_de")


def test_default_window_leaves_room_under_one_second():
    """A 1.0 s window cannot satisfy a sub-1 s requirement however fast the
    code is. The first version of this system reported 1.002 s and failed."""
    from cbmx.monitor import Monitor
    import inspect
    sig = inspect.signature(Monitor.__init__)
    assert sig.parameters["window_s"].default < 1.0


def test_link_budget_accounting_is_byte_exact():
    from cbmx.monitor import LinkBudget
    b = LinkBudget(10 * 1024 * 1024)
    for i in range(10):
        b.spend(float(i), 800)
    assert b.total == 8000
    r = b.report(3600.0)
    assert r["bytes_per_hour"] == pytest.approx(8000.0)
    assert r["within_budget"]
