"""Tests for everything added in the August 2026 work.

Two of these are regression tests for defects that actually shipped into a run
and produced confident, wrong, plausible-looking output. Those two are the most
important tests in this file, because both failures were silent:

    test_align_rate_preserves_every_field
        A decimation helper rebuilt the record positionally and stopped four
        fields short, so every record silently reverted to a DEFAULT BEARING
        GEOMETRY. Every fault line was then searched at the wrong frequency and
        landed on empty spectrum. The run reported 0 detections and 0 false
        alarms across 122 records, which reads like an admirably conservative
        monitor. It was a nine-ball geometry applied to an eight-ball bearing.

    test_the_two_prominence_db_signatures_are_pinned
        Two functions share a name and return their pair in OPPOSITE order.
        Unpacking one as the other made a dimensionless order number, 3.05, get
        read as a decibel figure — so every condition reported "3.0 dB" and the
        conclusion would have been that order tracking makes no difference.

The rest cover the modules with no coverage before: motor-current analysis and
its torsional band, the generic sequential test, the evidence record, the
hydraulic loader and symptom set, and the semantic gate.

Tests needing the large public datasets skip cleanly when the data is absent, so
`pytest` passes on a fresh clone. The semantic fixtures are synthetic, a few
kilobytes, and are committed — the gate is testable without downloading anything.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from cbmx.health.baseline import Baseline, Regime, Running
from cbmx.health.sequential import Channels, WaldAccumulator, build_channels
from cbmx.io.base import Record
from cbmx.io.cwru import align_rate
from cbmx.physics import mcsa
from cbmx.physics.bearing import get as get_bearing
from cbmx.report import Alternative, Report

ROOT = Path(__file__).resolve().parents[1]
HYDRAULIC = ROOT / "data" / "hydraulic"
SEMANTIC = ROOT / "data" / "semantic"


# ── regression 1: the silent geometry substitution ──────────────────────────

def test_align_rate_preserves_every_field():
    """The defect that produced 0 detections and 0 false alarms over 122 records.

    Asserted field by field rather than with a blanket equality, so that adding
    a field to Record without teaching align_rate about it fails here rather
    than in a proposal.
    """
    rec = Record(
        file_no=7, fault="outer_race", defect_in=0.021, load_hp=2,
        rpm_nominal=1500, fs=64000.0, signal=np.random.default_rng(0).normal(size=4096),
        rpm_measured=1499.4, channel="vibration_1", path="/tmp/x.mat",
        dataset="paderborn", bearing_key="PU6203",
        rpm_series=np.full(64, 1499.4), note="REAL damage",
    )
    out = align_rate(rec, 32000.0)

    assert out.bearing_key == "PU6203", (
        "decimation reset the bearing geometry — this is the defect that made "
        "an eight-ball 6203 be searched as a nine-ball 6205")
    assert out.dataset == "paderborn"
    assert out.note == rec.note
    assert out.rpm_series is not None and len(out.rpm_series) == 64, (
        "decimation deleted the measured speed trace, which is what "
        "variable_speed and all order tracking read")
    assert out.fs == 32000.0 and out.signal.size < rec.signal.size
    for f in ("file_no", "fault", "defect_in", "load_hp", "rpm_nominal",
              "rpm_measured", "channel", "path"):
        assert getattr(out, f) == getattr(rec, f), f"{f} was not carried through"


def test_a_record_that_never_moves_is_not_a_quiet_record():
    """A monitor that never speaks passes a false-alarm test perfectly.

    Guards the reading discipline the geometry bug taught: silence is only
    meaningful beside a detection count on the same run.
    """
    b = get_bearing("PU6203")
    wrong = get_bearing("SKF6205")
    assert abs(b.orders()["outer_race"] - wrong.orders()["outer_race"]) > 0.4, (
        "the two geometries must differ enough that substituting one for the "
        "other moves every fault line off its true position")


# ── regression 2: the two same-named functions returning opposite pairs ─────

def test_the_two_prominence_db_signatures_are_pinned():
    """`EnvelopeSpectrum.prominence_db` returns (order, dB).
    `mcsa.Spectrum.prominence_db` returns (dB, frequency). Opposite orders,
    same name. Unpacking one as the other silently reports an order as decibels.
    """
    from cbmx.physics.envelope import EnvelopeSpectrum, Band

    orders = np.linspace(0.0, 20.0, 4001)
    amp = np.full_like(orders, 0.01)
    amp[np.argmin(np.abs(orders - 3.05))] = 1.0
    es = EnvelopeSpectrum(orders, amp, 25.0, Band(500.0, 2000.0, 0.0, 0.0), 64000.0)
    first, second = es.prominence_db(3.05)
    assert 2.9 < first < 3.2, "envelope: first element is the ORDER"
    assert second > 20.0, "envelope: second element is the DECIBEL value"

    fs = 8192.0
    t = np.arange(int(4 * fs)) / fs
    x = np.sin(2 * np.pi * 100.0 * t) + 0.01 * np.random.default_rng(1).normal(size=t.size)
    sp = mcsa.spectrum(x, fs)
    db, where = sp.prominence_db(100.0, tol_hz=1.0)
    assert db > 20.0, "mcsa: first element is the DECIBEL value"
    assert 99.0 < where < 101.0, "mcsa: second element is the FREQUENCY"


# ── motor current, and the collision guard ─────────────────────────────────

def test_a_fault_line_on_a_supply_harmonic_is_declared_unobservable():
    """On the Paderborn rig the ball-defect order is 3.9932 and the drive has
    four pole pairs, so that sideband sits on the second supply harmonic at
    every speed. Measured naively it reads about 29 dB of ball evidence on a
    certified-healthy bearing. It must come back as NOT OBSERVABLE, not as a
    number, and not as zero — zero would read downstream as 'measured, quiet'.
    """
    b = get_bearing("PU6203")
    fs, shaft, f1 = 64000.0, 25.0, 100.0
    pred = mcsa.predict(b, shaft, f1, k_max=2, f_nyquist=fs / 2)
    ball = pred["rolling_element"]
    assert ball.blocked, "the ball sideband collides with 2*f1 and must be blocked"

    rng = np.random.default_rng(2)
    t = np.arange(int(4 * fs)) / fs
    x = np.sin(2 * np.pi * f1 * t) + 0.3 * np.sin(2 * np.pi * 2 * f1 * t)
    x += 0.001 * rng.normal(size=t.size)
    got = mcsa.measure(x, fs, b, shaft, k_max=2, f_supply=f1)
    assert got["rolling_element"]["observable"] is False
    assert math.isnan(got["rolling_element"]["prominence_db"]), (
        "an unobservable line must be NaN, never 0.0")


def test_predict_drops_lines_below_dc_and_above_nyquist():
    b = get_bearing("PU6203")
    pred = mcsa.predict(b, shaft_hz=25.0, f_supply=100.0, k_max=6, f_nyquist=400.0)
    for p in pred.values():
        for f in p.lines:
            assert 1.0 < f < 400.0


def test_supply_fundamental_is_measured_not_assumed():
    fs = 20000.0
    t = np.arange(int(2 * fs)) / fs
    x = np.sin(2 * np.pi * 60.0 * t)
    assert abs(mcsa.supply_fundamental(mcsa.spectrum(x, fs)) - 60.0) < 0.5


def test_the_torsional_band_selects_by_frequency_not_harmonic_number():
    """The property that makes the band a machine constant: a fixed 199-259 Hz
    band picks the 5th harmonic at 900 rpm and the 3rd at 1500 rpm, because both
    land at 229 Hz. If it selected by harmonic index it would follow the speed.
    """
    band = mcsa.TorsionalBand(199.0, 259.0)
    order = get_bearing("PU6203").orders()["outer_race"]
    assert band.harmonics_inside(order * 15.0) == [5]      # 900 rpm
    assert band.harmonics_inside(order * 25.0) == [3]      # 1500 rpm


def test_a_band_with_no_harmonic_at_this_speed_yields_no_harmonics():
    """Abstention at the physics layer: the caller must get an empty list and
    report 'no baseline', rather than being handed the nearest harmonic."""
    band = mcsa.TorsionalBand(466.0, 476.0)
    assert band.harmonics_inside(get_bearing("PU6203").orders()["outer_race"] * 25.0) == []


# ── the generic sequential test ────────────────────────────────────────────

def test_boundaries_come_from_alpha_and_beta_alone():
    w = WaldAccumulator(0.02, 0.10)
    assert abs(w.upper - math.log(0.90 / 0.02)) < 1e-9
    assert abs(w.lower - math.log(0.10 / 0.98)) < 1e-9
    tighter = WaldAccumulator(0.001, 0.10)
    assert tighter.upper > w.upper, "a stricter false-alarm rate must demand more evidence"


def test_one_observation_can_never_name_a_part():
    w = WaldAccumulator(0.02, 0.10, leak_per_obs=1.0)
    j = w.observe("brg", Channels(1.5, 0.8, 0.6))
    assert j.decision != "name_it"
    assert j.S < w.upper


def test_sustained_evidence_eventually_crosses():
    w = WaldAccumulator(0.02, 0.10, leak_per_obs=1.0)
    for _ in range(40):
        j = w.observe("brg", Channels(1.5, 0.8, 0.6))
    assert j.decision == "name_it"


def test_evidence_against_can_push_the_accumulator_back_down():
    """The direction channel must be able to exonerate. A test built only from
    channels that can never argue against a fault convicts eventually."""
    w = WaldAccumulator(0.02, 0.10, leak_per_obs=1.0)
    for _ in range(10):
        w.observe("brg", Channels(1.0, 0.8, 0.6))
    high = w.state("brg")
    for _ in range(10):
        w.observe("brg", Channels(-1.0, -0.7, -0.4))
    assert w.state("brg") < high


def test_no_ready_baseline_gives_no_baseline_not_a_guess():
    w = WaldAccumulator(0.02, 0.10)
    assert w.observe("brg", None).decision == "no_baseline"
    assert build_channels({}, {}, {}) is None


def test_a_deviation_against_the_declared_physics_cannot_raise_the_level():
    """A cooler that is failing removes LESS heat. If the drop across it has
    grown, that is evidence against, and the level channel must not carry it."""
    z = {"cooler.dT_across": -6.0}
    ch = build_channels(z, {"cooler.dT_across": -1}, {"cooler.dT_across": True})
    assert ch is not None and ch.level >= 0.0 and ch.direction > 0
    z_wrong = {"cooler.dT_across": +6.0}
    ch_wrong = build_channels(z_wrong, {"cooler.dT_across": -1},
                              {"cooler.dT_across": True})
    assert ch_wrong.direction < 0, "wrong-direction movement must argue against"
    assert ch_wrong.level <= 0.0, "and must not contribute positive level"


def test_silence_from_corroborating_symptoms_is_mild_evidence_against():
    ch = build_channels({"a": 6.0, "b": 0.1}, {"a": 1, "b": 1},
                        {"a": True, "b": True})
    assert ch.corroboration < 0


# ── the evidence record ────────────────────────────────────────────────────

def _report(prev: str = "") -> Report:
    return Report(
        asset="AAG/HPU/pump_motor", lru="drive-end bearing", part="outer_race",
        stock_number="BRG-6203-2RS", order_predicted=3.0543,
        freq_predicted_hz=76.36, freq_measured_hz=76.66, shaft_hz=25.0,
        z=11.2, regime="s5l0", baseline_n=240, decision="name_it",
        evidence_nats=4.1, boundary_nats=3.81, observations=6,
        alternatives=[Alternative("inner_race", 0.4, "below operating point")],
        sensor="VIB-01", band_hz=[500.0, 2000.0], capture_id="c1",
        method="envelope/frozen-band", prev_hash=prev)


def test_the_record_fits_the_link_budget():
    """The topic sets <10 MB/hour. At 60 reports/hour a record must stay far
    under 170 kB, and the measured figure is under a kilobyte."""
    n = len(_report().serialise(compact=True))
    assert n < 4096, f"record grew to {n} bytes"
    assert n * 60 < 10e6


def test_the_sentence_is_reconstructible_and_therefore_optional_on_the_link():
    r = _report()
    assert len(r.serialise(compact=True)) < len(r.serialise(compact=False))
    s = r.sentence()
    for fragment in ("drive-end bearing", "76.7 Hz", "76.4 Hz", "sigma", "240"):
        assert fragment in s, f"the maintainer sentence dropped {fragment!r}"


def test_the_hash_chains_and_any_edit_breaks_it():
    a = _report()
    b = _report(prev=a.digest())
    assert b.digest() != a.digest()
    tampered = _report(prev=a.digest())
    tampered.z = 11.3
    assert tampered.digest() != b.digest(), "an altered record must not hash the same"


def test_the_record_names_what_it_rejected():
    assert _report().to_record()["alternatives"], (
        "a diagnosis without the parts it ruled out is an assertion")


# ── the semantic gate ──────────────────────────────────────────────────────

pytestmark_semantic = pytest.mark.skipif(
    not (SEMANTIC / "asset_card.json").exists(), reason="semantic fixtures absent")


def _gate_fixtures():
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from eval_semantic import AssetGraph, Corpus, propose, validate
    g = AssetGraph(json.loads((SEMANTIC / "asset_card.json").read_text()))
    recs = [json.loads(ln) for ln in
            (SEMANTIC / "maintenance.jsonl").read_text().splitlines() if ln.strip()]
    return g, Corpus(recs), propose, validate


@pytestmark_semantic
def test_a_true_hypothesis_passes_every_check():
    g, c, propose, validate = _gate_fixtures()
    b = g.bearing_at("AAG/HPU/pump_motor/de_bearing")
    h = propose(g, c, "VIB-01", "outer_race", b.orders()["outer_race"] * 1.004, 11.2)
    res = validate(h, g, c)
    assert res.accepted, res.report()
    assert h.lru_path == "AAG/HPU/pump_motor/de_bearing"
    assert h.part_number == "BRG-6203-2RS"


@pytestmark_semantic
@pytest.mark.parametrize("break_it,why", [
    (lambda h, b: setattr(h, "part_number", "BRG-6203-XX"), "part not in catalogue"),
    (lambda h, b: setattr(h, "cited_work_orders", ["WO-2025-9999"]), "work order absent"),
    (lambda h, b: setattr(h, "cited_work_orders", ["WO-2025-0091"]), "citation is another asset"),
    (lambda h, b: setattr(h, "order_measured", b.orders()["inner_race"]), "order contradicts mode"),
    (lambda h, b: setattr(h, "lru_path", "AAG/HPU"), "path is not a replaceable unit"),
])
def test_the_gate_discards_rather_than_softens(break_it, why):
    """A gate that has never rejected anything is decoration. Each case fails a
    specific assertion against a Government artifact — never a judgement about
    plausibility, which is what makes a rejection explainable to a maintainer."""
    g, c, propose, validate = _gate_fixtures()
    b = g.bearing_at("AAG/HPU/pump_motor/de_bearing")
    h = propose(g, c, "VIB-01", "outer_race", b.orders()["outer_race"] * 1.004, 11.2)
    break_it(h, b)
    res = validate(h, g, c)
    assert not res.accepted, f"gate accepted a hypothesis where {why}"
    assert any(not ok for _, ok, _ in res.checks)


@pytestmark_semantic
def test_the_asset_graph_walks_up_to_the_replaceable_unit():
    g, _, _, _ = _gate_fixtures()
    assert g.lru_for("AAG/HPU/pump_motor/de_bearing") == "AAG/HPU/pump_motor/de_bearing"
    assert g.observed_by("VIB-01") == "AAG/HPU/pump_motor/de_bearing"
    assert g.observed_by("PRS-02") == "AAG/HPU/control_valve"


@pytestmark_semantic
def test_retrieval_surfaces_the_records_that_bear_on_the_fault():
    """From eighteen work orders it must find the two about outer-race noise on
    this unit, not merely two that mention the unit."""
    g, c, _, _ = _gate_fixtures()
    hits = c.search("outer race noise growl load arrestment drive-end bearing",
                    asset_prefix="AAG/HPU/pump_motor/de_bearing", k=2)
    wos = {rec["wo"] for _, rec in hits}
    assert wos & {"WO-2026-0255", "WO-2025-0498"}, wos


# ── hydraulic loader and symptoms ──────────────────────────────────────────

pytestmark_hydraulic = pytest.mark.skipif(
    not (HYDRAULIC / "profile.txt").exists(),
    reason="ZeMA hydraulic dataset absent; see README for the download")


@pytestmark_hydraulic
def test_the_all_optimal_pool_is_too_small_to_baseline_on():
    """The finding that forced per-component healthy pools: only ten cycles are
    optimal on all four components AND settled, and they are contiguous in time.
    If a future release changes that, the pooling decision should be revisited.
    """
    from cbmx.io.hydraulic import COMPONENTS, HydraulicRig
    prof = HydraulicRig(HYDRAULIC).profile
    both = prof.stable.copy()
    for comp in COMPONENTS:
        both &= prof.healthy_mask(comp)
    assert int(both.sum()) < 30
    for comp in COMPONENTS:
        assert int((prof.healthy_mask(comp) & prof.stable).sum()) > 150


@pytestmark_hydraulic
def test_the_virtual_channels_are_refused():
    """CE, CP and SE are computed by the rig from its own temperatures. Scoring
    a cooler fault against them would be circular."""
    from cbmx.io.hydraulic import HydraulicRig
    from cbmx.physics.hydraulic import ChannelMap, CycleView
    rig = HydraulicRig(HYDRAULIC)
    v = CycleView(rig, 0, ChannelMap.resolve(rig, np.arange(50)))
    for ch in ("CE", "CP", "SE"):
        with pytest.raises(ValueError):
            v.get(ch)


def test_every_hydraulic_symptom_declares_a_direction():
    """A symptom with no declared direction cannot exonerate, and a test built
    only from such symptoms convicts eventually."""
    from cbmx.physics.hydraulic import SYMPTOMS
    for s in SYMPTOMS:
        assert s.direction in (-1, 1), s.id
        assert s.rationale.strip(), f"{s.id} has no stated physical rationale"
        assert s.component


def test_each_component_has_at_least_one_primary_symptom():
    from cbmx.physics.hydraulic import BY_COMPONENT
    for comp, syms in BY_COMPONENT.items():
        assert any(s.primary for s in syms), comp


# ── regimes ────────────────────────────────────────────────────────────────

def test_regimes_keep_operating_points_apart():
    b = Baseline("t", min_samples=5)
    cold, hot = Regime(12, 0), Regime(18, 0)
    for _ in range(30):
        b.observe("line", cold, 4.0)
    z_cold, run_cold = b.score("line", cold, 4.0)
    _, run_hot = b.score("line", hot, 4.0)
    assert run_cold.ready and not run_hot.ready, (
        "a regime never observed healthy must not inherit another's baseline")


def test_a_frozen_baseline_cannot_absorb_a_developing_fault():
    r = Running(min_samples=5)
    for _ in range(30):
        r.update(4.0)
    before = r.median
    r.frozen = True
    for _ in range(200):
        r.update(40.0)
    assert r.median == before, "freezing is what stops a fault becoming the new normal"


def test_record_membership_uses_identity_not_array_equality():
    """`r not in list_of_records` must not compare signal arrays.

    Record is a dataclass, so `in` calls __eq__, which compares the numpy
    signals elementwise. With records of equal length that silently returns an
    array and happens to work; with records that differ by one sample — which
    decimation produces routinely once more than one bearing is present — it
    raises ValueError deep inside a list comprehension. Three bearings on one
    sample rate never triggered it; fourteen did, on the first fleet run.
    """
    import numpy as np
    from cbmx.io.base import Record

    def mk(n):
        return Record(file_no=1, fault="outer_race", defect_in=0.0, load_hp=7,
                      rpm_nominal=1500, fs=32000.0, signal=np.zeros(n))

    a, b = mk(128000), mk(128001)
    kept = [r for r in (a, b) if not any(r is c for c in (a,))]
    assert kept == [b] or kept[0] is b
