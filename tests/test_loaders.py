"""Regression tests for the two loaders that had none: XJTU-SY and the
VT/NSWC hydraulic actuator.

Neither dataset ships with this repository, so these tests build the file
layouts on disk from scratch. That is deliberate. A loader test that needs the
real download only runs on the machine that has it, which is the machine least
likely to catch a regression.

The XJTU tests are unusual in what they assert. That loader exists so the
run-to-failure evaluator's `--dataset` flag can be honest about a set this
project decided not to use, and the decision is recorded in
`PROGNOSIS_STOPPING_RULE.md`. The tests below therefore pin the *properties that
decision rests on* — an empty label table, a retrospective life fraction —
rather than only the parsing.
"""
from __future__ import annotations

import numpy as np
import pytest

from cbmx.io import actuator, xjtu


# ══════════════════════════════════════════════════ XJTU-SY ══════════════════

def _write_xjtu(root, condition="35Hz12kN", bearing="Bearing1_1", n=5,
                header=True, rows=64):
    """A minimal XJTU tree: <condition>/<bearing>/<index>.csv, one per minute."""
    d = root / condition / bearing
    d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(1, n + 1):
        h = np.vstack([rng.normal(0, 1, rows), rng.normal(0, 1, rows)]).T
        text = "\n".join(f"{a:.5f},{b:.5f}" for a, b in h)
        if header:
            text = "Horizontal_vibration_signals,Vertical_vibration_signals\n" + text
        (d / f"{i}.csv").write_text(text)
    return d


def test_scan_finds_bearings_under_their_condition(tmp_path):
    _write_xjtu(tmp_path)
    found = xjtu.scan(tmp_path)
    assert "35Hz12kN" in found
    assert "Bearing1_1" in found["35Hz12kN"]
    assert found["35Hz12kN"]["Bearing1_1"].n_captures == 5


def test_captures_come_back_in_life_order_not_string_order(tmp_path):
    """Capture files are named by integer index. Sorted as text, 10 precedes 2,
    and a life curve read in that order is not a life curve."""
    _write_xjtu(tmp_path, n=12)
    life = xjtu.scan(tmp_path)["35Hz12kN"]["Bearing1_1"]
    indices = [c.index for c in life.captures()]
    assert indices == sorted(indices)
    assert indices == list(range(1, 13))


def test_header_row_is_sniffed_rather_than_assumed(tmp_path):
    """Published mirrors differ on whether the CSV carries a header. Read
    blindly, the first sample becomes a column name and every later index is
    off by one."""
    a = tmp_path / "with"
    b = tmp_path / "without"
    _write_xjtu(a, header=True, rows=64)
    _write_xjtu(b, header=False, rows=64)
    xa = next(xjtu.scan(a)["35Hz12kN"]["Bearing1_1"].captures())
    xb = next(xjtu.scan(b)["35Hz12kN"]["Bearing1_1"].captures())
    assert xa.x.shape == xb.x.shape == (64,)
    assert np.isfinite(xa.x).all() and np.isfinite(xb.x).all()


def test_fault_labels_are_empty_so_no_accuracy_can_be_claimed():
    """An empty table makes the evaluator report lead time and decline to report
    part-naming accuracy, which is correct. A guessed table would make it print
    a number that means nothing."""
    assert xjtu.FAULT_LABELS == {}


def test_life_fraction_is_retrospective(tmp_path):
    """Fraction of life is only computable after the bearing has failed, which
    is why remaining-useful-life from run-to-failure data is retrospective and
    must be labelled as such."""
    _write_xjtu(tmp_path, n=10)
    life = xjtu.scan(tmp_path)["35Hz12kN"]["Bearing1_1"]
    assert life.fraction_of_life(0) == 0.0
    assert life.fraction_of_life(10) == pytest.approx(1.0)
    assert life.fraction_of_life(5) == pytest.approx(0.5)


def test_life_minutes_follows_the_published_capture_interval(tmp_path):
    _write_xjtu(tmp_path, n=7)
    life = xjtu.scan(tmp_path)["35Hz12kN"]["Bearing1_1"]
    assert xjtu.CAPTURE_INTERVAL_S == 60.0
    assert life.life_minutes == pytest.approx(7.0)


def test_describe_names_what_is_missing(tmp_path):
    msg = xjtu.describe(tmp_path)
    assert "no" in msg.lower() or "not" in msg.lower()


# ══════════════════════════════════ VT/NSWC hydraulic actuator ═══════════════

COLUMNS = ["Angle", "PG_1", "PG_2", "PG_3", "Temp_1", "Temp_2", "Temp_3",
           "Accel_1", "Accel_2", "Accel_3", "Internal_Temp"]


# `_phases` requires the stroke to exceed 2 degrees per sample in absolute
# angular velocity. That floor sits above the quantisation noise of a held
# angle, so a slow synthetic ramp is correctly read as "never moved". The
# fixture therefore strokes at about 3 deg/sample, as the real records do.
STROKE_SAMPLES = 40
STROKE_DEGREES = 120.0


def _write_actuator_cycle(path, n=900, stroke=True):
    """One tab-delimited actuation: a fast stroke, then a hold."""
    rng = np.random.default_rng(1)
    if stroke:
        angle = np.concatenate([
            np.linspace(0.0, STROKE_DEGREES, STROKE_SAMPLES),
            np.full(n - STROKE_SAMPLES, STROKE_DEGREES)])
    else:
        angle = np.full(n, STROKE_DEGREES)            # never moved
    angle = angle + rng.normal(0, 0.05, n)
    cols = {"Angle": angle, "Internal_Temp": np.full(n, 41.0)}
    for i in (1, 2, 3):
        cols[f"PG_{i}"] = 1000 + rng.normal(0, 5, n)
        cols[f"Temp_{i}"] = 40 + rng.normal(0, 0.2, n)
        cols[f"Accel_{i}"] = rng.normal(0, 0.3, n)
    head = "\t".join(COLUMNS)
    body = "\n".join("\t".join(f"{cols[c][k]:.5f}" for c in COLUMNS)
                     for k in range(n))
    path.write_text(head + "\n" + body)


def test_unit_table_covers_the_published_units():
    """Six units, and the two with a named defect are the ones the evaluation
    is about. If this table drifts, a damaged unit is silently scored as a
    baseline."""
    assert set(actuator.UNITS) == {
        "Act_1", "Act_2", "Act_3 (Gear Damage)", "Act_4 (Seal Defect)",
        "Act_5", "Act_6"}
    assert actuator.UNITS["Act_3 (Gear Damage)"][0] == "gear"
    assert actuator.UNITS["Act_4 (Seal Defect)"][0] == "seal"
    baselines = [u for u, (c, _) in actuator.UNITS.items() if c == "baseline"]
    assert len(baselines) == 4


def test_load_unit_reads_a_cycle_and_splits_its_phases(tmp_path):
    d = tmp_path / "Act_1"
    d.mkdir()
    _write_actuator_cycle(d / "cycle_001.txt")
    cycles = actuator.load_unit(str(tmp_path), "Act_1")
    assert len(cycles) == 1
    c = cycles[0]
    assert c.condition == "baseline"
    assert c.pg.shape[0] == 3 and c.temp.shape[0] == 3 and c.accel.shape[0] == 3
    assert c.motion.stop > c.motion.start
    assert c.hold.stop > c.hold.start
    assert c.moved is True


def test_a_cycle_that_never_moved_is_marked_as_such(tmp_path):
    """A record with no stroke carries no information about the actuator, and
    scoring it as though it did would put a healthy-looking point in the trend."""
    d = tmp_path / "Act_1"
    d.mkdir()
    _write_actuator_cycle(d / "cycle_001.txt", stroke=False)
    cycles = actuator.load_unit(str(tmp_path), "Act_1")
    assert cycles == [] or cycles[0].moved is False


def test_short_or_malformed_records_are_skipped_not_scored(tmp_path):
    """A truncated capture loads without error and lands in the trend as
    movement with no physical cause, so it is dropped at the loader."""
    d = tmp_path / "Act_1"
    d.mkdir()
    _write_actuator_cycle(d / "good.txt")
    _write_actuator_cycle(d / "too_short.txt", n=100)          # under the floor
    (d / "wrong_columns.txt").write_text("A\tB\n1\t2\n")
    assert len(actuator.load_unit(str(tmp_path), "Act_1")) == 1


def test_sample_rate_is_the_one_the_authors_state():
    assert actuator.FS == 1000.0


def test_a_slow_ramp_is_not_a_stroke(tmp_path):
    """The phase splitter has an absolute floor of 2 degrees per sample, above
    the quantisation noise of a held angle. A drift that never exceeds it is a
    hold, not an actuation, and must not be scored as one."""
    d = tmp_path / "Act_1"
    d.mkdir()
    rng = np.random.default_rng(2)
    n = 900
    slow = np.linspace(0.0, 80.0, n) + rng.normal(0, 0.05, n)   # 0.09 deg/sample
    cols = {"Angle": slow, "Internal_Temp": np.full(n, 41.0)}
    for i in (1, 2, 3):
        cols[f"PG_{i}"] = 1000 + rng.normal(0, 5, n)
        cols[f"Temp_{i}"] = 40 + rng.normal(0, 0.2, n)
        cols[f"Accel_{i}"] = rng.normal(0, 0.3, n)
    body = "\n".join("\t".join(f"{cols[c][k]:.5f}" for c in COLUMNS)
                     for k in range(n))
    (d / "slow.txt").write_text("\t".join(COLUMNS) + "\n" + body)
    cycles = actuator.load_unit(str(tmp_path), "Act_1")
    assert cycles and cycles[0].moved is False


def test_load_all_returns_every_unit_even_when_empty(tmp_path):
    """A missing unit directory must come back as an empty list rather than be
    absent, so a caller iterating the units cannot silently skip one."""
    got = actuator.load_all(str(tmp_path))
    assert set(got) == set(actuator.UNITS)
    assert all(v == [] for v in got.values())
