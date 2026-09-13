"""Regression tests for the NASA/IMS run-to-failure loader.

Every one of these guards a mistake this dataset actively invites.

The archive nests three deep — zip, then 7z, then three rar files — and the
third test unpacks into a folder called `4th_test/txt/`. Read naively, `scan`
names that test `txt`, the readme's failure label for `3rd_test/3` never
matches it, and the one bearing in that test that actually failed is scored as
a healthy control. Nothing errors. The run simply reports that the method found
nothing on set 3, and the reason is invisible.

The capture filenames are timestamps sorted as text by every tool that lists a
directory, and `2003.11.01` sorts before `2003.10.22`. A life curve read in
filename order is not a life curve, and it still looks like one on a plot.

The readme publishes eight channels for test 1 and four for tests 2 and 3.
Getting that wrong does not raise; it mixes two bearings into one channel,
which reads as a noisier bearing rather than as an error.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from cbmx.io import ims


# ── the misnamed third test ──────────────────────────────────────────────────

def test_fourth_test_folder_maps_to_the_readmes_third_set():
    """`3rd_test.rar` unpacks to `4th_test/txt/`. That is the archive's naming,
    not ours, and the labels are keyed on the readme's."""
    assert ims._test_name(Path("/d/ims/4th_test/txt")) == "3rd_test"
    assert ims._test_name(Path("/d/ims/4th_test")) == "3rd_test"
    assert ims._test_name(Path("/d/ims/1st_test")) == "1st_test"
    assert ims._test_name(Path("/d/ims/2nd_test")) == "2nd_test"


def test_container_directory_takes_its_parents_identity():
    """A folder called `txt` is packaging. Named as a test it would collide with
    any other `txt` folder and silently merge two experiments."""
    assert ims._test_name(Path("/d/ims/2nd_test/txt")) == "2nd_test"


def test_the_misnamed_folder_still_finds_its_failure_label():
    """The point of the mapping: the bearing that failed in set 3 must not be
    scored as a control."""
    assert ims.FAULT_LABELS[f"{ims._test_name(Path('/d/4th_test/txt'))}/3"] \
        == "outer_race"


# ── the labels, as transcribed ───────────────────────────────────────────────

def test_labels_match_the_readme_verbatim():
    """"inner race defect occurred in bearing 3 and roller element defect in
    bearing 4" (set 1); "outer race failure occurred in bearing 1" (set 2);
    "outer race failure occurred in bearing 3" (set 3)."""
    assert ims.FAULT_LABELS == {
        "1st_test/3": "inner_race",
        "1st_test/4": "rolling_element",
        "2nd_test/1": "outer_race",
        "3rd_test/3": "outer_race",
    }


def test_eight_bearings_survived_and_are_controls():
    """Twelve bearings across three tests, four with a reported failure. The
    other eight are the control that makes a lead time mean anything, and the
    stopping rule forbids dropping them."""
    total = 3 * 4
    assert total - len(ims.FAULT_LABELS) == 8


def test_module_ships_no_bearing_geometry():
    """The readme describes the rig, the channels and the failures, and never
    gives the ZA-2115's dimensions. Under PROGNOSIS_STOPPING_RULE.md the
    declared-physics half does not run, so no geometry may appear here. A future
    edit that adds it must also change this test, which is the point."""
    for name in dir(ims):
        assert "PITCH" not in name.upper()
        assert "BALL_D" not in name.upper()
        assert "N_BALLS" not in name.upper()
    assert not hasattr(ims, "GEOMETRY")
    assert not hasattr(ims, "BEARING")


def test_published_census_is_the_readmes_numbers():
    assert ims.PUBLISHED["1st_test"]["files"] == 2156
    assert ims.PUBLISHED["2nd_test"]["files"] == 984
    assert ims.PUBLISHED["3rd_test"]["files"] == 4448
    assert ims.PUBLISHED["1st_test"]["channels"] == 8
    assert ims.PUBLISHED["2nd_test"]["channels"] == 4


# ── time order ───────────────────────────────────────────────────────────────

def test_captures_are_ordered_by_timestamp_not_by_filename(tmp_path):
    """October 22 must precede November 1. Sorted as text it does not."""
    d = tmp_path / "2nd_test"
    d.mkdir()
    names = ["2003.11.01.00.00.00", "2003.10.22.12.06.24", "2003.10.31.23.59.59"]
    for n in names:
        (d / n).write_text("\n".join("1\t2\t3\t4" for _ in range(4)))
    # `scan` needs at least ten files before it will call a folder a test.
    for i in range(10):
        (d / f"2003.11.02.00.00.{i:02d}").write_text(
            "\n".join("1\t2\t3\t4" for _ in range(4)))
    found = ims.scan(tmp_path)
    stamps = found["2nd_test"][1].stamps
    assert stamps == sorted(stamps)
    assert stamps[0] == datetime(2003, 10, 22, 12, 6, 24)


def test_life_hours_comes_from_timestamps_not_an_assumed_interval(tmp_path):
    """The readme says every ten minutes "except the first 43 files were taken
    every 5 minutes", and set 1 resumes on the next working day after gaps of
    hours. An assumed interval is wrong by more than a day."""
    d = tmp_path / "1st_test"
    d.mkdir()
    # 22nd to the 31st, plus two impossible dates that a regex alone would
    # accept. October has 30 days after the 22nd only in a careless parser.
    for i in range(12):
        (d / f"2003.10.{22 + i:02d}.00.00.00").write_text(
            "\n".join("\t".join("1" * 1 for _ in range(8)) for _ in range(4)))
    life = ims.scan(tmp_path)["1st_test"][1]
    assert life.n_captures == 10, "2003.10.32 and 2003.10.33 must be rejected"
    assert life.life_hours == pytest.approx(9 * 24.0)


def test_partial_extraction_is_skipped_rather_than_scored(tmp_path):
    """Nine captures is not a test. Scoring it would report a life of minutes."""
    d = tmp_path / "1st_test"
    d.mkdir()
    for i in range(9):
        (d / f"2003.10.{22 + i:02d}.00.00.00").write_text("1\t2\t3\t4")
    assert ims.scan(tmp_path) == {}


# ── the unextracted-archive message ──────────────────────────────────────────

def test_describe_names_the_archives_still_to_extract(tmp_path):
    """'No capture files' and 'you have not finished unpacking' are different
    problems with the same symptom."""
    (tmp_path / "IMS.7z").write_bytes(b"")
    (tmp_path / "1st_test.rar").write_bytes(b"")
    msg = ims.describe(tmp_path)
    assert "IMS.7z" in msg and "1st_test.rar" in msg
    assert "three deep" in msg.lower() or "THREE deep" in msg


def test_describe_says_no_geometry_when_data_is_present(tmp_path):
    d = tmp_path / "2nd_test"
    d.mkdir()
    for i in range(12):
        (d / f"2004.02.{12 + i:02d}.00.00.00").write_text(
            "\n".join("1\t2\t3\t4" for _ in range(4)))
    msg = ims.describe(tmp_path)
    assert "geometry" in msg
    assert "part-naming accuracy is not reported" in msg
    assert "984" in msg          # the published count it is being checked against


# ── truncated captures ───────────────────────────────────────────────────────

def test_truncated_capture_is_flagged(tmp_path):
    """A killed extraction leaves one short file. It loads without error and
    lands in the life curve as movement that has no physical cause. This
    happened during the real extraction, which is why the check exists."""
    d = tmp_path / "2nd_test"
    d.mkdir()
    full = "\n".join("1\t2\t3\t4" for _ in range(2000))
    for i in range(12):
        (d / f"2004.02.{12 + i:02d}.00.00.00").write_text(full)
    (d / "2004.02.28.00.00.00").write_text(full[:200])
    bad = ims.short_captures(tmp_path)
    assert [p.name for p, _, _ in bad] == ["2004.02.28.00.00.00"]
    assert "TRUNCATED" in ims.describe(tmp_path)


def test_intact_extraction_flags_nothing(tmp_path):
    """Real captures vary by a few hundred bytes. The check must not fire on
    that, or it will be ignored the one time it matters."""
    import random
    d = tmp_path / "2nd_test"
    d.mkdir()
    rng = random.Random(0)
    for i in range(12):
        rows = 2000 + rng.randint(-3, 3)
        (d / f"2004.02.{12 + i:02d}.00.00.00").write_text(
            "\n".join("1\t2\t3\t4" for _ in range(rows)))
    assert ims.short_captures(tmp_path) == []
    assert "TRUNCATED" not in ims.describe(tmp_path)


# ── the scope of set 3 ───────────────────────────────────────────────────────

def test_set_three_is_clipped_to_the_documented_experiment(tmp_path):
    """The folder ships captures running two weeks past the end of the
    experiment the readme describes. The readme is the only source for both the
    failure label and the extent, so it governs both or neither."""
    d = tmp_path / "4th_test" / "txt"
    d.mkdir(parents=True)
    inside = [f"2004.03.{4 + i:02d}.09.27.46" for i in range(12)]      # early March
    outside = ["2004.04.10.00.00.00", "2004.04.18.02.42.55"]           # past 4 April
    for n in inside + outside:
        (d / n).write_text("\n".join("1\t2\t3\t4" for _ in range(4)))
    life = ims.scan(tmp_path)["3rd_test"][3]
    assert life.n_captures == 14
    clipped = ims.clip_to_published(life)
    assert clipped.n_captures == 12
    assert clipped.stamps[-1] <= ims.published_end("3rd_test")
    assert clipped.test == "3rd_test" and clipped.bearing == 3


def test_clipping_shortens_the_life_and_so_lowers_every_percentage(tmp_path):
    """The default is the conservative one. A call at a fixed hour leaves a
    smaller fraction of a shorter life, so clipping cannot flatter the result —
    which is why it is safe to have chosen it from the documentation."""
    d = tmp_path / "4th_test" / "txt"
    d.mkdir(parents=True)
    for i in range(12):
        (d / f"2004.03.{4 + i:02d}.09.27.46").write_text("1\t2\t3\t4")
    (d / "2004.04.18.02.42.55").write_text("1\t2\t3\t4")
    life = ims.scan(tmp_path)["3rd_test"][1]
    assert ims.clip_to_published(life).life_hours < life.life_hours


def test_clipping_leaves_the_documented_sets_untouched(tmp_path):
    """Sets 1 and 2 match the readme to the second, so clipping must be a no-op
    on them. Captures are placed an hour apart INSIDE each published window —
    the first draft of this test spread set 2 across twelve days, ran four of
    them past the readme's 19 February end, and was correctly clipped."""
    (tmp_path / "1st_test").mkdir()
    (tmp_path / "2nd_test").mkdir()
    for i in range(12):
        (tmp_path / "1st_test" / f"2003.10.22.{12 + i:02d}.06.24").write_text("1\t2")
        (tmp_path / "2nd_test" / f"2004.02.12.{10 + i:02d}.32.39").write_text("1\t2\t3\t4")
    found = ims.scan(tmp_path)
    for t in ("1st_test", "2nd_test"):
        life = found[t][1]
        assert life.n_captures == 12
        assert ims.clip_to_published(life).n_captures == 12, \
            f"{t} lies inside its published window and must not be clipped"


# ── one read per capture ─────────────────────────────────────────────────────

def test_columns_follow_the_readmes_channel_arrangement():
    """Set 1: bearing 1 is channels 1&2, bearing 3 is 5&6. Sets 2 and 3: one
    channel each. Getting this wrong mixes two bearings into one signal, which
    reads as a noisier bearing rather than as an error."""
    assert ims.columns_for("1st_test", 1) == slice(0, 2)
    assert ims.columns_for("1st_test", 3) == slice(4, 6)
    assert ims.columns_for("2nd_test", 4) == slice(3, 4)
    assert ims.columns_for("3rd_test", 3) == slice(2, 3)


def test_iter_test_reads_each_file_once(tmp_path, monkeypatch):
    """Four bearings share one file because they shared one shaft. Reading it
    per bearing quadruples the I/O across 9,464 files for nothing."""
    d = tmp_path / "2nd_test"
    d.mkdir()
    for i in range(12):
        (d / f"2004.02.{12 + i:02d}.00.00.00").write_text(
            "\n".join("1\t2\t3\t4" for _ in range(4)))
    life = ims.scan(tmp_path)["2nd_test"][1]
    calls = []
    real = ims.read_capture
    monkeypatch.setattr(ims, "read_capture", lambda p: (calls.append(p), real(p))[1])
    got = list(ims.iter_test(life))
    assert len(got) == 12 and len(calls) == 12
    assert got[0][2].shape[1] == 4


def test_iter_test_honours_every(tmp_path):
    d = tmp_path / "2nd_test"
    d.mkdir()
    for i in range(12):
        (d / f"2004.02.{12 + i:02d}.00.00.00").write_text("1\t2\t3\t4")
    life = ims.scan(tmp_path)["2nd_test"][1]
    assert [i for i, _, _ in ims.iter_test(life, every=3)] == [1, 4, 7, 10]
