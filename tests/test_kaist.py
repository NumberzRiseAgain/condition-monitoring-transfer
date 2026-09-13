"""Regression tests for the KAIST varying-speed loader.

Two of these are here because the corresponding mistakes were actually made
during the first run and were silent rather than loud.

The `_constant` recordings ship without a header row while the varying-speed
ones have one. Read blindly, the first data sample becomes four column names,
and a lookup by name then either fails loudly or -- far worse -- succeeds
against a column that has shifted.

The tachometer file and the signal file do not start together. Left
uncorrected, order tracking resamples against the wrong angle and the shaft
line sits about 4 dB above its floor where the aligned record reaches 33 dB.
Nothing errors; the run simply detects less and the reason is invisible.

The dataset tests are skipped when the data is absent, and say why.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from cbmx.io import kaist
from cbmx.physics import bearing as bearing_mod
from cbmx.physics.envelope import resample_to_angle

DATA = Path("data/kaist")
have_data = DATA.exists() and any(DATA.glob("vibration_*.csv"))
needs_data = pytest.mark.skipif(
    not have_data, reason="KAIST subset absent; put the CSVs under data/kaist")


# ── things that need no data ────────────────────────────────────────────────
def test_assumed_geometry_is_flagged_unverified():
    """The KAIST entry must never lose its caveat. If somebody quietly cleans
    up the `source` string, every result derived from it silently becomes a
    claim the dataset does not support."""
    bg = bearing_mod.get("KAIST6205U")
    assert "UNVERIFIED" in bg.source
    assert all(bg.check().values())
    # BPFO + BPFI is identically the ball count. Cheap, and it catches a
    # transposed dimension immediately.
    assert abs(bg.bpfo + bg.bpfi - bg.n_rolling) < 1e-9


def test_headerless_and_headed_files_both_read(tmp_path):
    hd = tmp_path / "vibration_normal_0.csv"
    hd.write_text("bearingA_x,bearingA_y,bearingB_x,bearingB_y\n1,2,3,4\n5,6,7,8\n")
    x, names = kaist.read_signal(hd, kind="vibration")
    assert names == kaist.DEFAULT_CHANNELS["vibration"]
    assert x.shape == (2, 4) and x[0, 0] == 1.0

    nh = tmp_path / "vibration_normal_constant.csv"
    nh.write_text("1,2,3,4\n5,6,7,8\n9,10,11,12\n")
    x2, names2 = kaist.read_signal(nh, kind="vibration")
    assert names2 == kaist.DEFAULT_CHANNELS["vibration"]
    # The first row is DATA, not a header. Three rows, not two.
    assert x2.shape == (3, 4) and x2[0, 0] == 1.0


def test_sibling_subsets_do_not_overwrite_each_other(tmp_path):
    """The three Mendeley subsets reuse identical file names. Flattened into one
    directory they would silently overwrite each other and the run would report
    a smaller n than it believed it had — the worst kind of quiet error for a
    false-alarm claim, which is entirely a claim about n."""
    hdr = "bearingA_x,bearingA_y,bearingB_x,bearingB_y\n1,2,3,4\n5,6,7,8\n"
    rpm = "time,rpm\n0,1000\n1,1200\n"
    (tmp_path / "vibration_normal_0.csv").write_text(hdr)
    (tmp_path / "rpm_normal_0.csv").write_text(rpm)
    for sub in ("Subset2", "Subset 3"):
        d = tmp_path / sub
        d.mkdir()
        (d / "vibration_normal_0.csv").write_text(hdr)
        (d / "rpm_normal_0.csv").write_text(rpm)

    idx = kaist.scan(tmp_path)
    reps = sorted(r for (k, c, r) in idx if k == "vibration")
    assert reps == ["0", "subset2-0", "subset3-0"], reps
    assert kaist.conditions_available(tmp_path)["normal"] == [
        "normal_0", "normal_subset2-0", "normal_subset3-0"]
    # and each one still finds its own tachometer, not the root's
    assert idx[("rpm", "normal", "subset2-0")].parent.name == "Subset2"


def test_alignment_recovers_a_planted_offset():
    """Synthesise a shaft tone under a varying speed, offset the tachometer,
    and check the search finds it. No fault, no bearing, no labels."""
    fs = 4096.0
    dur = 12.0
    t = np.arange(int(dur * fs)) / fs
    hz = 20.0 + 6.0 * np.sin(2 * np.pi * 0.15 * t)
    phase = 2 * np.pi * np.cumsum(hz) / fs
    x = np.sin(phase) + 0.4 * np.sin(2 * phase)
    x = np.column_stack([x, x, x, x])

    # The tachometer's timestamps run 0.40 s AHEAD of the signal's, so the
    # correction that undoes it is -0.40: `windows` looks up
    # tacho.at(t + offset). Asserting the sign, not just the magnitude, is the
    # point of the test -- an offset applied the wrong way round is worse than
    # none, and it looks identical in every summary statistic.
    planted = 0.40
    tt = np.arange(0.0, dur + 0.2, 0.05)
    tac = kaist.Tacho(tt - planted, np.interp(tt, t, hz) * 60.0)

    rec = kaist.Record("normal", "0", "vibration", x,
                       kaist.DEFAULT_CHANNELS["vibration"], fs, tac)
    off, aligned, unaligned = kaist.align_tacho(
        rec, "bearingA_x", search_s=1.0, coarse_s=0.05, n_windows=6)
    assert abs(off + planted) < 0.06, (off, -planted)
    assert aligned > unaligned + 5.0


def test_current_sampling_rate_is_recovered_from_the_tachometer():
    """`estimate_current_fs` ships unrun against the real data, so it gets a
    synthetic case with a known answer. Without this it is untested code that
    a reviewer would be right to discount.

    Build a three-phase current whose supply fundamental is locked to a varying
    shaft speed through 2 pole pairs, sample it at a rate the caller does not
    supply, and check the rate and the pole-pair count both come back.
    """
    fs_true, p_true, slip = 25_600.0, 2, 0.03
    dur = 40.0
    t = np.arange(int(dur * fs_true)) / fs_true
    rpm = 1200.0 + 600.0 * np.sin(2 * np.pi * 0.05 * t)
    f_elec = p_true * rpm / (60.0 * (1.0 - slip))
    phase = 2 * np.pi * np.cumsum(f_elec) / fs_true
    rng = np.random.default_rng(1)
    current = np.sin(phase) + 0.05 * rng.normal(size=t.shape[0])

    tt = np.arange(0.0, dur + 0.2, 0.109)
    tac = kaist.Tacho(tt, np.interp(tt, t, rpm))

    est = kaist.estimate_current_fs(current, tac)
    assert est.fs == fs_true, est.as_dict()
    assert est.pole_pairs == p_true, est.as_dict()
    assert est.residual_rpm < 40.0, est.as_dict()
    assert abs(est.slip_estimate - slip) < 0.02, est.as_dict()


def test_window_wander_is_zero_at_constant_speed():
    fs = 2048.0
    x = np.column_stack([np.random.default_rng(0).normal(size=4096)] * 4)
    tac = kaist.Tacho(np.array([0.0, 10.0]), np.array([1200.0, 1200.0]))
    rec = kaist.Record("normal", "constant", "vibration", x,
                       kaist.DEFAULT_CHANNELS["vibration"], fs, tac)
    w = next(kaist.windows(rec, "bearingA_x", seconds=1.0))
    assert w.wander_pct == pytest.approx(0.0, abs=1e-9)
    assert w.mean_hz == pytest.approx(20.0)


# ── things that need the data ───────────────────────────────────────────────
@needs_data
def test_published_sampling_rate_is_the_one_the_data_supports():
    """25.6 kHz is taken from the authors' paper, and the paper could be wrong
    or could describe a different subset. It is checkable without any label:
    only the correct rate puts the shaft line where order tracking can find it,
    because a wrong rate distorts the time axis the angle is integrated over.

    This is the test that would have caught a transcription error in the one
    constant this loader takes on trust.
    """
    x, _ = kaist.read_signal(DATA / "vibration_normal_1.csv",
                             int(25600 * 30), "vibration")
    tac = kaist.read_tacho(DATA / "rpm_normal_1.csv")
    sig = x[:, 0]

    def sharpness(fs: float) -> float:
        n = int(fs)
        vals = []
        for k in range(12):
            s = k * n
            if s + n > sig.shape[0]:
                break
            hz = tac.at(np.arange(s, s + n) / fs)
            if hz.min() <= 0:
                continue
            xa, spr = resample_to_angle(sig[s:s + n] - sig[s:s + n].mean(), fs,
                                        hz, samples_per_rev=512)
            X = np.abs(np.fft.rfft(xa * np.hanning(xa.shape[0])))
            o = np.fft.rfftfreq(xa.shape[0], d=1.0 / spr)
            near = np.abs(o - 1.0) < 0.05
            ring = (np.abs(o - 1.0) < 0.6) & (np.abs(o - 1.0) > 0.08)
            if near.any() and ring.any():
                vals.append(20 * math.log10(max(X[near].max(), 1e-18)
                                            / max(float(np.median(X[ring])), 1e-18)))
        return float(np.median(vals)) if vals else -math.inf

    best = max((12800.0, 20000.0, 25600.0, 32768.0, 51200.0), key=sharpness)
    assert best == kaist.FS_VIBRATION


@needs_data
def test_tacho_offset_is_nonzero_and_alignment_helps():
    rec = kaist.load(DATA, "vibration", "normal", "0", max_seconds=30.0)
    off, aligned, unaligned = kaist.align_tacho(rec, "bearingA_x")
    assert abs(off) > 0.05, "expected a real offset; the files are not aligned"
    assert aligned > unaligned + 5.0
