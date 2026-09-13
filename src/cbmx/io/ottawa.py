"""uOttawa bearing data under time-varying speed.

The most valuable of the public sets for this bid, and it is not close. Every
other dataset runs at a constant speed; this one sweeps the shaft up, down, or
both, inside a single ten-second record. An arrestment is a speed transient, so
a method that has only ever been shown at constant speed has not been tested for
what AAG actually does.

It is also the only public set that exercises the order-tracking path in
cbmx.physics.envelope, which until now has been written and never run in anger.
Expect it to matter: with the shaft sweeping, a fault line smears across many
bins in a plain spectrum and vanishes under the floor. Resampling against shaft
angle instead of time is what pins it back to one bin.

Naming, from the dataset paper: <health>-<speed profile>-<trial>.mat
    health : H healthy, I inner, O outer, B ball, C combined
    speed  : A increasing, B decreasing, C increasing-then-decreasing,
             D decreasing-then-increasing
Variables: Channel_1 vibration, Channel_2 tachometer. 200 kHz, 10 s.

The tachometer channel is a pulse train, not an rpm reading. `speed_from_tacho`
turns it into a per-sample speed by counting edges — and if that fails, the
loader says so rather than substituting a constant, because a constant speed on
a variable-speed record is the one assumption that would invalidate the whole
point of running this set.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import Record

HEALTH = {"H": "normal", "I": "inner_race", "O": "outer_race",
          "B": "rolling_element", "C": "combined"}
PROFILE = {"A": "increasing", "B": "decreasing",
           "C": "increasing-decreasing", "D": "decreasing-increasing"}


def _moving_average(x: np.ndarray, k: int) -> np.ndarray:
    """O(n) moving average via a cumulative sum.

    np.convolve is O(n*k). Here k is a quarter-second at 200 kHz — fifty
    thousand taps — against two million samples, which is 1e11 operations per
    record and roughly sixteen seconds each. Sixty records is a quarter of an
    hour of smoothing before any analysis starts. A prefix sum does the same
    thing in a few milliseconds.
    """
    k = max(1, int(k))
    if k <= 1 or x.size < k:
        return x
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = (c[k:] - c[:-k]) / k
    pad_l = (k - 1) // 2
    pad_r = x.size - out.size - pad_l
    return np.concatenate([np.full(pad_l, out[0]), out, np.full(max(0, pad_r), out[-1])])[:x.size]


def speed_from_tacho(tacho: np.ndarray, fs: float, ppr: int = 1,
                     smooth_s: float = 0.25) -> Optional[np.ndarray]:
    """Per-sample shaft speed in rpm, from a pulse train.

    Edge detection on a hysteresis threshold, then instantaneous rate from the
    interval between edges, then a light smooth. Returns None rather than a
    guess when there are too few edges — a fabricated speed trace would be worse
    than no speed trace.
    """
    x = np.asarray(tacho, dtype=np.float64).ravel()
    if x.size < 1000:
        return None

    # Vectorised. The obvious per-sample loop with hysteresis is two million
    # Python iterations per record here, times sixty records — minutes of pure
    # interpreter time before any analysis starts. Smooth first to kill the
    # multiple crossings that hysteresis was there to reject, then find rising
    # crossings of the midpoint with one diff.
    k = max(3, int(0.0002 * fs))                 # ~0.2 ms
    if k % 2 == 0:
        k += 1
    xs = _moving_average(x, k)
    lo, hi = np.percentile(xs, 5), np.percentile(xs, 95)
    if hi - lo < 1e-9:
        return None
    above = xs > (lo + 0.5 * (hi - lo))
    e = np.flatnonzero(np.diff(above.astype(np.int8)) == 1) + 1
    if e.size < 20:
        return None

    # Reject crossings closer together than half the median interval: those are
    # residual noise, and a spurious edge reads as an impossible speed spike.
    d = np.diff(e)
    keep = np.concatenate([[True], d > 0.5 * np.median(d)])
    e = e[keep].astype(np.float64)
    if e.size < 20:
        return None

    rpm_at_edge = (fs / (np.diff(e) * ppr)) * 60.0
    rpm = np.interp(np.arange(x.size), e[1:], rpm_at_edge,
                    left=rpm_at_edge[0], right=rpm_at_edge[-1])
    return _moving_average(rpm, max(3, int(smooth_s * fs)))


def load_file(path: str | Path, ppr: int = 1) -> Record:
    from scipy.io import loadmat

    path = Path(path)
    parts = path.stem.upper().split("-")
    if not parts or parts[0][:1] not in HEALTH:
        raise KeyError(f"{path.name}: name does not start with a health code "
                       f"{sorted(HEALTH)}; keep the distribution's filenames.")
    fault = HEALTH[parts[0][:1]]
    profile = PROFILE.get(parts[1][:1] if len(parts) > 1 else "", "unknown")
    trial = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1

    m = loadmat(str(path), squeeze_me=True)
    vib_key = next((k for k in ("Channel_1", "channel_1", "vibration")
                    if k in m), None)
    tach_key = next((k for k in ("Channel_2", "channel_2", "tacho")
                     if k in m), None)
    if vib_key is None:
        raise KeyError(f"{path.name}: no vibration channel; found "
                       f"{sorted(k for k in m if not k.startswith('__'))}")
    sig = np.asarray(m[vib_key], dtype=np.float64).ravel()
    fs = 200000.0

    rpm_series = None
    if tach_key is not None:
        rpm_series = speed_from_tacho(np.asarray(m[tach_key]), fs, ppr)
    if rpm_series is None:
        warnings.warn(
            f"{path.name}: could not recover speed from the tachometer. This is "
            "a VARIABLE-SPEED record, so a constant speed would invalidate it. "
            "Fix the pulses-per-revolution (--ppr) rather than proceeding."
        )
    mean_rpm = float(np.mean(rpm_series)) if rpm_series is not None else 0.0

    return Record(file_no=trial, fault=fault, defect_in=0.0, load_hp=0,
                  rpm_nominal=int(round(mean_rpm)), fs=fs, signal=sig,
                  rpm_measured=mean_rpm or None, channel="acc",
                  path=str(path), dataset="ottawa", bearing_key="ER16K",
                  rpm_series=rpm_series, note=f"speed profile: {profile}")


def load_dir(d: str | Path, ppr: int = 1) -> List[Record]:
    out = []
    for p in sorted(Path(d).rglob("*.mat")):
        try:
            out.append(load_file(p, ppr))
        except Exception as e:
            warnings.warn(f"skipping {p.name}: {e}")
    return out
