"""KAIST varying-speed bearing set — the one that removes the synthetic-speed caveat.

Every other bearing benchmark in this repository is recorded at a constant
shaft speed. That is convenient and it is not the problem. An arresting engine
accelerates from rest to a few hundred rpm and back to rest inside three
seconds, and a fault line that sits at 3.585 x shaft speed sweeps across a
kilohertz while it does. A method validated only at fixed speed has not been
validated for the machine in the topic.

This set is real bearing damage recorded while the shaft speed wanders randomly
between roughly 680 and 2460 rpm, with a tachometer channel alongside. That
gives three things nothing else here gives:

  1. Order tracking can be tested against the alternative rather than asserted.
     `envelope_spectrum` and `envelope_spectrum_order_tracked` differ only in
     whether the envelope is resampled against shaft angle. On a fixed-speed
     bench they agree; here they should not, and the size of the disagreement
     is a measurement rather than an argument.

  2. The speed reference itself becomes testable. The tachometer is ground
     truth, so a speed estimate recovered from the motor current -- which is
     what a deployed machine would actually have -- can be scored against it in
     rpm, not asserted to be adequate.

  3. Regimes mean something. Speed moves across the whole range inside a single
     recording, so one recording populates many speed bins, and the
     `no_baseline` path can be exercised on speeds genuinely never observed
     healthy rather than on a contrivance.

Layout, as published (three sibling Mendeley subsets; this reads subset 1):

    vibration_<cond>_<k>.csv   bearingA_x, bearingA_y, bearingB_x, bearingB_y
    current_<cond>_<k>.csv     current_R, current_S, current_T
    rpm_<cond>_<k>.csv         time, rpm

with <cond> in {normal, inner, outer} and a `_constant` variant recorded at
fixed speed, which is useful as a control on the order-tracking comparison:
whatever advantage order tracking has must vanish there.

Two things the dataset does not tell you, and how this module handles each.

**Sampling rate.** The Mendeley record does not state it. The authors' own
paper (Jung, Kim, Youn et al., "Real-Time Vibration-Based Bearing Fault
Diagnosis Under Time-Varying Speed Conditions", arXiv:2311.18547v2) gives
25.6 kHz for the vibration channels, and that is the only figure used here --
it is a constant with a citation, not a fit. It is checkable from the data: at
25.6 kHz the untruncated vibration files run about 298 s, and the tachometer
files independently run 297 s. Those two numbers were not chosen to agree.

The current channels' rate is *not* published and is *not* assumed. See
`estimate_current_fs`, which recovers it from the tachometer and reports its own
residual, so a wrong answer looks wrong.

**Bearing geometry.** The dataset gives no bearing model, ball count, ball
diameter, pitch diameter or contact angle -- neither the Mendeley record nor the
authors' paper. Geometry is what tells this system where to look, so it cannot
be inferred from the data without turning a declared search into a fitted one.
It is therefore a single required parameter, and the catalogue entry carries an
UNVERIFIED flag exactly as the uOttawa entry does. Declare it once, score once,
report the outcome either way.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

# Vibration sampling rate. Stated in the authors' paper, not measured here, and
# not a free parameter. arXiv:2311.18547v2.
FS_VIBRATION = 25_600.0

CONDITIONS = ("normal", "inner", "outer")

# The dataset's condition names, mapped to the part names the rest of cbmx uses
# -- the same keys `Bearing.orders()` returns, so a symptom id is never a string
# that only means something inside one loader.
CONDITION_TO_PART = {
    "normal": None,
    "inner": "inner_race",
    "outer": "outer_race",
}

_NAME = re.compile(r"^(vibration|current|rpm)_(normal|inner|outer)_(\d+|constant)\.csv$")


# ── tachometer ───────────────────────────────────────────────────────────────
@dataclass
class Tacho:
    """The speed reference, at its own much slower rate.

    Sampled about every 0.109 s against vibration every 39 us, so it is
    interpolated onto the signal grid wherever it is needed. Interpolation of a
    speed that is deliberately wandering is a real approximation and it is worth
    being explicit that it is one: between two tachometer samples the shaft
    turns roughly 2 to 4 revolutions, so sub-revolution speed structure is not
    observable from this channel at all. `wander_pct` reports how much the speed
    actually moved inside a window, which is the number that says whether order
    tracking should have mattered there.
    """

    t: np.ndarray                 # seconds
    rpm: np.ndarray

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0])

    @property
    def hz(self) -> np.ndarray:
        return self.rpm / 60.0

    def at(self, t: np.ndarray) -> np.ndarray:
        """Shaft speed in Hz on an arbitrary time grid, held flat outside."""
        return np.interp(t, self.t, self.rpm) / 60.0

    def span(self) -> Tuple[float, float]:
        return float(self.rpm.min()), float(self.rpm.max())


def read_tacho(path: str | Path) -> Tacho:
    import pandas as pd

    d = pd.read_csv(path)
    cols = {c.strip().lower(): c for c in d.columns}
    if "time" not in cols or "rpm" not in cols:
        raise ValueError(f"{path}: expected columns time,rpm; got {list(d.columns)}")
    t = d[cols["time"]].to_numpy(dtype=np.float64)
    r = d[cols["rpm"]].to_numpy(dtype=np.float64)
    ok = np.isfinite(t) & np.isfinite(r) & (r > 0)
    return Tacho(t[ok], r[ok])


# ── signal files ─────────────────────────────────────────────────────────────
def constant_speed_tacho(x: np.ndarray, fs: float, duration_s: float,
                         lo_hz: float = 5.0, hi_hz: float = 70.0) -> Tacho:
    """A flat speed reference for the `_constant` recordings.

    Subset 1 ships no `rpm_*_constant.csv`, so the fixed-speed control has no
    tachometer. The shaft line is recoverable anyway: residual imbalance puts
    energy at exactly 1 x shaft speed in every rotating machine, healthy or
    not, and at constant speed it is a single sharp line. Taking the dominant
    line in a plausible shaft band is label-free and health-free.

    This is an estimate and is reported as one. It is used only for the
    fixed-speed control, where the whole claim under test is that order
    tracking has *no* advantage -- a small error in a constant speed cannot
    manufacture that conclusion, because it applies identically to both arms.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = min(x.shape[0], int(8 * fs))
    w = np.hanning(n)
    X = np.abs(np.fft.rfft(x[:n] * w))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    band = (f >= lo_hz) & (f <= hi_hz)
    if not band.any():
        raise ValueError("no shaft band available at this sample rate")
    hz = float(f[np.argmax(np.where(band, X, -np.inf))])
    t = np.array([0.0, max(duration_s, 1.0)])
    return Tacho(t, np.array([hz * 60.0, hz * 60.0]))


DEFAULT_CHANNELS = {
    "vibration": ["bearingA_x", "bearingA_y", "bearingB_x", "bearingB_y"],
    "current": ["current_R", "current_S", "current_T"],
}


def read_signal(path: str | Path, max_samples: Optional[int] = None,
                kind: str = "vibration") -> Tuple[np.ndarray, List[str]]:
    """Read a vibration or current CSV as (n, channels) float64.

    These files are large -- the published vibration records are around 600 MB
    of ASCII each -- so the C parser is used and the column set is returned
    rather than guessed at by position.

    The `_constant` files in subset 1 ship WITHOUT a header row while the
    varying-speed files have one. Read blindly, that silently turns the first
    data sample into four column names and every later lookup by name fails --
    or worse, succeeds against the wrong column. So the first line is tested
    and the channel names are supplied from `DEFAULT_CHANNELS` when it is data.
    """
    import pandas as pd

    with open(path, "r") as fh:
        first = fh.readline().strip()
    try:
        [float(v) for v in first.split(",")]
        headerless = True
    except ValueError:
        headerless = False

    if headerless:
        d = pd.read_csv(path, nrows=max_samples, header=None)
        names = DEFAULT_CHANNELS.get(kind, [])[:d.shape[1]]
        if len(names) != d.shape[1]:
            names = [f"ch{i}" for i in range(d.shape[1])]
    else:
        d = pd.read_csv(path, nrows=max_samples)
        names = [str(c).strip() for c in d.columns]
    return d.to_numpy(dtype=np.float64), names


@dataclass
class Record:
    """One recording, and the speed that was measured while it ran."""

    condition: str                # normal | inner | outer
    replicate: str                # "0", "1", "2", "constant"
    kind: str                     # vibration | current
    x: np.ndarray                 # (n, channels)
    channels: List[str]
    fs: float
    tacho: Tacho
    path: str = ""
    # Seconds to add to a signal-file timestamp before looking it up in the
    # tachometer. Zero until `align_tacho` has been run; see that function for
    # why leaving it at zero is not a neutral choice.
    tacho_offset_s: float = 0.0
    align_db: Optional[Tuple[float, float]] = None   # (aligned, unaligned)

    @property
    def part(self) -> Optional[str]:
        return CONDITION_TO_PART[self.condition]

    @property
    def healthy(self) -> bool:
        return self.condition == "normal"

    @property
    def unit(self) -> str:
        """Recording identity. The dataset is one rig, so this is NOT a
        different machine -- it is the same rig with a different bearing fitted.
        Naming it `unit` anyway keeps the held-out logic honest: a baseline
        fitted on one recording and tested on another is still a cross-record
        comparison, and cross-record comparison is the thing that manufactures
        false alarms elsewhere in this repository."""
        return f"{self.condition}_{self.replicate}"

    @property
    def duration_s(self) -> float:
        return self.x.shape[0] / self.fs

    def channel(self, name: str) -> np.ndarray:
        try:
            return self.x[:, self.channels.index(name)]
        except ValueError:
            raise KeyError(f"{name!r} not in {self.channels}") from None

    def times(self) -> np.ndarray:
        return np.arange(self.x.shape[0]) / self.fs


def align_tacho(rec: "Record", channel: str, search_s: float = 3.0,
                coarse_s: float = 0.1, n_windows: int = 12,
                window_s: float = 1.0) -> Tuple[float, float, float]:
    """Recover the time offset between the signal file and its tachometer file.

    THIS IS NOT OPTIONAL AND IT IS NOT COSMETIC. The two files are not sample
    aligned in this subset: taking `time` in the rpm file at face value leaves
    the shaft line in the order domain sitting around 5 dB above its floor,
    where the same records aligned reach 25-38 dB. Order tracking against a
    speed reference that is a few hundred milliseconds out of step is worse
    than not order tracking at all, because the angle it integrates is wrong in
    a way that grows across the window.

    The offset is recovered from the SHAFT line -- 1x and 2x -- which every
    rotating machine produces from residual imbalance whether it is healthy or
    broken. No fault order, no fault label and no damaged recording enters this
    search, and the identical procedure runs on every record, so it cannot
    favour one class over another.

    Returns (offset_s, aligned_score_db, unaligned_score_db) so the size of the
    correction is visible rather than silently applied.
    """
    from ..physics.envelope import resample_to_angle

    x = rec.channel(channel)
    n = int(round(window_s * rec.fs))
    starts = np.linspace(0, max(x.shape[0] - n - 1, 0), n_windows).astype(int)

    def score(off: float) -> float:
        vals = []
        for s0 in starts:
            seg = x[s0:s0 + n]
            if seg.shape[0] < n:
                continue
            t = (np.arange(n) + s0) / rec.fs + off
            hz = rec.tacho.at(t)
            if np.min(hz) <= 0:
                continue
            xa, spr = resample_to_angle(seg - seg.mean(), rec.fs, hz, samples_per_rev=512)
            w = np.hanning(xa.shape[0])
            X = np.abs(np.fft.rfft(xa * w))
            o = np.fft.rfftfreq(xa.shape[0], d=1.0 / spr)
            tot = 0.0
            for k in (1.0, 2.0):
                near = np.abs(o - k) < 0.05
                ring = (np.abs(o - k) < 0.6) & (np.abs(o - k) > 0.08)
                if not near.any() or not ring.any():
                    continue
                tot += 20.0 * math.log10(max(float(X[near].max()), 1e-18)
                                         / max(float(np.median(X[ring])), 1e-18))
            vals.append(tot)
        return float(np.median(vals)) if vals else -math.inf

    grid = np.arange(-search_s, search_s + 1e-9, coarse_s)
    scores = [(float(g), score(float(g))) for g in grid]
    best_off, best = max(scores, key=lambda r: r[1])
    fine = np.arange(best_off - coarse_s, best_off + coarse_s + 1e-9, coarse_s / 10.0)
    for g in fine:
        s = score(float(g))
        if s > best:
            best, best_off = s, float(g)
    return best_off, best, score(0.0)


@dataclass
class Window:
    """One scored observation. The unit of everything downstream."""

    record: str                   # unit id
    condition: str
    index: int
    t0: float
    x: np.ndarray                 # single channel, time domain
    shaft_hz: np.ndarray          # same length, from the tachometer
    fs: float

    @property
    def mean_hz(self) -> float:
        return float(np.mean(self.shaft_hz))

    @property
    def wander_pct(self) -> float:
        """Peak-to-peak speed change inside this window, as a percentage of the
        mean. This is the covariate that decides whether order tracking can
        possibly help: at 0% the two spectra are the same computation."""
        lo, hi = float(self.shaft_hz.min()), float(self.shaft_hz.max())
        return 100.0 * (hi - lo) / max(self.mean_hz, 1e-9)

    @property
    def revolutions(self) -> float:
        return float(np.sum(self.shaft_hz) / self.fs)


def windows(rec: Record, channel: str, seconds: float = 1.0,
            overlap: float = 0.0, limit: Optional[int] = None
            ) -> Iterator[Window]:
    """Cut a recording into fixed-duration windows.

    Fixed *duration*, not fixed revolutions, and the choice matters. A
    fixed-revolution window would already be half of order tracking, applied
    before the comparison that is supposed to measure whether order tracking
    helps. Both arms get the same samples; only the envelope resampling differs.
    """
    x = rec.channel(channel)
    n = int(round(seconds * rec.fs))
    if n < 64:
        raise ValueError("window shorter than 64 samples")
    step = int(round(n * (1.0 - overlap)))
    if step < 1:
        raise ValueError("overlap must be below 1.0")
    t_all = rec.times()
    total = x.shape[0]
    k = 0
    for start in range(0, total - n + 1, step):
        if limit is not None and k >= limit:
            return
        sl = slice(start, start + n)
        t = t_all[sl]
        yield Window(rec.unit, rec.condition, k, float(t[0]), x[sl],
                     rec.tacho.at(t + rec.tacho_offset_s), rec.fs)
        k += 1


# ── directory discovery ──────────────────────────────────────────────────────
def scan(root: str | Path) -> Dict[Tuple[str, str, str], Path]:
    """Index a directory of KAIST CSVs by (kind, condition, replicate).

    The set ships as three sibling subsets — "divided into three parts because
    of storage limitations", per the Mendeley record — and each reuses the same
    file names. Dropping them all in one directory would silently overwrite one
    subset with another, and the run would report a smaller n than it thought it
    had, which is the worst kind of quiet error for a false-alarm claim.

    So a subdirectory is allowed, and its name is folded into the replicate:
    `data/kaist/subset2/vibration_normal_0.csv` becomes replicate `subset2-0`,
    addressed as `normal_subset2-0`. Files sitting directly in the root keep
    their bare replicate, so nothing about the single-subset layout changes.
    A subset directory must carry its own `rpm_*.csv` files, because the speed
    reference is matched to the recording and not to the condition.
    """
    out: Dict[Tuple[str, str, str], Path] = {}

    def take(p: Path, prefix: str = "") -> None:
        m = _NAME.match(p.name)
        if m:
            rep = f"{prefix}{m.group(3)}" if prefix else m.group(3)
            out[(m.group(1), m.group(2), rep)] = p

    root = Path(root)
    for p in sorted(root.glob("*.csv")):
        take(p)
    for d in sorted(x for x in root.iterdir() if x.is_dir()):
        tag = "".join(c for c in d.name.lower() if c.isalnum())
        if not tag:
            continue
        for p in sorted(d.glob("*.csv")):
            take(p, f"{tag}-")
    return out


def conditions_available(root: str | Path) -> Dict[str, List[str]]:
    """Which recordings exist, as the uids the tools take. Printed by the eval
    so a missing subset is visible as a shorter list rather than as a quietly
    smaller n."""
    idx = scan(root)
    out: Dict[str, List[str]] = {}
    for (kind, cond, rep) in sorted(idx):
        if kind == "vibration":
            out.setdefault(cond, []).append(f"{cond}_{rep}")
    return out


def load(root: str | Path, kind: str, condition: str, replicate: str = "0",
         max_seconds: Optional[float] = None, fs: Optional[float] = None) -> Record:
    """Load one recording together with its tachometer file.

    `max_seconds` caps the read, which is the difference between a two-minute
    experiment and a forty-minute one on files this size.
    """
    idx = scan(root)
    key = (kind, condition, replicate)
    if key not in idx:
        raise FileNotFoundError(
            f"no {kind}_{condition}_{replicate}.csv under {root}; "
            f"have {sorted({(k[0], k[1], k[2]) for k in idx})}")
    tkey = ("rpm", condition, replicate)
    if tkey not in idx and replicate != "constant":
        raise FileNotFoundError(f"no rpm_{condition}_{replicate}.csv under {root} "
                                f"-- the speed reference is not optional here")

    if fs is None:
        if kind != "vibration":
            raise ValueError("fs must be given for current records; the dataset "
                             "does not publish it -- see estimate_current_fs")
        fs = FS_VIBRATION

    nmax = None if max_seconds is None else int(round(max_seconds * fs))
    x, names = read_signal(idx[key], nmax, kind=kind)
    if tkey in idx:
        tacho = read_tacho(idx[tkey])
    else:
        tacho = constant_speed_tacho(x[:, 0], fs, x.shape[0] / fs)
    return Record(condition, replicate, kind, x, names, fs, tacho, str(idx[key]))


# ── the current channels' sampling rate, recovered rather than assumed ───────
@dataclass
class CurrentFsEstimate:
    fs: float
    pole_pairs: int
    residual_rpm: float           # rms disagreement with the tachometer
    slip_estimate: float
    n_frames: int
    candidates: Dict[float, float] = field(default_factory=dict)

    def as_dict(self) -> Dict:
        return {"fs_hz": round(self.fs, 1), "pole_pairs": self.pole_pairs,
                "residual_rpm_rms": round(self.residual_rpm, 1),
                "slip": round(self.slip_estimate, 4), "frames": self.n_frames}


def estimate_current_fs(
    current: np.ndarray,
    tacho: Tacho,
    fs_candidates: Sequence[float] = (12_800.0, 25_600.0, 51_200.0, 10_000.0,
                                      20_000.0, 40_000.0, 50_000.0, 65_536.0),
    pole_pairs: Sequence[int] = (1, 2, 3),
    frame_revs: float = 20.0,
) -> CurrentFsEstimate:
    """Recover the current channels' sampling rate from the tachometer.

    The dataset does not publish it, and guessing wrong puts every MCSA sideband
    in the wrong place while producing perfectly confident output -- the exact
    failure mode this repository keeps finding in its own code.

    It is recoverable because the supply fundamental is locked to shaft speed:
    for an induction motor with p pole pairs turning at n rpm with slip s,

        f_1  =  p * n / (60 * (1 - s))

    and the shaft speed is *deliberately varying* across the record, which is
    what makes this identifiable. Track the dominant current line in normalised
    frequency (cycles per sample, which needs no rate to compute), then ask
    which assumed rate makes that track proportional to the tachometer's own
    rpm curve. A wrong rate stretches the time axis, so the two curves stop
    lining up and the residual blows up.

    Reported as an rms disagreement in rpm so a bad answer is visible as a bad
    number rather than as silence.
    """
    x = np.asarray(current, dtype=np.float64)
    x = x - x.mean()
    best: Optional[CurrentFsEstimate] = None
    scores: Dict[float, float] = {}

    for fs in fs_candidates:
        n = x.shape[0]
        dur = n / fs
        if dur > tacho.duration_s * 1.05 or dur < 5.0:
            scores[fs] = float("inf")       # would run past the tachometer
            continue
        mean_hz = float(np.mean(tacho.at(np.linspace(0.0, dur, 64))))
        nfft = int(2 ** round(np.log2(max(1024.0, frame_revs * fs / max(mean_hz, 1.0)))))
        nfft = min(nfft, n)
        hop = nfft // 2
        starts = list(range(0, n - nfft + 1, hop))
        if len(starts) < 8:
            scores[fs] = float("inf")
            continue
        w = np.hanning(nfft)
        nu, tc = [], []
        for s0 in starts:
            X = np.abs(np.fft.rfft(x[s0:s0 + nfft] * w))
            f = np.fft.rfftfreq(nfft, d=1.0)          # cycles per sample
            band = (f > 1.0 / nfft * 4) & (f < 0.25)  # exclude DC leakage
            i = int(np.argmax(np.where(band, X, -np.inf)))
            nu.append(f[i])
            tc.append((s0 + nfft / 2) / fs)
        nu = np.asarray(nu)
        tc = np.asarray(tc)
        rpm_true = np.interp(tc, tacho.t, tacho.rpm)
        f_elec = nu * fs
        for p in pole_pairs:
            rpm_sync = 60.0 * f_elec / p
            # One slip scalar per candidate, fitted by least squares. Slip is a
            # property of the motor and load, not of the sampling rate, so it
            # cannot rescue a wrong fs -- a wrong fs distorts the *shape*.
            k = float(np.dot(rpm_sync, rpm_true) / max(np.dot(rpm_sync, rpm_sync), 1e-12))
            resid = float(np.sqrt(np.mean((k * rpm_sync - rpm_true) ** 2)))
            slip = 1.0 - k
            if not (0.0 <= slip < 0.15):        # physically implausible slip
                continue
            scores[fs] = min(scores.get(fs, float("inf")), resid)
            if best is None or resid < best.residual_rpm:
                best = CurrentFsEstimate(fs, p, resid, slip, len(starts))

    if best is None:
        raise RuntimeError("no candidate sampling rate is consistent with the "
                           "tachometer; the current channels cannot be used "
                           "without the rate from the dataset paper")
    best.candidates = scores
    return best
