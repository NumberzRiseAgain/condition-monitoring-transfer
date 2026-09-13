"""Motor current signature analysis — the same geometry, read off the wires.

Why this module exists, in one sentence: the Advanced Arresting Gear has no
vibration sensors on the deployed system, and it does have current and voltage
on the electric motor, so a method that cannot work from current cannot work on
the machine as it stands today.

The physics is not new and it is not ours — but WHICH physics applies depends
on where the bearing sits, and getting that wrong is the reason most MCSA work
on bearings disappoints.

    Bearings that carry the motor's own rotor. A defect displaces the rotor,
    the displacement modulates the air-gap permeance, and the permeance
    modulates the stator current. This is Schoen, Habetler, Kamran and
    Bartheld (1995), and it is a strong, direct coupling.

    Bearings anywhere else in the drive train — which is most bearings, and
    almost certainly the ones that matter on an arresting engine. There is no
    air gap involved. The ONLY path to the current is mechanical: the defect
    must produce shaft TORQUE ripple, the torque ripple must reach the motor,
    and the motor's current must respond to it. Every link in that chain
    attenuates, and the first one can be close to zero.

Either way the frequency is the same, and it is the one the geometry predicts —
the result appears in the current spectrum as a pair of sidebands about the
supply fundamental:

    f_sideband  =  | f_1  +/-  k * f_characteristic |            k = 1, 2, ...

with f_characteristic being exactly the BPFO, BPFI, ball-defect or cage
frequency that `physics/bearing.py` already computes from four caliper
dimensions. Schoen, Habetler, Kamran and Bartheld set this out in 1995 and the
result has been reproduced many times since. Nothing about it is fitted.

What that buys, and what it costs.

    It buys a day-one capability. Current is already instrumented, already
    wired, already telemetered. No installation, no ship availability, no new
    cabling through a wet space, no cost line for hardware.

    It costs sensitivity, and the cost is large. The mechanical-to-electrical
    coupling is weak: sideband amplitudes are routinely 40 to 60 dB below the
    fundamental, and the fundamental itself leaks across the very region where
    the sidebands sit. Published work is consistent that bearing faults are
    among the harder things to see in current — much harder than a broken rotor
    bar or a shorted turn.

Conceding that plainly is the point, not a hedge. A reviewer who knows drives
will not believe a proposal that claims vibration-equivalent performance from
current, and will discount everything else in it. The honest architecture is a
ladder: current gives coverage from day one, added vibration sensors give
sensitivity later, and the same evidence layer runs over both.

Two implementation choices that keep this from becoming a peak hunt.

    The supply fundamental is measured, not assumed. On an inverter-fed drive
    the electrical frequency is the mechanical speed times the pole-pair count,
    and the pole-pair count is often not in the dataset documentation. Guessing
    it would put every sideband in the wrong place. It is instead read off the
    current spectrum as the dominant line, and reported, so the number can be
    checked.

    The floor is local and the fundamental is excluded from it. Spectral leakage
    from a line 50 dB above its neighbourhood will otherwise set the noise floor
    for the sidebands and every prominence will read as zero. The floor is the
    median of an annulus around the line under test, with the immediate
    neighbourhood of both the line and the fundamental removed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .bearing import Bearing


@dataclass
class Spectrum:
    """One-sided amplitude spectrum, with the tools to interrogate one line."""

    freq: np.ndarray
    mag: np.ndarray

    @property
    def df(self) -> float:
        return float(self.freq[1] - self.freq[0])

    def _slice(self, lo: float, hi: float) -> slice:
        i = int(np.searchsorted(self.freq, lo))
        j = int(np.searchsorted(self.freq, hi))
        return slice(max(0, i), max(i + 1, j))

    def peak(self, at: float, tol_hz: float = 0.5) -> Tuple[float, float]:
        """Largest line within +/- tol of the prediction, and where it landed.

        A tolerance in Hz rather than in percent: the sideband position is set
        by an absolute frequency offset from the fundamental, so a percentage
        tolerance would be far too wide at high frequency and too narrow at low.
        """
        s = self._slice(at - tol_hz, at + tol_hz)
        if s.stop <= s.start:
            return 0.0, at
        seg = self.mag[s]
        k = int(np.argmax(seg))
        return float(seg[k]), float(self.freq[s][k])

    def floor(self, at: float, exclude_hz: float = 2.0, window_hz: float = 40.0,
              also_exclude: Tuple[float, ...] = ()) -> float:
        """Median magnitude in an annulus around `at`.

        `also_exclude` carries the supply fundamental and its neighbourhood.
        Leaving it in would let the strongest line in the spectrum set the floor
        for lines 50 dB below it, and every prominence would come out at zero —
        a failure that looks exactly like a healthy machine.
        """
        s = self._slice(at - window_hz, at + window_hz)
        f, m = self.freq[s], self.mag[s]
        keep = np.abs(f - at) > exclude_hz
        for x in also_exclude:
            keep &= np.abs(f - x) > exclude_hz
        m = m[keep]
        if m.size < 8:
            return float("nan")
        return float(np.median(m))

    def prominence_db(self, at: float, tol_hz: float = 0.5,
                      also_exclude: Tuple[float, ...] = ()) -> Tuple[float, float]:
        """How far the line at `at` stands above its own local floor, in dB."""
        p, where = self.peak(at, tol_hz)
        fl = self.floor(at, also_exclude=also_exclude)
        if not np.isfinite(fl) or fl <= 0 or p <= 0:
            return float("nan"), where
        return float(20.0 * np.log10(p / fl)), where


def spectrum(x: np.ndarray, fs: float, detrend: bool = True) -> Spectrum:
    x = np.asarray(x, dtype=np.float64)
    if detrend:
        x = x - x.mean()
    n = x.size
    w = np.hanning(n)
    X = np.abs(np.fft.rfft(x * w)) * (2.0 / w.sum())
    return Spectrum(np.fft.rfftfreq(n, 1.0 / fs), X)


def supply_fundamental(sp: Spectrum, lo: float = 5.0, hi: float = 400.0
                       ) -> float:
    """The dominant line — measured, never assumed.

    Bounded well away from DC so a residual offset cannot win, and below the
    switching frequency of any sane inverter so the carrier cannot.
    """
    s = sp._slice(lo, hi)
    k = int(np.argmax(sp.mag[s]))
    return float(sp.freq[s][k])


@dataclass
class SidebandPrediction:
    part: str
    order: float                 # multiples of shaft speed
    f_char: float                # Hz at this shaft speed
    lines: List[float]           # the sideband frequencies to interrogate
    k_max: int
    blocked: List[float] = field(default_factory=list)   # collided with k*f1


SUPPLY_HARMONICS = 12          # how many k*f1 lines to treat as occupied
GUARD_HZ = 1.5                 # how close is 'collides with'


def _supply_lines(f_supply: float, f_nyquist: float,
                  n: int = SUPPLY_HARMONICS) -> np.ndarray:
    """k * f1 for k = 1..n. These belong to the drive, not to the bearing."""
    ks = np.arange(1, n + 1) * f_supply
    return ks[ks < f_nyquist]


def predict(bearing: Bearing, shaft_hz: float, f_supply: float,
            k_max: int = 2, f_nyquist: float = np.inf
            ) -> Dict[str, SidebandPrediction]:
    """Where each fault would put energy in the current spectrum.

    Both signs of every order are included: the modulation is symmetric, and a
    method that only looks above the fundamental throws away half the evidence
    for no reason. Lines that fold below zero or above Nyquist are dropped
    rather than aliased into a neighbour's place.

    A line is also dropped when it lands within a guard band of a supply
    harmonic, and this is the part that matters. On the Paderborn rig the
    ball-defect frequency at 1500 rpm is 99.65 Hz against a 100 Hz electrical
    fundamental, so the upper sideband sits at 199.65 Hz — within a resolution
    cell of the 200 Hz second harmonic, which is enormous, permanent, and has
    nothing whatever to do with the bearing. Measured naively it reads as 29 dB
    of ball-defect evidence on a bearing that is certified healthy. Dropping the
    line and declaring the fault unobservable at this speed is the correct
    answer; reporting 29 dB is the failure mode this whole system exists to
    avoid, dressed up as a result.
    """
    sup = _supply_lines(f_supply, f_nyquist)
    out: Dict[str, SidebandPrediction] = {}
    for part, order in bearing.orders().items():
        fc = order * shaft_hz
        lines, blocked = [], []
        for k in range(1, k_max + 1):
            for sign in (-1, +1):
                f = f_supply + sign * k * fc
                if not (1.0 < f < f_nyquist):
                    continue
                if sup.size and float(np.min(np.abs(sup - f))) < GUARD_HZ:
                    blocked.append(f)
                    continue
                lines.append(f)
        out[part] = SidebandPrediction(part, order, fc, lines, k_max, blocked)
    return out


def measure(current: np.ndarray, fs: float, bearing: Bearing, shaft_hz: float,
            k_max: int = 2, f_supply: Optional[float] = None
            ) -> Dict[str, dict]:
    """Prominence in dB at each fault's sidebands, plus where they landed.

    The returned prominence is the MEAN over that fault's sideband set, not the
    maximum. Taking the maximum over four candidate lines is a four-fold search,
    and a four-fold search finds something on a healthy machine as reliably as
    on a broken one. Averaging asks the question the physics actually poses:
    is there energy at all of the places this defect must put it.
    """
    sp = spectrum(current, fs)
    f1 = f_supply if f_supply is not None else supply_fundamental(sp)
    pred = predict(bearing, shaft_hz, f1, k_max, f_nyquist=fs / 2.0)

    supply = tuple(_supply_lines(f1, fs / 2.0).tolist())
    out: Dict[str, dict] = {"_supply_hz": f1, "_shaft_hz": shaft_hz}
    for part, p in pred.items():
        vals, wheres = [], []
        for f in p.lines:
            db, where = sp.prominence_db(f, tol_hz=max(0.5, 3 * sp.df),
                                         also_exclude=supply)
            if np.isfinite(db):
                vals.append(db)
                wheres.append(where)
        out[part] = {
            "order": p.order,
            "f_char_hz": p.f_char,
            "n_lines": len(vals),
            "n_blocked": len(p.blocked),
            # NaN, not zero, when every line this fault would use is occupied by
            # the drive. Zero would be read downstream as 'measured, and quiet'.
            "prominence_db": (float(np.mean(vals)) if vals else float("nan")),
            "observable": bool(vals),
            "lines_hz": p.lines,
            "blocked_hz": p.blocked,
            "found_hz": wheres,
        }
    return out


# ── the torsional band ──────────────────────────────────────────────────────
#
# Everything above assumes the useful energy sits at the first or second
# sideband of the fault order. On this rig that assumption is wrong, and the
# measurement that shows why is worth writing down.
#
# An outer-race defect strikes radially. The rolling element hits a fixed point
# on the outer ring, the impulse is reacted by the housing, and almost none of
# it becomes shaft torque. Measured on Paderborn KA04, at the fault order:
#
#     radial force sensor      +23.8 dB over healthy
#     accelerometer            +26.5 dB
#     shaft torque              -0.2 dB          <- the signal dies here
#     motor phase current       +1.8 dB  (inside the healthy spread)
#
# Motor current responds to torque. If the fault puts nothing into torque, no
# amount of demodulation recovers it — Park's vector, envelope demodulation and
# raw sideband prominence all return nothing, because there is nothing to return.
#
# But the fault order is not the only place to look. Tracing every harmonic of
# BPFO through the torque channel at two speeds:
#
#     900 rpm    strongest torque response at the 5th harmonic — 229.0 Hz
#    1500 rpm    strongest torque response at the 3rd harmonic — 229.0 Hz
#
# Different harmonic numbers, the same frequency. That is a structural torsional
# resonance of the drive train, and it is the transmission path: the impulse
# train is broadband, the drive train passes the part of it that lands in the
# resonance, and the current carries whatever reaches the shaft — attenuated by
# about 13 dB, but there. In the sideband at |f1 - k*BPFO| for that k:
#
#     900 rpm,  k=5    healthy 4.6 dB -> damaged 12.9 dB
#    1500 rpm,  k=3    healthy 5.2 dB -> damaged 14.2 dB
#
# So motor current does work on this machine, under a rule that comes out of the
# physics rather than out of a search: find the torsional resonance once, freeze
# it, and read whichever harmonic of the geometry-derived order falls inside it.
# The orders still come from four caliper dimensions. Only the choice of which
# harmonic to read depends on the machine, and that choice is made once, at
# commissioning, on a small set of records that are then excluded from the test.
#
# This is the same discipline `physics/envelope.py` applies to vibration, moved
# to a different part of the spectrum: for vibration the band is the
# high-frequency bearing resonance, for current it is the low-frequency
# torsional one. Selecting the band per-window at run time would be a
# multiple-comparisons machine, and the data says so out loud — on the HEALTHY
# K001 bearing at 900 rpm, the 7th harmonic sideband reads 18.7 dB, higher than
# anything the damaged bearing produces at its own resonance. A detector free to
# pick its best harmonic each time would call that bearing faulty.


@dataclass
class TorsionalBand:
    """Where the drive train lets bearing energy through to the shaft.

    Commissioned once, then frozen. `lo`/`hi` are in Hz and are a property of
    the machine, not of the bearing or the speed — which is exactly why the
    harmonic index that lands inside it changes with speed and the band does not.
    """

    lo: float
    hi: float
    commissioned_on: int = 0
    note: str = ""

    @property
    def centre(self) -> float:
        return 0.5 * (self.lo + self.hi)

    def harmonics_inside(self, f_char: float, k_max: int = 12) -> List[int]:
        """Which harmonics of this fault order fall in the band at this speed."""
        return [k for k in range(1, k_max + 1) if self.lo <= k * f_char <= self.hi]

    def as_dict(self) -> Dict[str, float]:
        return {"lo_hz": self.lo, "hi_hz": self.hi,
                "commissioned_on": self.commissioned_on, "note": self.note}


def commission_torsional_band(damaged, healthy, bearing, width_hz: float = 60.0,
                              k_max: int = 14, f_lo: float = 50.0,
                              f_hi: float = 900.0, part: str = "outer_race"
                              ) -> TorsionalBand:
    """Find the band where the drive train delivers bearing energy to the CURRENT.

    `damaged` and `healthy` are sequences of (phase_current, fs, shaft_hz, f1).
    Commissioning is done on current rather than on torque deliberately: an
    arresting engine has current transducers and does not have a shaft torque
    transducer, so a commissioning procedure that needs one is a procedure that
    cannot be executed on the customer's machine.

    THE RULE: a frequency qualifies only if the damaged-minus-healthy contrast
    is present at EVERY speed the commissioning set contains, and the score is
    the WORST of those contrasts, not the average.

    That is a physical argument, not a statistical convenience. A transmission
    resonance is a property of the structure, so it sits at a fixed frequency
    and different harmonics of the fault order pass through it at different
    speeds. Anything that appears at one speed only is excitation or noise. On
    the Paderborn rig the rule matters: 503.7 Hz shows +5.7 dB of contrast at
    900 rpm and nothing at 1500 rpm, and the minimum-across-speeds rule discards
    it, while 229.0 Hz shows +7.1 dB at 900 rpm (5th harmonic) and +7.4 dB at
    1500 rpm (3rd harmonic) and survives.

    Three mistakes were made getting here and are worth naming, because each one
    produced a confident wrong band:

      Scoring absolute prominence rather than contrast. Every spectrum is peaky
      somewhere; what identifies a transmission path is energy the healthy
      machine does not also have.

      Summing over harmonics rather than taking the nearest. Harmonics are
      spaced f_char apart, so a fixed-width window holds more of them at high
      frequency than at low, and the search lands at the top of its own range
      for no reason but arithmetic. This found 464 Hz.

      Averaging the contrast across speeds rather than taking the worst. That
      let a one-speed artefact at 496 Hz outscore the real path, and the band it
      produced contained no harmonic at all at 1500 rpm — so the detector
      abstained on three quarters of the data and looked conservative rather
      than broken.

    This is a LABELLED, ONE-TIME step. The records used are excluded from every
    downstream measurement, exactly as `physics/envelope.py` commissions the
    vibration band. On a real machine it corresponds to one run with a known
    degraded bearing, or to a historical test record, after which the answer is
    frozen for the life of the deployment.
    """
    grid = np.arange(f_lo, f_hi, 1.0)

    def curve(samples) -> np.ndarray:
        """Sideband prominence at whichever fault harmonic is nearest each bin."""
        acc = np.full((len(samples), grid.size), np.nan)
        for i, (x, fs, fr, f1) in enumerate(samples):
            sp = spectrum(np.asarray(x, dtype=np.float64), fs)
            fc = bearing.orders()[part] * fr
            hs = np.array([k * fc for k in range(1, k_max + 1)
                           if f_lo <= k * fc <= f_hi])
            if not hs.size:
                continue
            guards = tuple(np.concatenate([np.arange(1, 40) * fr,
                                           np.arange(1, 20) * f1]).tolist())
            cache: Dict[float, float] = {}
            for j, g in enumerate(grid):
                f = float(hs[int(np.argmin(np.abs(hs - g)))])
                if abs(f - g) > width_hz / 2:
                    continue
                if f not in cache:
                    vals = []
                    for ff in (abs(f1 - f), f1 + f):
                        if ff < 2.0:
                            continue
                        db, _ = sp.prominence_db(ff, tol_hz=max(0.6, 3 * sp.df),
                                                 also_exclude=guards)
                        if np.isfinite(db):
                            vals.append(db)
                    cache[f] = float(np.mean(vals)) if vals else np.nan
                acc[i, j] = cache[f]
        # All-NaN columns are ordinary here: a grid point with no harmonic
        # within half a band-width at this speed simply has no measurement.
        # Suppressed rather than silenced blindly — the NaN survives and the
        # cross-speed minimum below treats it as 'no evidence at this speed'.
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(acc, axis=0)

    speeds = sorted({round(s[2], 1) for s in damaged})
    per_speed = []
    for v in speeds:
        d = [s for s in damaged if round(s[2], 1) == v]
        h = [s for s in healthy if round(s[2], 1) == v]
        if not d or not h:
            continue
        per_speed.append(curve(d) - curve(h))
    if not per_speed:
        raise ValueError("commissioning needs damaged and healthy records at "
                         "the same speed; got none in common")

    stack = np.vstack(per_speed)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        worst = np.nanmin(np.where(np.isfinite(stack), stack, np.nan), axis=0)
    worst = np.where(np.isfinite(worst), worst, -np.inf)
    if not np.isfinite(worst).any():
        raise ValueError("no frequency shows contrast at every speed; the "
                         "current has no usable transmission path here")
    # The score is flat across a plateau, because every grid point in a span
    # maps to the same nearest harmonic. Taking argmax picks the LEFT edge of
    # that plateau, which then drags a neighbouring harmonic with no contrast
    # into the band and dilutes the measurement. Take the centre of the plateau
    # instead — the frequencies that actually carry the energy.
    score = float(np.max(worst))
    plateau = grid[worst >= score - 0.5]
    centre = float(0.5 * (plateau.min() + plateau.max()))
    return TorsionalBand(centre - width_hz / 2, centre + width_hz / 2,
                         len(damaged) + len(healthy),
                         note=f"worst-case contrast across {len(per_speed)} "
                              f"speeds peaks at {centre:.0f} Hz (+{score:.1f} dB)")
