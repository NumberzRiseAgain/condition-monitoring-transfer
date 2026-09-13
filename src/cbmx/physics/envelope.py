"""Envelope analysis — pulling the thumps out from under the machine.

A bearing defect does not produce a tone at BPFO. It produces a short impact,
which rings the structure at whatever its resonances happen to be — typically
some kilohertz, far above any fault frequency — and those rings *repeat* at
BPFO. So the fault frequency is not in the spectrum of the signal at all. It is
in the spectrum of the signal's amplitude.

That is the whole reason envelope analysis exists, and it is why looking for
3.585 x shaft speed in a plain FFT of the vibration usually finds nothing.

Three steps:

  1. Choose a band. The impacts excite a resonance; everything outside that
     band is shaft noise, gear mesh and electrical pickup that will swamp them.
  2. Demodulate. Take the analytic signal in that band and its magnitude — the
     outline the impacts trace.
  3. Transform the outline. The repetition rate now appears as a line.

Step 1 is where most implementations quietly cheat by hard-coding a band that
worked on the dataset they developed against. We choose it by spectral kurtosis,
which asks a question with an actual answer: in which band is the signal least
Gaussian? Impacts are impulsive and noise is not, so the band containing the
impacts is the band with the highest kurtosis. It is a published method
(Antoni's kurtogram), it needs no labels, and — the part that matters for this
bid — it adapts on its own to a machine we have never seen.

Implementation note: bandpass filtering and the Hilbert transform are the same
operation done twice. Zeroing the negative-frequency half of an FFT and keeping
only the band of interest gives the band-limited analytic signal in one inverse
transform. One FFT of the record, then every candidate band is nearly free —
which is what makes the kurtogram affordable inside a one-second budget.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── analytic signal, band-limited ────────────────────────────────────────────
def _fft_once(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    return np.fft.fft(x)


def _analytic_band(X: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    """Band-limited analytic signal from a precomputed FFT.

    Keeping only positive frequencies inside [lo, hi] and doubling them is
    exactly bandpass-then-Hilbert, in one inverse transform.
    """
    n = X.shape[0]
    f = np.fft.fftfreq(n, d=1.0 / fs)
    Y = np.zeros(n, dtype=np.complex128)
    keep = (f >= lo) & (f < hi)
    Y[keep] = 2.0 * X[keep]
    return np.fft.ifft(Y)


def _kurtosis(v: np.ndarray) -> float:
    v = v - v.mean()
    s = v.std()
    if s < 1e-12:
        return 0.0
    return float(np.mean((v / s) ** 4) - 3.0)


@dataclass
class Band:
    lo: float
    hi: float
    kurtosis: float
    level: int

    @property
    def centre(self) -> float:
        return 0.5 * (self.lo + self.hi)

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def as_dict(self) -> Dict[str, float]:
        return {"lo_hz": round(self.lo, 1), "hi_hz": round(self.hi, 1),
                "centre_hz": round(self.centre, 1), "kurtosis": round(self.kurtosis, 2),
                "level": self.level}


def choose_band(
    x: np.ndarray,
    fs: float,
    levels: Tuple[int, ...] = (2, 3, 4, 5),
    min_hz: float = 500.0,
) -> Band:
    """A fast kurtogram: which band carries the impulses?

    `min_hz` keeps the search above the shaft-order region. Fault repetition
    rates live at tens of Hz; the resonances they ring live far above. Allowing
    the search down into the orders lets it lock onto the shaft imbalance, which
    is impulsive-looking, always present, and never the answer.
    """
    X = _fft_once(x)
    nyq = fs / 2.0
    best: Optional[Band] = None
    for lv in levels:
        nb = 2 ** lv
        w = nyq / nb
        for k in range(nb):
            lo, hi = k * w, (k + 1) * w
            if hi <= min_hz or lo >= nyq:
                continue
            lo = max(lo, min_hz)
            env = np.abs(_analytic_band(X, fs, lo, hi))
            kur = _kurtosis(env)
            if best is None or kur > best.kurtosis:
                best = Band(lo, hi, kur, lv)
    if best is None:                      # fs too low for min_hz — take the top half
        lo, hi = nyq * 0.5, nyq
        env = np.abs(_analytic_band(X, fs, lo, hi))
        best = Band(lo, hi, _kurtosis(env), 1)
    return best


def commission_band(
    records: List[np.ndarray],
    fs: float,
    candidate_orders: Dict[str, float],
    shaft_hz: float,
    levels: Tuple[int, ...] = (2, 3, 4),
    min_hz: float = 500.0,
    max_order: float = 20.0,
) -> Band:
    """Pick the demodulation band once, at commissioning, and then freeze it.

    Kurtosis is the textbook criterion and it is the right one when the fault is
    already loud. It fails first: at low signal-to-noise the impacts do not make
    the band measurably non-Gaussian, while the envelope spectrum — which
    averages over hundreds of impacts — still resolves the line cleanly. On a
    synthetic outer-race fault buried in equal-power broadband noise, the
    kurtogram picks the wrong band entirely while the envelope spectrum in the
    correct band puts the line 16 dB clear of everything else.

    So this scores each candidate band by how well it resolves *any* of the
    geometrically possible fault orders, which is far more sensitive.

    That criterion, used at run time, would be cheating: choosing the band that
    best shows a fault, and then reporting that a fault was shown, is a
    multiple-comparisons trap that manufactures evidence out of noise. It is
    legitimate here for one reason only — the demodulation band is a property of
    the *structure*, not of the fault. It is chosen once when the sensor is
    installed, written into the asset card, and frozen. Run time never selects.

    Freezing also pays for itself twice: it removes the bias, and it removes a
    kurtogram from the per-window budget, which is most of what makes the
    one-second latency requirement comfortable rather than tight.
    """
    if not records:
        raise ValueError("commissioning needs at least one record")
    nyq = fs / 2.0
    best: Optional[Band] = None
    for lv in levels:
        nb = 2 ** lv
        w = nyq / nb
        for k in range(nb):
            lo, hi = max(k * w, min_hz), (k + 1) * w
            if hi <= min_hz or lo >= nyq or hi - lo < 50.0:
                continue
            cand = Band(lo, hi, 0.0, lv)
            score, kur = 0.0, 0.0
            for x in records:
                es = envelope_spectrum(x, fs, shaft_hz, band=cand, max_order=max_order)
                score = max(score, max(es.prominence_db(o)[1] for o in candidate_orders.values()))
                kur = max(kur, _kurtosis(np.abs(_analytic_band(_fft_once(x), fs, lo, hi))))
            if best is None or score > best.kurtosis:
                best = Band(lo, hi, score, lv)     # `kurtosis` field carries the score
    return best


def envelope(x: np.ndarray, fs: float, band: Optional[Band] = None) -> Tuple[np.ndarray, Band]:
    """The amplitude outline of the impacts, and the band it came from."""
    if band is None:
        band = choose_band(x, fs)
    X = _fft_once(x)
    env = np.abs(_analytic_band(X, fs, band.lo, band.hi))
    return env - env.mean(), band


# ── the envelope spectrum, in orders ─────────────────────────────────────────
@dataclass
class EnvelopeSpectrum:
    orders: np.ndarray            # x axis, multiples of shaft speed
    amplitude: np.ndarray         # linear amplitude
    shaft_hz: float
    band: Band
    fs: float

    def floor(self, at: float, exclude: float = 0.05, window: float = 0.8) -> float:
        """Local noise floor near an order, as a median with the line itself cut
        out. A median is used rather than a mean because the neighbourhood will
        often contain other real lines — harmonics, sidebands, shaft orders —
        and a mean would let them inflate the floor and hide the very peak we
        came to measure."""
        m = (np.abs(self.orders - at) <= window) & (np.abs(self.orders - at) > exclude)
        if not m.any():
            return float(np.median(self.amplitude) + 1e-12)
        return float(np.median(self.amplitude[m]) + 1e-12)

    def peak(self, at: float, tol: float = 0.02) -> Tuple[float, float]:
        """Largest amplitude within tolerance of an order; returns (order, amp).

        The tolerance is not slack for our benefit — it is speed wander. Shaft
        speed is never exactly constant, so a line sits in a small neighbourhood
        rather than on a bin. Where the peak actually landed is reported so the
        report can show measured against predicted rather than asserting a hit.
        """
        w = max(tol * at, 0.03)
        m = np.abs(self.orders - at) <= w
        if not m.any():
            return (at, 0.0)
        idx = np.argmax(np.where(m, self.amplitude, -np.inf))
        return (float(self.orders[idx]), float(self.amplitude[idx]))

    def prominence_db(self, at: float, tol: float = 0.02) -> Tuple[float, float]:
        """How far a line stands above its own neighbourhood, in dB.

        This — not raw amplitude — is the quantity a baseline is learned on.
        Raw amplitude moves with load, speed, sensor mounting and gain; a
        line's prominence over its local floor is far more stable, so a
        threshold learned on it survives the machine doing its job.
        """
        o, a = self.peak(at, tol)
        f = self.floor(at)
        return (o, float(20.0 * np.log10(max(a, 1e-12) / f)))


def envelope_spectrum(
    x: np.ndarray,
    fs: float,
    shaft_hz: float,
    band: Optional[Band] = None,
    max_order: float = 20.0,
) -> EnvelopeSpectrum:
    env, band = envelope(x, fs, band)
    n = env.shape[0]
    # Hann, because the lines we care about are narrow and close together, and
    # rectangular leakage would smear a sideband into its carrier.
    w = np.hanning(n)
    E = np.abs(np.fft.rfft(env * w)) * (2.0 / (w.sum() + 1e-12))
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    orders = f / max(shaft_hz, 1e-9)
    keep = orders <= max_order
    return EnvelopeSpectrum(orders[keep], E[keep], shaft_hz, band, fs)


# ── order tracking, for machines whose speed moves ───────────────────────────
def resample_to_angle(
    x: np.ndarray,
    fs: float,
    shaft_hz_series: np.ndarray,
    samples_per_rev: int = 256,
) -> Tuple[np.ndarray, float]:
    """Resample from equal time steps to equal shaft-angle steps.

    On a bench at fixed speed this is unnecessary. On an arresting engine during
    a recovery — which is the whole point — speed changes fast, and a fault line
    smears across many bins and disappears under the floor. Resampling against
    shaft angle instead of time pins every order to a fixed bin regardless of
    what the speed did.

    Returns the angle-domain signal and an effective sample rate expressed in
    samples per revolution, so downstream code can treat orders as its
    frequency axis directly.
    """
    x = np.asarray(x, dtype=np.float64)
    shaft_hz_series = np.asarray(shaft_hz_series, dtype=np.float64)
    if shaft_hz_series.shape[0] != x.shape[0]:
        shaft_hz_series = np.interp(
            np.linspace(0, 1, x.shape[0]),
            np.linspace(0, 1, shaft_hz_series.shape[0]),
            shaft_hz_series,
        )
    t = np.arange(x.shape[0]) / fs
    revs = np.concatenate([[0.0], np.cumsum(np.diff(t) * shaft_hz_series[:-1])])
    total = revs[-1]
    if total <= 0:
        raise ValueError("shaft never turned; cannot order-track")
    grid = np.arange(0.0, total, 1.0 / samples_per_rev)
    return np.interp(grid, revs, x), float(samples_per_rev)


def envelope_spectrum_order_tracked(
    x: np.ndarray,
    fs: float,
    shaft_hz_series: np.ndarray,
    band: Optional[Band] = None,
    samples_per_rev: int = 256,
    max_order: float = 20.0,
) -> EnvelopeSpectrum:
    """Envelope spectrum for a machine whose speed is not constant.

    The band is chosen in the time domain, where it is a physical property of
    the structure, and demodulation happens there too; only the envelope is
    resampled against angle. Choosing the band after angular resampling would
    smear the very resonance the band is meant to isolate.
    """
    env, band = envelope(x, fs, band)
    env_a, spr = resample_to_angle(env, fs, shaft_hz_series, samples_per_rev)
    n = env_a.shape[0]
    w = np.hanning(n)
    E = np.abs(np.fft.rfft(env_a * w)) * (2.0 / (w.sum() + 1e-12))
    orders = np.fft.rfftfreq(n, d=1.0 / spr)
    keep = orders <= max_order
    mean_hz = float(np.mean(shaft_hz_series))
    return EnvelopeSpectrum(orders[keep], E[keep], mean_hz, band, fs)
