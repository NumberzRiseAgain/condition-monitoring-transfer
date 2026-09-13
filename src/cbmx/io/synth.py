"""Synthetic vibration with a planted defect.

The point of this file is a test whose answer is known exactly. Public
seeded-fault data is the evidence that goes in the proposal, but it cannot tell
us whether a disagreement is our bug or the dataset's noise. Here the ground
truth is not a label written by somebody — it is the number we generated with.

The model is the standard one for a localised bearing defect:

    x(t) = sum_k  A * w(theta_k) * s(t - t_k)  +  shaft orders  +  noise

    t_k       impact times, at the fault rate, with slip
    w(theta)  load-zone weighting where the defect moves; constant where it does not
    s(tau)    a decaying sinusoid — the structure ringing after being struck

Three details are deliberately included because leaving them out makes the
problem easier than it is, and a detector tuned on a too-easy signal fails on
the first real recording:

  Slip. Rolling elements do not roll perfectly; the actual repetition rate
  wanders by a percent or two. This is why fault lines are broad rather than
  sharp, and why any search has to have a tolerance. A generator without slip
  produces needle-thin lines that flatter the analysis.

  Load-zone modulation. An inner-race defect rotates through the load zone once
  per revolution and an outer-race defect does not. Reproducing that is what
  lets the sideband test be tested at all.

  Shaft orders. Real machines have imbalance and misalignment at 1x and 2x, and
  they are usually far larger than the fault. A method that only works when they
  are absent has not been tested.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from ..physics.bearing import Bearing


@dataclass
class SynthSpec:
    bearing: Bearing
    fault: str = "outer_race"        # outer_race | inner_race | rolling_element | cage | none
    shaft_hz: float = 29.17          # 1750 rpm
    fs: float = 12000.0
    seconds: float = 4.0
    severity: float = 1.0            # impact amplitude, relative to noise
    resonance_hz: float = 3200.0
    damping: float = 900.0           # ringdown rate, 1/s
    slip: float = 0.012              # fractional jitter in the repetition rate
    load_depth: float = 0.75         # how deep the load-zone modulation goes
    noise: float = 1.0               # broadband sigma
    shaft_order_amp: float = 2.5     # imbalance at 1x — usually larger than the fault
    speed_wander: float = 0.0        # fractional peak drift over the record
    seed: int = 0

    def truth(self) -> Dict[str, float]:
        """Everything the analysis is supposed to recover."""
        o = self.bearing.orders()
        return {
            "fault": self.fault,
            "fault_order": 0.0 if self.fault == "none" else o[self.fault],
            "fault_hz": 0.0 if self.fault == "none" else o[self.fault] * self.shaft_hz,
            "shaft_hz": self.shaft_hz,
            "resonance_hz": self.resonance_hz,
            "modulated": self.fault in ("inner_race", "rolling_element"),
        }


def _speed_profile(spec: SynthSpec, n: int, rng) -> np.ndarray:
    if spec.speed_wander <= 0:
        return np.full(n, spec.shaft_hz)
    t = np.linspace(0, 1, n)
    # A slow drift plus a slower ripple: enough to smear a spectrum, not enough
    # to be unphysical for a machine under changing load.
    drift = spec.speed_wander * (0.6 * np.sin(2 * np.pi * 0.35 * t) + 0.4 * t)
    return spec.shaft_hz * (1.0 + drift)


def generate(spec: SynthSpec) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Returns (signal, shaft_hz_per_sample, truth)."""
    rng = np.random.default_rng(spec.seed)
    n = int(spec.seconds * spec.fs)
    t = np.arange(n) / spec.fs
    shaft = _speed_profile(spec, n, rng)

    # Cumulative shaft revolutions — impacts are placed in angle, not in time,
    # which is what makes the speed profile actually mean something.
    revs = np.concatenate([[0.0], np.cumsum(np.diff(t) * shaft[:-1])])
    total_revs = revs[-1]

    x = np.zeros(n)
    impact_times = []

    if spec.fault != "none":
        order = spec.bearing.orders()[spec.fault]
        n_imp = int(total_revs * order)
        # Impact positions in revolutions, with slip.
        k = np.arange(1, n_imp + 1)
        pos = k / order
        pos = pos + rng.normal(0.0, spec.slip / order, size=pos.shape)
        pos = pos[(pos > 0) & (pos < total_revs)]

        # Where in the load zone did each impact happen?
        if spec.fault == "inner_race":
            phase = (pos % 1.0) * 2 * np.pi                    # once per shaft rev
        elif spec.fault == "rolling_element":
            phase = (pos * spec.bearing.ftf % 1.0) * 2 * np.pi  # at cage rate
        else:
            phase = np.zeros_like(pos)                          # stationary in the zone
        w = 1.0 - spec.load_depth * (1.0 - np.maximum(0.0, np.cos(phase)) ** 1.4)

        idx = np.searchsorted(revs, pos).clip(0, n - 1)
        impact_times = (idx / spec.fs).tolist()

        # One ringdown kernel, added at each impact. Building the kernel once and
        # slicing it is what keeps this fast enough to generate long records.
        klen = int(min(n, 6.0 * spec.fs / spec.damping))
        tk = np.arange(klen) / spec.fs
        kernel = np.exp(-spec.damping * tk) * np.sin(2 * np.pi * spec.resonance_hz * tk)
        for i, a in zip(idx, w):
            end = min(n, i + klen)
            x[i:end] += spec.severity * a * kernel[: end - i]

    # Shaft orders: imbalance at 1x, misalignment at 2x. Usually dominant.
    ang = 2 * np.pi * revs
    x += spec.shaft_order_amp * np.sin(ang)
    x += 0.45 * spec.shaft_order_amp * np.sin(2 * ang + 0.7)

    x += rng.normal(0.0, spec.noise, size=n)

    truth = spec.truth()
    truth["n_impacts"] = len(impact_times)
    truth["duration_s"] = spec.seconds
    truth["impacts_per_rev_actual"] = (
        len(impact_times) / total_revs if total_revs > 0 else 0.0
    )
    return x, shaft, truth
