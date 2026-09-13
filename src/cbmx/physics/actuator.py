"""Declared symptoms for a hydraulic rotary actuator, with a direction each.

WRITTEN BEFORE ANY SCORING. Each symptom below is a statement about fluid power
or gear mechanics, and each carries the direction the fault is expected to move
it. None was chosen by looking at how well it separated the labels, which is the
only property that makes the direction channel in the sequential test mean
anything: a deviation that goes the wrong way has to be able to push the
accumulator back down.

SEAL DEFECT — internal leakage past the rotor seal.
  A leaking seal lets fluid cross from the working chamber to the return side
  without doing work. Four consequences follow, and they do not all point the
  same way, which is the point:

    hold_decay    pressure falls faster once the stroke has finished, because
                  the chamber cannot hold what it has.            direction +1
    working_dP    less differential is sustained across the actuator while it
                  is moving.                                       direction -1
    transit       the stroke takes longer, because part of the supplied flow
                  crosses the seal instead of turning the rotor.   direction +1
    temp_rise     throttled leakage dissipates energy as heat.     direction +1

GEAR DAMAGE — tooth damage in the rack-and-pinion drive.
    accel_rms     a damaged mesh puts more energy into the housing. direction +1
    accel_kurt    tooth defects are impulsive, not merely louder.   direction +1
    press_ripple  torque ripple from a damaged mesh is visible in the working
                  pressure, which is why part of this is observable without an
                  accelerometer at all.                            direction +1
    angle_jerk    the stroke is less smooth.                       direction +1

The two sensor arms are declared here rather than at scoring time:

    ARM_INSTALLED   pressure and temperature, with angle used only to segment
                    the stroke and as operating-state context. This is what the
                    deployed arresting gear already carries.
    ARM_VIBRATION   the same, plus the three-axis accelerometer.

`accel_rms` and `accel_kurt` are the only symptoms that require the second arm.
Everything else is available on installed-like sensing, which is what makes the
comparison in tools/eval_actuator.py a fair one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from ..io.actuator import Cycle


@dataclass(frozen=True)
class Symptom:
    id: str
    component: str
    direction: int                # +1 fault raises it, -1 fault lowers it
    arm: str                      # "installed" or "vibration"
    primary: bool
    fn: Callable[[Cycle], float]


# ── helpers ────────────────────────────────────────────────────────────────

def _working_pg(c: Cycle) -> int:
    """Which pressure gauge is the working line, decided at commissioning.

    Chosen as the channel with the largest excursion during the stroke, on
    healthy cycles, and then frozen by the caller. Deciding it per cycle would
    let the detector pick whichever channel looked worst that day.
    """
    if not c.moved:
        return 0
    span = [float(np.ptp(c.pg[i, c.motion])) for i in range(3)]
    return int(np.argmax(span))


def _robust_slope(y: np.ndarray) -> float:
    if y.size < 20:
        return 0.0
    x = np.arange(y.size, dtype=float)
    x -= x.mean()
    yy = y - np.median(y)
    denom = float((x * x).sum())
    return float((x * yy).sum() / denom) if denom > 0 else 0.0


def _kurtosis(x: np.ndarray) -> float:
    if x.size < 20:
        return 0.0
    d = x - x.mean()
    s = d.std()
    return float((d ** 4).mean() / (s ** 4)) if s > 1e-9 else 0.0


# ── the symptoms ───────────────────────────────────────────────────────────

def _hold_decay(c: Cycle, w: int) -> float:
    seg = c.pg[w, c.hold]
    if seg.size < 50:
        return np.nan
    return -_robust_slope(seg)            # positive = falling faster


def _working_dP(c: Cycle, w: int) -> float:
    if not c.moved:
        return np.nan
    others = [i for i in range(3) if i != w]
    hi = float(np.mean(c.pg[w, c.motion]))
    lo = float(np.mean([np.mean(c.pg[i, c.motion]) for i in others]))
    return hi - lo


def _transit(c: Cycle, w: int) -> float:
    if not c.moved:
        return np.nan
    return (c.motion.stop - c.motion.start) / 1000.0


def _temp_rise(c: Cycle, w: int) -> float:
    t = c.temp[0]
    if t.size < 200:
        return np.nan
    n = max(50, t.size // 10)
    return float(np.median(t[-n:]) - np.median(t[:n]))


def _press_ripple(c: Cycle, w: int) -> float:
    if not c.moved:
        return np.nan
    seg = c.pg[w, c.motion].astype(float)
    if seg.size < 40:
        return np.nan
    return float(np.std(np.diff(seg)))


def _angle_jerk(c: Cycle, w: int) -> float:
    if not c.moved:
        return np.nan
    a = c.angle[c.motion].astype(float)
    if a.size < 40:
        return np.nan
    return float(np.std(np.diff(a, n=2)))


def _accel_rms(c: Cycle, w: int) -> float:
    if not c.moved:
        return np.nan
    a = c.accel[:, c.motion].astype(float)
    a = a - a.mean(axis=1, keepdims=True)
    return float(np.sqrt((a ** 2).mean()))


def _accel_kurt(c: Cycle, w: int) -> float:
    if not c.moved:
        return np.nan
    return float(max(_kurtosis(c.accel[i, c.motion].astype(float)) for i in range(3)))


SYMPTOMS: List[Symptom] = [
    Symptom("seal.hold_decay",   "seal", +1, "installed", True,  _hold_decay),
    Symptom("seal.working_dP",   "seal", -1, "installed", True,  _working_dP),
    Symptom("seal.transit",      "seal", +1, "installed", False, _transit),
    Symptom("seal.temp_rise",    "seal", +1, "installed", False, _temp_rise),
    Symptom("gear.press_ripple", "gear", +1, "installed", True,  _press_ripple),
    Symptom("gear.angle_jerk",   "gear", +1, "installed", False, _angle_jerk),
    Symptom("gear.accel_rms",    "gear", +1, "vibration", True,  _accel_rms),
    Symptom("gear.accel_kurt",   "gear", +1, "vibration", False, _accel_kurt),
]

ARMS = {
    "installed": {"installed"},
    "vibration": {"installed", "vibration"},
}


def measure(c: Cycle, working: int) -> Dict[str, float]:
    return {s.id: float(s.fn(c, working)) for s in SYMPTOMS}
