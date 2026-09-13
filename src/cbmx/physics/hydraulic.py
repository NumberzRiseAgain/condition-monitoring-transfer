"""What each fault does to the circuit, written down before anything is scored.

This file is the hydraulic counterpart of `physics/bearing.py`, and it exists
for the same reason. In the bearing case, geometry says which frequency a defect
lands on, so the search is one line wide and a hit means something. Without that,
a monitor has 43,680 numbers per cycle and four labels, and whatever it finds
will be significant at some threshold. That is not detection, it is a
multiple-comparisons machine with a plausible story attached.

There is no geometry here. What there is instead is a hydraulic circuit whose
failure modes have known signatures in first-year fluid power:

  cooler         removes heat. Failing, it removes less. Oil runs hotter and the
                 temperature drop across it shrinks.
  valve          switches. Failing, it switches more slowly. The pressure edge
                 on the switched line gets less steep and the transition takes
                 longer.
  pump leakage   internal leakage returns fluid to suction instead of delivering
                 it. Less flow for the same shaft power, and the lost energy
                 turns into heat.
  accumulator    stores fluid under gas pre-charge and smooths the supply. Losing
                 pre-charge, it stops smoothing: supply pressure ripples more,
                 and it stiffens — pressure follows demand more abruptly.

Every symptom below was written from that paragraph, with a declared direction
of fault, before any symptom was scored against any label. That ordering is the
whole discipline; a symptom picked because it separated the classes well is a
fitted parameter wearing the costume of a physical argument.

Two consequences are accepted rather than engineered around:

  Some symptoms will be weak. The published work on this rig reports cooler and
  valve as easy and accumulator as hard. If the accumulator symptoms here are
  weak, the correct output is a low detection rate and an honest statement of
  it, not a better feature.

  Some faults will be confounded. Pump leakage heats the oil, and so does a
  failing cooler. A monitor that reports 'the oil is hot' has said nothing
  useful. The attribution has to come from the symptom that only one of them can
  produce — flow per watt for the pump, temperature drop across the cooler —
  and where two components can produce the same evidence, the honest answer is
  to name neither.

Channel choice. Which physical port each numbered sensor sits on is not in the
distributed documentation. Rather than guess, the channels used below are picked
by a rule applied to the BASELINE CYCLES ONLY and stated in the code: the
working pressure line is the pressure channel with the highest mean, the
switched line is the pressure channel with the largest within-cycle range, the
hottest and coldest temperature channels bracket the cooler. No label is
consulted in making any of those choices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np


@dataclass(frozen=True)
class Symptom:
    """One measurable quantity, and what a fault does to it.

    `direction` is +1 if the fault makes the number go up and -1 if down. It is
    declared here, from the physics, and the evidence test uses it as a channel:
    a symptom that moves the wrong way is evidence *against* that component,
    not merely a smaller amount of evidence for it. Without that, any large
    excursion in any direction accumulates toward a diagnosis.
    """

    id: str
    component: str
    direction: int
    unit: str
    rationale: str
    fn: Callable[["CycleView"], float]
    primary: bool = True


class CycleView:
    """Everything one cycle's symptoms are allowed to see.

    A thin wrapper rather than raw arrays, so that a symptom cannot quietly
    reach for a virtual channel or for the profile row.
    """

    def __init__(self, rig, i: int, channels: "ChannelMap"):
        self._rig = rig
        self.i = i
        self.ch = channels

    def get(self, sensor: str) -> np.ndarray:
        if sensor in ("CE", "CP", "SE"):
            raise ValueError(f"{sensor} is a virtual channel computed by the "
                             f"rig from the temperatures; scoring against it "
                             f"would be circular")
        return self._rig.cycle(sensor, self.i)


@dataclass
class ChannelMap:
    """Which numbered sensor plays which physical role.

    Resolved once, from unlabelled statistics over the baseline cycles.
    """

    working_pressure: str        # highest mean pressure — the load line
    switched_pressure: str       # largest within-cycle swing — downstream of the valve
    working_flow: str            # flow with the largest swing — the switched circuit
    cooling_flow: str            # the steadier flow — the cooling circuit
    hot_temp: str                # highest mean temperature
    cold_temp: str               # lowest mean temperature
    motor_power: str = "EPS1"
    vibration: str = "VS1"
    excluded: tuple = ()         # channels found dead

    @classmethod
    def resolve(cls, rig, idx: np.ndarray) -> "ChannelMap":
        """idx: the cycle indices of the baseline pool. No labels are read."""
        pres = [s for s in ("PS1", "PS2", "PS3", "PS4", "PS5", "PS6")]
        stats = {}
        for s in pres + ["FS1", "FS2", "TS1", "TS2", "TS3", "TS4"]:
            a = np.asarray(rig.sensor(s)[idx], dtype=np.float64)
            stats[s] = (float(a.mean()), float(np.ptp(a, axis=1).mean()))

        dead = tuple(s for s in pres if stats[s][0] == 0.0 and stats[s][1] == 0.0)
        live_p = [s for s in pres if s not in dead]

        working = max(live_p, key=lambda s: stats[s][0])
        switched = max((s for s in live_p if s != working),
                       key=lambda s: stats[s][1])
        wflow = max(("FS1", "FS2"), key=lambda s: stats[s][1])
        cflow = "FS2" if wflow == "FS1" else "FS1"
        temps = ["TS1", "TS2", "TS3", "TS4"]
        hot = max(temps, key=lambda s: stats[s][0])
        cold = min(temps, key=lambda s: stats[s][0])
        return cls(working, switched, wflow, cflow, hot, cold, excluded=dead)

    def describe(self) -> str:
        return (f"working pressure {self.working_pressure} · switched "
                f"{self.switched_pressure} · working flow {self.working_flow} · "
                f"cooling flow {self.cooling_flow} · hot {self.hot_temp} · "
                f"cold {self.cold_temp}"
                + (f" · DEAD {','.join(self.excluded)}" if self.excluded else ""))


# ── helpers ─────────────────────────────────────────────────────────────────

def _edge_metrics(x: np.ndarray, fs: float) -> tuple:
    """Steepest transition in a signal, and how long the transition takes.

    Robust to the sample-level noise that would otherwise make max|dx/dt| a
    measure of the noise floor: the derivative is taken over a 50 ms span
    rather than between adjacent samples.
    """
    span = max(1, int(0.05 * fs))
    d = (x[span:] - x[:-span]) / (span / fs)
    if d.size == 0:
        return 0.0, 0.0
    slew = float(np.abs(d).max())

    rng = float(x.max() - x.min())
    if rng <= 0:
        return slew, 0.0
    lo, hi = x.min() + 0.1 * rng, x.min() + 0.9 * rng
    in_transit = float(np.mean((x > lo) & (x < hi)))
    return slew, in_transit


def _steady_ripple(x: np.ndarray, fs: float) -> float:
    """Ripple in the loaded part of the cycle, with the trend removed.

    Taking the standard deviation of the whole cycle would mostly measure the
    load step, which is commanded and has nothing to do with the accumulator.
    So: keep only samples in the upper half of the pressure range, detrend
    linearly, and report the robust spread of what is left.
    """
    if x.size < 20:
        return 0.0
    thr = x.min() + 0.5 * (x.max() - x.min())
    m = x > thr
    if m.sum() < 20:
        return 0.0
    y = x[m]
    t = np.arange(y.size, dtype=np.float64)
    a, b = np.polyfit(t, y, 1)
    r = y - (a * t + b)
    return float(np.median(np.abs(r - np.median(r))) * 1.4826)


# ── the symptoms ────────────────────────────────────────────────────────────

def _cooler_dT_across(c: CycleView) -> float:
    """Temperature drop the cooler achieves. Falls as the cooler fails."""
    return float(c.get(c.ch.hot_temp).mean() - c.get(c.ch.cold_temp).mean())


def _cooler_temp_level(c: CycleView) -> float:
    """Mean oil temperature across all four probes. Rises as the cooler fails.

    Confounded on purpose, and marked non-primary because of it: internal pump
    leakage also heats the oil. Kept because it corroborates, never because it
    attributes.
    """
    return float(np.mean([c.get(s).mean() for s in ("TS1", "TS2", "TS3", "TS4")]))


def _valve_slew(c: CycleView) -> float:
    """Steepest pressure rate on the switched line. Falls as the valve lags."""
    x = c.get(c.ch.switched_pressure)
    return _edge_metrics(x, 100.0)[0]


def _valve_transit(c: CycleView) -> float:
    """Fraction of the cycle the switched line spends mid-transition.

    Rises as the valve lags: a slow valve spends longer between states.
    """
    x = c.get(c.ch.switched_pressure)
    return _edge_metrics(x, 100.0)[1]


def _pump_flow_per_watt(c: CycleView) -> float:
    """Delivered flow per watt of motor power. Falls with internal leakage.

    This is the one symptom that separates pump leakage from a cooler fault.
    Both raise oil temperature; only leakage costs delivered flow at constant
    input power.
    """
    p = float(c.get(c.ch.motor_power).mean())
    if p <= 0:
        return float("nan")
    return float(c.get(c.ch.working_flow).mean()) / p * 1000.0   # l/min per kW


def _pump_power(c: CycleView) -> float:
    """Mean motor power. Non-primary: it moves for several reasons."""
    return float(c.get(c.ch.motor_power).mean())


def _accum_ripple(c: CycleView) -> float:
    """Supply-pressure ripple under load. Rises as pre-charge is lost.

    An accumulator with its gas charge intact absorbs the pump's delivery
    pulsation and the demand steps. Losing charge, it stops absorbing and the
    ripple appears on the working line.
    """
    return _steady_ripple(c.get(c.ch.working_pressure), 100.0)


def _accum_stiffness(c: CycleView) -> float:
    """How abruptly the working line follows demand. Rises as charge is lost.

    Measured as the steepest pressure rate on the working line — not the
    switched line, which is the valve's business.
    """
    return _edge_metrics(c.get(c.ch.working_pressure), 100.0)[0]


SYMPTOMS: List[Symptom] = [
    Symptom("cooler.dT_across", "cooler", -1, "degC",
            "a failing cooler removes less heat, so the drop across it shrinks",
            _cooler_dT_across, primary=True),
    Symptom("cooler.temp_level", "cooler", +1, "degC",
            "oil runs hotter — but so it does with pump leakage, so this "
            "corroborates and never attributes",
            _cooler_temp_level, primary=False),

    Symptom("valve.slew", "valve", -1, "bar/s",
            "a lagging valve produces a less steep pressure edge",
            _valve_slew, primary=True),
    Symptom("valve.transit", "valve", +1, "fraction",
            "a lagging valve spends longer between states",
            _valve_transit, primary=True),

    Symptom("pump.flow_per_kW", "pump", -1, "l/min/kW",
            "internal leakage returns fluid to suction: less delivered flow "
            "for the same input power",
            _pump_flow_per_watt, primary=True),
    Symptom("pump.power", "pump", +1, "W",
            "input power shifts with leakage, but with much else besides",
            _pump_power, primary=False),

    Symptom("accum.ripple", "accumulator", +1, "bar",
            "a discharged accumulator stops smoothing the supply",
            _accum_ripple, primary=True),
    Symptom("accum.stiffness", "accumulator", +1, "bar/s",
            "without stored volume the line follows demand more abruptly",
            _accum_stiffness, primary=True),
]


BY_COMPONENT: Dict[str, List[Symptom]] = {}
for _s in SYMPTOMS:
    BY_COMPONENT.setdefault(_s.component, []).append(_s)


def compute_matrix(rig, channels: ChannelMap, n_cycles: int,
                   progress_every: int = 0) -> np.ndarray:
    """(n_cycles, n_symptoms) float64. NaN where a symptom is undefined."""
    out = np.full((n_cycles, len(SYMPTOMS)), np.nan)
    for i in range(n_cycles):
        v = CycleView(rig, i, channels)
        for j, s in enumerate(SYMPTOMS):
            try:
                out[i, j] = s.fn(v)
            except Exception:
                out[i, j] = np.nan
        if progress_every and (i + 1) % progress_every == 0:
            print(f"    {i+1}/{n_cycles} cycles", flush=True)
    return out


def regime_variables(rig, n_cycles: int, channels: ChannelMap) -> np.ndarray:
    """(n_cycles,) mean oil temperature — the nuisance variable to bin on.

    Everything in a hydraulic circuit depends on oil temperature: viscosity sets
    leakage, valve response and pressure drop. Comparing a cold cycle against a
    warm baseline would produce an alarm on every cold start, which is precisely
    the nuisance alarm that gets a monitor switched off.

    It is used as a regime for valve, pump and accumulator. It is deliberately
    NOT used for the cooler, because for the cooler the oil temperature is
    downstream of the fault: binning by it would condition on the outcome and
    file every cooler fault into a regime with no baseline. That is a real
    trade — the cooler judgement carries the temperature nuisance the other
    three do not — and it is stated rather than absorbed.
    """
    t = np.zeros(n_cycles)
    probes = ("TS1", "TS2", "TS3", "TS4")
    arrs = [rig.sensor(s) for s in probes]
    for i in range(n_cycles):
        t[i] = float(np.mean([float(np.asarray(a[i]).mean()) for a in arrs]))
    return t
