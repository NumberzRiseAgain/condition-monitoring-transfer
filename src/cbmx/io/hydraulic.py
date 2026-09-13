"""ZeMA / UCI hydraulic test rig — 2205 cycles, seventeen channels, four
components degraded independently.

Why this dataset is in the repository at all.

Everything else here is a bearing on a bench: a shaft turning at a constant
speed, an accelerometer, a defect ground into a race. That is the right place to
prove the *physics* half of cbmx, because bearing geometry predicts exactly
where the energy will land and the prediction can be checked.

It is the wrong place to prove the other half. The Advanced Arresting Gear does
not present as a bench bearing. It presents as a machine that performs a
repeated, bounded, heavily-loaded cycle, instrumented with pressure,
temperature and motor current — and, at the time of writing, with no vibration
sensors on the deployed system at all. A method that only works when someone
hands it an accelerometer is not a method that can be fielded on it.

This rig is the closest public analogue that exists:

  repeated fixed-duration load cycles          an arrestment is a cycle
  pressure, temperature, motor power           the sensors AAG already has
  four components degrading independently      several failure modes at once
  graded severity, not present/absent          the interesting regime
  a 'stable' flag on each cycle                the rig admits when it is unsure

What does not carry over is the physics library: there is no bearing here, no
geometry, no order to predict. That is the point of running it. If the
governance layer — regimes, robust baselines, freezing, sequential evidence,
abstention — only works when bolted to bearing physics, then it is not a layer,
it is a feature of the bearing code. Running it over a machine with no bearings
in it is the test of that claim, and it is a test that can fail.

Layout of the files as distributed. Each sensor is one tab-delimited matrix:
rows are cycles (2205 of them, in time order), columns are samples within the
60-second cycle. The number of columns differs per sensor because the rates do:
6000 at 100 Hz, 600 at 10 Hz, 60 at 1 Hz. `profile.txt` carries one row per
cycle with the four component conditions and the stable flag.

Reading 556 MB of text on every run is slow enough to discourage re-running the
evaluation, which is the wrong incentive, so the first load converts each
matrix to a .npy and every later load memory-maps it.

Two channels are excluded from anything that scores, and the exclusion is
deliberate rather than incidental:

  CE, CP  'virtual' cooling efficiency and cooling power. These are not
          measurements; they are computed by the rig from the temperature
          channels. Using them to detect a cooler fault would be scoring
          against a quantity derived from the thing being detected. Every
          published result on this dataset that reports near-perfect cooler
          classification should be read with that in mind.
  SE      efficiency factor, likewise virtual.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


# name -> (samples per cycle, sampling rate Hz, physical quantity, unit)
SENSORS: Dict[str, tuple] = {
    "PS1":  (6000, 100.0, "pressure", "bar"),
    "PS2":  (6000, 100.0, "pressure", "bar"),
    "PS3":  (6000, 100.0, "pressure", "bar"),
    "PS4":  (6000, 100.0, "pressure", "bar"),
    "PS5":  (6000, 100.0, "pressure", "bar"),
    "PS6":  (6000, 100.0, "pressure", "bar"),
    "EPS1": (6000, 100.0, "motor_power", "W"),
    "FS1":  (600,   10.0, "volume_flow", "l/min"),
    "FS2":  (600,   10.0, "volume_flow", "l/min"),
    "TS1":  (60,     1.0, "temperature", "degC"),
    "TS2":  (60,     1.0, "temperature", "degC"),
    "TS3":  (60,     1.0, "temperature", "degC"),
    "TS4":  (60,     1.0, "temperature", "degC"),
    "VS1":  (60,     1.0, "vibration", "mm/s"),
    # virtual — loadable, never scored. See the module docstring.
    "CE":   (60,     1.0, "cooling_efficiency_VIRTUAL", "%"),
    "CP":   (60,     1.0, "cooling_power_VIRTUAL", "kW"),
    "SE":   (60,     1.0, "efficiency_factor_VIRTUAL", "%"),
}

VIRTUAL = ("CE", "CP", "SE")
MEASURED = tuple(s for s in SENSORS if s not in VIRTUAL)

N_CYCLES = 2205
CYCLE_SECONDS = 60.0

# profile.txt column -> (name, value meaning 'optimal', ordering of severity)
# Severity is expressed as 0 = healthy and increasing integers = worse, so that
# every component can be reported on one scale regardless of whether its native
# units go up (bar) or down (%) with health.
COMPONENTS: Dict[str, dict] = {
    "cooler": {
        "col": 0,
        "healthy_value": 100,
        # % efficiency: 100 full, 20 reduced, 3 close to total failure
        "severity": {100: 0, 20: 1, 3: 2},
        "labels": {100: "full efficiency", 20: "reduced efficiency",
                   3: "close to total failure"},
    },
    "valve": {
        "col": 1,
        "healthy_value": 100,
        "severity": {100: 0, 90: 1, 80: 2, 73: 3},
        "labels": {100: "optimal switching", 90: "small lag",
                   80: "severe lag", 73: "close to total failure"},
    },
    "pump": {
        "col": 2,
        "healthy_value": 0,
        "severity": {0: 0, 1: 1, 2: 2},
        "labels": {0: "no leakage", 1: "weak leakage", 2: "severe leakage"},
    },
    "accumulator": {
        "col": 3,
        "healthy_value": 130,
        "severity": {130: 0, 115: 1, 100: 2, 90: 3},
        "labels": {130: "optimal pressure", 115: "slightly reduced",
                   100: "severely reduced", 90: "close to total failure"},
    },
}

STABLE_COL = 4


@dataclass
class Profile:
    """One row per cycle: the four component conditions and the stable flag."""

    raw: np.ndarray                     # (2205, 5) int

    def value(self, component: str, i: int) -> int:
        return int(self.raw[i, COMPONENTS[component]["col"]])

    def severity(self, component: str, i: int) -> int:
        return COMPONENTS[component]["severity"][self.value(component, i)]

    def severity_series(self, component: str) -> np.ndarray:
        col = self.raw[:, COMPONENTS[component]["col"]]
        table = COMPONENTS[component]["severity"]
        return np.array([table[int(v)] for v in col], dtype=int)

    def condition_label(self, component: str, i: int) -> str:
        return COMPONENTS[component]["labels"][self.value(component, i)]

    @property
    def stable(self) -> np.ndarray:
        """True where the rig reported that static conditions were reached.

        The rig flags 756 of its 2205 cycles as possibly not settled. Learning a
        baseline from those would widen every distribution with transients that
        are nothing to do with component health, which is the ordinary way a
        self-tuning monitor talks itself out of ever reporting anything. They
        are excluded from baseline fitting and reported separately at scoring.
        """
        return self.raw[:, STABLE_COL] == 0

    def healthy_mask(self, component: str) -> np.ndarray:
        """Cycles where THIS component is at its optimal condition.

        Not 'cycles where everything is optimal' — there are only 21 of those,
        10 of them settled, and they are contiguous in time. Ten adjacent cycles
        is not a baseline; it is one operating point observed for ten minutes,
        and any threshold learned from it would be an artefact of that ten
        minutes. Per-component healthy pools give 599 to 1221 cycles each,
        spread across the whole run.

        The price of that choice is stated rather than hidden: a pool that is
        healthy for the pump may contain a degraded cooler. Regime binning and
        robust statistics are what absorb that, and the false-alarm rate
        measured on held-out healthy cycles is what shows whether they did.
        """
        col = COMPONENTS[component]["col"]
        return self.raw[:, col] == COMPONENTS[component]["healthy_value"]


class HydraulicRig:
    """Lazy, memory-mapped access to the sensor matrices."""

    def __init__(self, root: str | Path, cache: Optional[str | Path] = None):
        self.root = Path(root)
        self.cache = Path(cache) if cache else self.root / "_npy"
        self.cache.mkdir(parents=True, exist_ok=True)
        self._mm: Dict[str, np.ndarray] = {}
        self._profile: Optional[Profile] = None

    # -- profile -------------------------------------------------------------
    @property
    def profile(self) -> Profile:
        if self._profile is None:
            raw = np.loadtxt(self.root / "profile.txt", dtype=int)
            if raw.shape != (N_CYCLES, 5):
                raise ValueError(f"profile.txt is {raw.shape}, expected "
                                 f"({N_CYCLES}, 5) — wrong file or truncated")
            self._profile = Profile(raw)
        return self._profile

    # -- sensors -------------------------------------------------------------
    def _npy(self, sensor: str) -> Path:
        return self.cache / f"{sensor}.npy"

    def build_cache(self, sensors=MEASURED, verbose: bool = True) -> None:
        for s in sensors:
            p = self._npy(s)
            if p.exists():
                continue
            src = self.root / f"{s}.txt"
            if not src.exists():
                raise FileNotFoundError(src)
            if verbose:
                print(f"  converting {s}.txt ...", flush=True)
            a = np.loadtxt(src, dtype=np.float32)
            n_cols = SENSORS[s][0]
            if a.shape != (N_CYCLES, n_cols):
                raise ValueError(f"{s}: got {a.shape}, expected "
                                 f"({N_CYCLES}, {n_cols})")
            np.save(p, a)
            del a

    def sensor(self, name: str) -> np.ndarray:
        """(2205, n) memory-mapped float32. Rows are cycles."""
        if name not in self._mm:
            p = self._npy(name)
            if not p.exists():
                self.build_cache([name])
            self._mm[name] = np.load(p, mmap_mode="r")
        return self._mm[name]

    def cycle(self, name: str, i: int) -> np.ndarray:
        return np.asarray(self.sensor(name)[i], dtype=np.float64)

    def t(self, name: str) -> np.ndarray:
        n, fs = SENSORS[name][0], SENSORS[name][1]
        return np.arange(n) / fs

    # -- integrity -----------------------------------------------------------
    def dead_channels(self, sensors=MEASURED) -> List[str]:
        """Channels that never move across the whole run.

        PS4 reads exactly zero in the first cycle. A channel that is dead is not
        a channel that is quiet, and a monitor that scores it will report that
        nothing is ever wrong with whatever it was supposed to watch. Checking
        for it costs one pass and is the sort of thing that otherwise surfaces
        after a proposal has been submitted.
        """
        dead = []
        for s in sensors:
            a = self.sensor(s)
            if float(np.ptp(np.asarray(a[::37], dtype=np.float64))) == 0.0:
                dead.append(s)
        return dead

    def summary(self) -> str:
        p = self.profile
        out = [f"hydraulic rig: {N_CYCLES} cycles x {CYCLE_SECONDS:.0f}s",
               f"  measured channels: {', '.join(MEASURED)}",
               f"  virtual (never scored): {', '.join(VIRTUAL)}",
               f"  settled cycles: {int(p.stable.sum())} of {N_CYCLES}"]
        for c in COMPONENTS:
            h = int(p.healthy_mask(c).sum())
            hs = int((p.healthy_mask(c) & p.stable).sum())
            out.append(f"  {c:<12} healthy {h:>5}   healthy+settled {hs:>5}")
        both = p.stable.copy()
        for c in COMPONENTS:
            both &= p.healthy_mask(c)
        out.append(f"  all four optimal AND settled: {int(both.sum())}"
                   f"   <- why per-component pools are used instead")
        return "\n".join(out)
