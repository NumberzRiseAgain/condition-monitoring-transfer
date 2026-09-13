"""What is normal for this machine — learned online, unlabelled, from itself.

This is the half of the system that needs data, and it is deliberately the
smaller half. Physics already said where to look. The only remaining question is
how much energy at that place is too much, and that question has an answer that
can be learned from the machine while it is healthy: no labelled failures, no
fleet model, no training run.

Three design decisions carry most of the weight.

Regimes. A bearing line at 40% load and 900 rpm has a different normal level
from the same line at full load and 1800 rpm. Learning one global threshold
across all of them produces a distribution wide enough to hide any real fault.
So statistics are kept per operating regime — coarse bins of speed and load —
and a reading is only ever compared against its own regime. On an arresting
engine this matters more than on a bench: every recovery is a different weight
and a different engaging speed, so a single threshold would be almost
meaningless.

Robust statistics. Mean and standard deviation are the wrong estimators here,
because the very thing we are looking for — an occasional large reading — drags
them upward and raises the threshold it was supposed to trip. A developing fault
would train the system to accept itself. Median and median-absolute-deviation do
not have that failure mode, so those are what is stored.

Freezing. Once a regime has been called anomalous, its baseline stops updating.
Without that, a slow degradation is absorbed as the new normal — the classic and
completely silent failure of every self-tuning condition monitor. The machine
gets worse, the baseline follows it up, and nothing is ever reported.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple


@dataclass
class Regime:
    """Coarse operating point. Deliberately coarse: fine bins never fill."""

    speed_bin: int
    load_bin: int

    @property
    def key(self) -> str:
        return f"s{self.speed_bin}l{self.load_bin}"

    @classmethod
    def of(cls, shaft_hz: float, load: float = 0.0,
           speed_step_hz: float = 5.0, load_step: float = 0.25) -> "Regime":
        return cls(int(shaft_hz // speed_step_hz), int(load // load_step))


@dataclass
class Running:
    """Streaming median and MAD, via P-square-style quantile tracking.

    Storing every sample would be exact and would also mean unbounded memory on
    a box that must run for months between port calls. These estimators are a
    few floats per line per regime and converge quickly enough to be useful
    after a few dozen observations.
    """

    n: int = 0
    median: float = 0.0
    mad: float = 0.0
    lo: float = math.inf
    hi: float = -math.inf
    frozen: bool = False
    min_samples: int = 30

    # Step sizes decay as 1/n so early samples move the estimate a lot and later
    # ones refine it — the right behaviour for a machine that must be useful
    # after an hour but keep improving over a deployment.
    def update(self, x: float) -> None:
        if self.frozen:
            return
        self.n += 1
        self.lo = min(self.lo, x)
        self.hi = max(self.hi, x)
        if self.n == 1:
            self.median = x
            self.mad = 0.0
            return
        step = max(0.02, 1.0 / self.n)
        self.median += step * (1.0 if x > self.median else -1.0) * max(self.mad, 0.05)
        dev = abs(x - self.median)
        self.mad += step * (dev - self.mad)

    @property
    def ready(self) -> bool:
        """Below this the estimate is not worth thresholding against, and the
        honest thing is to say so rather than emit a confident number.

        Configurable because real datasets do not care what number looked tidy:
        the CWRU healthy records give about ten seconds per load, which lands
        just under a hard-coded 30 and would silently disable every line. A
        threshold that data cannot meet should be a stated parameter, not a
        constant somebody has to find.
        """
        return self.n >= self.min_samples

    def z(self, x: float) -> float:
        """Robust z-score. 1.4826 converts MAD to a Gaussian-equivalent sigma,
        so a threshold expressed in these units means what a reader expects."""
        s = max(self.mad * 1.4826, 0.5)      # floor: never divide by a fluke
        return (x - self.median) / s

    def as_dict(self) -> Dict:
        return {"n": self.n, "median_db": round(self.median, 2),
                "mad_db": round(self.mad, 2), "ready": self.ready,
                "frozen": self.frozen,
                "seen_lo": None if self.lo == math.inf else round(self.lo, 2),
                "seen_hi": None if self.hi == -math.inf else round(self.hi, 2)}


class Baseline:
    """Per (symptom, regime) normality for one asset."""

    def __init__(self, asset_id: str, speed_step_hz: float = 5.0,
                 load_step: float = 0.25, min_samples: int = 30):
        self.asset_id = asset_id
        self.speed_step_hz = speed_step_hz
        self.load_step = load_step
        self.min_samples = min_samples
        self.stats: Dict[str, Running] = {}

    @staticmethod
    def _k(symptom_id: str, regime: Regime) -> str:
        return f"{symptom_id}@{regime.key}"

    def regime(self, shaft_hz: float, load: float = 0.0) -> Regime:
        return Regime.of(shaft_hz, load, self.speed_step_hz, self.load_step)

    def get(self, symptom_id: str, regime: Regime) -> Running:
        return self.stats.setdefault(self._k(symptom_id, regime),
                                     Running(min_samples=self.min_samples))

    def observe(self, symptom_id: str, regime: Regime, prominence_db: float) -> None:
        self.get(symptom_id, regime).update(prominence_db)

    def freeze(self, symptom_id: str, regime: Regime) -> None:
        """Stop learning this line here. Called the moment evidence crosses, so
        a developing fault cannot teach the system that it is normal."""
        self.get(symptom_id, regime).frozen = True

    def score(self, symptom_id: str, regime: Regime, prominence_db: float
              ) -> Tuple[float, Running]:
        r = self.get(symptom_id, regime)
        return (r.z(prominence_db) if r.ready else 0.0), r

    # -- persistence ---------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({
            "asset_id": self.asset_id,
            "speed_step_hz": self.speed_step_hz,
            "load_step": self.load_step,
            "min_samples": self.min_samples,
            "stats": {k: v.__dict__ for k, v in self.stats.items()},
        }, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Baseline":
        d = json.loads(Path(path).read_text())
        b = cls(d["asset_id"], d.get("speed_step_hz", 5.0), d.get("load_step", 0.25),
                d.get("min_samples", 30))
        for k, v in d.get("stats", {}).items():
            r = Running()
            r.__dict__.update(v)
            b.stats[k] = r
        return b

    def coverage(self) -> Dict:
        ready = sum(1 for r in self.stats.values() if r.ready)
        return {"lines_tracked": len(self.stats), "lines_ready": ready,
                "min_samples": self.min_samples,
                "max_n": max((r.n for r in self.stats.values()), default=0),
                "lines_frozen": sum(1 for r in self.stats.values() if r.frozen),
                "regimes": len({k.split("@")[1] for k in self.stats})}
