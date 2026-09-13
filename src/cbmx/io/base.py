"""One record type, shared by every dataset loader.

Each public dataset stores its labels differently — in the filename, in a struct
field, in an accompanying spreadsheet — and each uses its own sample rate,
bearing and speed convention. Normalising all of that at the loader boundary is
what lets the same evaluation run against four datasets without a single
conditional downstream.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Record:
    file_no: int
    fault: str                 # normal | inner_race | outer_race | rolling_element | cage
    defect_in: float           # 0.0 where the dataset does not grade severity
    load_hp: int               # or a load index; only ever used to bin regimes
    rpm_nominal: int
    fs: float
    signal: np.ndarray
    rpm_measured: Optional[float] = None
    channel: str = "DE"
    path: str = ""
    dataset: str = ""
    bearing_key: str = "SKF6205"     # which geometry in the catalogue applies
    rpm_series: Optional[np.ndarray] = None   # per-sample speed, where measured
    note: str = ""

    @property
    def shaft_hz(self) -> float:
        return (self.rpm_measured or self.rpm_nominal) / 60.0

    @property
    def is_healthy(self) -> bool:
        return self.fault == "normal"

    @property
    def variable_speed(self) -> bool:
        """True where the record carries a measured speed trace that actually
        moves. This is the property that matters for AAG: an arrestment is a
        speed transient, and a method that only works at constant speed has not
        been tested for it."""
        if self.rpm_series is None or len(self.rpm_series) < 10:
            return False
        r = np.asarray(self.rpm_series)
        return float(r.max() - r.min()) / max(1.0, float(r.mean())) > 0.05

    @property
    def label(self) -> str:
        if self.is_healthy:
            return f"normal @ {self.load_hp}"
        sz = f" {self.defect_in:.3f}in" if self.defect_in else ""
        return f"{self.fault}{sz} @ {self.load_hp}"

    def describe(self) -> str:
        rpm = (f"{self.rpm_measured:.0f} meas" if self.rpm_measured
               else f"{self.rpm_nominal} nom")
        v = " VAR-SPEED" if self.variable_speed else ""
        return (f"[{self.dataset}:{self.file_no:>5}] {self.label:<32} "
                f"{self.fs/1000:6.1f}kHz {rpm:>10}  "
                f"{len(self.signal)/self.fs:6.1f}s{v}")
