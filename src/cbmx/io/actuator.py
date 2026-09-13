"""Loader for the Virginia Tech NSI / Luna Labs / NSWC Philadelphia hydraulic
rotary actuator dataset.

Six Moog Flo-Tork 15,000 in-lbf industrial rotary actuators, monitored by a Luna
Labs eCBM node at 1 kHz. Each file is one actuation: a stroke followed by a hold.

    https://github.com/vtnsi/actuator_dataset   (MIT licence)

Condition and load come from the directory names and from the benchmark code's
own label dictionaries in `CNN_train_script.py` and `CNN_train_load_script.py`,
so the pairing below is the authors' own, not our inference:

    Act_1  baseline        butterfly valve, 4 inch
    Act_4  SEAL DEFECT     butterfly valve, 4 inch     <- matched pair
    Act_2  baseline        ball valve, 2.5 inch
    Act_3  GEAR DAMAGE     ball valve, 2.5 inch        <- matched pair
    Act_5  baseline        no load
    Act_6  baseline        no load                     <- healthy/healthy control

Every actuator is one physical unit in one fixed condition, so no unit has a
healthy prefix followed by a fault. That is the whole difficulty of this set and
it is faced directly in tools/eval_actuator.py rather than worked around.

Columns, as published: Time, Accel_1..3, Angle, Temp_1..3, PG_1..3, Lim_1..3,
Internal_Temp. Values are raw counts from the node's ADC except Internal_Temp,
which is degrees Celsius. Nothing here converts counts to engineering units,
because every symptom below is a ratio or a robust z-score against a baseline
measured in the same counts.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

FS = 1000.0                      # Hz, stated by the dataset authors

UNITS = {
    "Act_1": ("baseline", "butterfly"),
    "Act_2": ("baseline", "ball"),
    "Act_3 (Gear Damage)": ("gear", "ball"),
    "Act_4 (Seal Defect)": ("seal", "butterfly"),
    "Act_5": ("baseline", "none"),
    "Act_6": ("baseline", "none"),
}


@dataclass
class Cycle:
    """One actuation, already split into its motion and hold phases."""

    unit: str
    condition: str
    load: str
    path: str
    t: np.ndarray
    angle: np.ndarray
    pg: np.ndarray                # (3, N)
    temp: np.ndarray              # (3, N)
    accel: np.ndarray             # (3, N)
    internal_temp: float
    motion: slice
    hold: slice

    @property
    def moved(self) -> bool:
        return (self.motion.stop - self.motion.start) >= 40


def _phases(angle: np.ndarray) -> tuple:
    """Split a record into the stroke and the hold that follows it.

    The stroke is found from the angle trace alone, never from pressure, so the
    same segmentation is used by both sensor arms. If segmentation depended on a
    channel only one arm can see, the two arms would not be comparable.
    """
    v = np.abs(np.gradient(angle.astype(float)))
    if v.max() <= 0:
        return slice(0, 0), slice(0, len(angle))

    # The threshold has to sit above the quantisation noise of a held angle,
    # not merely above zero. Taking first-to-last index above a low threshold
    # picks up isolated noise crossings near the end of the record and returns
    # a "stroke" that is the whole file: transit then reads 2.99 s of a 3.00 s
    # capture and the hold phase is empty. Use the longest contiguous run.
    noise = float(np.median(v)) * 6.0
    thr = max(0.15 * float(v.max()), noise, 2.0)
    above = v > thr
    best_a = best_b = 0
    a = None
    for i, x in enumerate(above):
        if x and a is None:
            a = i
        elif not x and a is not None:
            if i - a > best_b - best_a:
                best_a, best_b = a, i
            a = None
    if a is not None and len(above) - a > best_b - best_a:
        best_a, best_b = a, len(above)
    if best_b - best_a < 20:
        return slice(0, 0), slice(0, len(angle))
    b = min(best_b + 25, len(angle))            # let the stroke settle
    return slice(best_a, b), slice(b, len(angle))


def load_unit(root: str, unit: str) -> List[Cycle]:
    cond, load = UNITS[unit]
    out: List[Cycle] = []
    for p in sorted(glob.glob(os.path.join(root, unit, "*.txt"))):
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception:
            continue
        need = ["Angle", "PG_1", "PG_2", "PG_3", "Temp_1", "Temp_2", "Temp_3",
                "Accel_1", "Accel_2", "Accel_3", "Internal_Temp"]
        if any(c not in df.columns for c in need) or len(df) < 500:
            continue
        ang = df["Angle"].to_numpy(float)
        m, h = _phases(ang)
        out.append(Cycle(
            unit=unit, condition=cond, load=load, path=p,
            t=np.arange(len(ang)) / FS,
            angle=ang,
            pg=np.vstack([df[f"PG_{i}"].to_numpy(float) for i in (1, 2, 3)]),
            temp=np.vstack([df[f"Temp_{i}"].to_numpy(float) for i in (1, 2, 3)]),
            accel=np.vstack([df[f"Accel_{i}"].to_numpy(float) for i in (1, 2, 3)]),
            internal_temp=float(np.nanmedian(df["Internal_Temp"].to_numpy(float))),
            motion=m, hold=h,
        ))
    return out


def load_all(root: str) -> Dict[str, List[Cycle]]:
    return {u: load_unit(root, u) for u in UNITS}
