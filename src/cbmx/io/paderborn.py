"""Paderborn (KAT) bearing dataset — real damage, and motor current.

Two properties make this the most important validation set for DV087.

Real damage. Most of the fault records here come from accelerated lifetime
tests, not from a defect somebody machined in. Seeded faults are clean, sharp
and generous; real spalls are distributed, irregular and much harder. A method
validated only on seeded faults has been validated on the easy case, and a
reviewer who knows this field will ask.

Motor current, recorded alongside vibration. AAG is instrumented for voltage and
current, and motor current signature analysis has the same structure as our
vibration method — a physics-predicted frequency, a learned normal level, an
evidence test. Being able to show the architecture running on current as well as
vibration is what makes the multi-sensor claim in the topic concrete instead of
aspirational.

Naming: N15_M07_F10_KA01_1.mat
    N15   1500 rpm       M07  0.7 Nm load torque
    F10   1000 N radial  KA01 bearing code       1  trial
Bearing codes: K0xx healthy, KAxx outer ring, KIxx inner ring. Codes from
accelerated life tests (real damage) are listed in REAL_DAMAGE below.

The .mat holds one struct named after the file stem, whose `Y` field is an array
of channel structs each with `Name` and `Data`. That layout is awkward to reach
through scipy and varies slightly between releases, so the loader searches by
channel name rather than by index and reports what it found when it cannot.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np

from .base import Record

# Codes whose damage came from an accelerated lifetime test rather than a
# machined defect. These are the records that matter most.
REAL_DAMAGE = {"KA04", "KA15", "KA16", "KA22", "KA30",
               "KI04", "KI14", "KI16", "KI17", "KI18", "KI21"}

NAME_RE = re.compile(r"^N(\d+)_M(\d+)_F(\d+)_([A-Z]+\d+)_(\d+)$", re.I)

# Channel names as published. Vibration is the default; current is what makes
# this set worth more than the others.
CHANNELS = {
    "vibration": ("vibration_1",),
    "current": ("phase_current_1", "phase_current_2"),
    "speed": ("speed",),
    "torque": ("torque",),
    "temperature": ("temp_2_bearing_module",),
}


def _fault_of(code: str) -> str:
    c = code.upper()
    if c.startswith("K0") or c.startswith("K00"):
        return "normal"
    if c.startswith("KA"):
        return "outer_race"
    if c.startswith("KI"):
        return "inner_race"
    if c.startswith("KB"):
        return "combined"
    return "unknown"


def _channels(mat_struct) -> Dict[str, np.ndarray]:
    """Pull every named channel out of the nested Y array."""
    out: Dict[str, np.ndarray] = {}
    Y = getattr(mat_struct, "Y", None)
    if Y is None:
        return out
    items = Y if isinstance(Y, (list, np.ndarray)) else [Y]
    for ch in np.atleast_1d(items):
        name = getattr(ch, "Name", None)
        data = getattr(ch, "Data", None)
        if name is None or data is None:
            continue
        try:
            out[str(name).strip()] = np.asarray(data, dtype=np.float64).ravel()
        except Exception:
            continue
    return out


def load_file(path: str | Path, channel: str = "vibration") -> Record:
    from scipy.io import loadmat

    path = Path(path)
    m0 = NAME_RE.match(path.stem)
    if not m0:
        raise KeyError(f"{path.name}: name does not match Paderborn's "
                       "N##_M##_F##_CODE_# convention; keep the distribution's "
                       "filenames or the labels would be a guess.")
    rpm = int(m0.group(1)) * 100
    torque = int(m0.group(2)) / 10.0
    force = int(m0.group(3)) * 100
    code = m0.group(4).upper()
    trial = int(m0.group(5))

    m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    key = next((k for k in m if not k.startswith("__")), None)
    chans = _channels(m[key]) if key else {}
    if not chans:
        raise KeyError(f"{path.name}: could not read the Y channel array. "
                       f"Top-level keys: {[k for k in m if not k.startswith('__')]}")

    wanted = CHANNELS.get(channel, (channel,))
    sig = None
    for w in wanted:
        for have, data in chans.items():
            if have.lower() == w.lower():
                sig = data
                break
        if sig is not None:
            break
    if sig is None:
        raise KeyError(f"{path.name}: no '{channel}' channel. "
                       f"Present: {sorted(chans)}")

    # 64 kHz for vibration, 64 kHz for current in the published set. Stated
    # rather than inferred, because inferring it from length would be wrong the
    # moment a record is trimmed.
    fs = 64000.0

    return Record(file_no=trial, fault=_fault_of(code), defect_in=0.0,
                  load_hp=int(torque * 10), rpm_nominal=rpm, fs=fs,
                  signal=sig, rpm_measured=float(rpm), channel=channel,
                  path=str(path), dataset="paderborn", bearing_key="PU6203",
                  note=(f"{code} {'REAL damage' if code in REAL_DAMAGE else 'artificial'}"
                        f", {torque:.1f}Nm, {force}N"))


def load_dir(d: str | Path, channel: str = "vibration",
             real_damage_only: bool = False) -> List[Record]:
    out = []
    for p in sorted(Path(d).rglob("*.mat")):
        try:
            r = load_file(p, channel)
        except Exception as e:
            warnings.warn(f"skipping {p.name}: {e}")
            continue
        if real_damage_only and "REAL" not in r.note and not r.is_healthy:
            continue
        out.append(r)
    return out
