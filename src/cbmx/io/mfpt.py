"""MFPT bearing fault data — a different rig, a different bearing.

Small, quick, and valuable for one reason: it uses a NICE bearing with eight
rollers and completely different proportions from the CWRU 6205. If the
geometry-first method is right, it should transfer with no change at all — the
asset card gets different numbers and nothing else moves. If it does not
transfer, we have been fitting to CWRU without noticing.

The geometry in the catalogue is verified rather than transcribed: it reproduces
MFPT's own published 81.12 Hz outer-race and 118.875 Hz inner-race frequencies
at a 25 Hz shaft, to three decimals. That check is the difference between a
dimension you believe and one you have merely copied.

Layout: each .mat holds a struct `bearing` with fields
    gs    the acceleration signal
    sr    sample rate
    rate  shaft rate, already in Hz (not rpm — a trap)
    load  applied load, lb
Directory names carry the condition, so the loader reads the path as well as
the file.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np

from .base import Record


def _field(struct, name):
    try:
        v = struct[name]
        while isinstance(v, np.ndarray) and v.dtype == object and v.size == 1:
            v = v[0]
        while isinstance(v, np.ndarray) and v.size == 1 and v.dtype != object:
            v = v.ravel()[0]
        return v
    except Exception:
        return None


def _condition(path: Path) -> Optional[str]:
    s = (str(path.parent).lower() + " " + path.stem.lower())
    if "baseline" in s:
        return "normal"
    if "outerrace" in s or "outer race" in s:
        return "outer_race"
    if "innerrace" in s or "inner race" in s:
        return "inner_race"
    return None


def load_file(path: str | Path) -> Record:
    from scipy.io import loadmat

    path = Path(path)
    fault = _condition(path)
    if fault is None:
        raise KeyError(f"{path.name}: cannot tell the condition from the path. "
                       "MFPT encodes it in the directory name; keep the "
                       "distribution's folder structure.")
    m = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    key = next((k for k in m if not k.startswith("__")), None)
    if key is None:
        raise KeyError(f"{path.name}: no data variables")
    b = m[key]

    def get(name):
        return getattr(b, name, None) if hasattr(b, name) else _field(b, name)

    gs = get("gs")
    if gs is None:
        raise KeyError(f"{path.name}: no 'gs' signal; found "
                       f"{[a for a in dir(b) if not a.startswith('_')]}")
    sig = np.asarray(gs, dtype=np.float64).ravel()
    sr = float(get("sr") or 97656.0)
    # `rate` is shaft speed in Hz here, not rpm. Treating it as rpm puts every
    # order out by a factor of sixty and is completely silent.
    rate_hz = float(get("rate") or 25.0)
    load = get("load")
    load = int(float(load)) if load is not None else 0

    return Record(file_no=abs(hash(path.stem)) % 100000, fault=fault,
                  defect_in=0.0, load_hp=load, rpm_nominal=int(round(rate_hz * 60)),
                  fs=sr, signal=sig, rpm_measured=rate_hz * 60.0,
                  channel="acc", path=str(path), dataset="mfpt",
                  bearing_key="MFPTNICE",
                  note=f"{path.parent.name}/{path.name}")


def load_dir(d: str | Path) -> List[Record]:
    out = []
    for p in sorted(Path(d).rglob("*.mat")):
        try:
            out.append(load_file(p))
        except Exception as e:
            warnings.warn(f"skipping {p.name}: {e}")
    return out
