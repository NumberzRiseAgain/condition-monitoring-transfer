"""Case Western Reserve seeded-fault bearing data.

Real recordings from a real machine, with the defect location, defect size and
motor load all documented. That combination is why this dataset — and not a
simulator — is what the accuracy claim in the proposal has to rest on. A Navy
reviewer can download the same files and check us.

Three traps live in this dataset, and all three are in the published literature
as silent errors rather than as failures:

  Sample rate. The fault records in the "12k Drive End" set are sampled at
  12 kHz. The normal baseline records are sampled at 48 kHz. Learning a baseline
  from one and testing against the other puts every order in the wrong place and
  the demodulation band on the wrong part of the spectrum. This loader records
  fs per file and `align_rate` decimates rather than letting them mix.

  Speed. Load and speed move together — 1797 rpm at no load down to about
  1730 rpm at 3 hp. Orders are relative to shaft speed, so using a nominal rpm
  for all four introduces a 4% error, which is twice the tolerance a fault line
  is searched with. Every file carries its own measured RPM and that is what is
  used.

  Bearing. Drive-end records are an SKF 6205 (nine balls); fan-end records are a
  6203 (eight). Using one geometry for both is a mistake that produces confident
  wrong answers rather than an error.

The file-number-to-condition table below is transcribed from the CWRU download
page. Transcription is exactly the kind of thing that goes wrong quietly, so the
loader cross-checks each file's own RPM against the table's expected value and
warns on disagreement rather than trusting either blindly.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ── the catalogue ───────────────────────────────────────────────────────────
# (file number, fault, defect inches, motor load hp, nominal rpm, fs)
# fault: normal | inner_race | rolling_element | outer_race
#
# VERIFY THIS TABLE against https://engineering.case.edu/bearingdatacenter
# before any number derived from it goes in a proposal. It is transcribed, and
# a transcription error here would be invisible downstream.
_ROWS: List[Tuple[int, str, float, int, int, int]] = [
    # Normal baseline — note 48 kHz, unlike the fault records below.
    (97,  "normal", 0.000, 0, 1797, 48000),
    (98,  "normal", 0.000, 1, 1772, 48000),
    (99,  "normal", 0.000, 2, 1750, 48000),
    (100, "normal", 0.000, 3, 1730, 48000),

    # 12 kHz drive end — inner race
    (105, "inner_race", 0.007, 0, 1797, 12000),
    (106, "inner_race", 0.007, 1, 1772, 12000),
    (107, "inner_race", 0.007, 2, 1750, 12000),
    (108, "inner_race", 0.007, 3, 1730, 12000),
    (169, "inner_race", 0.014, 0, 1797, 12000),
    (170, "inner_race", 0.014, 1, 1772, 12000),
    (171, "inner_race", 0.014, 2, 1750, 12000),
    (172, "inner_race", 0.014, 3, 1730, 12000),
    (209, "inner_race", 0.021, 0, 1797, 12000),
    (210, "inner_race", 0.021, 1, 1772, 12000),
    (211, "inner_race", 0.021, 2, 1750, 12000),
    (212, "inner_race", 0.021, 3, 1730, 12000),

    # 12 kHz drive end — rolling element
    (118, "rolling_element", 0.007, 0, 1797, 12000),
    (119, "rolling_element", 0.007, 1, 1772, 12000),
    (120, "rolling_element", 0.007, 2, 1750, 12000),
    (121, "rolling_element", 0.007, 3, 1730, 12000),
    (185, "rolling_element", 0.014, 0, 1797, 12000),
    (186, "rolling_element", 0.014, 1, 1772, 12000),
    (187, "rolling_element", 0.014, 2, 1750, 12000),
    (188, "rolling_element", 0.014, 3, 1730, 12000),
    (222, "rolling_element", 0.021, 0, 1797, 12000),
    (223, "rolling_element", 0.021, 1, 1772, 12000),
    (224, "rolling_element", 0.021, 2, 1750, 12000),
    (225, "rolling_element", 0.021, 3, 1730, 12000),

    # 12 kHz drive end — outer race, defect centred at 6 o'clock (the load zone)
    (130, "outer_race", 0.007, 0, 1797, 12000),
    (131, "outer_race", 0.007, 1, 1772, 12000),
    (132, "outer_race", 0.007, 2, 1750, 12000),
    (133, "outer_race", 0.007, 3, 1730, 12000),
    (197, "outer_race", 0.014, 0, 1797, 12000),
    (198, "outer_race", 0.014, 1, 1772, 12000),
    (199, "outer_race", 0.014, 2, 1750, 12000),
    (200, "outer_race", 0.014, 3, 1730, 12000),
    (234, "outer_race", 0.021, 0, 1797, 12000),
    (235, "outer_race", 0.021, 1, 1772, 12000),
    (236, "outer_race", 0.021, 2, 1750, 12000),
    (237, "outer_race", 0.021, 3, 1730, 12000),
]


from .base import Record  # noqa: E402  (after the file-map table above)


def catalogue() -> List[Dict]:
    return [dict(file_no=n, fault=f, defect_in=d, load_hp=l, rpm=r, fs=s)
            for n, f, d, l, r, s in _ROWS]


def _meta(file_no: int) -> Optional[Tuple]:
    for row in _ROWS:
        if row[0] == file_no:
            return row
    return None


def load_file(path: str | Path, channel: str = "DE") -> Record:
    """Read one CWRU .mat.

    Variables are named by file number, e.g. X097_DE_time, X097_FE_time,
    X097RPM. The numbering inside a file does not always match the file's own
    name — some files carry a differently-numbered variable — so the variable is
    found by suffix rather than by constructing the expected name.
    """
    from scipy.io import loadmat

    path = Path(path)
    file_no = int("".join(c for c in path.stem if c.isdigit()) or 0)
    m = loadmat(str(path))

    sig_keys = [k for k in m if k.endswith(f"_{channel}_time")]
    if not sig_keys:
        avail = sorted(k for k in m if not k.startswith("__"))
        raise KeyError(f"{path.name}: no {channel} channel. Present: {avail}")
    sig = np.asarray(m[sig_keys[0]], dtype=np.float64).ravel()

    rpm = None
    rpm_keys = [k for k in m if k.upper().endswith("RPM")]
    if rpm_keys:
        try:
            rpm = float(np.asarray(m[rpm_keys[0]]).ravel()[0])
        except Exception:
            rpm = None

    meta = _meta(file_no)
    if meta is None:
        raise KeyError(
            f"{path.name}: file {file_no} is not in the catalogue. Add it to "
            "_ROWS with its documented condition, or the label would be a guess."
        )
    _, fault, defect, load, rpm_nom, fs = meta

    # Cross-check the transcription against the file itself.
    if rpm is not None and abs(rpm - rpm_nom) > 60:
        warnings.warn(
            f"{path.name}: file reports {rpm:.0f} rpm, catalogue says {rpm_nom}. "
            "The catalogue row is probably wrong — verify against the CWRU page."
        )

    return Record(file_no=file_no, fault=fault, defect_in=defect, load_hp=load,
                  rpm_nominal=rpm_nom, fs=float(fs), signal=sig,
                  rpm_measured=rpm, channel=channel, path=str(path),
                  dataset="cwru", bearing_key="SKF6205" if channel == "DE" else "SKF6203")


def load_dir(d: str | Path, channel: str = "DE") -> List[Record]:
    out = []
    for p in sorted(Path(d).glob("*.mat")):
        try:
            out.append(load_file(p, channel))
        except (KeyError, Exception) as e:      # a bad file must not kill a run
            warnings.warn(f"skipping {p.name}: {e}")
    return out


def align_rate(rec: Record, target_fs: float) -> Record:
    """Decimate to a common rate.

    The normal baseline is 48 kHz and the fault records are 12 kHz. Mixing them
    silently puts every order in the wrong bin. Integer decimation with an
    anti-alias filter is used rather than naive subsampling, which would fold
    the very high-frequency resonance energy the envelope analysis depends on
    straight back down into the band of interest.
    """
    if abs(rec.fs - target_fs) < 1e-6:
        return rec
    ratio = rec.fs / target_fs
    q = int(round(ratio))
    if abs(ratio - q) > 1e-6 or q < 1:
        raise ValueError(f"{rec.fs} -> {target_fs} is not an integer decimation")
    try:
        from scipy.signal import decimate
        sig = decimate(rec.signal, q, ftype="fir", zero_phase=True)
    except Exception:                            # scipy absent: honest fallback
        from numpy import convolve
        k = np.hanning(8 * q + 1)
        k /= k.sum()
        sig = convolve(rec.signal, k, mode="same")[::q]
    # dataclasses.replace, NOT a positional rebuild.
    #
    # This used to construct the new Record positionally and stop at `path`,
    # which silently reset `dataset`, `bearing_key`, `rpm_series` and `note` to
    # their defaults. The default bearing_key is "SKF6205", so every Paderborn
    # record — a 6203 with eight rolling elements — came out of decimation
    # carrying the CWRU nine-ball geometry, and outer race was searched at
    # 3.5848 x shaft instead of 3.0543.
    #
    # The failure was silent, and worse than silent, it looked good: every fault
    # line landed on empty spectrum, so the run reported 0 detections and 0
    # false alarms over 122 records. A monitor that never speaks passes a
    # false-alarm test perfectly. That is the argument for always reading the
    # detection column beside the false-alarm column, and for making this
    # rebuild field-preserving by construction rather than by remembering.
    #
    # `rpm_series` matters just as much: it is what `variable_speed` and the
    # order tracking read, so on a variable-speed dataset this step was deleting
    # the only evidence that the transient case works at all.
    from dataclasses import replace
    return replace(rec, fs=target_fs, signal=sig)


def windows(rec: Record, window_s: float, hop_s: Optional[float] = None):
    """Successive windows, with their start time."""
    n = int(window_s * rec.fs)
    hop = int((hop_s or window_s) * rec.fs)
    for i in range(0, len(rec.signal) - n + 1, hop):
        yield i / rec.fs, rec.signal[i:i + n]
