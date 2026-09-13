"""XJTU-SY accelerated run-to-failure bearings — the only set here that ends in a failure.

Every other dataset in this repository is a snapshot: a bearing in a condition,
recorded for a few seconds. This one is a *history*. Fifteen bearings were run
until they failed, with 32,768 samples captured at 25.6 kHz once per minute for
the whole of each life. That is what makes it the only public data we have that
can say anything about the topic's second verb: "detect faults **and predict
possible failures**."

Wang, Wang, Yan, Hou et al., *A Hybrid Prognostics Approach for Estimating
Remaining Useful Life of Rolling Element Bearings*, IEEE Trans. Reliability;
data at `biaowang.tech/xjtu-sy-bearing-datasets`. Three operating conditions,
five bearings each:

    condition 1   2100 rpm   12 kN
    condition 2   2250 rpm   11 kN
    condition 3   2400 rpm   10 kN

Two accelerometers, horizontal and vertical, on the housing.

**This loader discovers the layout rather than assuming it.** Published mirrors
differ in how they nest the folders and in whether the CSVs carry a header, and
a loader that hard-codes one mirror's shape fails on another with a message
about a missing directory rather than about the real problem. `scan` walks for
CSV files whose name is an integer — the capture index, which is also the minute
— and takes the bearing identity from the folder above and the condition from
the folder above that. `describe` prints what it found, so the shape can be
checked before anything is scored.

**What is NOT in here, deliberately.**

*Bearing geometry.* The rig uses an LDK UER204, and this module ships no
dimensions for it. Guessing them would repeat the KAIST situation with none of
the excuse: the numbers are obtainable, they are simply not in this repository
yet. `tools/eval_prognosis_xjtu.py` runs its geometry-free half without them and
refuses the declared-order half until a catalogue entry exists.

*Failure-mode labels.* The dataset documents which part failed on each bearing.
Those labels belong in `FAULT_LABELS` below, transcribed from the dataset's own
documentation — not from memory, and not inferred from the signals, which would
be fitting the label to the data. Until they are filled in, the evaluation
reports lead time and monotonicity, and declines to report part-naming accuracy.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# Stated by the dataset, not measured here.
FS = 25_600.0                 # Hz
SAMPLES_PER_CAPTURE = 32_768  # 1.28 s
CAPTURE_INTERVAL_S = 60.0     # one capture per minute

CHANNELS = ("horizontal", "vertical")

# Transcribe from the dataset documentation. Key is the bearing folder name as
# it appears on disk, e.g. "Bearing1_1". Value is one of the part names the rest
# of cbmx uses: outer_race, inner_race, rolling_element, cage — or a tuple where
# the documentation reports more than one.
#
# LEAVE THIS EMPTY UNTIL IT IS READ OFF THE DOCUMENTATION. An empty table makes
# the evaluation say "no labels, no accuracy claim", which is correct. A guessed
# table makes it print a number that means nothing.
FAULT_LABELS: Dict[str, Tuple[str, ...]] = {}

_INT_NAME = re.compile(r"^(\d+)$")


@dataclass
class Capture:
    """One 1.28 s recording, and where it sits in the bearing's life."""

    index: int                 # 1-based capture number, also the minute
    x: np.ndarray              # horizontal
    y: np.ndarray              # vertical
    path: str = ""

    @property
    def minutes(self) -> float:
        return (self.index - 1) * CAPTURE_INTERVAL_S / 60.0

    def channel(self, name: str) -> np.ndarray:
        if name.startswith("h"):
            return self.x
        if name.startswith("v"):
            return self.y
        raise KeyError(f"channel {name!r}; have {CHANNELS}")


@dataclass
class BearingLife:
    """Every capture for one bearing, in order, ending at failure."""

    condition: str
    bearing: str
    files: List[Path] = field(default_factory=list)

    @property
    def n_captures(self) -> int:
        return len(self.files)

    @property
    def life_minutes(self) -> float:
        return self.n_captures * CAPTURE_INTERVAL_S / 60.0

    @property
    def label(self) -> Tuple[str, ...]:
        return FAULT_LABELS.get(self.bearing, ())

    def fraction_of_life(self, index: int) -> float:
        """Where a capture sits in this bearing's life, 0 at commissioning and
        1 at failure. Only computable *after* the bearing has failed, which is
        why remaining-useful-life numbers from run-to-failure data are always
        retrospective and must be labelled as such."""
        return index / max(self.n_captures, 1)

    def captures(self, limit: Optional[int] = None,
                 every: int = 1) -> Iterator[Capture]:
        for i, p in enumerate(self.files, start=1):
            if (i - 1) % every:
                continue
            if limit is not None and i > limit:
                return
            x, y = read_capture(p)
            yield Capture(i, x, y, str(p))


def read_capture(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Two channels out of one capture CSV.

    Mirrors differ on whether the header row is present, so it is tested rather
    than assumed — the same trap the KAIST `_constant` files sprang, where a
    missing header silently turns the first data sample into a column name.
    """
    import pandas as pd

    with open(path, "r") as fh:
        first = fh.readline().strip()
    try:
        [float(v) for v in first.split(",")]
        headerless = True
    except ValueError:
        headerless = False

    d = pd.read_csv(path, header=None if headerless else 0)
    a = d.to_numpy(dtype=np.float64)
    if a.shape[1] < 2:
        raise ValueError(f"{path}: expected two channels, got {a.shape[1]}")
    return a[:, 0], a[:, 1]


def scan(root: str | Path) -> Dict[str, Dict[str, BearingLife]]:
    """Discover conditions and bearings under `root`, however it is nested.

    A capture file is any `<integer>.csv`. Its parent folder names the bearing,
    and that folder's parent names the operating condition. Captures are ordered
    by their integer index, not lexically — `10.csv` sorts before `9.csv` as a
    string, and a life curve read in that order is not a life curve.
    """
    out: Dict[str, Dict[str, BearingLife]] = {}
    for p in Path(root).rglob("*.csv"):
        m = _INT_NAME.match(p.stem)
        if not m:
            continue
        bearing = p.parent.name
        condition = p.parent.parent.name
        life = out.setdefault(condition, {}).setdefault(
            bearing, BearingLife(condition, bearing))
        life.files.append(p)
    for cond in out.values():
        for life in cond.values():
            life.files.sort(key=lambda q: int(q.stem))
    return out


def describe(root: str | Path) -> str:
    """What is actually on disk, before anything is scored."""
    found = scan(root)
    if not found:
        return (f"No capture files under {root}.\n"
                f"Expected <condition>/<bearing>/<n>.csv, for example\n"
                f"  35Hz12kN/Bearing1_1/1.csv\n"
                f"Extract preserving the folder structure; a flattened copy\n"
                f"loses both the bearing identity and the capture order.")
    lines = [f"{root}"]
    total = 0
    for cond in sorted(found):
        lines.append(f"  {cond}")
        for name in sorted(found[cond], key=_natural):
            life = found[cond][name]
            total += life.n_captures
            lab = ", ".join(life.label) if life.label else "no label in FAULT_LABELS"
            lines.append(f"    {name:<16} {life.n_captures:>5} captures  "
                         f"{life.life_minutes:>7.0f} min to failure   {lab}")
    lines.append(f"  {total} captures across "
                 f"{sum(len(v) for v in found.values())} bearings")
    if not FAULT_LABELS:
        lines.append("")
        lines.append("  FAULT_LABELS is empty, so no part-naming accuracy will be")
        lines.append("  reported. Transcribe the failure modes from the dataset's")
        lines.append("  own documentation into src/cbmx/io/xjtu.py.")
    return "\n".join(lines)


def _natural(s: str) -> Tuple:
    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s))
