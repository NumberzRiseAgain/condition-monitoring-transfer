"""NASA/IMS bearings — four bearings on one shaft, run until they failed.

Center for Intelligent Maintenance Systems, University of Cincinnati, published
through the **NASA Prognostics Center of Excellence** data repository. Three
test-to-failure experiments: four Rexnord ZA-2115 double-row bearings on one
shaft, 2000 rpm constant, 6000 lb radial load, a one-second snapshot at 20 kHz
captured roughly every ten minutes until a bearing failed. Tests run for days,
so a life is hundreds or thousands of snapshots long.

**Why this set rather than the obvious alternative.** The other public
run-to-failure bearing data is XJTU-SY, hosted on a personal website in the PRC
with Baidu Netdisk among its mirrors. For a proposal whose own §7.6 states that
execution is 100% U.S. and whose Volume 7 discloses foreign affiliations, the
provenance question is not worth one topic row. This set is U.S.-origin,
NASA-hosted, and citing the NASA Prognostics Center of Excellence in a Navy
volume reads better than citing a personal file share.

**What makes it the right technical choice too.** The bearings fail naturally
over days rather than being crushed to death in an hour, so a lead time measured
here is a lead time a maintainer could actually schedule against. The failure
modes are documented per bearing. And the bearing is a catalogue part with
published dimensions, which is the first run-to-failure set here whose geometry
would not carry an UNVERIFIED flag.

Layout, as the archive actually ships — it nests three deep, and the third
test is in a folder named for a fourth:

    4.+Bearings.zip
      └ 4. Bearings/IMS.7z
          ├ Readme Document for IMS Bearing Data.pdf
          ├ 1st_test.rar  →  1st_test/2003.10.22.12.06.24    8 cols: b1-b4, x and y
          ├ 2nd_test.rar  →  2nd_test/2004.02.12.10.32.39    4 cols: one per bearing
          └ 3rd_test.rar  →  4th_test/txt/2004.03.04.09.27.46   4 cols, misnamed folder

`3rd_test.rar` really does unpack to `4th_test/txt/`. That is the archive's own
naming, and it is what anyone else extracting this file will see, so nothing is
renamed on disk — `scan` maps it back to the readme's "Set No. 3" through
`_TEST_ALIASES` and `_CONTAINER_DIRS` below, where the mapping is visible in
code rather than hidden in an extraction step nobody kept.

Tab-delimited ASCII, no header, 20,480 rows. The filename is the capture
timestamp and is the only ordering information there is — sorted as text,
`2003.11.01` precedes `2003.10.22`, and a life curve read in that order is not a
life curve. `scan` parses the timestamp and sorts by it.

**What the readme does and does not give us.**

*It gives the failure modes*, per bearing, per test, and `FAULT_LABELS` below is
transcribed from it verbatim. Every bearing not in that table survived its test
and is a control, which is the whole reason a lead time here means anything.

*It does not give the bearing geometry.* The Rexnord ZA-2115's pitch diameter,
ball diameter, ball count and contact angle appear nowhere in the readme — it
describes the rig, the channels and the failures, and stops. Under the rule
declared in `PROGNOSIS_STOPPING_RULE.md` before this data was downloaded, the
declared-physics half therefore **does not run**, and this module ships no
geometry. What is reported is lead time and monotonicity from geometry-free
indicators; part-naming accuracy is not reported, because it cannot be computed
without inventing the numbers the rule exists to stop us inventing.

The defect orders can be had from the dataset authors' own paper (Qiu, Lee and
Lin, *Journal of Sound and Vibration* 289 (2006) 1066-1090, cited by the
readme). Using them would be a legitimate, disclosable change — but it is a
change made *after* seeing the data, and the volume reports it as one if it is
ever made. It has not been made.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# Stated by the dataset, not measured here.
FS = 20_000.0                 # Hz
SAMPLES_PER_CAPTURE = 20_480  # 1.024 s
SHAFT_RPM = 2000.0            # constant
RADIAL_LOAD_LB = 6000.0

# Columns per bearing, by test. Test 1 records two accelerometers per bearing;
# tests 2 and 3 record one. Getting this wrong silently mixes two bearings into
# one channel, which looks like a noisier bearing rather than like an error.
CHANNELS_PER_BEARING = {"1st_test": 2, "2nd_test": 1, "3rd_test": 1, "4th_test": 1}

# Transcribed verbatim from "Readme Document for IMS Bearing Data.pdf", which
# ships inside IMS.7z. Its exact words, per set:
#
#   Set 1: "inner race defect occurred in bearing 3 and roller element defect
#           in bearing 4"
#   Set 2: "outer race failure occurred in bearing 1"
#   Set 3: "outer race failure occurred in bearing 3"
#
# Key is "<test>/<bearing>". Every bearing NOT listed here ran to the end of its
# test without a reported defect and is scored as a control. Three of the four
# bearings on each shaft are controls, and they are what make the fourth mean
# something — see PROGNOSIS_STOPPING_RULE.md, "Bearings will not be dropped".
FAULT_LABELS: Dict[str, str] = {
    "1st_test/3": "inner_race",
    "1st_test/4": "rolling_element",
    "2nd_test/1": "outer_race",
    "3rd_test/3": "outer_race",
}

# Also from the readme, and used only to check the census against what the
# publisher says should be there. A silent short extraction is otherwise
# indistinguishable from a short test.
PUBLISHED = {
    "1st_test": dict(files=2156, channels=8, start="2003.10.22 12:06:24",
                     end="2003.11.25 23:39:56"),
    "2nd_test": dict(files=984, channels=4, start="2004.02.12 10:32:39",
                     end="2004.02.19 06:22:39"),
    "3rd_test": dict(files=4448, channels=4, start="2004.03.04 09:27:46",
                     end="2004.04.04 19:01:57"),
}


def published_end(test: str) -> Optional[datetime]:
    """Last capture of the experiment the readme actually describes."""
    p = PUBLISHED.get(test)
    if not p:
        return None
    return datetime.strptime(p["end"], "%Y.%m.%d %H:%M:%S")


def clip_to_published(life: "BearingLife") -> "BearingLife":
    """`life`, truncated to the recording window the readme states.

    **Why this exists, and why it is the default.** Set 3 ships 6,324 captures
    running to 18 April 2004. The readme describes Set No. 3 as 4,448 captures
    ending 4 April, and it is the readme — nothing else — that tells us an outer
    race failure occurred in bearing 3. Scoring the extra fortnight would measure
    a lead time against a denominator no document supports, using a label the
    same document scopes to a shorter experiment.

    It is also the choice that hurts the result. A first call at a fixed wall
    clock hour leaves a *smaller* fraction of a 754-hour life than of a 1,073-hour
    one, so clipping lowers every percentage this evaluation reports. That is the
    right direction for a default to be wrong in, and it is chosen on the
    documentation rather than on which number came out better — the full folder
    is scored too, and both are reported.
    """
    end = published_end(life.test)
    if end is None:
        return life
    keep = [(p, t) for p, t in zip(life.files, life.stamps) if t <= end]
    out = BearingLife(life.test, life.bearing)
    out.files = [p for p, _ in keep]
    out.stamps = [t for _, t in keep]
    return out


# The archive's folder names, mapped to the readme's set names. `3rd_test.rar`
# unpacks to `4th_test/txt/`; nothing is renamed on disk, so this is where the
# two vocabularies are reconciled.
_TEST_ALIASES = {"4th_test": "3rd_test"}

# Directory names that are packaging, not a test. A capture folder named `txt`
# takes its identity from its parent.
_CONTAINER_DIRS = {"txt", "data", "ascii"}

_TS = re.compile(r"^(\d{4})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})\.(\d{2})$")


def _test_name(d: Path) -> str:
    """The readme's name for the test whose captures sit in directory `d`."""
    name = d.name
    if name.lower() in _CONTAINER_DIRS:
        name = d.parent.name
    return _TEST_ALIASES.get(name, name)


def _stamp(p: Path) -> Optional[datetime]:
    m = _TS.match(p.name)
    if not m:
        return None
    y, mo, d, h, mi, s = (int(v) for v in m.groups())
    try:
        return datetime(y, mo, d, h, mi, s)
    except ValueError:
        return None


@dataclass
class Capture:
    """One 1.024 s snapshot, and where it sits in the test."""

    index: int                        # 1-based, in time order
    when: datetime
    x: np.ndarray                     # (samples, channels) for this bearing
    path: str = ""

    @property
    def primary(self) -> np.ndarray:
        return self.x[:, 0]


@dataclass
class BearingLife:
    """Every snapshot for one bearing of one test, in time order."""

    test: str
    bearing: int                      # 1-4
    files: List[Path] = field(default_factory=list)
    stamps: List[datetime] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.test}/{self.bearing}"

    @property
    def n_captures(self) -> int:
        return len(self.files)

    @property
    def label(self) -> str:
        return FAULT_LABELS.get(self.key, "")

    @property
    def life_hours(self) -> float:
        """Wall-clock life, from the timestamps rather than from a nominal
        capture interval. The published interval is 'mostly every ten minutes',
        and 'mostly' is doing work: test 1 changes interval partway through. A
        lead time computed from an assumed interval would be wrong by hours."""
        if len(self.stamps) < 2:
            return 0.0
        return (self.stamps[-1] - self.stamps[0]).total_seconds() / 3600.0

    def hours_at(self, i: int) -> float:
        if not self.stamps:
            return 0.0
        i = min(max(i, 0), len(self.stamps) - 1)
        return (self.stamps[i] - self.stamps[0]).total_seconds() / 3600.0

    def captures(self, every: int = 1, limit: Optional[int] = None
                 ) -> Iterator[Capture]:
        n_ch = CHANNELS_PER_BEARING.get(self.test, 1)
        lo = (self.bearing - 1) * n_ch
        for i, (p, ts) in enumerate(zip(self.files, self.stamps), start=1):
            if (i - 1) % every:
                continue
            if limit is not None and i > limit:
                return
            a = read_capture(p)
            if a.shape[1] < lo + n_ch:
                continue
            yield Capture(i, ts, a[:, lo:lo + n_ch], str(p))


def columns_for(test: str, bearing: int) -> slice:
    """Which columns of a capture belong to `bearing`, per the readme's
    "Channel Arrangement" line for that set."""
    n = CHANNELS_PER_BEARING.get(test, 1)
    lo = (bearing - 1) * n
    return slice(lo, lo + n)


def iter_test(life: "BearingLife", every: int = 1
              ) -> Iterator[Tuple[int, datetime, np.ndarray]]:
    """Every capture of a test, read **once**, whole.

    `BearingLife.captures` reads a file per bearing, which on this dataset means
    opening all 9,464 files four times each — roughly 30 GB of parsing to get
    scalars out of. The four bearings share one file because they shared one
    shaft, so anything scoring all four should read it once and slice. Yields
    (1-based index in time order, timestamp, full array).
    """
    for i, (p, ts) in enumerate(zip(life.files, life.stamps), start=1):
        if (i - 1) % every:
            continue
        yield i, ts, read_capture(p)


def read_capture(path: str | Path) -> np.ndarray:
    """One snapshot as (samples, channels).

    Tab-delimited with no header in every release seen, but the delimiter is
    sniffed rather than assumed — a whitespace-delimited copy read as
    tab-delimited yields one column of strings and fails far from here.
    """
    import pandas as pd

    with open(path, "r") as fh:
        first = fh.readline()
    sep = "\t" if "\t" in first else r"\s+"
    d = pd.read_csv(path, sep=sep, header=None, engine="python" if sep != "\t" else "c")
    return d.to_numpy(dtype=np.float64)


def scan(root: str | Path) -> Dict[str, Dict[int, BearingLife]]:
    """Discover tests and bearings under `root`, in true time order."""
    out: Dict[str, Dict[int, BearingLife]] = {}
    for d in sorted(Path(root).rglob("*")):
        if not d.is_dir():
            continue
        files = [(p, _stamp(p)) for p in d.iterdir() if p.is_file()]
        files = [(p, t) for p, t in files if t is not None]
        if len(files) < 10:
            continue
        files.sort(key=lambda pt: pt[1])
        test = _test_name(d)
        n_ch = CHANNELS_PER_BEARING.get(test)
        if n_ch is None:
            # Unknown test folder: infer channels per bearing from column count.
            try:
                cols = read_capture(files[0][0]).shape[1]
            except Exception:                                    # noqa: BLE001
                continue
            n_ch = 2 if cols >= 8 else 1
            CHANNELS_PER_BEARING[test] = n_ch
        for b in range(1, 5):
            life = BearingLife(test, b)
            life.files = [p for p, _ in files]
            life.stamps = [t for _, t in files]
            out.setdefault(test, {})[b] = life
    return out


_ARCHIVE_SUFFIXES = {".zip", ".rar", ".7z", ".gz", ".tar", ".tgz", ".bz2"}


def _nested_archives(root: str | Path) -> List[Path]:
    """Archives still sitting unextracted under `root`.

    The NASA release ships as a zip whose contents are themselves archives, so
    the first extraction leaves a directory of `.rar`/`.7z` files and no capture
    anywhere. That looks identical to 'you downloaded nothing', and telling the
    two apart by hand costs a round trip."""
    try:
        return sorted(p for p in Path(root).rglob("*")
                      if p.is_file() and p.suffix.lower() in _ARCHIVE_SUFFIXES)[:12]
    except OSError:
        return []


def short_captures(root: str | Path, tol: float = 0.5) -> List[Tuple[Path, int, int]]:
    """Capture files whose size is far below the median for their folder.

    Every capture in a set holds 20,480 rows and they come out within a few
    hundred bytes of each other, so a file at half the median size is a file
    whose write was interrupted. This matters because a truncated capture does
    not raise: it loads as a short array, its RMS and kurtosis are computed over
    whatever survived, and it lands in the life curve as a point that moved for
    no physical reason. An extraction killed partway through is the ordinary way
    to get one, and the only cheap way to see it is to look at the sizes.

    Returns (path, its size, the folder's median) for each offender.
    """
    bad: List[Tuple[Path, int, int]] = []
    for d in sorted(Path(root).rglob("*")):
        if not d.is_dir():
            continue
        sizes = [(p, p.stat().st_size) for p in d.iterdir()
                 if p.is_file() and _TS.match(p.name)]
        if len(sizes) < 10:
            continue
        med = sorted(s for _, s in sizes)[len(sizes) // 2]
        bad += [(p, s, med) for p, s in sizes if s < med * tol]
    return bad


def describe(root: str | Path) -> str:
    found = scan(root)
    if not found:
        nested = _nested_archives(root)
        if nested:
            names = "\n".join(f"    {p.relative_to(Path(root))}" for p in nested)
            return (f"No capture files under {root}, but there are archives left\n"
                    f"unextracted inside it:\n{names}\n\n"
                    f"The NASA release nests archives THREE deep: the zip holds\n"
                    f"IMS.7z, which holds 1st_test.rar, 2nd_test.rar, 3rd_test.rar\n"
                    f"and the readme PDF. Extract all of them, then run this probe\n"
                    f"again. On macOS `tar -xf <file>` reads both .7z and .rar;\n"
                    f"otherwise The Unarchiver, or `brew install p7zip unar`.")
        return (f"No capture files under {root}.\n"
                f"Expected <test>/<timestamp> files, for example\n"
                f"  1st_test/2003.10.22.12.06.24\n"
                f"Extract the archive preserving its folder structure.\n"
                f"A folder is only accepted as a test if it holds at least 10\n"
                f"timestamp-named files, so a partial extraction is skipped\n"
                f"silently rather than scored.")
    lines = [f"{root}", ""]
    lines.append("  Census, against the counts the readme publishes. A short")
    lines.append("  extraction and a short test look identical without this.")
    lines.append("")
    for test in sorted(found):
        any_life = found[test][1]
        pub = PUBLISHED.get(test)
        n = any_life.n_captures
        if pub is None:
            check = "not a set the readme names"
        elif n == pub["files"]:
            check = f"matches the published {pub['files']}"
        else:
            check = f"PUBLISHED {pub['files']} — differs by {n - pub['files']:+d}"
        lines.append(f"  {test}: {n} captures, {any_life.life_hours:.1f} h, "
                     f"{CHANNELS_PER_BEARING.get(test, 1)} channel(s) per bearing")
        lines.append(f"      {check}")
        if any_life.stamps:
            lines.append(f"      {any_life.stamps[0]:%Y.%m.%d %H:%M:%S} to "
                         f"{any_life.stamps[-1]:%Y.%m.%d %H:%M:%S}"
                         + (f"   readme: {pub['start']} to {pub['end']}" if pub else ""))
        for b in sorted(found[test]):
            lab = found[test][b].label
            lines.append(f"      bearing {b}: "
                         + (f"{lab} (the failure)" if lab
                            else "survived — control"))
        lines.append("")
    short = short_captures(root)
    if short:
        lines.append(f"  ** {len(short)} TRUNCATED capture(s) — an extraction that")
        lines.append("     stopped partway. These load without error and put a point")
        lines.append("     in the life curve that moved for no physical reason.")
        for p, s, med in short[:6]:
            lines.append(f"       {p.name}  {s:,} bytes, folder median {med:,}")
        lines.append("     Re-extract that test's archive over the top, then re-probe.")
        lines.append("")
    n_fault = sum(1 for t in found for b in found[t] if found[t][b].label)
    n_ctrl = sum(1 for t in found for b in found[t] if not found[t][b].label)
    lines.append(f"  {n_fault} bearings with a reported failure, {n_ctrl} controls.")
    lines.append("")
    lines.append("  No ZA-2115 geometry: the readme does not publish the bearing's")
    lines.append("  dimensions, so per PROGNOSIS_STOPPING_RULE.md the declared-physics")
    lines.append("  half does not run and part-naming accuracy is not reported.")
    lines.append("  Lead time and monotonicity are.")
    return "\n".join(lines)
