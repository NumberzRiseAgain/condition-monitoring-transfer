"""The asset as a graph — what can fail, what replaces it, what watches it.

The topic asks for isolation to the Lowest Replaceable Unit. An LRU is not a
physics concept; it is a supply concept. It is the smallest thing a sailor can
pull off a shelf and fit, and which one that is depends on the Navy's own
provisioning, not on where the crack happens to be. A bearing race is where the
defect lives; the *bearing* is usually what gets replaced; on some assemblies
the whole pump cartridge is the replaceable item and the bearing inside it is
not stocked at all.

So this module keeps two things apart that are easy to conflate:

    the SYMPTOM      "energy at 3.585 x shaft, no sidebands"  — physics
    the LRU          "P/N 6205-2RS, drive-end bearing"        — supply

and stores the mapping between them explicitly, with the part number the Navy
uses. When Matthew gives us the real breakdown, only the asset card changes.
Nothing in the physics or the evidence layer knows a part number exists.

That separation is also the honest answer to a question a reviewer will ask:
what happens when the LRU is coarser than the defect? The answer is that the
report names the LRU that will actually be pulled, and carries the finer
diagnosis underneath it as supporting detail. Claiming to isolate a fault to a
part the Navy cannot order is worse than useless — it reads as a system built
by people who have never held a maintenance manual.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..physics.bearing import Bearing


@dataclass
class LRU:
    """Something a maintainer can actually remove and replace."""

    id: str
    name: str
    part_number: str = ""
    parent: Optional[str] = None
    notes: str = ""

    def path(self, tree: Dict[str, "LRU"]) -> List[str]:
        """Walk up to the asset root — this is what a report shows so the
        reader knows where in the machine to go."""
        out, cur, seen = [], self, set()
        while cur is not None and cur.id not in seen:
            seen.add(cur.id)
            out.append(cur.name)
            cur = tree.get(cur.parent) if cur.parent else None
        return list(reversed(out))


@dataclass
class Symptom:
    """A physically-predicted observable, and the LRU it implicates."""

    id: str
    part: str                 # outer_race | inner_race | rolling_element | cage
    lru: str                  # LRU id
    order: float              # multiple of shaft speed, from geometry
    expect_sidebands: bool
    sideband_spacing: float   # in orders
    description: str
    evidence_for: str = ""    # what a maintainer should look for on removal


@dataclass
class SensorPoint:
    id: str
    kind: str                 # accelerometer | current | pressure | temperature
    location: str
    fs: float
    watches: List[str] = field(default_factory=list)   # LRU ids
    band_lo_hz: Optional[float] = None                 # frozen at commissioning
    band_hi_hz: Optional[float] = None

    @property
    def commissioned(self) -> bool:
        return self.band_lo_hz is not None and self.band_hi_hz is not None


@dataclass
class Asset:
    id: str
    name: str
    lrus: Dict[str, LRU]
    sensors: Dict[str, SensorPoint]
    bearings: Dict[str, Bearing]          # LRU id -> geometry
    symptoms: List[Symptom] = field(default_factory=list)
    shaft_ratio: Dict[str, float] = field(default_factory=dict)  # LRU id -> speed vs input
    raw: Dict = field(default_factory=dict)

    # -- construction --------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path) -> "Asset":
        d = yaml.safe_load(Path(path).read_text())
        lrus = {
            x["id"]: LRU(id=x["id"], name=x["name"], part_number=x.get("part_number", ""),
                         parent=x.get("parent"), notes=x.get("notes", ""))
            for x in d.get("lrus", [])
        }
        sensors = {
            x["id"]: SensorPoint(
                id=x["id"], kind=x["kind"], location=x.get("location", ""),
                fs=float(x["fs"]), watches=list(x.get("watches", [])),
                band_lo_hz=x.get("band_lo_hz"), band_hi_hz=x.get("band_hi_hz"),
            )
            for x in d.get("sensors", [])
        }
        bearings, ratios = {}, {}
        for x in d.get("bearings", []):
            bearings[x["lru"]] = Bearing(
                designation=x.get("designation", x["lru"]),
                n_rolling=int(x["n_rolling"]),
                ball_diameter=float(x["ball_diameter"]),
                pitch_diameter=float(x["pitch_diameter"]),
                contact_angle_deg=float(x.get("contact_angle_deg", 0.0)),
                source=x.get("source", ""),
            )
            ratios[x["lru"]] = float(x.get("shaft_ratio", 1.0))

        a = cls(id=d["asset"], name=d.get("name", d["asset"]), lrus=lrus,
                sensors=sensors, bearings=bearings, shaft_ratio=ratios, raw=d)
        a.build_symptoms()
        return a

    def build_symptoms(self) -> None:
        """Derive every observable from geometry. No hand-written frequencies —
        if a number in a report is wrong, it is because a dimension in the asset
        card is wrong, and that is a fixable, checkable kind of wrong."""
        HUMAN = {
            "outer_race": ("outer race", "spalling or brinelling on the stationary race, "
                                         "usually in the load zone"),
            "inner_race": ("inner race", "spalling on the rotating race; check for "
                                         "fretting at the shaft fit"),
            "rolling_element": (
                "a rolling element",
                "a flat or spall on one ball; inspect all elements and the "
                "cage pockets"),
            "cage": ("the cage", "cracked or worn cage; often follows another defect "
                                 "rather than starting one"),
        }
        self.symptoms = []
        for lru_id, b in self.bearings.items():
            ratio = self.shaft_ratio.get(lru_id, 1.0)
            for part, order in b.orders().items():
                human, look_for = HUMAN[part]
                self.symptoms.append(Symptom(
                    id=f"{lru_id}::{part}",
                    part=part,
                    lru=lru_id,
                    order=order * ratio,
                    # Only a defect that moves through the load zone modulates.
                    expect_sidebands=part in ("inner_race", "rolling_element"),
                    sideband_spacing=(
                        (1.0 * ratio) if part == "inner_race"
                        else (b.ftf * ratio)),
                    description=f"defect on {human} of {b.designation}",
                    evidence_for=look_for,
                ))

    # -- queries -------------------------------------------------------------
    def symptoms_for_sensor(self, sensor_id: str) -> List[Symptom]:
        s = self.sensors[sensor_id]
        return [sy for sy in self.symptoms if sy.lru in s.watches]

    def lru_path(self, lru_id: str) -> List[str]:
        return self.lrus[lru_id].path(self.lrus)

    def part_number(self, lru_id: str) -> str:
        return self.lrus[lru_id].part_number

    def describe(self) -> str:
        L = [f"{self.name}  [{self.id}]"]
        for s in self.sensors.values():
            band = (f"{s.band_lo_hz:.0f}-{s.band_hi_hz:.0f} Hz"
                    if s.commissioned else "NOT COMMISSIONED")
            L.append(f"  sensor {s.id}  {s.kind} @ {s.location}  "
                     f"{s.fs:.0f} Hz  band {band}")
            for lru in s.watches:
                L.append(f"      watches {self.lrus[lru].name}"
                         f"  [{self.lrus[lru].part_number or 'no P/N'}]")
        L.append(f"  {len(self.symptoms)} symptoms derived from geometry:")
        for sy in self.symptoms:
            sb = f"  sidebands ±{sy.sideband_spacing:.3f}" if sy.expect_sidebands else ""
            L.append(f"      {sy.order:7.3f}x  {sy.description}{sb}")
        return "\n".join(L)
