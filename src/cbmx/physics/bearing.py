"""Bearing kinematics — where to look, derived from a datasheet.

This file is the reason the bid is feasible. Everything in it comes from four
dimensions a caliper can measure, and none of it comes from data. Before we have
ever seen the machine, we know which frequency belongs to which replaceable
part, and that is the fault isolation the topic asks for.

For a bearing with n rolling elements of diameter d on a pitch diameter D at
contact angle phi, write r = (d/D)cos(phi). Then, per shaft revolution:

    FTF  = (1 - r) / 2                  the cage, and the ball orbit rate
    BPFO = n(1 - r) / 2                 a defect on the outer race
    BPFI = n(1 + r) / 2                 a defect on the inner race
    BSF  = (1 - r^2) / (2 d/D)          how fast a ball spins

Two things about this that matter more than the algebra.

The first is that BPFO is exactly n * FTF, and BPFI is exactly n * (1 - FTF).
The cage carries the balls round at some fraction of shaft speed; a stationary
outer-race defect is struck once per ball that passes it, and an inner-race
defect rotates against that traffic and so is struck more often. Everything
else is bookkeeping, and stating it this way lets an engineer check a number in
their head, which matters when they are deciding whether to believe us.

The second is BSF, which is a trap. BSF is the rate at which a ball *spins*. A
defect on that ball strikes the outer race once per spin and the inner race once
per spin, so the observable impact rate is 2 x BSF. Published tables — the Case
Western table among them — quote the doubled figure under the name "BSF" without
saying so. Searching a spectrum at half the right frequency finds nothing, and a
failing bearing then reads as healthy. This module keeps the two separate and
names them: `bsf` is the spin, `ball_defect` is what you look for.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Bearing:
    """Geometry, and the part names an operator would order by."""

    designation: str
    n_rolling: int
    ball_diameter: float          # d, any consistent unit
    pitch_diameter: float         # D, same unit
    contact_angle_deg: float = 0.0
    source: str = ""              # where the dimensions came from — cite it

    def __post_init__(self):
        if self.n_rolling < 3:
            raise ValueError("a bearing needs at least 3 rolling elements")
        if not (0 < self.ball_diameter < self.pitch_diameter):
            raise ValueError("ball diameter must be positive and below pitch diameter")
        if not (0 <= self.contact_angle_deg < 90):
            raise ValueError("contact angle must be in [0, 90)")

    @property
    def r(self) -> float:
        """(d/D)cos(phi) — the one derived quantity everything uses."""
        return (self.ball_diameter / self.pitch_diameter) * math.cos(
            math.radians(self.contact_angle_deg)
        )

    # -- orders: multiples of shaft speed ------------------------------------
    @property
    def ftf(self) -> float:
        """Cage / fundamental train frequency. Also the ball orbit rate."""
        return 0.5 * (1.0 - self.r)

    @property
    def bpfo(self) -> float:
        """Outer race. Identically n * ftf — the balls sweeping past a fixed point."""
        return 0.5 * self.n_rolling * (1.0 - self.r)

    @property
    def bpfi(self) -> float:
        """Inner race. Identically n * (1 - ftf) — the defect running against traffic."""
        return 0.5 * self.n_rolling * (1.0 + self.r)

    @property
    def bsf(self) -> float:
        """Ball SPIN rate. Not the impact rate — see ball_defect."""
        return (1.0 - self.r * self.r) / (2.0 * self.ball_diameter / self.pitch_diameter)

    @property
    def ball_defect(self) -> float:
        """What a defect on a ball actually produces: 2 x BSF, because the ball
        strikes the inner race and the outer race once per spin. This is the
        number to search for, and the one published tables quietly report."""
        return 2.0 * self.bsf

    def orders(self) -> Dict[str, float]:
        """Every fault order, keyed by the part that fails."""
        return {
            "outer_race": self.bpfo,
            "inner_race": self.bpfi,
            "rolling_element": self.ball_defect,
            "cage": self.ftf,
        }

    def hz(self, shaft_hz: float) -> Dict[str, float]:
        """The same, in Hz, at a given shaft speed."""
        return {k: v * shaft_hz for k, v in self.orders().items()}

    def hz_at_rpm(self, rpm: float) -> Dict[str, float]:
        return self.hz(rpm / 60.0)

    # -- sanity ---------------------------------------------------------------
    def check(self) -> Dict[str, bool]:
        """Identities that must hold. Cheap, and they catch a mistyped dimension
        or a transposed d and D immediately — which is worth having, because a
        wrong geometry produces confident, plausible, wrong answers rather than
        an error."""
        return {
            "bpfo_is_n_times_ftf": abs(self.bpfo - self.n_rolling * self.ftf) < 1e-9,
            "bpfi_is_n_minus_bpfo": abs(self.bpfi - (self.n_rolling - self.bpfo)) < 1e-9,
            "bpfi_exceeds_bpfo": self.bpfi > self.bpfo,
            "ftf_below_half": 0.0 < self.ftf < 0.5,
            "ball_defect_is_twice_bsf": abs(self.ball_defect - 2 * self.bsf) < 1e-12,
        }

    def describe(self) -> str:
        o = self.orders()
        w = max(len(k) for k in o)
        lines = [
            f"{self.designation}  n={self.n_rolling}  d={self.ball_diameter:g}  "
            f"D={self.pitch_diameter:g}  phi={self.contact_angle_deg:g}deg",
            f"  {'cage (FTF)':<{w}}  {self.ftf:8.4f} x shaft",
        ]
        for k in ("outer_race", "inner_race", "rolling_element"):
            lines.append(f"  {k:<{w}}  {o[k]:8.4f} x shaft")
        lines.append(f"  {'ball spin':<{w}}  {self.bsf:8.4f} x shaft   "
                     f"(impacts at {self.ball_defect:.4f}, twice this)")
        if self.source:
            lines.append(f"  dimensions: {self.source}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Catalogue. Dimensions are cited, because a fault frequency is only as
# trustworthy as the geometry behind it and a reviewer will want to check.
#
# The two below are the bearings in the Case Western Reserve seeded-fault
# dataset, which is the public evidence base for this bid. Note that the 6203
# has EIGHT rolling elements, not nine: assuming nine (a natural slip, since its
# neighbour in the same test rig has nine) makes every one of its published
# multipliers irreconcilable. That inconsistency is a useful canary — if the
# identities in check() fail, the geometry is wrong, not the physics.
# ---------------------------------------------------------------------------
CATALOGUE: Dict[str, Bearing] = {
    "SKF6205": Bearing(
        designation="SKF 6205-2RS JEM",
        n_rolling=9, ball_diameter=0.3126, pitch_diameter=1.537,
        contact_angle_deg=0.0,
        source="CWRU bearing data centre, drive-end bearing (inches)",
    ),
    "SKF6203": Bearing(
        designation="SKF 6203-2RS JEM",
        n_rolling=8, ball_diameter=0.2656, pitch_diameter=1.122,
        contact_angle_deg=0.0,
        source="CWRU bearing data centre, fan-end bearing (inches)",
    ),
    # MFPT's NICE bearing. Verified: these dimensions reproduce MFPT's own
    # published 81.12 Hz outer / 118.875 Hz inner at a 25 Hz shaft, which is the
    # only way to be sure a transcribed geometry is right.
    "MFPTNICE": Bearing(
        designation="NICE bearing (MFPT test rig)",
        n_rolling=8, ball_diameter=0.235, pitch_diameter=1.245,
        contact_angle_deg=0.0,
        source="MFPT fault data sets documentation (inches)",
    ),
    # uOttawa variable-speed rig. VERIFY before quoting: the dataset paper gives
    # the bearing as an ER-16K and these dimensions are the manufacturer's, not
    # a figure the dataset itself publishes. cbmx prints the orders it derives —
    # check them against the paper before any number leaves the terminal.
    "ER16K": Bearing(
        designation="ER-16K (uOttawa variable-speed rig)",
        n_rolling=9, ball_diameter=0.2812, pitch_diameter=1.516,
        contact_angle_deg=0.0,
        source="MB ER-16K manufacturer dimensions (inches) — UNVERIFIED against the dataset paper",
    ),
    # Paderborn uses a 6203, the same designation as the CWRU fan end — but the
    # dimensions below are NOT borrowed from SKF's catalogue. They are the ones
    # the dataset's own paper prints for the bearing actually in the rig:
    # Lessmeier, Kimotho, Zimmer & Sextro, "Condition Monitoring of Bearing
    # Damage in Electromechanical Drive Systems by Using Motor Current Signals
    # of Electric Motors: A Benchmark Data Set for Data-Driven Classification",
    # PHM Society European Conference 2016, Table 1: pitch circle diameter
    # 28.55 mm, rolling element diameter 6.75 mm, 8 elements, 0 degrees, FAG.
    #
    # This entry previously used SKF 6203 dimensions as a stand-in, which put
    # the raceway orders about 0.1% out and the ball-defect order 0.14% out.
    # Small enough to stay invisible for ever, which is precisely the argument
    # for citing the dataset's own paper rather than a catalogue that happens to
    # agree to three figures.
    #
    # Note also that the paper states damage at the rolling elements was never
    # observed in the lifetime tests. There is no ball-defect record anywhere in
    # this dataset to test against.
    # KAIST varying-speed rig. THE DATASET PUBLISHES NO GEOMETRY. Neither the
    # Mendeley record (10.17632/vxkj334rzv) nor the authors' paper
    # (arXiv:2311.18547v2) gives a bearing model, ball count, ball diameter,
    # pitch diameter or contact angle; the Data in Brief article behind them is
    # paywalled. A 6205 is the near-universal research bearing for a rig of this
    # class and these are SKF's catalogue dimensions for it.
    #
    # That is a declaration, not a measurement, and it is the weakest link in
    # any KAIST result this repository produces. It is declared ONCE, before
    # scoring, and it is not revised if the result is disappointing -- revising
    # geometry until the peaks line up is fitting the physics to the labels,
    # which is the one thing this method may not do. Anything derived from this
    # entry must carry the UNVERIFIED caveat into the report.
    "KAIST6205U": Bearing(
        designation="6205 assumed (KAIST varying-speed rig)",
        n_rolling=9, ball_diameter=7.94, pitch_diameter=39.04,
        contact_angle_deg=0.0,
        source="SKF 6205 catalogue dimensions (mm) — UNVERIFIED: the KAIST "
               "dataset publishes no bearing geometry at all",
    ),
    # The KAT set contains bearings from TWO manufacturers with different pitch
    # circle diameters, which the dataset's own per-bearing fact sheets state and
    # the 2016 paper's single Table 1 does not: FAG at 28.55 mm (KA15, KI16) and
    # MTK at 29.05 mm (KA16, KA22, KA30, KI04, KI14, KI17, KI18). Applying one
    # geometry to all of them is a 1.75% error on every raceway order — inside
    # the peak-search tolerance, so it produces no error message and no obviously
    # wrong number. Exactly the class of mistake this module exists to prevent.
    "PU6203MTK": Bearing(
        designation="MTK 6203 (Paderborn KAT, MTK-supplied bearings)",
        n_rolling=8, ball_diameter=6.75, pitch_diameter=29.05,
        contact_angle_deg=0.0,
        source="KAT per-bearing damage fact sheet shipped with each archive (mm)",
    ),
    "PU6203": Bearing(
        designation="FAG 6203 (Paderborn KAT bearing test rig)",
        n_rolling=8, ball_diameter=6.75, pitch_diameter=28.55,
        contact_angle_deg=0.0,
        source="Lessmeier et al. 2016, PHME, Table 1 (mm, as published)",
    ),
}

# Published multipliers, for regression. If our arithmetic ever drifts from the
# numbers the dataset's own documentation states, we want a test failure and not
# a quiet disagreement.
PUBLISHED = {
    "SKF6205": {"inner_race": 5.4152, "outer_race": 3.5848, "cage": 0.3983,
                "rolling_element": 4.7135},
    "SKF6203": {"inner_race": 4.9469, "outer_race": 3.0530, "cage": 0.3817,
                "rolling_element": 3.9874},
}


def get(name: str) -> Bearing:
    key = name.upper().replace("-", "").replace(" ", "")
    if key not in CATALOGUE:
        raise KeyError(f"unknown bearing {name!r}; have {sorted(CATALOGUE)}")
    return CATALOGUE[key]


@dataclass
class Harmonics:
    """Where to look in a spectrum, and what counts as a hit.

    A real defect does not put energy at exactly one line. It puts energy at the
    fault frequency and its harmonics, and — for anything whose defect moves
    through the load zone — at sidebands spaced one shaft revolution apart around
    each of those. Searching only the fundamental throws away most of the
    evidence and most of the discrimination.
    """

    centre: float                 # order, multiples of shaft speed
    harmonics: int = 3
    sidebands: int = 2
    sideband_spacing: float = 1.0     # in orders; one shaft revolution
    tolerance: float = 0.02           # fractional, absorbs speed wander

    def lines(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for h in range(1, self.harmonics + 1):
            f = self.centre * h
            out[f"h{h}"] = f
            for s in range(1, self.sidebands + 1):
                out[f"h{h}-{s}fr"] = f - s * self.sideband_spacing
                out[f"h{h}+{s}fr"] = f + s * self.sideband_spacing
        return {k: v for k, v in out.items() if v > 0}

    def band(self, order: float) -> tuple:
        """Half-width window around a line, widened by the tolerance."""
        w = max(self.tolerance * order, 0.03)
        return (order - w, order + w)


def harmonics_for(b: Bearing, part: str, **kw) -> Harmonics:
    """Sidebands are only expected where the defect moves relative to the load
    zone. An outer-race defect is stationary in it, so asking for sidebands there
    invites false confidence from whatever noise happens to sit nearby."""
    centre = b.orders()[part]
    defaults = {
        "outer_race": dict(harmonics=4, sidebands=0),
        "inner_race": dict(harmonics=3, sidebands=2, sideband_spacing=1.0),
        "rolling_element": dict(harmonics=3, sidebands=2, sideband_spacing=b.ftf),
        "cage": dict(harmonics=2, sidebands=0),
    }[part]
    defaults.update(kw)
    return Harmonics(centre=centre, **defaults)
