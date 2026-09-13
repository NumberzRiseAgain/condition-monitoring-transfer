"""Attributing energy to the part that produced it, when lines collide.

The single-line search asks each candidate defect one question: how much energy
is at your frequency? On CWRU that produces a specific, reproducible error —
every rolling-element fault reads as an inner-race fault, and it reads that way
by a decisive margin rather than a near-tie:

    rolling_element 0.021in @ 0hp
        outer 3.7    inner 16.3    ball 2.1    cage 0.4   ->  inner_race, +12.6 dB

The cause is geometry and it can be shown without any data:

    ball defect order              4.7134
    its +2 cage sideband           5.5101      <-- lands here
    inner race window       5.3069 - 5.5235

A ball defect's own fundamental is weak, because a spall on a rolling element
only strikes hard when it happens to be in the load zone. What is strong is its
sideband structure — and one of those sidebands sits inside the inner-race
window. The single-line search sees a big peak in inner-race's window and has no
way to know it belongs to the ball.

The fix is to stop asking about one line and start asking about a *family*. Each
defect predicts a whole pattern — a fundamental, its harmonics, and sidebands at
a spacing the geometry also fixes. A real inner-race fault puts energy at 5.415
AND at 5.415±1.0 (shaft) AND at 10.83. A ball fault leaking into that window
puts energy at 5.510 and at nothing else the inner-race family predicts. Scoring
the pattern rather than the peak separates them.

Two rules make this work and both are geometric, not fitted:

  Contested lines are dropped. Where two families both predict energy within
  tolerance of the same order, that bin cannot distinguish them, so neither is
  allowed to count it. This costs sensitivity and buys specificity, which is the
  right trade for a system whose output is a maintenance action.

  A family must keep its fundamental or a harmonic. A hypothesis supported only
  by sidebands is not a diagnosis, it is a coincidence.

Honesty note, and it matters for the proposal: this change was made *because* of
what CWRU showed. The mechanism is geometric and would have been correct before
we ever saw the data, but the decision to look for it was not. Any accuracy
number measured after this point is developed-on-CWRU, not held out from it.
Validating on a second dataset — Paderborn or MFPT — is what would restore a
genuine held-out claim, and that is worth doing before the volume is written.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


from .bearing import Bearing, harmonics_for
from .envelope import EnvelopeSpectrum

# What each member of a family is worth. The fundamental carries the diagnosis;
# harmonics corroborate it; sidebands say the defect is moving through the load
# zone the way the geometry says it should.
WEIGHTS = {"fundamental": 1.0, "harmonic": 0.55, "sideband": 0.5}


@dataclass
class LineEvidence:
    name: str
    order: float
    kind: str                  # fundamental | harmonic | sideband
    prominence_db: float
    measured_order: float
    contested_with: List[str] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return not self.contested_with


@dataclass
class Attribution:
    scores: Dict[str, float]                   # part -> family score, dB
    lines: Dict[str, List[LineEvidence]]       # part -> its family
    dropped: List[Tuple[str, str, float]]      # (part, line, order) dropped as contested

    def best(self) -> Tuple[str, float]:
        if not self.scores:
            return ("", 0.0)
        k = max(self.scores, key=self.scores.get)
        return (k, self.scores[k])

    def margin(self) -> float:
        v = sorted(self.scores.values(), reverse=True)
        return (v[0] - v[1]) if len(v) > 1 else float("inf")

    def explain(self, part: str) -> Dict:
        """The arithmetic behind one hypothesis, for the report."""
        fam = self.lines.get(part, [])
        used = [ln for ln in fam if ln.usable]
        return {
            "score_db": round(self.scores.get(part, 0.0), 2),
            "lines_used": len(used),
            "lines_dropped_as_contested": [
                {"line": ln.name, "order": round(ln.order, 3),
                 "also_claimed_by": ln.contested_with}
                for ln in fam if not ln.usable
            ],
            "detail": [
                {"line": ln.name, "kind": ln.kind,
                 "order_predicted": round(ln.order, 4),
                 "order_measured": round(ln.measured_order, 4),
                 "db": round(ln.prominence_db, 2)}
                for ln in used
            ],
        }


def _kind(name: str) -> str:
    if name == "h1":
        return "fundamental"
    return "sideband" if "fr" in name or "-" in name or "+" in name else "harmonic"


def _overlaps(a: float, b: float, tol: float) -> bool:
    wa = max(tol * a, 0.03)
    wb = max(tol * b, 0.03)
    return abs(a - b) <= (wa + wb) * 0.5


def attribute(es: EnvelopeSpectrum, bearing: Bearing, tol: float = 0.02,
              parts: Optional[List[str]] = None) -> Attribution:
    parts = parts or list(bearing.orders().keys())

    families: Dict[str, Dict[str, float]] = {}
    for p in parts:
        h = harmonics_for(bearing, p)
        h.tolerance = tol
        families[p] = {k: v for k, v in h.lines().items() if 0.1 < v <= 20.0}

    # Which lines cannot distinguish between hypotheses?
    contested: Dict[Tuple[str, str], List[str]] = {}
    for p, fam in families.items():
        for name, order in fam.items():
            clash = []
            for q, other in families.items():
                if q == p:
                    continue
                for oname, oorder in other.items():
                    if _overlaps(order, oorder, tol):
                        clash.append(f"{q}:{oname}")
                        break
            if clash:
                contested[(p, name)] = clash

    lines: Dict[str, List[LineEvidence]] = {}
    scores: Dict[str, float] = {}
    dropped: List[Tuple[str, str, float]] = []

    for p, fam in families.items():
        ev: List[LineEvidence] = []
        for name, order in fam.items():
            mo, db = es.prominence_db(order, tol)
            c = contested.get((p, name), [])
            ev.append(LineEvidence(name, order, _kind(name), db, mo, c))
            if c:
                dropped.append((p, name, order))
        lines[p] = ev

        usable = [ln for ln in ev if ln.usable]
        anchor = [ln for ln in usable if ln.kind in ("fundamental", "harmonic")]
        if not anchor:
            # Sidebands alone are a coincidence, not a diagnosis.
            scores[p] = -99.0
            continue
        num = sum(WEIGHTS[ln.kind] * ln.prominence_db for ln in usable)
        den = sum(WEIGHTS[ln.kind] for ln in usable)
        scores[p] = num / max(den, 1e-9)

    return Attribution(scores, lines, dropped)


def collisions(bearing: Bearing, tol: float = 0.02) -> List[Dict]:
    """Which lines collide, from geometry alone. No data required, and worth
    printing at commissioning: a bearing whose families overlap badly cannot be
    diagnosed to the element by vibration alone, and saying so up front is far
    better than discovering it in a confusion matrix."""
    parts = list(bearing.orders().keys())
    fams = {}
    for p in parts:
        h = harmonics_for(bearing, p)
        h.tolerance = tol
        fams[p] = {k: v for k, v in h.lines().items() if 0.1 < v <= 20.0}
    out = []
    seen = set()
    for p, fam in fams.items():
        for name, order in fam.items():
            for q, other in fams.items():
                if q == p:
                    continue
                for oname, oorder in other.items():
                    if _overlaps(order, oorder, tol):
                        key = tuple(sorted([f"{p}:{name}", f"{q}:{oname}"]))
                        if key in seen:
                            continue
                        seen.add(key)
                        out.append({"a": f"{p}:{name}", "a_order": round(order, 4),
                                    "b": f"{q}:{oname}", "b_order": round(oorder, 4),
                                    "separation": round(abs(order - oorder), 4)})
    return sorted(out, key=lambda d: d["separation"])
