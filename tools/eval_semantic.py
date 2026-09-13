#!/usr/bin/env python3
"""The semantic layer, end to end, with the gate doing real work.

The topic asks for hybrid physics AND semantic reasoning — knowledge graphs over
the asset, the maintenance domain and mission context. A volume that describes
that architecture without exercising it leaves the evaluator's central question
open. This runs it.

    detected residual
        -> resolve the containing assembly and the Lowest Replaceable Unit
        -> retrieve the maintenance history that bears on it
        -> resolve the part number through the parts catalogue
        -> assemble a hypothesis WITH CITATIONS
        -> deterministic validation gate accepts or DISCARDS
        -> only what survives is shown to a maintainer

Then it does the part that matters more: it feeds the gate four hypotheses that
are wrong in four different ways and shows each one rejected, with the reason.
A gate that has never rejected anything is decoration.

WHAT IS AND IS NOT EXERCISED HERE. The retrieval, the graph traversal, the
part-number resolution, the hypothesis assembly and the whole validation gate
run for real, on a synthetic asset card and eighteen mock work orders. What is
NOT exercised is a language model rendering the final sentence: that step is
stubbed deterministically, because what needs demonstrating is the gate's
behaviour, not fluency. In the fielded system the retrieval and hypothesis
assembly are interpretDB's, with this same gate downstream and unchanged.

WHY THE DATA IS SYNTHETIC AND WHY THAT IS FINE HERE. No Government AAG data has
been seen; it is available post-award. The asset card and work orders below are
invented, and their shape rather than their content is what is being tested —
that the traversal resolves an LRU, that citations must exist and match, and
that a part number must resolve. Every identifier is fictional and the file says
so at the top.

Usage:
    PYTHONPATH=src python3 tools/eval_semantic.py
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cbmx.physics.bearing import Bearing                            # noqa: E402


# ── the asset graph ─────────────────────────────────────────────────────────

class AssetGraph:
    """The Government-owned artifact. The platform reads it; it owns nothing."""

    def __init__(self, card: Dict):
        self.card = card
        self.by_path: Dict[str, Dict] = {}
        self.parent: Dict[str, Optional[str]] = {}
        self._index(card, None)
        self.parts = card["parts_catalogue"]
        self.sensors = card["sensors"]

    def _index(self, node: Dict, parent: Optional[str]) -> None:
        p = node.get("path")
        if p:
            self.by_path[p] = node
            self.parent[p] = parent
        for c in node.get("children", []):
            self._index(c, p if p else parent)

    def lru_for(self, path: str) -> Optional[str]:
        """Walk up until a node marked as a replaceable unit is found."""
        cur = path
        while cur is not None:
            n = self.by_path.get(cur)
            if n and n.get("lru"):
                return cur
            cur = self.parent.get(cur)
        return None

    def bearing_at(self, path: str) -> Optional[Bearing]:
        n = self.by_path.get(path) or {}
        g = n.get("bearing")
        if not g:
            return None
        return Bearing(designation=n.get("name", path),
                       n_rolling=g["n_rolling"],
                       ball_diameter=g["ball_diameter_mm"],
                       pitch_diameter=g["pitch_diameter_mm"],
                       contact_angle_deg=g["contact_angle_deg"],
                       source="asset card")

    def observed_by(self, sensor: str) -> Optional[str]:
        s = self.sensors.get(sensor)
        return s["observes"] if s else None


# ── retrieval over the maintenance corpus ───────────────────────────────────

_WORD = re.compile(r"[a-z0-9]+")


def _toks(s: str) -> List[str]:
    return _WORD.findall(s.lower())


class Corpus:
    """TF-IDF cosine retrieval over free-text work orders.

    Deliberately simple and inspectable. The point being demonstrated is not
    retrieval sophistication — it is that a retrieved record must survive the
    gate, and a gate cannot be tested against a retriever nobody can audit.
    """

    def __init__(self, records: List[Dict]):
        self.records = records
        self.docs = [_toks(r["text"] + " " + r["asset"] + " " + r["action"])
                     for r in records]
        df = Counter()
        for d in self.docs:
            df.update(set(d))
        n = len(self.docs)
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}
        self.vecs = [self._vec(d) for d in self.docs]

    def _vec(self, toks: List[str]) -> Dict[str, float]:
        tf = Counter(toks)
        v = {w: (1 + math.log(c)) * self.idf.get(w, 1.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def search(self, query: str, asset_prefix: Optional[str] = None, k: int = 3):
        q = self._vec(_toks(query))
        scored = []
        for rec, v in zip(self.records, self.vecs):
            if asset_prefix and not rec["asset"].startswith(asset_prefix):
                continue
            s = sum(q.get(w, 0.0) * x for w, x in v.items())
            scored.append((s, rec))
        scored.sort(key=lambda t: -t[0])
        return scored[:k]


# ── the hypothesis, and the gate ────────────────────────────────────────────

@dataclass
class Hypothesis:
    lru_path: str
    fault_mode: str
    part_number: str
    cited_work_orders: List[str]
    rationale: str
    order_predicted: float = 0.0
    order_measured: float = 0.0
    sensor: str = ""


@dataclass
class GateResult:
    accepted: bool
    checks: List[tuple] = field(default_factory=list)   # (name, passed, detail)

    def report(self) -> str:
        return "\n".join(
            f"      {'PASS' if ok else 'FAIL'}  {name:<28} {detail}"
            for name, ok, detail in self.checks)


def validate(h: Hypothesis, g: AssetGraph, c: Corpus,
             tol: float = 0.02) -> GateResult:
    """Every claim in the hypothesis must resolve against a Government artifact.

    No check here is about plausibility. Each one asks whether a specific
    assertion corresponds to something that exists. A hypothesis failing any
    check is DISCARDED — not shown with lower confidence, not hedged. A gate
    that softens rather than rejects is a gate that always passes.
    """
    r = GateResult(True)

    node = g.by_path.get(h.lru_path)
    ok = node is not None
    r.checks.append(("asset path resolves", ok,
                     h.lru_path if ok else f"{h.lru_path} not in asset card"))

    ok_lru = bool(node and node.get("lru"))
    r.checks.append(("path is a replaceable unit", ok_lru,
                     "marked lru" if ok_lru else "not a replaceable unit"))

    b = g.bearing_at(h.lru_path) if node else None
    if b is not None:
        orders = b.orders()
        ok_mode = h.fault_mode in orders
        r.checks.append(("fault mode exists for this part", ok_mode,
                         f"{h.fault_mode}" if ok_mode
                         else f"{h.fault_mode} is not a mode of this component"))
        if ok_mode:
            pred = orders[h.fault_mode]
            err = abs(h.order_measured - pred) / pred if pred else 1.0
            ok_geo = err <= tol
            r.checks.append(("measured order matches geometry", ok_geo,
                             f"predicted {pred:.4f}x, measured "
                             f"{h.order_measured:.4f}x, {err*100:.2f}% off "
                             f"(tolerance {tol*100:.0f}%)"))
        else:
            r.checks.append(("measured order matches geometry", False,
                             "not evaluated — fault mode invalid"))
    else:
        r.checks.append(("fault mode exists for this part", False,
                         "no component physics on this node"))

    exp_part = node.get("part_number") if node else None
    ok_part_node = exp_part == h.part_number
    r.checks.append(("part number matches asset card", ok_part_node,
                     f"{h.part_number}" if ok_part_node
                     else f"card says {exp_part}, hypothesis says {h.part_number}"))

    ok_cat = h.part_number in g.parts
    r.checks.append(("part resolves in catalogue", ok_cat,
                     g.parts[h.part_number]["nomenclature"] if ok_cat
                     else f"{h.part_number} not in parts catalogue"))

    known = {rec["wo"]: rec for rec in c.records}
    for wo in h.cited_work_orders:
        rec = known.get(wo)
        if rec is None:
            r.checks.append((f"citation {wo}", False, "work order does not exist"))
            continue
        same = rec["asset"].startswith(h.lru_path) or h.lru_path.startswith(rec["asset"])
        r.checks.append((f"citation {wo}", same,
                         f"{rec['date']} {rec['action']}" if same
                         else f"cites {rec['asset']}, hypothesis is {h.lru_path}"))

    ok_sensor = g.observed_by(h.sensor) == h.lru_path if h.sensor else False
    r.checks.append(("sensor observes this unit", ok_sensor,
                     f"{h.sensor} -> {g.observed_by(h.sensor)}"))

    r.accepted = all(ok for _, ok, _ in r.checks)
    return r


def propose(g: AssetGraph, c: Corpus, sensor: str, fault_mode: str,
            order_measured: float, z: float) -> Hypothesis:
    """Physics has already decided. This connects that decision to the asset."""
    observed = g.observed_by(sensor)
    lru = g.lru_for(observed) if observed else None
    node = g.by_path.get(lru, {})
    b = g.bearing_at(lru) if lru else None
    query = (f"{fault_mode.replace('_', ' ')} noise growl load arrestment "
             f"{node.get('name', '')}")
    hits = c.search(query, asset_prefix=lru, k=2)
    cites = [rec["wo"] for _, rec in hits]
    prior = "; ".join(f"{rec['wo']} ({rec['date']}, {rec['action']})"
                      for _, rec in hits)
    return Hypothesis(
        lru_path=lru or "", fault_mode=fault_mode,
        part_number=node.get("part_number", ""), cited_work_orders=cites,
        order_predicted=b.orders()[fault_mode] if b and fault_mode in b.orders() else 0.0,
        order_measured=order_measured, sensor=sensor,
        rationale=(f"{node.get('name','?')} shows {z:.1f} sigma at the "
                   f"{fault_mode.replace('_',' ')} order on {sensor}. "
                   f"Maintenance history for this unit: {prior}."))


def show(title: str, h: Hypothesis, res: GateResult) -> None:
    print(f"\n── {title}")
    print(f"   LRU        {h.lru_path}")
    print(f"   fault mode {h.fault_mode}   order {h.order_measured:.4f}x "
          f"(predicted {h.order_predicted:.4f}x)")
    print(f"   part       {h.part_number}")
    print(f"   citations  {', '.join(h.cited_work_orders) or '(none)'}")
    print("   gate:")
    print(res.report())
    verdict = ("ACCEPTED — shown to maintainer" if res.accepted
               else "DISCARDED — not shown")
    print(f"   VERDICT    {verdict}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/semantic")
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    root = Path(a.root)
    g = AssetGraph(json.loads((root / "asset_card.json").read_text()))
    recs = [json.loads(ln)
            for ln in (root / "maintenance.jsonl").read_text().splitlines()
            if ln.strip()]
    c = Corpus(recs)

    print(f"asset card: {len(g.by_path)} nodes, "
          f"{sum(1 for n in g.by_path.values() if n.get('lru'))} replaceable units, "
          f"{len(g.parts)} catalogue entries, {len(g.sensors)} sensors")
    print(f"maintenance corpus: {len(recs)} work orders\n")
    print("SYNTHETIC DATA — every identifier is invented; see the asset card header.")
    print("The Government AAG card replaces it wholesale in Phase II.")

    # ── the real case ──────────────────────────────────────────────────────
    # A detection has already been made deterministically: outer-race energy on
    # VIB-01 at 3.0543x shaft, 11.2 sigma above this unit's own baseline.
    b = g.bearing_at("AAG/HPU/pump_motor/de_bearing")
    measured = b.orders()["outer_race"] * 1.004          # 0.4% off, real slip
    h = propose(g, c, "VIB-01", "outer_race", measured, 11.2)
    res = validate(h, g, c)
    print("\n" + "═" * 74)
    print("CASE 1 — a true detection, connected to the asset and its history")
    print("═" * 74)
    show("hypothesis proposed by the semantic layer", h, res)
    print(f"\n   rationale as assembled:\n      {h.rationale}")
    cat = g.parts[h.part_number]
    print(f"\n   resolved part: {h.part_number} — {cat['nomenclature']}, "
          f"NSN {cat['nsn']}" + (f", supersedes {cat['supersedes']}"
                                 if cat['supersedes'] else ""))

    # ── the adversarial cases ─────────────────────────────────────────────
    print("\n" + "═" * 74)
    print("CASES 2-5 — hypotheses that are wrong in four different ways")
    print("═" * 74)
    bad = []

    h2 = propose(g, c, "VIB-01", "outer_race", measured, 11.2)
    h2.part_number = "BRG-6203-XX"
    bad.append(("bad part number — does not exist in the catalogue", h2))

    h3 = propose(g, c, "VIB-01", "outer_race", measured, 11.2)
    h3.cited_work_orders = ["WO-2025-9999"]
    bad.append(("bogus citation — work order does not exist", h3))

    h4 = propose(g, c, "VIB-01", "outer_race", measured, 11.2)
    h4.cited_work_orders = ["WO-2025-0091"]           # the water twister valve
    bad.append(("citation exists but belongs to a different asset", h4))

    h5 = propose(g, c, "VIB-01", "outer_race", measured, 11.2)
    h5.order_measured = b.orders()["inner_race"]      # wrong line for this mode
    bad.append(("measured order does not match the declared fault mode", h5))

    results = {}
    for title, hb in bad:
        rb = validate(hb, g, c)
        show(title, hb, rb)
        results[title] = rb.accepted

    print("\n" + "═" * 74)
    n_rej = sum(1 for v in results.values() if not v)
    print(f"gate accepted the true hypothesis: {res.accepted}")
    print(f"gate rejected {n_rej} of {len(results)} malformed hypotheses")
    print("\nEvery rejection is a specific failed assertion against a Government")
    print("artifact — a part that is not in the catalogue, a work order that does")
    print("not exist, a citation belonging to another asset, a frequency the")
    print("geometry does not predict. None is a judgement about plausibility.")

    if a.json:
        Path(a.json).write_text(json.dumps(
            {"true_case_accepted": res.accepted,
             "adversarial": {k: {"accepted": v} for k, v in results.items()},
             "n_work_orders": len(recs),
             "n_lrus": sum(1 for n in g.by_path.values() if n.get("lru"))},
            indent=2))
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
