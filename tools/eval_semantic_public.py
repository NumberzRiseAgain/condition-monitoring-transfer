#!/usr/bin/env python3
"""E17 — the same semantic gate, on a published maintenance corpus.

WHY THIS EXISTS. E3 exercises the validation gate on a synthetic asset card and
eighteen invented work orders. The gate is real; the substrate is not, and an
evaluator is entitled to ask whether the traversal survives data somebody else
published. This runs the identical gate — `validate()` imported from
`eval_semantic.py`, not re-implemented — against the S1000D Issue 6 Bike Sample
Data Set: a real illustrated parts catalogue with real part numbers, and real
procedural data modules with real step prose, published by ASD/AIA/A4A.

WHAT IS EXERCISED, AND WHAT IS NOT. The published corpus carries a parts
breakdown and maintenance procedures. It carries no component geometry and no
sensors, so three of the gate's checks cannot be evaluated on it:

    fault mode exists for this part      needs bearing geometry
    measured order matches geometry      needs bearing geometry
    sensor observes this unit            needs a sensor map

Those three are evaluated on the seed asset card in E3 and on measured data in
E1/E2. This experiment scores the other five, which are the traceability half:

    asset path resolves                  the LRU exists in the breakdown
    path is a replaceable unit           it is something a maintainer can change
    part number matches asset card       the hypothesis names the right part
    part resolves in catalogue           the part number exists
    citation <wo>                        the cited procedure exists AND belongs
                                         to this unit

The exclusion is declared in the output JSON under `checks_excluded`. Nothing is
tuned: the gate code is untouched and the corpus is read as published.

HOW THE ASSET CARD IS BUILT. Mechanically, from two things the publisher
already encodes. The tree is the Standard Numbering System carried in every data
module code: model, system, sub-system, assembly. The replaceable units are the
items the publisher lists under a procedure's required spares, each attached to
the assembly of the module that lists it. The illustrated parts data modules add
the top-level breakdown by indenture -- an item at indenture n is a child of the
most recent item at indenture n-1 -- and supply catalogue nomenclature. No node,
name or part number is authored here, and nothing is matched by similarity.

HOW THE CORPUS IS BUILT. One record per procedural data module: the data module
code as the work-order identifier, the issue date as the date, the information
name as the action, the top-level step prose as the text, and the assembly path
taken from the module's own code. A citation therefore belongs to a unit when the
publisher filed it under that unit's assembly, which is what the gate checks.

    python3 tools/eval_semantic_public.py --csdb <CSDB dir> --json ../results/e17_semantic_public.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_semantic import AssetGraph, Corpus, Hypothesis, validate    # noqa: E402

TRACEABILITY_CHECKS = ("asset path resolves", "path is a replaceable unit",
                       "part number matches asset card", "part resolves in catalogue")
EXCLUDED = ("fault mode exists for this part", "measured order matches geometry",
            "sensor observes this unit")


def local(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def text_of(el) -> str:
    return " ".join("".join(el.itertext()).split()) if el is not None else ""


def first(root, tag):
    for e in root.iter():
        if local(e.tag) == tag:
            return e
    return None


def seg(part_no: str) -> str:
    """Part numbers contain '/', which is the path separator here."""
    return part_no.replace("/", "-")


def parse(csdb: Path):
    ipds, procs = [], []
    for name in sorted(os.listdir(csdb)):
        if not name.lower().endswith(".xml"):
            continue
        try:
            root = ET.parse(csdb / name).getroot()
        except ET.ParseError:
            continue
        tags = {local(e.tag) for e in root.iter()}
        code = first(root, "dmCode")
        if code is None:
            continue
        if "illustratedPartsCatalog" in tags:
            ipds.append((name, root, code.attrib))
        if "procedure" in tags and "mainProcedure" in tags:
            procs.append((name, root, code.attrib))
    return ipds, procs


def dmc(attrs: Dict[str, str]) -> str:
    return ("DMC-{modelIdentCode}-{systemDiffCode}-{systemCode}-{subSystemCode}"
            "{subSubSystemCode}-{assyCode}-{disassyCode}{disassyCodeVariant}-"
            "{infoCode}{infoCodeVariant}-{itemLocationCode}").format(**attrs)


def sns_path(attrs: Dict[str, str]) -> List[str]:
    """The publisher's own breakdown, read out of the data module code."""
    model = attrs["modelIdentCode"]
    system = attrs["systemCode"]
    sub = f"{attrs['subSystemCode']}{attrs['subSubSystemCode']}"
    assy = attrs["assyCode"]
    return ["ASSET", f"ASSET/{model}", f"ASSET/{model}/{system}",
            f"ASSET/{model}/{system}/{sub}", f"ASSET/{model}/{system}/{sub}/{assy}"]


def spares_of(root) -> List[Dict[str, str]]:
    out = []
    for e in root.iter():
        if local(e.tag) != "spareDescr":
            continue
        name = text_of(first(e, "name"))
        for x in e.iter():
            if local(x.tag) == "partNumber" and text_of(x):
                out.append({"part_number": text_of(x), "name": name or text_of(x)})
    return out


def build_card(ipds, procs):
    """Assemble the card the gate reads: SNS tree, publisher spares as the
    replaceable units, catalogue from the illustrated parts data plus the spares."""
    root_node = {"path": "ASSET", "name": "published sample product", "children": []}
    by_path = {"ASSET": root_node}
    catalogue: Dict[str, Dict] = {}

    def ensure(path: str, name: str, parent: str) -> Dict:
        node = by_path.get(path)
        if node is None:
            node = {"path": path, "name": name, "children": []}
            by_path[path] = node
            by_path[parent]["children"].append(node)
        return node

    # -- the breakdown the data module codes encode -------------------------
    for _fname, root, attrs in procs:
        chain = sns_path(attrs)
        tech = text_of(first(root, "techName"))
        for depth in range(1, len(chain)):
            label = tech if depth == len(chain) - 1 else chain[depth].rsplit("/", 1)[-1]
            ensure(chain[depth], label, chain[depth - 1])

    # -- catalogue nomenclature and the top-level breakdown from the IPDs ----
    for fname, root, attrs in ipds:
        model = attrs["modelIdentCode"]
        ensure(f"ASSET/{model}", model, "ASSET")
        stack: Dict[int, str] = {0: f"ASSET/{model}"}
        for csn in [e for e in root.iter() if local(e.tag) == "catalogSeqNumber"]:
            try:
                ind = int(csn.get("indenture", "1"))
            except ValueError:
                ind = 1
            for item in [e for e in csn.iter() if local(e.tag) == "itemSeqNumber"]:
                ref = first(item, "partRef")
                if ref is None or not ref.get("partNumberValue"):
                    continue
                pn = ref.get("partNumberValue")
                name = text_of(first(item, "descrForPart")) or pn
                parent_path = stack.get(ind - 1, f"ASSET/{model}")
                path = f"{parent_path}/{seg(pn)}"
                node = ensure(path, name, parent_path)
                node["part_number"] = pn
                stack[ind] = path
                catalogue.setdefault(pn, {
                    "nomenclature": name, "source": f"{fname} (illustrated parts data)",
                    "quantity": text_of(first(item, "quantityPerNextHigherAssy")) or "1"})

    # -- replaceable units: what the publisher lists as spares ---------------
    n_lru = 0
    for fname, root, attrs in procs:
        assy = sns_path(attrs)[-1]
        for sp in spares_of(root):
            pn, name = sp["part_number"], sp["name"]
            path = f"{assy}/{seg(pn)}"
            node = ensure(path, name, assy)
            if not node.get("lru"):
                node["lru"] = True
                node["part_number"] = pn
                n_lru += 1
            catalogue.setdefault(pn, {
                "nomenclature": name, "source": f"{fname} (required spares)",
                "quantity": "1"})

    card = {"path": "ASSET", "name": root_node["name"], "children": root_node["children"],
            "parts_catalogue": catalogue, "sensors": {}}
    return card, by_path, n_lru


def build_corpus(procs) -> List[Dict]:
    records = []
    for fname, root, attrs in procs:
        tech = text_of(first(root, "techName"))
        info = text_of(first(root, "infoName"))
        issue = first(root, "issueDate")
        date = ("%s-%s-%s" % (issue.get("year"), issue.get("month"), issue.get("day"))
                if issue is not None and issue.get("year") else "undated")
        main = first(root, "mainProcedure")
        steps = []
        for st in list(main):
            if local(st.tag) == "proceduralStep":
                para = first(st, "para")
                steps.append(text_of(para) if para is not None else text_of(st))
        records.append({"wo": dmc(attrs), "asset": sns_path(attrs)[-1],
                        "action": info or "procedure", "date": date,
                        "text": f"{tech} {info} " + " ".join(steps),
                        "file": fname, "n_steps": len(steps)})
    return records


def subset_verdict(res):
    """Accept on the checks the published substrate can carry, and name the first
    check that failed. A class of mutants all rejected by one check is a weaker
    result than four classes rejected by four checks, so the caller reports which."""
    for name, ok, _ in res.checks:
        if name in EXCLUDED:
            continue
        if not ok:
            return False, (name if not name.startswith("citation ") else "citation")
    return True, None


def corpus_digest(csdb: Path) -> str:
    h = hashlib.sha256()
    for name in sorted(os.listdir(csdb)):
        p = csdb / name
        if p.is_file():
            h.update(name.encode())
            h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csdb", required=True, help="directory of published S1000D data modules")
    ap.add_argument("--json", default="")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    csdb = Path(a.csdb).resolve()
    ipds, procs = parse(csdb)

    card, by_path, n_lru = build_card(ipds, procs)
    part_paths = {n["part_number"]: p for p, n in by_path.items() if n.get("part_number")}
    g = AssetGraph(card)
    records = build_corpus(procs)
    c = Corpus(records)

    print(f"published corpus   {csdb}")
    print(f"  illustrated parts data modules  {len(ipds)}")
    print(f"  procedural data modules         {len(procs)}")
    print(f"  catalogue entries               {len(card['parts_catalogue'])}")
    print(f"  replaceable units (publisher's spares lists)  {n_lru}")
    print(f"  checks excluded on this substrate: {', '.join(EXCLUDED)}\n")

    lru_paths = [p for p, n in by_path.items() if n.get("lru")]
    true_cases = []
    adversarial = {k: [] for k in ("phantom part", "wrong-assembly part",
                                   "citation does not exist",
                                   "citation belongs to another unit")}
    rejected_by = {k: {} for k in adversarial}
    other_parts = [n["part_number"] for p, n in by_path.items() if n.get("part_number")]

    for lru in sorted(lru_paths):
        node = by_path[lru]
        assembly = lru.rsplit("/", 1)[0]
        hits = c.search(f"{node['name']} remove install replace", asset_prefix=assembly, k=2)
        cites = [rec["wo"] for _, rec in hits]
        if not cites:
            continue
        h = Hypothesis(lru_path=lru, fault_mode="n/a", part_number=node["part_number"],
                       cited_work_orders=cites, rationale="", sensor="")
        res = validate(h, g, c)
        ok, _why = subset_verdict(res)
        true_cases.append((lru, node["part_number"], cites, ok))
        if a.verbose and not ok:
            print(f"  REJECTED true case {lru}\n{res.report()}")

        foreign = next((p for p in other_parts
                        if p != node["part_number"] and part_paths[p] and
                        not part_paths[p].startswith(lru) and not lru.startswith(part_paths[p])), None)
        foreign_wo = next((r["wo"] for r in records
                           if not r["asset"].startswith(lru) and not lru.startswith(r["asset"])), None)
        mutants = [
            ("phantom part", Hypothesis(lru, "n/a", "ZZ-NOT-A-PART-0000", cites, "", sensor="")),
            ("citation does not exist", Hypothesis(lru, "n/a", node["part_number"],
                                                   ["DMC-NOSUCH-AAA-D00-00-00-00AA-000A-A"], "", sensor="")),
        ]
        if foreign:
            mutants.append(("wrong-assembly part",
                            Hypothesis(lru, "n/a", foreign, cites, "", sensor="")))
        if foreign_wo:
            mutants.append(("citation belongs to another unit",
                            Hypothesis(lru, "n/a", node["part_number"], [foreign_wo], "", sensor="")))
        for label, mh in mutants:
            accepted, why = subset_verdict(validate(mh, g, c))
            adversarial[label].append(accepted)
            if not accepted:
                rejected_by[label][why] = rejected_by[label].get(why, 0) + 1

    n_true = len(true_cases)
    n_true_ok = sum(1 for *_, ok in true_cases if ok)
    print(f"TRUE HYPOTHESES     {n_true_ok} of {n_true} accepted")
    print("ADVERSARIAL         one mutation each, everything else identical")
    summary = {}
    for label, outcomes in adversarial.items():
        rejected = sum(1 for o in outcomes if not o)
        summary[label] = {"n": len(outcomes), "rejected": rejected,
                          "rejected_by_check": rejected_by[label]}
        rate = f"{rejected}/{len(outcomes)}" if outcomes else "0/0"
        by = ", ".join(f"{k} ({v})" for k, v in sorted(rejected_by[label].items()))
        print(f"  {label:<34} rejected {rate:<8} by: {by}")

    out = {
        "experiment": "E17 semantic gate on a published maintenance corpus",
        "gate": "validate() imported unchanged from tools/eval_semantic.py",
        "corpus": {"path": str(csdb), "files": len(os.listdir(csdb)),
                   "digest": corpus_digest(csdb),
                   "ipd_modules": len(ipds), "procedural_modules": len(procs)},
        "catalogue_entries": len(card["parts_catalogue"]),
        "replaceable_units": n_lru,
        "checks_scored": list(TRACEABILITY_CHECKS) + ["citation <wo>"],
        "checks_excluded": {k: "not carried by the published corpus" for k in EXCLUDED},
        "true_hypotheses": {"n": n_true, "accepted": n_true_ok},
        "adversarial": summary,
        "per_unit": [{"lru": l, "part_number": p, "citations": ci, "accepted": ok}
                     for l, p, ci, ok in true_cases],
    }
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(out, indent=2) + "\n")
        print(f"\nwrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
