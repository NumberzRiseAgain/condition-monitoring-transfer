#!/usr/bin/env python3
"""Check everything cbmx needs, and say exactly what is missing.

    python tools/preflight.py

Exits 0 if the no-dataset commands will run. Datasets are reported as present
or absent per tool; absent is not an error, because most of them are large and
licensed individually.
"""
from __future__ import annotations

import importlib
import os
import sys

OK, NO, WARN = "  ok  ", " MISS ", " warn "

REQUIRED = [("numpy", "1.24"), ("scipy", "1.10"), ("yaml", None),
            ("pandas", "2.0")]

# tool -> (path it reads, what the path is)
DATASETS = {
    "data/semantic/asset_card.json": "semantic layer fixtures (committed to the repo)",
    "data/paderborn":                "Paderborn KAT bearing set (CC BY-NC 4.0)",
    "data/hydraulic/profile.txt":    "ZeMA hydraulic rig via UCI",
    "data/cwru":                     "Case Western bearing set",
    "data/ottawa":                   "uOttawa variable-speed set",
    "data/kaist":                    "KAIST varying-speed bearing set, subset 1",
}

NEEDS = {
    "cbmx demo / physics":  [],
    "tools/eval_semantic":  ["data/semantic/asset_card.json"],
    "tools/eval_latency":   ["data/paderborn"],
    "tools/eval_dataset":   ["data/paderborn"],
    "tools/eval_mcsa":      ["data/paderborn"],
    "tools/eval_current_band": ["data/paderborn"],
    "tools/eval_transient": ["data/paderborn"],
    "tools/bench_edge":     ["data/paderborn"],
    "tools/eval_hydraulic": ["data/hydraulic/profile.txt"],
    "tools/eval_prognosis": ["data/hydraulic/profile.txt"],
    "tools/eval_cwru":      ["data/cwru"],
    "tools/eval_actuator":  ["--data given on the command line"],
    "tools/eval_kaist":     ["data/kaist"],
}


def main() -> int:
    bad = 0
    print("\ncbmx preflight\n" + "=" * 62)

    v = sys.version_info
    good = (v.major, v.minor) >= (3, 9)
    print("%s python %d.%d.%d   (needs >= 3.9)"
          % (OK if good else NO, v.major, v.minor, v.micro))
    bad += 0 if good else 1

    if (v.major, v.minor) == (3, 9):
        print("%s python 3.9 works, but pip will resolve older numpy/scipy/pandas"
              % WARN)
        print("        pins, because current wheels need 3.10+. If you have a")
        print("        newer python, prefer it.")

    print("%s running inside a virtualenv"
          % (OK if sys.prefix != sys.base_prefix else WARN))
    if sys.prefix == sys.base_prefix:
        print("        not fatal, but 'pip install -e .' will touch system python")

    # pip gained PEP 660 (editable installs driven from pyproject.toml alone) in
    # 21.3. macOS ships 21.2.4 — one release below the line — and fails with
    # 'File "setup.py" or "setup.cfg" not found'. Catch it here rather than
    # letting it look like a broken repository.
    try:
        import pip
        pv = tuple(int(x) for x in pip.__version__.split(".")[:2])
        if pv < (21, 3):
            print("%s pip %s is TOO OLD for 'pip install -e .'" % (NO, pip.__version__))
            print("        pip 21.3+ is required (PEP 660). Fix, in this venv:")
            print("            python -m pip install --upgrade pip setuptools wheel")
            bad += 1
        else:
            print("%s pip %s" % (OK, pip.__version__))
    except Exception:
        print("%s could not determine the pip version" % WARN)

    for mod, minv in REQUIRED:
        try:
            m = importlib.import_module(mod)
            print("%s %-8s %s" % (OK, mod, getattr(m, "__version__", "")))
        except ImportError:
            print("%s %-8s  -> pip install -e \".[dev]\"" % (NO, mod))
            bad += 1

    try:
        import cbmx                                            # noqa: F401
        print("%s cbmx importable" % OK)
    except ImportError:
        print("%s cbmx NOT importable -> pip install -e \".[dev]\"" % NO)
        bad += 1

    try:
        import pytest                                          # noqa: F401
        print("%s pytest" % OK)
    except ImportError:
        print("%s pytest (dev extra not installed)" % WARN)

    print("\ndatasets\n" + "-" * 62)
    have = {}
    for path, what in DATASETS.items():
        p = os.path.exists(path)
        have[path] = p
        print("%s %-30s %s" % (OK if p else NO, path, what))

    print("\nwhat you can run right now\n" + "-" * 62)
    for tool, needs in NEEDS.items():
        missing = [n for n in needs if n.startswith("data/") and not have.get(n)]
        if not needs:
            print("  RUN     %s" % tool)
        elif needs and not needs[0].startswith("data/"):
            print("  RUN     %-24s (%s)" % (tool, needs[0]))
        elif missing:
            print("  skip    %-24s needs %s" % (tool, ", ".join(missing)))
        else:
            print("  RUN     %s" % tool)

    print("\n" + "=" * 62)
    if bad:
        print("%d prerequisite(s) missing. Fix those first.\n" % bad)
        return 1
    print("Prerequisites satisfied. The no-dataset commands will run.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
