#!/usr/bin/env python3
"""Write results/manifest.json: what ran, on what, from which sources.

The run identifier is the first 16 hex digits of sha256 over the sorted source
digests plus the seed, so the same sources and the same seed always give the same
identifier, and a changed line anywhere in src/, tools/ or tests/ gives a new one.

    python3 manifest.py --seed 101 --results ../results --status ../results/_run_status.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_GLOBS = ["src/**/*.py", "tools/*.py", "tests/*.py", "run_e2_demo.py",
                "reproduce.sh", "manifest.py", "requirements.txt", "configs/*.yaml"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--status", required=True,
                    help="JSON written by reproduce.sh: experiments run / skipped, datasets present")
    a = ap.parse_args()
    results = Path(a.results)

    sources = {}
    for g in SOURCE_GLOBS:
        for p in sorted(HERE.glob(g)):
            if p.is_file() and "__pycache__" not in p.parts:
                sources[str(p.relative_to(HERE))] = sha256(p)
    h = hashlib.sha256()
    for k in sorted(sources):
        h.update(f"{k} {sources[k]}\n".encode())
    h.update(f"seed {a.seed}\n".encode())
    run_id = h.hexdigest()[:16]

    libs = {}
    for name in ("numpy", "scipy", "pandas", "yaml", "pytest"):
        try:
            mod = __import__(name)
            libs[name] = getattr(mod, "__version__", "?")
        except Exception:
            libs[name] = "absent"

    status = json.loads(Path(a.status).read_text()) if Path(a.status).is_file() else {}
    result_files = {p.name: sha256(p) for p in sorted(results.glob("e*.json"))}

    manifest = {
        "study": "cbmx -- explainable condition-based maintenance at the edge (ARRESTLINE, DON26BZ05-DV087)",
        "run_identifier": run_id,
        "seed": a.seed,
        "written_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": libs,
        "experiments_run": status.get("run", []),
        "experiments_skipped_for_missing_data": status.get("skipped", []),
        "datasets_present": status.get("datasets_present", []),
        "datasets_absent": status.get("datasets_absent", []),
        "result_files": result_files,
        "prior_runs_carried_forward": "results/prior_runs_2026-08/ (August 2026) and results/fresh_2026-09-09/ "
                                      "(9 September 2026, the run on the machine that held the datasets) -- outputs "
                                      "and logs from this same code, NOT part of this run, hashed separately in "
                                      "results/_prior_run_digests.txt",
        "source_digests": sources,
    }
    (results / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest.json written: run_identifier {run_id}, seed {a.seed}, "
          f"{len(sources)} source files, {len(result_files)} result files, "
          f"{len(manifest['experiments_run'])} experiments run, "
          f"{len(manifest['experiments_skipped_for_missing_data'])} skipped for missing data")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
