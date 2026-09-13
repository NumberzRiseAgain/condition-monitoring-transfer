#!/usr/bin/env python3
"""E2 -- the synthetic end-to-end demo, captured as a result file.

Runs `cbmx demo` (commission from geometry, learn normal from the machine's own
healthy signal, then plant each fault in turn) with a fixed seed, keeps the whole
transcript, and parses the four contract numbers the demo prints into JSON so the
evidence pack can hash them.

Synthetic signals only. These numbers show the pipeline runs end to end at the
stated cadence and link budget; they are not a claim about real machinery.

    PYTHONPATH=src python3 run_e2_demo.py --seed 101 --json ../results/e2_demo.json
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from cbmx import cli  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--json", required=True)
    ap.add_argument("--transcript", default="")
    a = ap.parse_args()

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = cli.main(["demo", "--seed", str(a.seed)])
    text = buf.getvalue()
    sys.stdout.write(text)
    if a.transcript:
        Path(a.transcript).write_text(text)

    def grab(pattern, cast=float):
        m = re.search(pattern, text)
        return cast(m.group(1)) if m else None

    out = {
        "experiment": "E2 synthetic end-to-end demo (cbmx demo)",
        "seed": a.seed,
        "synthetic": True,
        "exit_code": rc,
        "latency_worst_s": grab(r"latency\s+worst\s+([0-9.]+)\s*s"),
        "latency_target_s": 1.0,
        "link_peak_mb_per_hour": grab(r"link\s+peak\s+([0-9.]+)\s*MB/hour"),
        "link_target_mb_per_hour": 10.0,
        "lru_isolation": grab(r"LRU isolation\s+(\d+/\d+)", str),
        "quiet_machine_bytes": grab(r"quiet machine\s+(\d+)\s*bytes", int),
        "quiet_machine_reports": grab(r"quiet machine\s+\d+\s*bytes,\s*(\d+)\s*reports", int),
        "chain_verifies": ("chain verifies: True" in text),
        "note": "Synthetic signals. Shows the pipeline runs end to end within the "
                "topic's latency and link budgets; not a measurement on real machinery.",
    }
    Path(a.json).write_text(json.dumps(out, indent=2))
    ok = rc == 0 and out["latency_worst_s"] is not None and out["link_peak_mb_per_hour"] is not None
    print(f"\nwrote {a.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
