#!/usr/bin/env python3
"""One table across every dataset that ran. Written for pasting into a volume."""
from __future__ import annotations

import json
from pathlib import Path

ORDER = ["cwru_line", "cwru_family", "mfpt_line", "mfpt_family",
         "ottawa_family", "paderborn_vib", "paderborn_cur"]


def main() -> int:
    runs = Path("runs")
    found = []
    for j in sorted(runs.glob("*.json")):
        try:
            found.append((j.stem, json.loads(j.read_text())))
        except Exception:
            continue
    if not found:
        print("No results in runs/. Nothing completed.")
        return 1
    found.sort(key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99)

    print("=" * 92)
    print("CBMX — PUBLIC DATASET RESULTS")
    print("=" * 92)
    print(f"\n{'run':<20}{'bearing':<26}{'correct':>9}{'judged':>8}"
          f"{'WRONG':>7}{'false alarms':>14}{'var-speed':>11}")
    print("-" * 92)
    for name, d in found:
        judged = d["total"] - d.get("declined", 0)
        print(f"{name:<20}{d.get('bearing','')[:24]:<26}"
              f"{d['correct']}/{d['total']:<7}{judged:>8}"
              f"{d.get('wrong_part','?'):>7}"
              f"{str(d.get('false_alarms','?'))+'/'+str(d.get('healthy','?')):>14}"
              f"{d.get('variable_speed_records',0):>11}")
    print()
    print("WRONG = a named part that was the wrong part. For a maintenance system")
    print("that is the number that costs money: it pulls a good LRU. 'nothing' is")
    print("a miss, which costs nothing but time.")
    print()
    tot_fa = sum(d.get("false_alarms", 0) for _, d in found)
    print(f"Total false alarms across every healthy record, every dataset: {tot_fa}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
