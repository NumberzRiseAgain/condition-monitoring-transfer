#!/usr/bin/env python3
"""Does the failing bearing speak before its shaft-mates?

    python3 tools/eval_rtf_discrimination.py --json results/e16_ims_readme.json \
                                             --json results/e16_ims_full.json \
                                             --out  results/e16_ims_discrimination.json

`eval_runtofailure.py` reports a lead time against the end of a bearing's own
life. That answers "how much warning" and it does not answer the question an
evaluator asks next: **was the warning about this bearing, or about the rig?**

On IMS four bearings share one shaft and one file. A defect metres of steel away
still shakes the accelerometer. So an indicator with no bearing-level
discrimination at all will still post a good lead time on the failing unit,
because the whole shaft goes loud together near the end of a test and the
failing unit is somewhere in that crowd.

The statistic here is the one that separates those two worlds, and it needs no
new data because it is an ordering, not a measurement: **within each test, did
the documented failure raise its call before every control on the same shaft?**

- Failure leads, by a margin large against the test's own length: the call is
  about the bearing. This is detection.
- A control leads, or the whole shaft calls within a few captures: the call is
  about the rig. This is an end-of-test alarm wearing a detector's clothes, and
  the lead time on the failing unit is not evidence of anything.

The margin is reported in captures and in hours, signed. Negative means a
surviving bearing spoke first. Nothing here is swept or tuned; it reads the
call indices already written by the run and sorts them.

**The hours come from `lead_hours`, not from a capture count.** IMS does not
sample on a fixed interval — set 1 changes cadence partway through — so
converting a capture margin with a mean spacing is wrong, and wrong by more
than a factor of two on set 1. Each unit's `lead_hours` is measured from the
capture timestamps by the run itself, so the difference between two units'
lead times is a true wall-clock margin. The capture count is still reported
beside it because it says how many observations separated the two calls, which
the hours alone do not.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def by_test(units):
    t = {}
    for u in units:
        t.setdefault(u["unit"].split("/")[0], []).append(u)
    return t


def analyse(doc):
    out = []
    for test, us in sorted(by_test(doc["units"]).items()):
        caps = us[0]["captures"]
        life = us[0]["life_h"]
        called = sorted([u for u in us if u["first_call"] is not None],
                        key=lambda u: u["first_call"])
        fails = [u for u in us if u["is_failure"]]
        row = {"test": test, "captures": caps, "life_h": life,
               "failures": [u["unit"] for u in fails],
               "n_called": len(called), "n_units": len(us)}
        if not called or not fails:
            row.update(discriminates=None, first_caller=None, margin_captures=None,
                       margin_h=None, note="no call in scope" if not called
                       else "no documented failure in this test")
            out.append(row)
            continue
        first = called[0]
        first_fail = min((u for u in called if u["is_failure"]),
                         key=lambda u: u["first_call"], default=None)
        first_ctrl = min((u for u in called if not u["is_failure"]),
                         key=lambda u: u["first_call"], default=None)
        row["first_caller"] = first["unit"]
        row["first_caller_is_failure"] = bool(first["is_failure"])
        if first_fail is None:
            row.update(discriminates=False, margin_captures=None, margin_h=None,
                       note="the documented failure never spoke; only controls did")
        elif first_ctrl is None:
            row.update(discriminates=True, margin_captures=None, margin_h=None,
                       note="every control stayed silent for the whole test")
        else:
            m = first_ctrl["first_call"] - first_fail["first_call"]
            # Signed wall-clock margin from the run's own timestamp-derived
            # lead times: failure leads by however much less life it had left.
            mh = first_fail["lead_hours"] - first_ctrl["lead_hours"]
            row.update(discriminates=bool(m > 0), margin_captures=int(m),
                       margin_h=round(mh, 2),
                       note=("failure led the first control by "
                             f"{m} captures" if m > 0 else
                             f"a surviving bearing led the failure by {-m} captures"))
        row["call_order"] = [
            {"unit": u["unit"], "role": "failure" if u["is_failure"] else "control",
             "first_call": u["first_call"], "called_at_h": round(
                 u["life_h"] - u["lead_hours"], 2), "label": u["label"]}
            for u in called]
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", required=True,
                    help="a result file written by eval_runtofailure.py; repeatable")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    report = {}
    for p in a.json:
        doc = json.loads(Path(p).read_text())
        report[doc.get("scope", Path(p).stem)] = analyse(doc)

    for scope, rows in report.items():
        print(f"\n  scope: {scope}")
        print("  " + "-" * 74)
        print(f"  {'test':<10}{'first to call':<16}{'discriminates':<15}{'margin':<14}note")
        print("  " + "-" * 74)
        for r in rows:
            d = {True: "yes", False: "NO", None: "-"}[r["discriminates"]]
            marg = "-" if r["margin_captures"] is None else \
                f"{r['margin_captures']:+d} ({r['margin_h']:+.1f}h)"
            print(f"  {r['test']:<10}{str(r['first_caller'] or '-'):<16}{d:<15}{marg:<14}{r['note']}")
        ok = sum(1 for r in rows if r["discriminates"] is True)
        n = sum(1 for r in rows if r["discriminates"] is not None)
        print(f"\n  bearing-level discrimination on {ok} of {n} tests")
        if n and ok < n:
            print("  Where a control leads, the lead time on the failing unit is not")
            print("  evidence of bearing-level detection. Report it per test; a pooled")
            print("  median across tests hides exactly this.")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(report, indent=2, default=float))
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
