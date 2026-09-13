# cbmx run log — 30 August 2026 — NASA/IMS run-to-failure

`tools/eval_runtofailure.py`, both scopes, via `run_ims.sh`. Console output in
`runs/ims_readme.log` and `runs/ims_full.log`; per-bearing detail in the
matching `.json`. Census in `runs/ims_census.log`.

Data: Center for Intelligent Maintenance Systems, University of Cincinnati,
published through the NASA Prognostics Center of Excellence. Three
test-to-failure experiments, four Rexnord ZA-2115 bearings per shaft, 2000 rpm,
6000 lb radial load, 1.024 s snapshots at 20 kHz roughly every ten minutes.
9,464 captures on disk; 7,588 in the documented scope.

---

## 0. What was declared before the data was downloaded

`PROGNOSIS_STOPPING_RULE.md`, written 29 August:

- **If the median life remaining at first call is below 10%, prognosis is not
  claimed.**
- The alarm threshold will not be lowered until a number appears.
- The persistence requirement (three consecutive captures) will not be relaxed.
- No bearing will be dropped.
- The declared-physics half will not run without a ZA-2115 catalogue entry taken
  from the archive's own documentation.

All five held. Nothing below was tuned.

---

## 1. The result

**The rule fired.** Median life remaining at first call is **1%** in the
documented scope and **2%** across the full folder. Both are far below the 10%
threshold, so **prognosis is not claimed**. The volume says, in these words or
close to them: *on this data the indicator is a failure detector, not a
prognostic*, and remaining-useful-life stays a Phase II task.

### Documented scope — the readme's own experiment (primary)

| unit | what | caps | life h | rho(rms) | first call | lead | % left |
|---|---|---:|---:|---:|---:|---:|---:|
| 1st_test/b1 | control | 2156 | 827.6 | 0.65 | never | — | — |
| 1st_test/b2 | control | 2156 | 827.6 | 0.55 | 2119 | 12.9 h | 2% |
| 1st_test/b3 | **inner_race** | 2156 | 827.6 | 0.76 | 2127 | 11.5 h | 1% |
| 1st_test/b4 | **rolling_element** | 2156 | 827.6 | 0.79 | 2152 | 0.4 h | 0% |
| 2nd_test/b1 | **outer_race** | 984 | 163.8 | 0.81 | 619 | **60.7 h** | **37%** |
| 2nd_test/b2 | control | 984 | 163.8 | 0.39 | 950 | 5.5 h | 3% |
| 2nd_test/b3 | control | 984 | 163.8 | 0.14 | 970 | 2.2 h | 1% |
| 2nd_test/b4 | control | 984 | 163.8 | 0.75 | 965 | 3.0 h | 2% |
| 3rd_test/b1 | control | 4448 | 753.6 | −0.21 | never | — | — |
| 3rd_test/b2 | control | 4448 | 753.6 | −0.86 | never | — | — |
| 3rd_test/b3 | **outer_race** | 4448 | 753.6 | 0.53 | **never** | — | — |
| 3rd_test/b4 | control | 4448 | 753.6 | 0.72 | never | — | — |

Spoke on **3 of 4** documented failures. Lead time median **11.5 h**, range
0.4–60.7 h. Life remaining median **1%**, range 0–37%. Median rho(rms) 0.60.

### Full folder — sensitivity check

Spoke on 4 of 4; lead time median 19.2 h; life remaining median **2%**;
controls spoke on 7 of 8. Set 3's outer-race failure is detected here, at 3% of
life remaining, and two of its three shaft-mates call within an hour of it.

**The two scopes agree on the verdict.** The scope question was settled from the
documentation before scoring precisely so it could not be settled by the answer,
and in the event it did not change the answer. Worth recording: the care was
still correct, and it cost nothing.

---

## 2. The controls, which is where this gets interesting

Four bearings share each shaft and the readme names which one failed, so eight
bearings survived and are controls. This is a correction to what
`eval_runtofailure.py` originally asserted — that run-to-failure data offers no
control at all.

**Controls spoke on 4 of 8 in the documented scope, 7 of 8 in the full folder.**

The geometry-free indicators are not merely late. They are **not specific to the
bearing that failed**. In set 2, all three surviving bearings call within an hour
of each other near the end of a test where only bearing 1 was damaged. In set 3's
full scope, bearings 1, 2 and 4 all call alongside bearing 3.

### An honest limit of our own metric

The report separates "before the failure on its shaft spoke" (a false alarm)
from "after" (possible structural transmission). On this data that boundary is
too sharp for what it is measuring:

- `1st_test/b2` calls at 814.7 h; the inner-race failure on its shaft calls at
  816.1 h. A margin of **1.4 hours** out of 828.
- `3rd_test/b4` (full scope) calls at 1046.3 h against b3's 1046.4 h — **six
  minutes**.

Classifying those as unambiguous false alarms rather than cross-talk is an
artefact of capture ordering, not a finding. The defensible statement is the
weaker and truer one: **near end of life these indicators light up across the
whole shaft, and which bearing crosses first is within the noise.**

---

## 3. What this is evidence for

Not for our prognostic claim — there isn't one, and the stopping rule is why.

It is evidence about the **generic condition-indicator approach**: RMS,
kurtosis and spectral crest, fitted per asset and thresholded. That is what most
condition-monitoring products ship, and on the best-documented public
run-to-failure bearing data it (a) announces failure in the last 1–2% of life
and (b) cannot say which of four bearings is the bad one.

Both are the failures the declared-physics method exists to avoid, and §3.1's
Paderborn result is the direct contrast: 0 false alarms of 400 healthy records
across unseen bearings, 0 wrong-part namings, from orders declared by geometry
rather than from an energy threshold.

**The limitation, stated plainly.** We cannot run our own method on this data.
The readme does not publish the ZA-2115's geometry and the stopping rule forbids
inventing it, so what IMS shows is that the generic approach falls over here —
not that ours would not. The head-to-head is Paderborn's, on Paderborn's data.
Claiming more than that from this run would be exactly the move this evidence
base exists to refuse.

---

## 4. Data provenance and handling

- `4.+Bearings.zip` → `4. Bearings/IMS.7z` → `1st_test.rar`, `2nd_test.rar`,
  `3rd_test.rar`, `Readme Document for IMS Bearing Data.pdf`. Three levels.
- `3rd_test.rar` unpacks to a folder named **`4th_test/txt/`**. Nothing was
  renamed on disk; `ims._TEST_ALIASES` maps it to the readme's Set No. 3.
  Unmapped, the label for `3rd_test/3` never matches and the one bearing that
  failed in set 3 is silently scored as a control.
- Census matched the published counts exactly for sets 1 and 2. Set 3 ships
  6,324 captures for an experiment the readme documents as 4,448; clipping to
  the readme's stated end date yields **exactly 4,448**.
- No truncated captures (`ims.short_captures`).
- Failure labels transcribed verbatim from the readme, not inferred from signals.
- 96 unit tests pass, 12 of them new for this loader.
