# Every parameter, and which of them move between datasets

A reviewer's first question about any result that spans four datasets is: what
did you tune on each one? This is the answer, in a form that can be checked
against the source rather than taken on trust. File and line references are to
this repository as of 29 August 2026.

The short version: **the parameters that decide whether to speak are the same
everywhere and live in one file. The parameters that describe the data differ
between datasets, and each difference has a reason stated here.** Claiming
nothing at all differs would be false, and would be found out in five minutes.

---

## 1. The decision parameters — identical across every dataset

None of these has ever been given a per-dataset value. They are class
attributes on the two evidence classes and there is no command-line flag that
changes them.

| parameter | value | where it lives | what it does |
|---|---|---|---|
| α, tolerated false-alarm rate | **0.02** | every tool | with β, sets the upper Wald boundary |
| β, tolerated miss rate | **0.10** | every tool | with α, sets the lower Wald boundary |
| upper boundary | **+3.81 nats** | derived, `log((1−β)/α)` | not a threshold anyone chose |
| lower boundary | **−2.28 nats** | derived, `log(β/(1−α))` | the exoneration boundary |
| `Z_OPERATING` | **3.0** | `health/evidence.py:82`, `health/sequential.py:169` | robust sigmas above baseline before a reading counts toward a fault |
| `W_LEVEL` | **1.0** | `evidence.py:75`, `sequential.py:97` | weight on the level channel |
| `W_SIDEBAND` / `W_DIRECTION` | **0.45** | `evidence.py:76`, `sequential.py:98` | weight on the physics-direction channel |
| `W_AGREEMENT` / `W_CORROBORATION` | **0.35** | `evidence.py:77`, `sequential.py:99` | weight on the corroboration channel |
| `K`, nats per unit of evidence | **0.55** | `evidence.py:83`, `sequential.py:101` | converts combined evidence to a log-likelihood step |
| level saturation | **[−1.0, +1.5]** | `evidence.py:102` | stops one bad window forcing a call |
| robust sigma floor | **0.5** | `baseline.py`, `Running.z` | never divide by a fluke |
| MAD → sigma factor | **1.4826** | `baseline.py`, `Running.z` | Gaussian equivalence |

α and β are the only two a maintainer is expected to argue about, and
everything else about when to speak follows from them. That is the property to
point at when somebody asks how the sensitivity was chosen.

## 2. The physics — declared per machine, never fitted

Fault orders come from four caliper dimensions through
`physics/bearing.py`, and the symptom set and its directions are written down
before any scoring. Nothing in either is selected by looking at label
separation. What differs per dataset is the *machine*, which is the point:

| dataset | geometry source | verified? |
|---|---|---|
| Paderborn KAT | Lessmeier et al. 2016, Table 1 — the dataset's own paper | yes |
| CWRU | CWRU bearing data centre | yes, against published multipliers |
| MFPT | MFPT documentation | yes, reproduces MFPT's own 81.12 / 118.875 Hz |
| uOttawa | ER-16K manufacturer dimensions | **no** — flagged UNVERIFIED |
| KAIST | assumed 6205; the dataset publishes none | **no** — flagged UNVERIFIED |
| VT/NSWC actuator | not a bearing; 8 declared hydraulic symptoms | n/a |

The two UNVERIFIED entries carry that word in the catalogue `source` string, a
test asserts it survives, and the tools print a banner. See
`tests/test_kaist.py::test_assumed_geometry_is_flagged_unverified`.

## 3. The data-handling parameters — these do differ

Sample rates differ by a factor of forty across these sets and record lengths
by a factor of three hundred. A window length that is correct for one is wrong
for another. Each value below is a statement about the data, not about the
sensitivity of the test.

| | Paderborn | CWRU | ZeMA hydraulic | VT/NSWC actuator | KAIST |
|---|---|---|---|---|---|
| sample rate | 64 kHz | 12/48 kHz | 1–100 Hz | 1 kHz | 25.6 kHz |
| observation | 0.75 s window | 0.75 s window | one 60 s duty cycle | one actuation cycle | 1.0 s window |
| `min_samples` | 30 | 30 | 25 | 25 | 25 |
| regime variable | speed 5 Hz × load 0.25 | speed × load | duty cycle | one bin per matched-load experiment | speed, 10 Hz bins |
| demodulation band | 0.5–2 kHz, frozen | commissioned, frozen | n/a, no envelope | n/a | 11.2–12.8 kHz, commissioned on healthy, frozen |

**`min_samples` is 25 on three sets and 30 on two, and that is a real
difference.** The `Running.ready` docstring in `health/baseline.py` records
why it is a stated parameter rather than a constant: the CWRU healthy records
give about ten seconds per load, which lands just under a hard-coded 30 and
would silently disable every line. A readiness threshold that the data cannot
meet should be visible in a signature, not buried. Raising or lowering it makes
the system *abstain* more or less; it does not make it more or less willing to
name a fault, because the boundaries are set by α and β alone.

**The regime for the actuator is one bin, and that is a concession, not a
choice.** `Internal_Temp` is constant within each actuator and different
between them, so binning on it made the regime a proxy for the unit and the
system abstained on 100% of targets. The 2.7 °C difference between reference
and target is reported as an uncontrolled covariate. `eval_actuator.py:51`.

**The KAIST band is commissioned rather than fixed**, and it was checked
against the alternative: the same run in the two runner-up bands and in the
full 0.5–12.8 kHz with no selection at all gives the identical verdict — inner
9/9, outer 0/9, zero alarms. So the band is not producing the result, and it is
also not a hidden knob that could. `RUNLOG_2026-08-29_kaist.md` §4.

## 4. What was changed after seeing a result, and what was not

Being able to answer this precisely is worth more than a claim of purity.

**Changed after seeing results, and disclosed:**

- The actuator stroke segmentation, after transit time read 2.99 s of a 3.00 s
  capture — a coding defect, corrected, not a tuning change.
- The actuator regime, from temperature bins to one bin, for the reason above.
- The KAIST tachometer offset, once the first run detected nothing. The
  correction is fitted per record but on the *shaft* line, which exists
  regardless of health, and the identical procedure runs on healthy and damaged
  recordings.

**Not changed, on principle:**

- The KAIST geometry. The assumed 6205 was declared once and kept, and the
  outer-race result is a failure under that assumption. Revising geometry until
  the peaks agree would be fitting physics to labels.
- The declared symptom directions. `angle_jerk` on the actuator was declared +1
  and measured −2.27; that is reported as a wrong declaration, not corrected
  into a right one.
- α and β, anywhere, ever.

## 5. Reproducing this table

```bash
grep -n "Z_OPERATING\|W_LEVEL\|W_SIDEBAND\|W_AGREEMENT\|K = " src/cbmx/health/evidence.py
grep -n "MIN_SAMPLES\|ALPHA\|BETA" tools/eval_*.py
grep -n "alpha\|beta\|min-samples\|window" tools/eval_kaist.py | head -20
```

Anything the table claims is identical should appear once. Anything it claims
differs should appear with the value shown.
