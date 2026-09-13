# cbmx — explainable condition-based maintenance at the edge

**Physics says where to look. The machine's own history says what is normal
there. A sequential evidence test says when it is worth speaking. And where an
operating point has never been observed healthy, it says so instead of guessing.**

Feasibility codebase behind the ARRESTLINE submission to **DON26BZ05-DV087**
(Navy, NAWCAD Lakehurst — edge-deployed explainable CBM+ for carrier-based
systems). Closes **23 September 2026 at noon ET** — noon, not end of day.

Every number in the proposal traces to a command in here.

---

## Why it is built this way

Component geometry — ball count, ball diameter, pitch diameter, contact angle —
predicts the characteristic *orders* at which a defect on each part appears;
shaft speed converts an order into a frequency. So the search is **one line
wide**, and a hit means something, **before any data from the machine has been
seen**.

Without that constraint a monitor has tens of thousands of numbers per cycle and
a handful of labels, and whatever it finds will be significant at some threshold.
That is not detection. It is a multiple-comparisons machine with a story attached.

Data is only needed for the second question — *how much energy at that place is
too much* — and that is learnable from the asset itself, online, unlabelled,
while it is healthy. No labelled failure. No fleet model. No training run.

---

## Layout

```
src/cbmx/
  physics/bearing.py     geometry -> fault orders; catalogue with cited sources
  physics/envelope.py    band commissioning, envelope spectrum, order tracking
  physics/mcsa.py        motor current: sidebands, supply-harmonic guard,
                         torsional-band commissioning
  physics/hydraulic.py   fluid-power symptoms, each with a DECLARED direction
  health/baseline.py     per-regime robust baselines, and the freeze
  health/evidence.py     sequential test with bearing-specific channels
  health/sequential.py   the same test with the bearing specifics removed
  graph/asset.py         asset tree to the Lowest Replaceable Unit
  io/                    one loader per public dataset, normalising to Record
  report.py              the evidence record that crosses the link
  monitor.py, cli.py     the pipeline and its entry point

tools/                   one script per experiment; each prints what it did
tests/                   110 tests, including a regression for every silent bug
scripts/                 tidy.py (lint autofix), used before every commit
data/semantic/           synthetic asset card + mock work orders (committed)
```

New loaders since the first commit: `io/kaist.py` (varying speed, with
tachometer alignment), `io/ims.py` (NASA run-to-failure), `io/ottawa.py`,
`io/actuator.py`. New tools: `eval_kaist.py`, `eval_runtofailure.py`,
`bench_kaist.py`, `eval_actuator.py`, `preflight.py`.

---

## Install and test

```bash
pip install -e ".[dev]"
PYTHONPATH=src python3 -m pytest tests/ -q       # 108 passed, 4 skipped
PYTHONPATH=src python3 -m flake8 src/cbmx tools tests  # 0 findings
```

**108 passed, 4 skipped on a fresh clone with no datasets at all.** With the
hydraulic and Paderborn sets present it is 110 and 2: the four that skip are the
ones that need a real download, and they say which one is missing rather than
failing. Every loader test builds its file layout in `tmp_path`, so a loader
regression is caught on the machine that made it.

---

## The datasets are not in this repository

About 1.9 GB together, each redistributable only under its own licence. Download
them and put them where the loaders expect:

| Set | What it gives | Where | Licence |
|---|---|---|---|
| Paderborn (KAT) | Real accelerated-lifetime damage; **synchronous vibration and motor current at 64 kHz** | `data/paderborn/<CODE>/*.mat` | **CC BY-NC 4.0 — non-commercial**; contact the authors for other use |
| ZeMA hydraulic (UCI) | 2,205 cycles; pressure, flow, temperature, motor power; four components at graded severity | `data/hydraulic/*.txt` + `profile.txt` | UCI ML Repository terms |
| Case Western | Seeded bearing faults, vibration | `data/cwru/*.mat` | CWRU Bearing Data Center terms |
| Ottawa | Variable-speed bearing data | `data/ottawa/*.mat` | Mendeley Data terms |
| KAIST | Varying speed, 613–2480 rpm, real seeded damage, tachometer | `data/kaist/*.csv` | Mendeley 10.17632/vxkj334rzv |
| NASA/IMS | Run to failure over days; 4 bearings per shaft, 3 of them controls | `data/ims/<test>/<timestamp>` | NASA Prognostics Center of Excellence |

**XJTU-SY is deliberately not used.** It is the better-known run-to-failure set
and is hosted on a personal website in the PRC with Baidu Netdisk among its
mirrors. `io/xjtu.py` exists so the evaluator's `--dataset` flag is honest about
what it will not load; see `PROGNOSIS_STOPPING_RULE.md`.

Paderborn ships as RAR. `7zip-rar` **segfaults** on these archives and `unar`
will not install on current Debian/Ubuntu; use **`bsdtar`** from
`libarchive-tools`, which extracts each in about nine seconds. `bsdtar` also
streams, so it extracts every file fully contained in a *truncated prefix* of an
archive — useful on a slow link, and verified byte-identical against a full
extraction.

The first hydraulic run converts the text matrices to `.npy` and memory-maps
them afterwards. That cache is gitignored.

---

## Reproducing every number in the proposal

```bash
# everything at once: runs every dataset present, prints got-vs-expected
bash verify.sh

# real damage at fleet scale, VIBRATION, leave-one-BEARING-out
PYTHONPATH=src python3 tools/eval_dataset.py --dataset paderborn \
    --data data/pu_N15_M07_F10 --holdout bearing --real-damage-only --band 500 2000

# the same run with the band chosen automatically — it detects NOTHING
PYTHONPATH=src python3 tools/eval_dataset.py --dataset paderborn \
    --data data/pu_N15_M07_F10 --holdout bearing --real-damage-only

# varying speed: order tracking against fixed speed, on the same records
PYTHONPATH=src python3 tools/eval_kaist.py --data data/kaist --bearing KAIST6205U

# run to failure, both scopes; the stopping rule decides what may be claimed
bash run_ims.sh

# the governance layer on a machine with no bearings in it
PYTHONPATH=src python3 tools/eval_hydraulic.py
PYTHONPATH=src python3 tools/eval_hydraulic.py --no-freeze     # the ablation
PYTHONPATH=src python3 tools/eval_hydraulic.py --time-split    # abstention

# MOTOR CURRENT: why the obvious method fails, and the band that works
PYTHONPATH=src python3 tools/eval_mcsa.py
PYTHONPATH=src python3 tools/eval_current_band.py

# the arrestment transient, and how well speed must be known
PYTHONPATH=src python3 tools/eval_transient.py

# prognosis precondition, and honest false-alarm bounds
PYTHONPATH=src python3 tools/eval_prognosis.py

# the two stated thresholds, measured
PYTHONPATH=src taskset -c 0 python3 tools/bench_edge.py
PYTHONPATH=src taskset -c 0 python3 tools/eval_latency.py

# the semantic layer and the gate that discards its output — no datasets needed
PYTHONPATH=src python3 tools/eval_semantic.py
```

---

## Results, as measured

| | |
|---|---|
| Real damage, **vibration**, 17 bearing codes, leave-one-bearing-out | **0 false alarms** on 400 healthy records across 5 unseen bearings at 4 conditions, and on 120 across 6 at the rated condition |
| Detection by damage extent | grade 2 or 3: **every bearing, every record**. Grade 1: **1 of 8 bearings**. The floor sits between them |
| Plastic-deformation damage | **0 of 4**, which the physics predicts: indentations do not produce repetitive impacts |
| Band chosen automatically instead of frozen | **0 of 890** detected, with a perfect false-alarm record. This is why the band is commissioned once |
| **Vibration against motor current**, same captures | 13/36 against **0/36** bearing-condition pairs; 12/12 against 0/12 at damage extent ≥2 |
| Valve drift and pump leakage, **pressure/flow/motor power** | detected at graded severity with no bearing physics present |
| Freeze ablation | without it: **0/108, 0/24, 0/9** fault episodes named. The monitor goes silent for ever |
| Abstention, unseen regimes | 83–87% *no baseline*, **0/668 false alarms** (≤0.45%) |
| Motor current, direct | **nothing** — z between −0.5 and +0.2. Torque at the fault order moves −0.2 dB while radial force moves +23.8 dB |
| Motor current, torsional band | z = +3.2 at rated load; ~16 captures to decide |
| Speed transient, **real** varying-speed data | inner race **9/9** record-folds, 0 alarms on 3 of 3 held-out healthy recordings, band-invariant across four bands |
| A second varying-speed rig (uOttawa) | **0 of 33** in four bands. The only set whose geometry we could not verify is the only one that detects nothing |
| Prognosis, NASA/IMS | **claim withdrawn.** Median life remaining at first call 1%, against a 10% threshold fixed before download. Half the surviving control bearings also alarmed |
| Latency, one core | L1 1.7 ms · **L2 252 ms** · L3 627 ms |
| Evidence record | **873 bytes**; 0.052 MB/hour at 60 reports/hour; 2,347× smaller than the capture |

Failures sit beside successes deliberately. A feasibility case reporting only
what worked says nothing about where the method's edges are.

---

## Two bugs worth knowing about, both silent

Both have regression tests in `tests/test_v2_modules.py`, and each is the reason
for a design rule rather than a footnote.

**A decimation helper reset the bearing geometry.** `align_rate` rebuilt each
`Record` positionally and stopped four fields short, so every record reverted to
the default nine-ball geometry — applied to an eight-ball bearing. Every fault
line landed on empty spectrum. The run reported **0 detections and 0 false alarms
across 122 records**, which reads like an admirably conservative monitor. It also
deleted `rpm_series`, which is what all order tracking reads.

> **Never read a false-alarm rate without the detection rate beside it. A monitor
> that never speaks passes a false-alarm test perfectly.**

**Two functions with one name returning opposite pairs.**
`EnvelopeSpectrum.prominence_db` returns `(order, dB)`;
`mcsa.Spectrum.prominence_db` returns `(dB, frequency)`. Unpacked the wrong way,
an order of 3.05 was read as 3.05 dB, and every condition returned the same
number.

> **A result that does not move when the inputs move is not a result.**

---

## What this is not

Not an autonomous maintenance authority — it recommends, people act. No run-time
band selection: the band is commissioned once and frozen, because a detector free
to pick its best band each window will eventually find a fault in a healthy
machine. On the healthy reference bearing one sideband reads 18.7 dB, higher than
the damaged bearing produces at its own resonance.

`α` and `β` set the **operating point** of the sequential test. The accumulated
quantity is a bounded three-channel evidence score, **not** a calibrated
log-likelihood ratio from fitted densities, so `α` does not by itself guarantee
an operational false-alarm probability. Measured rates with exact one-sided 95%
upper bounds are what may be quoted. Calibrating the increments is Phase II work.
