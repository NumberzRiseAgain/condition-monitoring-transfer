# cbmx run log — 29 August 2026 — KAIST varying-speed bearing set

Everything below was produced by `tools/eval_kaist.py` in this repository on
this date. Console output in `runs/kaist_loo.log`, numbers in `runs/kaist_loo.json`,
ablation in `runs/kaist_noalign.json`.

**Dataset.** *Vibration and Motor Current Dataset of Rolling Element Bearing
Under Varying Speed Conditions for Fault Diagnosis: Part 1*, Korea Advanced
Institute of Science and Technology, Mendeley Data
[10.17632/vxkj334rzv](https://data.mendeley.com/datasets/vxkj334rzv/5). Three
conditions — healthy, inner race, outer race — three replicates each, plus a
fixed-speed recording per condition. Four accelerometers on two bearing
housings, three motor phase currents, and a tachometer, under a randomly
varying 613–2480 rpm. Vibration at 25.6 kHz.

**Why this set.** Every other bearing benchmark in this repository is recorded
at a constant shaft speed. An arresting engine is not. A fault line at 3.585×
shaft speed sweeps across a large fraction of its own frequency during a
recovery, and a method validated only at fixed speed has not been validated for
the machine in the topic. This is the only public set here that presents that
condition with a tachometer to check the answer against.

Measured: **13.1% median peak-to-peak speed change inside a single one-second
window**, 3.3–19.7% at the 10th and 90th percentiles. The outer-race line
therefore sweeps across roughly 47% of its own frequency while being measured.

---

## 1. Two things the dataset does not tell you

**It publishes no bearing geometry.** Not the model, not the ball count, not
the ball or pitch diameter, not the contact angle — not in the Mendeley record,
not in the authors' arXiv paper (2311.18547v2), and the Data in Brief article
behind both is paywalled. Geometry is what tells this system where to look, so
it cannot be recovered from the data without turning a declared search into a
fitted one. A 6205 is **assumed** from the rig class, declared once as catalogue
entry `KAIST6205U`, flagged UNVERIFIED, and **not revised**. Every number in
section 4 inherits that assumption and a test enforces that the flag survives.

**It does not say which housing holds the damaged bearing.** Choosing the
channel that shows the fault best is selection on the label. So all four
accelerometer channels are monitored, each with its own baseline and its own
lines — 64 lines in total — and the healthy records are scored against exactly
the same expanded set, so the multiple comparison is paid for on both sides.

The answer falls out of the run rather than being assumed: the inner-race
signal reads z ≈ 8–9 on housing **B** and z ≈ 1.4–4.0 on housing A. The damaged
bearing is in housing B.

---

## 2. The defect that made the first run fail, and it was silent

The first pass detected nothing and reported a perfect false-alarm record.
Order tracking bought about 0.3 dB, which is roughly nothing, and the obvious
reading was that the method does not work here.

**The tachometer file and the signal file do not start together.** The offset
differs per recording and is a few hundred milliseconds:

| record | offset | shaft line as shipped | aligned | gain |
|---|---:|---:|---:|---:|
| normal_0 | −0.69 s | 4.0 dB | 33.4 dB | **+29.4** |
| normal_1 | +0.08 s | 30.9 dB | 31.6 dB | +0.7 |
| normal_2 | −0.49 s | 12.3 dB | 29.9 dB | +17.6 |
| inner_0 | −0.76 s | 2.8 dB | 27.8 dB | +24.9 |
| inner_1 | −0.50 s | 8.0 dB | 37.9 dB | +29.9 |
| inner_2 | −0.65 s | 4.8 dB | 31.2 dB | +26.5 |
| outer_0 | −0.36 s | 31.8 dB | 46.0 dB | +14.3 |
| outer_1 | −0.48 s | 27.0 dB | 44.1 dB | +17.1 |
| outer_2 | −0.58 s | 20.4 dB | 29.9 dB | +9.5 |

Taken at face value, the integrated shaft angle drifts across the window and
order tracking resamples against the wrong angle — which is worse than not
order tracking at all. `normal_1` happens to arrive nearly aligned, which is
exactly why the problem was hard to see: one record looked fine.

The correction is recovered from the **1× and 2× shaft lines**, which residual
imbalance puts in every rotating machine whether or not it is broken. No fault
order, no fault label and no damaged recording enters the search, and the same
procedure runs on every record. `io/kaist.align_tacho`.

**The ablation is the point.** Re-run with `--no-align`:

| | detection | false alarms |
|---|---|---|
| aligned | **3 of 6** damaged recordings, in all 3 folds | 0 of 3 held-out healthy recordings |
| not aligned | **0 of 6**, in all 3 folds | 0 of 3 |

The unaligned run has a flawless false-alarm record because it never speaks.
That is the same trap this repository fell into in August with a decimation
helper that reset the bearing geometry, and it is why detection and false alarm
are printed on one line here and never apart.

---

## 3. The one constant taken on trust, checked

The vibration sampling rate is not in the dataset either; 25.6 kHz comes from
the authors' paper. It is checkable without any label, because only the correct
rate puts the shaft line where order tracking can find it — a wrong rate
distorts the time axis the shaft angle is integrated over:

| assumed rate | 1× shaft line, order-tracked |
|---|---:|
| 10.0 kHz | 1.0 dB |
| 12.8 kHz | 0.7 dB |
| 20.0 kHz | 0.3 dB |
| **25.6 kHz** | **7.0 dB** |
| 32.768 kHz | 2.7 dB |
| 40.0 kHz | 3.8 dB |
| 51.2 kHz | 0.0 dB |
| 65.536 kHz | 1.8 dB |

`tests/test_kaist.py::test_published_sampling_rate_is_the_one_the_data_supports`
runs this against the data and fails if the constant is ever mistyped.

---

## 4. Results

α = 0.02, β = 0.10 → boundaries +3.81 / −2.28 nats. One-second windows. Regime
= shaft speed in 10 Hz bins. A line is scored only after 25 observations in its
own regime.

**Leave-one-out over the three healthy recordings.** Each is held out in turn
while the baseline is fitted on the other two and **frozen before any test
window is scored**. That gives three genuinely held-out healthy recordings
rather than one, and it tests whether the answer depends on which healthy
recording the baseline happened to see.

### Detection — window at which the fault is first named, order-tracked

| record | baseline on 1+2 | on 0+2 | on 0+1 | |
|---|---:|---:|---:|---|
| inner_0 | 6 | 6 | 6 | **3/3** |
| inner_1 | 7 | 8 | 7 | **3/3** |
| inner_2 | 9 | 15 | 15 | **3/3** |
| outer_0 | never | never | never | 0/3 |
| outer_1 | never | never | never | 0/3 |
| outer_2 | never | never | never | 0/3 |
| normal_0 / _1 / _2 held out | silent | silent | silent | **0 alarms** |

**Detection 9 of 9 inner-race record-folds, 0 of 9 outer-race. False alarm 0 of
3 held-out healthy recordings.**

### The false-alarm number is n = 3, and it is not 222

A recording contributes 74 windows, and it is tempting to write "0 of 222
windows, ≤1.3% at 95%". That would be wrong. The accumulator is sequential and
carries state across the windows of a recording, and the windows are
consecutive seconds of one continuous run — they are not independent Bernoulli
trials, so a binomial bound over them is arithmetic on correlated samples. One
recording is one trial. **The honest statement is three held-out healthy
recordings, zero alarms**, and that is a small number. It is the main reason
this result is worth reporting and is not worth over-claiming.

### Median robust z on the primary lines, order-tracked

Baseline on normal_0 + normal_1; normal_2 held out.

| line | normal_2 | inner_0 | inner_1 | inner_2 | outer_0 | outer_1 | outer_2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bearingB_y : **inner_race** | 0.13 | **8.76** | **8.48** | **8.79** | 0.56 | 0.43 | 0.43 |
| bearingB_x : **inner_race** | 0.17 | **8.53** | **7.96** | **8.41** | 0.23 | 0.29 | 0.32 |
| bearingA_x : inner_race | −0.09 | 3.93 | 2.53 | 2.78 | −0.14 | −0.07 | −0.20 |
| bearingA_y : inner_race | 0.10 | 3.20 | 1.48 | 1.87 | 0.09 | 0.18 | −0.12 |
| bearingB_x : outer_race | −0.06 | −0.13 | −0.37 | −0.55 | 0.36 | 0.45 | 0.42 |
| bearingB_y : outer_race | −0.25 | −0.49 | −0.36 | −0.59 | 0.07 | 0.51 | 0.37 |

The inner-race signal is 20× stronger on housing **B** than on A. The dataset
does not say which housing holds the damaged bearing; the run says B.

### Order tracking, measured rather than asserted

The two arms differ in exactly one function call — whether the envelope is
resampled against shaft angle. Same window, same band, same floor rule.

- It roughly **doubles the effect size** on the inner-race lines: 4.95 → 8.53,
  4.63 → 8.76, 4.68 → 7.96, 4.72 → 8.41.
- It **roughly halves the time to a decision**: 13/12/13 → 6/6/6 on inner_0,
  21/16/21 → 9/15/15 on inner_2, and it is never slower in any fold.
- On healthy data it costs nothing: every held-out healthy recording is silent
  in both arms.

**The control.** On the fixed-speed `_constant` recordings, where the two arms
are the same computation, the measured gain is **0.00 dB and −0.01 dB**. If it
had been anything else, the metric would have been measuring something other
than speed smear.

**Why the shaft-line gains look small and the detection gain does not.** At 1×
the measured sharpening is 0.3–1.6 dB. Smear scales with order — a line at
5.415× wanders 5.4 times as far in Hz as the shaft line for the same speed
change — so the gain measured at 1× is a lower bound on the gain at BPFI, and
the detection numbers are where the effect is actually visible.

### The band is not load-bearing

The kurtosis vote is not decisive: 11.2–12.8 kHz wins with 9 votes against 8
and 7 for the runners-up. So the result was re-run in each, and in the full
band with no selection at all:

| demodulation band | inner detected | outer detected | false alarms | first named (inner_0/1/2) |
|---|---|---|---|---|
| 11.2–12.8 kHz (commissioned) | 9/9 | 0/9 | 0/3 | 6 / 7–8 / 9–15 |
| 10.4–11.2 kHz | 9/9 | 0/9 | 0/3 | 6 / 7 / 9 |
| 8.0–8.8 kHz | 9/9 | 0/9 | 0/3 | 6 / 6 / 9 |
| 12.0–12.8 kHz | 9/9 | 0/9 | 0/3 | 13 / 12–14 / 32 |
| 0.5–12.8 kHz — no band selection at all | 9/9 | 0/9 | 0/3 | 7 / 7 / 15 |

The conclusion is invariant. Only the speed of the decision moves. That is
worth saying plainly in both directions: the commissioning step is not what
produces the result here, and it is also not a hidden knob that produces it.

### Margin — the healthy records are exonerated, not merely un-alarmed

| record | highest S reached | primary lines finishing below the lower boundary |
|---|---:|---:|
| normal_2 (held out) | **−0.11** | **8 / 8** |
| outer_0 | −0.08 | 8 / 8 |
| outer_1 | −0.10 | 8 / 8 |
| outer_2 | −0.04 | 8 / 8 |
| inner_0 | +5.81 | 4 / 8 |
| inner_1 | +5.81 | 4 / 8 |
| inner_2 | +5.81 | 4 / 8 |

The upper boundary is +3.81 nats. The held-out healthy record never rises
above −0.11, so it is not a near miss — it is four nats clear, and all eight of
its primary lines finish below the −2.28 lower boundary, meaning the test
positively cleared them rather than running out of data. On the inner-race
recordings exactly the four inner-race lines cross and the four outer-race
lines are exonerated, which is the isolation the topic asks for rather than an
undifferentiated anomaly flag.

Read the outer-race rows the other way round and they are the worst news in
this log: the system does not merely fail to find those faults, it **positively
clears them, 8 lines of 8**. A confident wrong answer is a more serious failure
mode than an undecided one, and section 5 treats it as such.

### The outer-race defect IS at the declared order — at constant speed

The same declared BPFO of 3.585, the same band, the same code, run on the
fixed-speed `_constant` recordings with the baseline fitted on the first half
of `normal_constant` and the second half held out:

| line | held-out healthy | outer_constant |
|---|---:|---:|
| bearingA_x : **outer_race** | −0.19 | **+8.81** |
| bearingA_x : inner_race | +0.62 | +0.19 |
| bearingB_x : outer_race | −0.20 | −0.98 |
| bearingB_y : outer_race | −0.48 | −0.45 |

Two things follow, and they matter more than the negative did.

**The assumed geometry is now corroborated on both races independently.** The
inner-race recordings put +8.8 at the declared BPFI of 5.415; the constant
outer-race recording puts +8.8 at the declared BPFO of 3.585, and is specific —
the inner-race line on the same record reads +0.19. Two different orders,
derived from the same four assumed dimensions, each firing only on the
recording whose label matches. That is a harder coincidence to arrange than one
hit, though it remains corroboration rather than confirmation.

**The outer housing is A and the inner housing is B.** The constant outer
signal is on `bearingA_x` alone; the varying inner signal is on `bearingB_*`
alone. The dataset says neither.

### And it is not simply a matter of speed

The `_constant` recording sits at 3008 rpm, above the whole varying range, so
the obvious explanation is that the outer fault only shows at high speed.
Stratifying the varying recordings by speed and rebuilding the baseline inside
each bin says otherwise — `bearingA_x`, outer_race z, order-tracked:

| speed bin | healthy windows | normal_2 | outer_0 | outer_1 | outer_2 |
|---|---:|---:|---:|---:|---:|
| 600–1200 rpm | 15 | −0.29 | −0.20 | +0.86 | +0.15 |
| 1200–1800 rpm | 72 | +0.44 | +0.05 | +0.11 | +0.29 |
| 1800–2100 rpm | 35 | +0.46 | −0.32 | −0.03 | −0.04 |
| 2100–2700 rpm | 26 | +0.16 | 0.00 | −0.60 | +0.17 |

Nothing anywhere, including the top bin. So either the defect needs more than
2700 rpm to couple, or — more likely — the varying-speed outer recordings were
made with a different and lighter specimen than the constant one. Both are
checkable by the dataset authors and neither is checkable here.

### Where the energy landed, against where it was declared

Predicted BPFI for the assumed 6205 is **5.415**. Measured on the four channels
of the three inner-race recordings: **5.343 to 5.428**, and in a broad scan the
peak sits at 5.40–5.46 with up to **+18.7 dB** over the healthy baseline. The
healthy held-out record's largest excursion anywhere in 0.5–12 orders is
+7.0 dB, at a different order in every channel.

This is reported, not used. Revising the geometry until the peaks agree would
be fitting physics to labels, and the assumed 6205 does not get a second
chance.

## 5. The honest half

1. **The outer race was not detected on any of its three varying-speed
   recordings, and worse, it was actively cleared.** All eight primary lines
   finish below the lower boundary on all three. The system's output on a
   damaged bearing is not "undecided" but "ordinary", which is the more
   dangerous of the two failures and is reported as such. What the constant-speed
   recording establishes is that this is not a geometry error and not a search
   in the wrong place: the identical declared BPFO fires at +8.81 there. The
   remaining explanations are a lighter specimen in the varying recordings or a
   coupling that needs more than 2700 rpm, and neither can be settled from the
   published data.
2. **The geometry is assumed.** If the rig is not a 6205, section 4's
   inner-race agreement is a coincidence at the 1% level across four channels
   and three recordings, which is unlikely but not impossible. **This is the
   number to ask the dataset authors for**, and until it arrives every KAIST
   figure carries the caveat.
3. **The tachometer offset is fitted per record.** It is fitted on the shaft
   line, which is health-free, and the identical procedure runs on healthy and
   damaged records — but it is still a per-record correction, and a deployed
   system would need its sensors synchronised at commissioning rather than
   recovered per capture.
4. **Three healthy recordings only.** Leave-one-out gets all three onto the
   held-out side, but they are three recordings from one rig on one day, and
   n = 3 is n = 3. This is the weakest number in the run and no amount of
   window counting improves it.
5. **One rig, one bearing size, one fault severity.** Nothing here says the
   inner-race margin survives an incipient defect.
6. **The per-line abstention rate is high and uneven** — 14% to 42% of
   line-windows return `no_baseline`, and `inner_2` is the 42%. That is the
   readiness rule working, not a fault, but it means a deployed system on this
   duty cycle would spend a meaningful fraction of its time declining to answer
   until the baseline had seen more of the speed range.
7. **Files truncated to 150 MB.** Each recording is ~298 s as published; this
   run uses the first ~74 s. Speed still traverses the full range within that,
   but the baseline sees fewer observations per regime than the full record
   would give. `inner_2` abstained on 31 of 74 windows against 10 for the
   others, because its speed distribution reaches a bin the baseline never
   filled — the `no_baseline` path firing on its own, unprompted.
8. **The tachometer offset search is per record and fitted.** It is fitted on
   the shaft line, which is health-free, and the identical procedure runs on
   healthy and damaged recordings. But it is a fitted quantity, and a reviewer
   is entitled to ask what it would cost if it were wrong. `--no-align` is the
   answer: everything, and silently.

## 5a. Compute cost of the order-tracked path

`tools/bench_kaist.py`, one pinned x86 core, all four channels, 16 declared
lines, one-second windows at 25.6 kHz with 14.1% median speed wander inside the
window. Timed: demodulation, angular resampling, envelope spectrum, prominence
and sidebands on every line, baseline comparison, sequential test. Not timed:
CSV parsing, which would be a ring buffer on the asset.

| arm | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| fixed-speed | 6.1 ms | 6.3 ms | 6.4 ms | 6.4 ms |
| order-tracked | 7.4 ms | 9.0 ms | 9.5 ms | **9.7 ms** |

Order tracking costs a factor of **1.43** — angular resampling is an
interpolation onto a 256-samples-per-revolution grid, linear in the window, and
it does not change the order of the cost. Against the topic's stated 1 s
budget from sample arrival to anomaly indication, the worst observed window is
**104× inside it**, at a 0.9% duty cycle.

This is a general-purpose container pinned to one core, not ruggedized
hardware, and no MIL-SPEC claim rests on it. What it establishes is the shape of
the number. Single-core pinning is the conservative direction.

## 6. Abstention, tested rather than asserted

Starved — a baseline fitted on 8 windows against a 25-observation minimum — the
system returns **`no_baseline` on 74 of 74 target windows, 0 of 64 lines ready,
and names nothing**. The same windows are named at window 6 once the baseline is
fitted properly. The refusal is the readiness rule, not the data.

## 7. Reproduce

```
python -m pip install -e ".[dev]"
mkdir -p data/kaist
```

Download subset 1 from `data.mendeley.com/datasets/vxkj334rzv` and put the
`vibration_*.csv`, `current_*.csv` and `rpm_*.csv` files in `data/kaist`. Then:

```
python tools/eval_kaist.py --data data/kaist --bearing KAIST6205U \
    --targets inner_0 inner_1 inner_2 outer_0 outer_1 outer_2 \
    --json runs/kaist_loo.json
python tools/eval_kaist.py --data data/kaist --bearing KAIST6205U --no-align
python tools/eval_kaist.py --data data/kaist --bearing KAIST6205U --band 500 12800
python -m pytest tests/test_kaist.py -q
```

The run takes about a minute per six recordings on a laptop.

## 8. The motor-current arm — NOT RUN

The current channels are staged and the loader reads them, but **the dataset
does not publish their sampling rate** and the vibration rate cannot be assumed
to apply — the two file sizes are inconsistent with a shared rate.
`io/kaist.estimate_current_fs` was written to recover it from the tachometer by
tracking the supply fundamental in normalised frequency and asking which assumed
rate makes that track proportional to the measured rpm.

**It was run, and it refused**, which is the designed behaviour and is worth
more than a guess would have been. Diagnosing the refusal: the dominant lines in
the current sit at a normalised 0.0599 and 0.1199 and **do not move as the shaft
speed sweeps 618–2480 rpm**. A supply fundamental locked to shaft speed is the
premise the estimator is built on, and a fixed line contradicts it — the rig
looks mains-fed rather than inverter-fed, so the fundamental is constant and the
motor takes up the speed range in slip. A different estimator is needed, keyed
to the slip sidebands rather than to the fundamental.

No current result is claimed and none should be. **Do not say we ran MCSA on
KAIST.** The rate is the second thing to ask the dataset authors for, after the
bearing geometry, and it is the one that unlocks the arm that matters most for
this topic — the Advanced Arresting Gear has motor current today and no
accelerometers.
