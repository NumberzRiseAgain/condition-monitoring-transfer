# cbmx run log — 29 August 2026 — VT/NSWC hydraulic rotary actuator

Everything below was produced by `tools/eval_actuator.py` in this repository on
this date, against a public dataset the Government can download and re-run.

**Dataset.** Virginia Tech National Security Institute, Luna Labs and Naval
Surface Warfare Center Philadelphia Division. Six Moog Flo-Tork 15,000 in-lbf
industrial rotary actuators, Luna Labs eCBM node, 1 kHz. Inlet/outlet pressure,
three temperatures, angular position, three-axis acceleration.
`github.com/vtnsi/actuator_dataset`, MIT licence. 598 records, 130 MB.

**Why this set and not another bearing benchmark.** It is a hydraulic actuator
with real seal damage and real gear damage, on the sensor types arresting gear
already carries, published in part by a Navy warfare centre. It closes a gap the
volume currently fills by analogy.

---

## 1. The difficulty this dataset presents, faced rather than avoided

Every actuator is **one physical unit in one fixed condition**. No unit has a
healthy prefix followed by its own fault. So a baseline must come from a
*different* unit — which is exactly the comparison our own method says
manufactures false alarms, and exactly the trap the dataset authors warn about:
a model can learn the actuator rather than the fault, and their own held-out
experiments show it.

The matched pairs come from the authors' own label dictionaries, not from us:

| Load | Healthy | Damaged |
|---|---|---|
| 4-inch butterfly valve | Act_1 | **Act_4, seal defect** |
| 2.5-inch ball valve | Act_2 | **Act_3, gear damage** |
| No load | Act_5 | Act_6 — **both healthy** |

The third row is the control, and it is what makes the first two mean anything.

---

## 2. Two implementation defects found before any result was believed

**The stroke segmentation returned the whole file.** `_phases` took the first
and last index whose angle derivative exceeded a threshold. Isolated
quantisation noise late in the record dragged the stroke to the end, so transit
time read **2.99 s of a 3.00 s capture** and the hold phase was empty — which
made `seal.hold_decay` NaN on three of six units. Fixed by taking the longest
contiguous run above a threshold set from the noise floor. True stroke is
~400 ms; hold is ~2.55 s.

**The regime variable was a proxy for the unit.** The natural regime for a
hydraulic rig is oil temperature, which is what the ZeMA work uses. Here
`Internal_Temp` is constant within each actuator and different between them,
because each unit was recorded in one session: Act_1 sits at 24.3–24.4 °C,
Act_4 at 27.0–27.4 °C. Binning on it makes the regime identical to the unit,
every target cycle lands in a bin the baseline never saw, and the system
**abstained on 100% of target cycles** — which it did, on the first run, before
this was found.

> That abstention was mechanically correct and diagnostically useless. It is
> also an unplanned demonstration that the `no_baseline` path fires when the
> operating point is genuinely unseen. The regime is now one bin per matched-load
> experiment, and the 2.7 °C temperature difference between reference and target
> is reported as an **uncontrolled covariate**, which is what it is.

---

## 3. The portability screen — decided on healthy data only

Before any damaged unit was opened: fit on Act_5 (healthy), score Act_6
(healthy), and disqualify any symptom that already reads like a fault between
two healthy units. **No fault label is used to make this decision**, so it cannot
be tuning in disguise.

| Symptom | median &#124;z&#124;, healthy vs healthy | portable |
|---|---|---|
| seal.hold_decay | 1.00 | yes |
| seal.working_dP | 0.18 | yes |
| seal.transit | 0.05 | yes |
| seal.temp_rise | 0.71 | yes |
| gear.press_ripple | 0.26 | yes |
| gear.angle_jerk | 0.60 | yes |
| gear.accel_rms | 0.71 | yes |
| gear.accel_kurt | 0.32 | yes |

All eight survive. Cross-unit comparison on this rig does not, by itself,
manufacture a fault signal.

---

## 4. Results

α = 0.02, β = 0.10. Baselines fitted on the first 50% of the reference unit and
**frozen before any test cycle is scored**.

### Median robust z on the target, against the same baseline

| Symptom | dir | Act_4 SEAL | Act_3 GEAR | Act_6 HEALTHY |
|---|---|---|---|---|
| **gear.press_ripple** | +1 | **+21.42** | **+26.51** | **0.00** |
| gear.accel_rms *(vibration only)* | +1 | +0.20 | **+1.90** | +0.38 |
| gear.angle_jerk | +1 | −2.27 | −1.52 | −0.18 |
| gear.accel_kurt | +1 | −0.50 | −0.33 | +0.05 |
| seal.hold_decay | +1 | +1.04 | +0.44 | +0.24 |
| seal.working_dP | −1 | +1.07 | +0.39 | −0.09 |
| seal.temp_rise | +1 | +0.93 | +0.02 | +0.35 |
| seal.transit | +1 | −0.06 | −0.07 | −0.05 |

Held-out healthy cycles of the reference unit read +0.38 and +0.23 on
`press_ripple`. The damaged units read +21 and +27 against that.

### Sequential test

| Experiment | arm | names target at | held-out healthy | one-shot FA |
|---|---|---|---|---|
| Gear damage, ball load | installed | **cycle 5** | never | 0/53 (≤5.50%) |
| Gear damage, ball load | + vibration | **cycle 4** | never | 0/53 |
| Seal defect, butterfly | installed | cycle 5 *(gear symptoms)* | never | 0/52 (≤5.60%) |
| Seal defect, butterfly | seal symptoms | **never** | never | 0/52 |
| **Healthy / healthy control** | installed | **never** | never | 0/53 (≤5.50%) |
| **Healthy / healthy control** | + vibration | **never** | never | 0/53 |

---

## 5. What this establishes, and what it does not

**Established.** Damage on both units is detected **from the sensors the
deployed gear already carries** — pressure, no accelerometer — sequentially,
within five actuations, with **zero firings across 77 cycles of the
healthy/healthy control** and zero on held-out healthy cycles of the reference.
The control is the reason the detection is worth stating.

**Vibration buys specificity, not detection.** `accel_rms` rises on the gear
unit (+1.90) and stays silent on the seal unit (+0.20) — physically right, since
a leaking seal is not a mechanical mesh fault. Installed sensing detects that
*something* is wrong; the accelerometer is what separates *which*. That is the
T4 argument, measured rather than asserted.

**Not established — and this is the honest half.**

1. **Our declared seal physics failed.** hold_decay, working_dP, transit and
   temp_rise all sit at |z| ≈ 1 or below on the seal unit. The leak was not
   visible in pressure decay, differential, stroke time or temperature as we
   declared them. The seal unit is detected only by the *gear* symptom set.
2. **Detection is not specific on installed sensors alone.** `press_ripple`
   fires at +21 on the seal unit and +27 on the gear unit. Without an
   accelerometer the system can say a unit is damaged but not which fault.
3. **One declared direction was wrong.** `angle_jerk` was declared +1 and
   measured −2.27 and −1.52 on the damaged units — consistently the other way.
   Same class of error as the ZeMA accumulator, and the remedy is the same:
   correct the declared physics rather than loosen the test.
4. **The temperature covariate is uncontrolled**, at about 2.7 °C between
   reference and target on the butterfly pair.
5. **One control pair only.** Act_5/Act_6 are the only two healthy units on the
   same load, and they differ by ~0.5 °C where the seal pair differs by 2.7 °C.
   The control is therefore easier than the test.

## 5a. Abstention, tested properly

The `no_baseline` path fires when a line's statistics are not ready. Starved of
observations — fitted on 20 cycles against a 25-observation minimum — the system
returns **no_baseline on 100% of 105 target cycles and names nothing**, with
0 of 8 lines ready. The same cycles are named at cycle 5 once the baseline is
fitted on 52. **The refusal is the readiness rule, not the data.**

An earlier version of this check applied a no-load baseline to butterfly-load
cycles and called that an unseen regime. Once the regime was collapsed to one
bin per experiment it stopped testing anything and printed 0% of 0%, which reads
like a failure and was merely vacuous. Replaced.

## 5b. Independently reproduced

Run on a second machine (macOS, Apple silicon, Python 3.9.6, numpy 2.0.2,
scipy 1.13.1, pandas 2.3.3) against the numbers above (Linux, Python 3.11.15,
numpy 2.4.6, scipy 1.17.1, pandas 3.0.5). **Every figure identical**, including
the portability screen to two decimals and both sequential detection cycles.

## 6. Reproduce

```
git clone https://github.com/vtnsi/actuator_dataset
PYTHONPATH=src python3 tools/eval_actuator.py --data actuator_dataset/Data
```

## 7. KAIST varying-speed set — since run; see the separate log

At the time of writing this log the KAIST set (Mendeley DOI
10.17632/vxkj334rzv and its two sibling subsets) could not be fetched:
`data.mendeley.com` and `zenodo.org` are refused by this environment's egress
policy, which is why the actuator set could be run and that one could not.

It was subsequently downloaded outside this environment and supplied. It has
now been run, and it is what removes the synthetic-speed-profile caveat: real
bearing damage under a randomly varying 613–2480 rpm with a tachometer as
ground truth. Inner race detected on 3 of 3 recordings with 0 false alarms;
outer race detected on 0 of 3. See **`RUNLOG_2026-08-29_kaist.md`**, which is
the authority for what may and may not be claimed from it.
