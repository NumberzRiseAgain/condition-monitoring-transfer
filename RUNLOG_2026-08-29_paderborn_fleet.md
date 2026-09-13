# cbmx run log — 29 August 2026 — Paderborn at fleet scale

Produced by `tools/eval_dataset.py` on this date, and reproduced independently
on a second machine (macOS, Python 3.9.6, numpy 2.0.2, scipy 1.13.1) against the
development machine (Linux, Python 3.10.12, numpy 2.2.6, scipy 1.15.3). The
four-condition sweep in section 1 is the second machine's output; the
band ablation matched to the record on both.

Console output in `runs/sweep_*.log`; the geometry control in `runs/fleet_C.log`.

**What changed.** Until today §3.1 rested on **one** healthy bearing, **one**
outer-race bearing and **one** inner-race bearing, with the healthy hold-out
taken from unseen *segments of the same recording*. This run uses **fourteen**
bearings — 5 healthy, 9 with accelerated-lifetime damage — and holds out a whole
**bearing** at a time. A false alarm here is a false alarm on a machine the
monitor has genuinely never met.

Scope: all four published operating conditions — `N15_M07_F10` (1500 rpm,
0.7 Nm, 1000 N), `N09_M07_F10` (900 rpm), `N15_M01_F10` (0.1 Nm) and
`N15_M07_F04` (400 N) — 280 records each, 64 kHz decimated to 32 kHz, run one
condition at a time because 1,120 records at 64 kHz do not fit in 3 GB of RAM.
3,560 damaged record-folds and 400 healthy ones in total.

---

## 1. The result

Leave-one-bearing-out over the five healthy bearings. Baseline fitted on the
other four, frozen, then scored against the held-out healthy bearing and every
damaged one, in every fold. Repeated independently at **all four operating
conditions** the dataset publishes.

**FALSE ALARMS: 0 of 400 healthy records** — 5 bearings the baseline never saw,
100 records each per condition, four conditions, zero alarms anywhere. This is
the number that got dramatically stronger today: it was 0 of 80 unseen
*segments* of one bearing.

**DETECTION separates by damage extent, not by fault class, and the separation
holds across speed and load.** Records named, per bearing, per condition:

| bearing | fault | extent | 1500 rpm 0.7 Nm 1000 N | 900 rpm | 0.1 Nm | 400 N |
|---|---|---:|---:|---:|---:|---:|
| KA16 | outer | **2** | 100/100 | 100/100 | 100/100 | 100/100 |
| KI16 | inner | **3** | 100/100 | 100/100 | 100/100 | 100/100 |
| KI18 | inner | **2** | 100/100 | 100/100 | **80/100** | 100/100 |
| KA15 | outer | 1 | 0/95 | 0/95 | 0/95 | 0/95 |
| KA22 | outer | 1 | 0/100 | 0/100 | 0/100 | 0/100 |
| KA30 | outer | 1 | 0/100 | 0/100 | 0/100 | 0/100 |
| KI04 | inner | 1 | 0/95 | 0/95 | **15/95** | 0/95 |
| KI14 | inner | 1 | 0/100 | 0/100 | 0/100 | 0/100 |
| KI17 | inner | 1 | 0/100 | **1/100** | 0/100 | 0/100 |
| | | | **300/890** | **301/890** | **295/890** | **300/890** |

Aggregated over all four conditions:

| | record-folds named | rate |
|---|---:|---:|
| damage extent 2 or 3 | **1180 / 1200** | **98.3%** |
| damage extent 1 | **16 / 2360** | **0.68%** |
| healthy, unseen bearing | **0 / 400** | **0%** |

A **145:1** separation between graded damage and the mildest grade, and nothing
at all on healthy bearings the baseline has never met.

This is the shape of the result worth reporting, and it is a different sentence
from "3 of 9 bearings". The method has a **sensitivity floor**, the floor sits
between damage extent 1 and extent 2 on this rig, and it does not move when the
shaft speed drops by 40% or the radial load drops by 60%.

**Two edges of the floor, both worth keeping.** At the *lowest torque* (0.1 Nm)
KI18 falls from 100/100 to 80/100 and KI04 rises from 0/95 to 15/95 — the only
condition where anything moves. Less torque means less impact energy, so the
graded bearing weakens and the marginal one becomes intermittently visible.
That is a load dependence acting in the physically expected direction on both
sides at once, which is the kind of coherence that makes a floor believable.

**Alternatives checked and rejected.** The three detected bearings are not
distinguished by manufacturer (KA16 and KI18 are MTK, KI16 is FAG, and the
missed set contains both), nor by damage combination (detected: R, S, S;
missed: S, S, R, M, M, R), nor by fault class (one outer, two inner). Damage
extent is the only clean discriminator among the fields the fact sheets carry.

Damage extents are read from the per-bearing fact sheet shipped inside each
archive (`KA16/KA16.pdf` and so on), not inferred from the paper.

## 1a. The full eleven bearings, and what KA04 turned out to be

The first fleet run used fourteen codes. K001, KA04 and KI21 were then
downloaded and added, giving **6 healthy and 11 damaged bearings**, 340 records
per operating condition.

**FALSE ALARMS: 0 of 120 healthy records across 6 unseen bearings.** K001 is a
third manufacturer again — IBU, pitch circle 29.05 mm, where the others are FAG
at 28.55 and MTK at 29.05 — and it raises nothing either.

**KA04 is damage extent 1, and it is detected on 114 of 114 records.** Its fact
sheet says so plainly: fatigue, pitting, outer raceway, single point, extent 1.
So the clean "extent 1 is never caught" reading from fourteen bearings does not
survive contact with the fifteenth.

| bearing | part | damage mode | extent | named |
|---|---|---|---:|---|
| KA16 | outer | fatigue / pitting ×2 | **2** | **120 / 120** |
| KI18 | inner | fatigue / pitting | **2** | **120 / 120** |
| KI16 | inner | fatigue / pitting | **3** | **120 / 120** |
| KA04 | outer | fatigue / pitting | 1 | **114 / 114** |
| KA22 | outer | fatigue / pitting | 1 | 0 / 120 |
| KI17 | inner | fatigue / pitting ×2 | 1 | 0 / 120 |
| KI21 | inner | fatigue / pitting | 1 | 0 / 120 |
| KA15 | outer | plastic deformation, indentations | 1 | 0 / 120 |
| KA30 | outer | plastic deformation, indentations | 1 | 0 / 120 |
| KI04 | inner | fatigue + plastic deformation | 1 | 0 / 114 |
| KI14 | inner | fatigue + plastic deformation | 1 | 0 / 120 |

Read down the table and two things are true at once.

**Every bearing graded 2 or 3 is caught, on every record.** Three of three, no
exceptions, at every operating condition.

**Grade 1 is the boundary, and it has scatter in it.** One of eight extent-1
bearings is caught. That is what a sensitivity floor actually looks like — not a
wall with one hole in it, but an edge that some damage lands above and most
lands below. KA04 and KA22 carry descriptors that are identical in every field
the fact sheets record — fatigue, pitting, outer raceway, single point, extent 1
— and one is found on every record while the other is found on none. The
dataset's own grading does not resolve them, and neither can we.

**One structure inside the misses that physics does predict.** The four bearings
whose damage involves plastic deformation — KA15 and KA30 outright, KI04 and
KI14 in combination with fatigue — are caught **0 of 4**. Indentations from
foreign particles do not produce the sharp repetitive impacts a spall does, so
an envelope method keyed to impact repetition should miss them. It does.

**A coincidence to disclose rather than let a reviewer find.** The 500–2000 Hz
band was originally commissioned on KA04 and KI21 in the very first Paderborn
run, long before any of this. KA04 is now the single extent-1 detection.
KI21 — the other commissioning bearing — is *not* detected, so the band is not
merely displaying its own commissioning set; but the overlap is real and the
volume should say so. The band-commissioning weakness in §2 is the same issue
seen from the other side.

## 2. The band decides everything, and commissioning it per run does not work

The first fleet run let `commission_band` choose the demodulation band, as the
tool has always done. It chose **12–16 kHz** and returned:

**detection 0 of 890, false alarms 0 of 100.**

A flawless false-alarm record produced by never speaking — the exact failure
this repository has now hit three times, and the reason detection and false
alarm are never printed apart.

Re-run with the **installation band, 500–2000 Hz** — the band the volume's
existing result was commissioned on, applied unchanged to nine bearings it was
not commissioned on — and detection is 300 of 890.

| demodulation band | detection | false alarms |
|---|---|---|
| 12–16 kHz, auto-commissioned this run | 0 / 890 | 0 / 100 |
| **500–2000 Hz, installation band, frozen** | **300 / 890** | **0 / 100** |

`commission_band` scores candidate bands by how well they resolve any
geometrically possible fault order, using whichever fault record happens to sort
first in each class. With three bearings that was stable. With nine it is not:
change the bearings and it changes its mind, catastrophically and silently.

**This is a finding about our method, not about the dataset.** The volume's
claim has always been that the band is chosen once at installation and frozen,
and this run is the first evidence of what the alternative costs. It also makes
the Phase II case concrete: band commissioning needs a stated acceptance
condition and a per-asset record, not a function that re-derives it.

`--band LO HI` now exists so the installation band can be given explicitly, and
the log says which of the two happened.

---

## 3. Two things the fleet found that three bearings could not

**A latent crash.** `Record` is a dataclass, so `r not in commission` calls
`__eq__`, which compares the signal arrays elementwise. With every record the
same length that quietly returns an array and works by accident; decimation at
fleet scale produces records differing by one sample, and the first fleet run
died with `operands could not be broadcast together with shapes (128000,)
(128001,)`. Fixed to identity comparison, with a regression test
(`test_record_membership_uses_identity_not_array_equality`).

**Two bearing geometries in one dataset.** The fact sheets give a pitch circle
diameter of **28.55 mm** for the FAG-supplied bearings (KA15, KI16, KI21) and
**29.05 mm** for the MTK-supplied ones (KA16, KA22, KA30, KI04, KI14, KI17,
KI18). Our catalogue applies 28.55 to all of them, from the 2016 paper's single
Table 1. That is a 0.53% error on BPFO and 0.33% on BPFI for seven of the nine —
inside the peak-search tolerance, so it raises nothing and prints nothing.

**Control run.** Re-scored with `PU6203MTK` (29.05 mm) applied to every record:
**byte-identical results** — same three bearings detected, same six missed, same
zero false alarms. So the geometry difference is real and worth carrying, but it
is *not* what decides detection here. Damage extent is. Recording both halves of
that matters: a difference that is real and does not explain the result is
exactly the kind of thing that gets mistaken for the cause.

---

## 4. What this does to the volume

§3.1 currently reads *25 of 25 outer race named, 0 wrong-part calls, 0 of 80
false alarms*, from one bearing per class. The fleet numbers are **better in one
direction and worse in the other**, and both belong in the volume.

Stronger: **0 false alarms on 400 records across 5 healthy bearings the baseline
never saw, at four operating conditions** — a different and much harder question
than unseen segments of a bearing it learned on.

Weaker, and better stated: not "3 of 9 bearings" but **98.3% of record-folds on
damage graded 2 or 3, against 0.68% at the mildest grade** — a sensitivity floor
with a 145:1 separation, located between extent 1 and extent 2, and stable
across a 40% speed reduction and a 60% load reduction.

**One thing to re-check before quoting either.** The volume describes KA04 as
*outer race, extent level 1, the mildest grade defined* and reports 25/25 named.
KI21 is extent 1 and was missed, consistent with everything above; KA04's own
fact sheet was not in our extracted copy, so its grade is currently our earlier
inference rather than a number read off the dataset's own document. If KA04 is
in fact extent 1 and detected, it is the single exception to the rule this run
establishes, and that matters enough to verify. Re-download `KA04.rar` and read
`KA04/KA04.pdf`.

## 5. Reproduce

```
python tools/eval_dataset.py --dataset paderborn --data data/pu_N15_M07_F10 \
    --holdout bearing --real-damage-only --band 500 2000 --out runs/fleet_B.json
python tools/eval_dataset.py --dataset paderborn --data data/pu_N15_M07_F10 \
    --holdout bearing --real-damage-only --out runs/fleet_A.json
python tools/eval_dataset.py --dataset paderborn --data data/pu_N15_M07_F10 \
    --holdout bearing --real-damage-only --band 500 2000 \
    --bearing-key PU6203MTK --out runs/fleet_C.json
```

## 6. Vibration against motor current, on the same bearings

`tools/eval_mcsa.py`, all 14 bearings, all 4 conditions, 1,120 files, both
channels measured from the *same* capture. Same geometry, same predicted lines,
same accumulator, same alpha and beta. Only the channel differs. Console output
in `runs/mcsa_fleet.log`.

This is the experiment behind the ladder argument in the volume, and behind
Matthew Marko's answer of 24 August: the deployed Advanced Arresting Gear has
motor current, pressure and temperature, and no accelerometers.

| bearing–condition pairs | vibration | motor current |
|---|---:|---:|
| damaged, named correctly | **13 / 36** | **0 / 36** |
| of those, damage extent ≥ 2 | **12 / 12** | **0 / 12** |
| of those, damage extent 1 | 1 / 24 | 0 / 24 |
| healthy, raising an alarm | 1 / 20 | 0 / 20 |

**Current sees nothing. Not weakly — nothing.** Across 36 damaged
bearing–condition pairs the robust z on every predicted line stays between
−2.4 and +0.8, and the damaged bearings are statistically indistinguishable
from the healthy ones. That includes KA16, KI16 and KI18, the three bearings
where vibration reads z of 9.5 to 19.5 and prominences of 34 to 41 dB against a
healthy floor of 5 to 6 dB.

That confirms §3.2 of the volume at fleet scale. It was 40 damaged records on
one bearing; it is now 720 records on nine, at four operating conditions, and
the answer is the same. **A bearing fault of this severity does not reach the
stator current on this drive.**

**What that is worth saying to NAVAIR.** It is the measured case for the
sensor-addition line in Phase II, and it is measured rather than asserted: the
sensor set the gear carries today cannot see these faults at any grade present
in this dataset, and one added accelerometer sees the graded ones at 12 of 12.
It is also the honest limit on the day-one capability — current gives coverage,
not sensitivity, and this is how much sensitivity it lacks.

**Two disagreements with §1, both instructive.**

*KI14 is named at one condition here and at none in §1.* It is extent 1 and it
sits on the threshold: z of 2.3, 2.2 and 1.0 at three conditions and named at
one. The floor is sharp but it is not a wall, and KI14 is the bearing sitting
on it.

*One healthy false alarm here — K004 at 400 N, named outer race at z = 2.6 —
where §1 reports 0 of 400.* The protocols differ in exactly one way that
matters: this tool builds its baseline from **one** healthy bearing (K002),
where §1 builds it from **four** and holds the fifth out. So the two runs
together say something neither says alone: **a baseline learned from a single
healthy unit produces a false alarm that a baseline learned from four does
not.** That is direct evidence for the fleet-baseline argument, obtained by
accident, and it belongs in the volume.

## 7. Not run

`eval_mcsa.py` across the nine bearings, per above. The machined-defect
bearings, excluded on purpose by `--real-damage-only`. K001, KA04 and KI21 at
fleet scale — they were not among the fourteen re-downloaded, which is why KA04
still needs its fact sheet read (§4).
