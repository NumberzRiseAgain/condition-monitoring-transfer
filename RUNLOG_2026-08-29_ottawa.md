# cbmx run log — 29 August 2026 — uOttawa variable speed

`tools/eval_dataset.py --dataset ottawa`. Console output in `runs/ott_*.log`.

**Dataset.** *Bearing Vibration Data under Time-varying Rotational Speed
Conditions*, Huang & Baddour, University of Ottawa, Mendeley Data
[10.17632/v43hmbwxpm](https://data.mendeley.com/datasets/v43hmbwxpm/2). Sixty
records: five health states — healthy, inner race, outer race, ball, combined —
by four speed profiles — increasing, decreasing, up-then-down, down-then-up —
by three trials. 200 kHz, 10 s. `Channel_1` vibration, `Channel_2` an encoder
pulse train.

**Why it was run.** Variable-speed validation was the volume's newest and most
AAG-relevant claim and it rested on one rig. This is a second rig, with a
different bearing, a different tachometer convention, and speed profiles that
sweep rather than wander.

---

## 1. The result

**0 of 33 damaged records detected. 0 of 12 healthy records alarmed. 0
wrong-part calls.** Repeated in four demodulation bands — 500–2000 Hz,
2–6 kHz, 6–12 kHz, and the 43.75–50 kHz the healthy-kurtosis commissioner
chose. The verdict does not move.

A run that never speaks passes a false-alarm test perfectly. It is reported
here as a failure, and the rest of this log is about which of three possible
causes it is.

## 2. Two defects found on the way, neither of them the answer

**The tachometer was decoded at the wrong resolution.** Run with the loader's
default `--ppr 1`, `Channel_2` yields shaft speeds of **129,000 to 1,400,000
rpm** — physically impossible, and the run duly returned `no_baseline` on 33 of
33 damaged records because every window landed in a speed bin the baseline had
never seen. The Mendeley record states the answer plainly: **"CPR of the encoder
is 1024."** At `--ppr 1024` the shaft reads 2–23 Hz, which is a bearing rig.

Worth recording: a label-free identification — scoring candidate resolutions by
the sharpness of the shaft line in the order domain, the same technique that
confirmed KAIST's 25.6 kHz — preferred **1000** (28.1 dB) over **1024**
(26.3 dB). It would have picked the wrong value by a small margin. Where a
parameter is published, the publication wins.

**Every window was being binned on its record's mean speed.** On a constant-speed
rig those are the same number. On a rig that sweeps its whole range inside ten
seconds the record mean describes no window in it, so healthy records landed in
one set of regimes and damaged records in another and nothing was comparable.
`window_shaft_hz` now takes each window's own speed from the record's speed
trace. Paderborn is unchanged by this — 25/40, 0 wrong-part, 0/80 — because on a
constant-speed rig it is a no-op.

Both were real defects. Neither produced a detection when fixed.

## 3. Which of three causes it is

A broad scan of the order domain, damaged against healthy, band 2–6 kHz,
order-tracked, says the signal is not simply absent:

| records | strongest lines above the healthy baseline |
|---|---|
| outer race | 2.05 (+4.4 dB), **6.03 (+10.2 dB)**, 8.07 (+3.4), 10.32 (+3.7) |
| inner race | 2.43 (+6.9), 3.03 (+6.2), **5.43 (+8.9)**, 7.43 (+6.6) |
| ball | 1.36 (+3.0), 1.98 (+3.4), 2.41 (+3.9), 6.05 (+2.9) |

Declared, from the catalogue's ER-16K entry: **BPFO 3.665, BPFI 5.335, ball
5.206**.

**The inner race lands where it should.** The strongest inner-race line is at
5.43 against a declared 5.335 — 1.8% out, inside the 2% search tolerance. So
the geometry is approximately right for that race, and the failure there is one
of *margin*: +8.9 dB over the healthy baseline is not enough to carry the
sequential test against this baseline's spread.

**The outer race does not.** Its strongest line is at 6.03, against a declared
3.665. That is not a harmonic of the declared order and not within any
tolerance. Either the geometry is wrong or the line belongs to something else.

## 4. What this is actually evidence of

The catalogue entry for this rig has always carried a caveat, written long
before the set was run:

> ER-16K (uOttawa variable-speed rig) — MB ER-16K manufacturer dimensions
> (inches), **UNVERIFIED against the dataset paper**

**This is the only dataset in the evidence base whose geometry comes from a
manufacturer catalogue rather than the dataset's own documentation, and it is
the only dataset where nothing is detected.** Set against the others:

| dataset | geometry source | result |
|---|---|---|
| Paderborn | the dataset's own paper, Table 1 | detected, 3 of 3 graded bearings |
| KAIST | assumed 6205, but corroborated by **both** race orders firing only where their labels match | detected, inner race 9/9 |
| **uOttawa** | **manufacturer catalogue, never verified** | **0 of 33** |

That is the method's central claim meeting its own contrapositive. Geometry is
what tells this system where to look; when the geometry is right it finds the
fault, and when the geometry is unverified it finds nothing and says so rather
than producing a confident wrong answer. A method that degraded gracefully into
plausible-looking output here would be far more dangerous.

**It is not proof.** The margin failure on the inner race is consistent with
geometry being roughly right and sensitivity being short, which is a different
story from geometry being wrong. Both would look like this.

## 5. What would settle it

The dataset paper (Huang & Baddour, *Data in Brief* 21, 2018) names the bearing
as an ER-16K but the Mendeley record does not publish its dimensions. **Ask the
authors for ball count, ball diameter, pitch diameter and contact angle**, as
has been asked of the KAIST and Paderborn authors. If the corrected geometry
puts BPFO at 6.03/2 or thereabouts, this becomes a detection result and a much
stronger one for having been wrong first.

Until then the volume says: run, not detected, geometry unverified, and the
variable-speed claim continues to rest on KAIST alone.

## 6. Not run

The twelve combined-fault records, which the evaluation excludes as a class.
