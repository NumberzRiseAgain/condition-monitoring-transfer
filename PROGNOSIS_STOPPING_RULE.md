# Declared before the run-to-failure data was downloaded

Written 29 August 2026, before any run-to-failure data was on disk. Nothing below
has been informed by looking at it, which is the only reason it is worth
writing down.

## What will be claimed, and only this

A **lead time** — how much of each bearing's life remained when the system first
spoke persistently — reported as a distribution across every bearing scored,
with the bearings it never spoke about counted in the denominator.

## The stopping rule

**If the median life remaining at first call is below 10%, prognosis is not
claimed.** The volume will say, in these words or close to them: *the indicator
is a failure detector, not a prognostic, on this data*, and remaining-useful-life
stays a Phase II task with the precondition measured in §3.7.

That threshold is chosen now, on the reasoning that a warning arriving in the
last tenth of a bearing's life cannot be scheduled against and is therefore
worth nothing to a maintenance planner — not on any property of the data.

## What will not be done

**The alarm threshold will not be lowered until a number appears.** `--z-alarm`
is 6.0 robust sigmas and the baseline fraction is the first 15% of each
bearing's own life. If those produce nothing, that is the result. Sweeping them
and reporting the best is fitting, and it is the specific failure this whole
evidence base exists to avoid.

**The persistence requirement will not be relaxed.** A call requires three
consecutive captures above threshold. A single-capture crossing is noise, and a
lead time measured from a spike the system would not itself have acted on is a
fiction.

**Bearings will not be dropped.** Every bearing in every test is scored and
reported, including the ones that never degrade — on the NASA/IMS set three of
the four bearings on each shaft survive the test, and they are the control that
makes the fourth mean anything.

## What is legitimate to change afterwards, and to disclose

The demodulation band, if the geometry-free indicators show the fault is
somewhere the band excludes — because that is a property of the rig's structure
and not of its labels. Any such change is reported as a change.

The **declared-physics half will not run at all** unless a catalogue entry for
the Rexnord ZA-2115 exists, taken from the documentation file the archive itself
ships. Inventing its dimensions would repeat the uOttawa situation, where
manufacturer-catalogue geometry produced 0 of 33 with no way to tell a geometry
error from a sensitivity limit.

## Which dataset, and why not the obvious one

The dataset is **NASA/IMS** — Center for Intelligent Maintenance Systems,
University of Cincinnati, published through the NASA Prognostics Center of
Excellence. Not XJTU-SY, which is the better-known run-to-failure set and is
hosted on a personal website in the PRC with Baidu Netdisk among its mirrors.
For a volume whose §7.6 states that execution is 100% U.S., that provenance is
not worth one topic row. The NASA set is also technically better here: bearings
that fail naturally over days give lead times a maintainer could schedule
against, where XJTU's fail in hours.

## Why this file exists

Three times in this evidence base a run has produced zero detections with a
perfect false-alarm record, and each time the temptation was to adjust something
until it spoke. Twice the right answer was a real defect — a decimation helper
resetting the geometry, a tachometer misaligned by half a second. Once it was
the honest answer, and adjusting would have manufactured a result.

Prognosis is where that temptation is strongest, because remaining-useful-life
is the most over-claimed number in this field and nobody checks it. Writing the
rule down before the data arrives is the cheapest possible defence against
our own hindsight.

---

# Outcome, recorded 30 August 2026

The data was downloaded and scored. **The rule fired.**

Median life remaining at first call: **1%** in the documented scope, **2%**
across the full folder. The threshold was 10%. **Prognosis is not claimed.**

Nothing above was changed to reach that. `--z-alarm` stayed at 6.0, the baseline
fraction stayed at the first 15%, persistence stayed at three consecutive
captures, no bearing was dropped, and the declared-physics half did not run —
the archive's readme describes the rig, the channels and the failures and never
gives the ZA-2115's dimensions, which is precisely the condition under which
this file said that half would not run.

One thing this file did not anticipate, and which turned out to matter more than
the lead time: **the surviving bearings speak too.** Four of eight in the
documented scope, seven of eight across the full folder. The geometry-free
indicators are not just late, they are not specific to the bearing that failed.
That is a result about generic condition indicators, not about our method, and
the distinction is drawn carefully in `RUNLOG_2026-08-30_ims.md` §3 — we cannot
run our method here, so we cannot claim it would have done better on this data.

One scope decision was made before scoring and is recorded here because it could
otherwise look like a choice made afterwards: set 3 ships 6,324 captures for an
experiment its readme documents as 4,448, and the documented window was made
primary on the grounds that the readme is also the only source for the failure
label. Clipping to the readme's end date yields exactly 4,448. Both scopes were
run and both are reported. They agree.
