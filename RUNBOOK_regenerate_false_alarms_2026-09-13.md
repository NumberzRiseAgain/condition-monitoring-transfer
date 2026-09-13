# Regenerating the false-alarm table

Closes `paper/BLOCKER_false_alarm_table_2026-09-13.md`. Run on the machine that has the
ZeMA and Paderborn data. Paste the whole terminal output back; it is the run log.

## What changed in the code first

`tools/eval_prognosis.py` Part B no longer contains any literal counts. It reads every
count from the run artifact that produced it and **exits rather than guess** if a file or a
key is missing. It also writes a canonical `evidence.json` that the paper, Volume 2 and the
certification pack should all read, so the two documents cannot drift apart again.

A second defect was found while doing this and is fixed in the same file. The
Clopper-Pearson upper bound fell back to a bisection over an incomplete-beta series when
scipy was absent. That series diverges for the shapes this module asks for: at `0/180` it
returned **93.25%** where the answer is **1.65%**, silently, and only on some inputs. The
bound is now computed from the binomial identity, and when scipy is present both routes are
computed and must agree or the run stops. All six bounds the paper quotes reproduce exactly.

## Commands

```bash
cd <path to this repository>

# 0 — environment, once. Never a bare python3: the Mac's is 3.9.6 with nothing in it.
python3 -m venv .venv
.venv/bin/python -m pip install -q --upgrade pip setuptools wheel
.venv/bin/python -m pip install -q numpy scipy pyyaml pandas
.venv/bin/python -c "import numpy,scipy;print('numpy',numpy.__version__,'scipy',scipy.__version__)"
mkdir -p runs

# 1 — hydraulic, the frozen-baseline run that carries the held-out healthy cycles
PYTHONPATH=src .venv/bin/python tools/eval_hydraulic.py \
    --root data/hydraulic \
    --json runs/hydraulic_freeze.json 2>&1 | tee runs/hydraulic_freeze.log

# 2 — Paderborn rated, the band an installation declares from geometry
PYTHONPATH=src .venv/bin/python tools/eval_dataset.py \
    --dataset paderborn --holdout bearing --real-damage-only \
    --data data/pu_N15_M07_F10 --band 500 2000 \
    --out runs/paderborn_fleet_rated_band500_2000.json \
    2>&1 | tee runs/paderborn_fleet_rated_band500_2000.log

# 3 — Paderborn rated, band auto-commissioned from healthy records only
PYTHONPATH=src .venv/bin/python tools/eval_dataset.py \
    --dataset paderborn --holdout bearing --real-damage-only \
    --data data/pu_N15_M07_F10 --commission healthy \
    --out runs/paderborn_fleet_rated.json \
    2>&1 | tee runs/paderborn_fleet_rated.log

# 4 — the motor-current channel, same records
PYTHONPATH=src .venv/bin/python tools/eval_dataset.py \
    --dataset paderborn --holdout bearing --real-damage-only \
    --data data/pu_N15_M07_F10 --channel current --band 500 2000 \
    --out runs/paderborn_fleet_current.json \
    2>&1 | tee runs/paderborn_fleet_current.log

# 5 — regenerate the table and write the canonical evidence file
PYTHONPATH=src .venv/bin/python tools/eval_prognosis.py \
    --root data/hydraulic \
    --hydraulic runs/hydraulic_freeze.json \
    --paderborn-vib runs/paderborn_fleet_rated_band500_2000.json \
    --paderborn-cur runs/paderborn_fleet_current.json \
    --json runs/prognosis.json \
    --evidence runs/evidence.json 2>&1 | tee runs/prognosis.log
```

If step 1 or 2 cannot find its data directory, it says so and stops. Change `--root` or
`--data` to match where the sets actually sit; do not edit the script.

Step 5 prints, for the run log, every withdrawn literal beside the regenerated count and
marks each `same` or `CHANGED`.

## What the answer is going to be, for the hydraulic half

`results/fresh_2026-09-09/hydraulic_freeze.json` is a computed artifact already on this tree,
so the hydraulic half can be read today without re-running anything. Reading it:

| component | k / n | rate | 95% upper | no-baseline cycles |
|---|---:|---:|---:|---:|
| accumulator | 0 / 180 | 0.00% | 1.65% | 0 |
| cooler | 0 / 244 | 0.00% | 1.22% | 0 |
| pump | 2 / 244 | 0.82% | 2.56% | 3 |
| **valve** | **44 / 184** | **23.91%** | **29.65%** | 20 |
| **all four pooled** | **46 / 852** | **5.40%** | **6.85%** | 23 |

The withdrawn table said *0 false alarms in 668 held-out healthy cycles, a one-sided 95%
upper bound of 0.45%*, across "all four components". The valve was the component missing from
it, and the valve is where the alarms are.

Re-run step 1 anyway. The point is a provenance chain that starts at the raw data on a machine
we control, not a number that happens to be right.

## Two decisions this run does not make

1. **Which Paderborn protocol the table quotes.** The 9 Sep runs give 0/120 across six unseen
   bearings on the rated full-17 set, and 0/400 across five unseen bearings on the 14-code
   fleet. Both are real and they answer slightly different questions. The withdrawn 0/80 and
   0/74 match neither. Pick one, say which, and quote it everywhere.
2. **Whether a no-baseline cycle belongs in the denominator.** The system refused to judge 23
   held-out healthy cycles, 20 of them on the valve. They are not alarms. Whether they are
   trials is a methodological choice, and the paper's own "refuse outside the commissioned
   envelope" framing argues for reporting both numbers rather than quietly picking one.

## After the run

- `runs/evidence.json` is the canonical file. Point `paper/make_numbers.py` at it, and
  generate the Volume 2 figures from it too, so the paper and the proposal cannot diverge.
- File the run with the evidence pack so the numbers are hashed on this tree.
- Volume 2 `latex_v3/Part1_body.tex` lines 644 and 1187-1188 still assert the withdrawn
  figures as measured. They have to be updated from the same source in the same round.
