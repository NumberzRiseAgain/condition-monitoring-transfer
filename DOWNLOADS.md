# Everything still to download, and what each one buys

The analysis container's egress proxy refuses `data.mendeley.com`, `zenodo.org`
and `groups.uni-paderborn.de` — all three return 403 — so every download below
is yours. Nothing here needs new code except XJTU-SY, which is being written now.

Work top to bottom. Each section ends with the exact command to run, and the
commands are paste-safe: no trailing `#` comments, because interactive zsh does
not strip them.

Put everything under `~/Downloads/cbmx_run/data/`.

---

## 1. Paderborn — the three missing codes

**Twenty minutes, 500 MB, and it settles a question the volume currently gets
wrong-footed on.** KA04 is the bearing Part 1's original *25 of 25* rests on.
Its damage grade is our inference, not a number read off the dataset's own fact
sheet — and if it turns out to be extent 1 *and* detected, it is the single
exception to the sensitivity floor the fleet run just established.

```bash
cd ~/Downloads/paderborn_rar
for C in K001 KA04 KI21; do
  curl -L -O "https://groups.uni-paderborn.de/kat/BearingDataCenter/$C.rar"
done
for f in K001.rar KA04.rar KI21.rar; do unar -q -o ~/Downloads/paderborn "$f"; done
```

Then read the grades straight out of the shipped fact sheets:

```bash
cd ~/Downloads/paderborn
for c in K001 KA04 KI21; do
  echo "== $c"
  pdftotext -layout "$c/$c.pdf" - | grep -iE "Extent of damage|Damage combination|Pitch circle|Manufacturer -"
done
```

If `pdftotext` is missing, open the three PDFs by hand and send me the *Extent of
damage*, *Damage combination* and *Pitch circle diameter* rows.

Then fold them into the fleet run:

```bash
cd ~/Downloads/cbmx_run
for C in N15_M07_F10 N09_M07_F10 N15_M01_F10 N15_M07_F04; do
  for f in ~/Downloads/paderborn/*/${C}_*.mat; do
    ln -sfn "../../../paderborn/$(basename "$(dirname "$f")")/$(basename "$f")" "data/pu_$C/$(basename "$f")"
  done
done
bash run_fleet.sh
```

That takes the fleet from 5 healthy / 9 damaged to **6 healthy / 11 damaged**,
which is a sixth leave-one-out fold and two more damaged bearings.

---

## 2. uOttawa variable speed — the second rig

**This is the most valuable download on the list.** Variable-speed validation is
the most AAG-relevant claim in the volume and it currently rests on one rig.
uOttawa sweeps the shaft up, down, and both ways inside a single ten-second
record — harder than KAIST, which merely wanders. A second rig with a different
bearing and a different tachometer convention turns the claim from a property of
KAIST into a property of the method.

Mendeley Data, **DOI 10.17632/v43hmbwxpm** — *Bearing vibration data collected
under time-varying rotational speed conditions* (Huang & Baddour). Take the
latest version. A couple of GB.

Sixty `.mat` files named `<health>-<profile>-<trial>.mat`:

| letter | health | | letter | speed profile |
|---|---|---|---|---|
| H | healthy | | A | increasing |
| I | inner race | | B | decreasing |
| O | outer race | | C | increasing then decreasing |
| B | ball | | D | decreasing then increasing |
| C | combined | | | |

Keep those filenames — the loader refuses to guess labels from anything else.
Each file holds `Channel_1` (vibration) and `Channel_2` (a tachometer **pulse
train**, not an rpm reading), 200 kHz, 10 s.

```bash
mkdir -p ~/Downloads/cbmx_run/data/ottawa
```

Extract every `.mat` into that directory, then:

```bash
cd ~/Downloads/cbmx_run
PYTHONPATH=src .venv/bin/python tools/eval_dataset.py \
    --dataset ottawa --data data/ottawa --ppr 1 \
    --out runs/ottawa.json 2>&1 | tee runs/ottawa.log
```

**Two things to watch, and both are honest failures if they happen.** `--ppr` is
the tachometer's pulses per revolution; if the recovered speed looks wrong the
loader says so rather than substituting a constant, because a constant speed on
a variable-speed record would invalidate the entire point of running this set.
And the bearing geometry in our catalogue (`ER16K`) is **manufacturer data, not
dataset-published** — the same UNVERIFIED status as KAIST. Send me the log either
way.

---

## 3. XJTU-SY run-to-failure — the prognosis gap

**The only dataset that moves "predict possible failures" — the topic's own
words — out of the Phase II column.** Fifteen bearings run to actual failure,
25.6 kHz, 32,768 samples (1.28 s) captured once per minute until the bearing
dies. Failure modes labelled per bearing, including **cage fracture**, which
nothing in our evidence base has ever tested.

Download from `biaowang.tech/xjtu-sy-bearing-datasets` — the page lists Google
Drive, Dropbox, MEGA and Baidu mirrors. Several GB.

Three operating conditions, five bearings each:

| condition | speed | radial load |
|---|---|---|
| 1 | 2100 rpm | 12 kN |
| 2 | 2250 rpm | 11 kN |
| 3 | 2400 rpm | 10 kN |

```bash
mkdir -p ~/Downloads/cbmx_run/data/xjtu
```

Extract preserving the per-bearing folder structure — the folder name carries
the bearing identity and the CSV name carries the minute index, and the loader
needs both to build a degradation curve. Do **not** flatten it.

The loader and evaluation for this are being written now. Download it and I will
have the code ready.

**A caution I would rather state before the data arrives than after.** Remaining
useful life is the single most over-claimed number in this field. If the health
indicator does not move monotonically with time on these bearings, that is the
result, and it goes in the volume as the result. A soft prognosis number would
undercut the one thing this volume is genuinely selling, which is that every
figure in it holds up when pushed.

---

## 4. KAIST subsets 2 and 3, and the untruncated subset 1

Raises the variable-speed false-alarm claim from **three** held-out healthy
recordings to **nine**, and restores each recording from 74 s to its full 298 s —
four times the windows, which should also collapse the 42% abstention rate on
`inner_2`.

| subset | DOI |
|---|---|
| 1 | `data.mendeley.com/datasets/vxkj334rzv` |
| 2 | `data.mendeley.com/datasets/x3vhp8t6hg` |
| 3 | `data.mendeley.com/datasets/j8d8pfkvj2` |

Take the latest version of each — subset 1 is now at version 7 and the copy we
used was earlier, so redownloading it is worth doing on its own. About 16 GB
total. **Check free space first.**

**Do not flatten the three subsets together.** They reuse identical filenames,
so a flat extract silently overwrites two thirds of the data and the run reports
a smaller `n` than it thinks it has. Keep them apart:

```bash
mkdir -p ~/Downloads/cbmx_run/data/kaist/subset2
mkdir -p ~/Downloads/cbmx_run/data/kaist/subset3
```

Subset 1's CSVs go directly in `data/kaist/`; subsets 2 and 3 into their own
subdirectories. The loader folds the directory name into the replicate, so
subset 2's healthy recording 0 becomes `normal_subset2-0`. Each subset folder
needs its **own** `rpm_*.csv` files — the speed reference is matched to the
recording, not to the condition.

Confirm the census before the long run:

```bash
cd ~/Downloads/cbmx_run
PYTHONPATH=src .venv/bin/python tools/eval_kaist.py --data data/kaist \
    --bearing KAIST6205U --targets inner_0 --seconds 20 --min-samples 8 | head -20
```

Then, with the replicate names that census prints:

```bash
PYTHONPATH=src .venv/bin/python tools/eval_kaist.py --data data/kaist \
    --bearing KAIST6205U --seconds 290 \
    --healthy normal_0 normal_1 normal_2 \
              normal_subset2-0 normal_subset2-1 normal_subset2-2 \
              normal_subset3-0 normal_subset3-1 normal_subset3-2 \
    --targets inner_0 inner_1 inner_2 outer_0 outer_1 outer_2 \
    --json runs/kaist_all.json 2>&1 | tee runs/kaist_all.log
```

Nine folds over full-length recordings is roughly thirty times the current run.
Start it and walk away.

---

## 5. Case Western — only to settle a sentence

Part 1 §7.4 states the Case Western set is used under its published terms, and
**no Case Western result appears anywhere in the volume.** That sentence is
currently describing work the reader cannot find.

Two honest options, and the first is cheaper: **delete the clause.** The second
is to re-download from the CWRU Bearing Data Center, drop the `.mat` files flat
into `data/cwru/`, and run:

```bash
PYTHONPATH=src .venv/bin/python tools/eval_dataset.py --dataset cwru \
    --data data/cwru --out runs/cwru.json 2>&1 | tee runs/cwru.log
```

Only worth the download if you want a CWRU number in §3. Otherwise strike the
clause and the volume is consistent again.

---

## Not recommended

**MFPT.** The loader and a verified geometry exist, and the evidence base already
has four bearing rigs. A fifth demonstrates portability that four already
demonstrate, and costs pages in a volume sitting at exactly 20.

**The Paderborn machined-defect bearings** — twelve codes. The volume's argument
is specifically about accelerated-lifetime damage, so adding seeded faults
dilutes the claim rather than strengthening it. `--real-damage-only` enforces
that in code.

---

## Order of work

1. **Paderborn ×3** — twenty minutes, settles the KA04 question
2. **uOttawa** — the second variable-speed rig
3. **XJTU-SY** — prognosis, once the loader lands
4. **KAIST 2+3** — raises `n` from 3 to 9
5. **CWRU** — only if you want the clause kept

Send me each log as it finishes and I will fold the numbers in as they arrive
rather than all at the end.
