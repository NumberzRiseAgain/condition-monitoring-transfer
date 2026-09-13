# Getting the rest of the KAIST data

Two jobs, one download session. Both raise the number that is currently the
weakest thing in the run.

| | what it fixes | how much |
|---|---|---|
| Subsets 2 and 3 | n = 3 healthy recordings | up to n = 9 |
| Un-truncated files | 74 s of each 298 s recording | 4× the windows per record |

The second one matters more than it sounds. `inner_2` currently abstains on 42%
of its line-windows because its speed distribution reaches a regime bin the
baseline never filled. Four times the data fills those bins.

---

## 1. Space

Nine untruncated vibration files run about 600 MB each, and there are three
subsets. Check before you start:

```bash
df -h ~
du -sh ~/Downloads/kaist
```

You want **35 GB free** to be comfortable: roughly 16 GB of archives, which you
delete after extracting, and 16 GB of CSVs. If that is tight, see §5 — you do
not need the current files to redo the vibration result.

## 2. Download

Three separate Mendeley records, all from the same group:

| subset | DOI landing page |
|---|---|
| Subset 1 | `data.mendeley.com/datasets/vxkj334rzv` |
| Subset 2 | `data.mendeley.com/datasets/x3vhp8t6hg` |
| Subset 3 | `data.mendeley.com/datasets/j8d8pfkvj2` |

Take the **latest version** of each — Subset 1 is now at version 7 and the copy
we used was an earlier one, so redownloading is worth it on its own. Use the
"Download all" button on each page.

`data.mendeley.com` is blocked from the analysis environment, which is why this
step is yours rather than mine.

## 3. Extract, keeping them apart

**Do not flatten the three subsets into one folder.** They reuse identical file
names — `vibration_normal_0.csv` exists in all three — so a flat extract
silently overwrites two thirds of your data and the run reports a smaller n
than it thinks it has. A test now guards this
(`test_sibling_subsets_do_not_overwrite_each_other`) but the guard only helps if
the files are in separate directories to begin with.

```bash
mkdir -p ~/Downloads/kaist_all/subset1
mkdir -p ~/Downloads/kaist_all/subset2
mkdir -p ~/Downloads/kaist_all/subset3
```

Extract each download into its matching folder. The archives are nested — an
outer zip containing `part1.zip` and so on — so you may need two passes. Then
flatten *within* each subset folder only, so the CSVs sit directly inside it:

```bash
cd ~/Downloads/kaist_all
for d in subset1 subset2 subset3; do
  find "$d" -mindepth 2 -name '*.csv' -exec mv -n {} "$d"/ \;
  find "$d" -mindepth 1 -type d -empty -delete
done
```

Check you got what you expect:

```bash
for d in subset1 subset2 subset3; do
  echo "$d: $(ls $d/vibration_*.csv 2>/dev/null | wc -l) vibration, $(ls $d/rpm_*.csv 2>/dev/null | wc -l) rpm"
done
```

Every subset folder needs its **own** `rpm_*.csv` files. The speed reference is
matched to the recording, not to the condition, and the tool refuses to proceed
without one rather than guessing.

## 4. Point the repository at it

Subset 1 goes in the root, the other two go in subdirectories. The loader folds
the directory name into the replicate, so Subset 2's healthy recording 0 becomes
`normal_subset2-0`.

```bash
cd /path/to/cbmx
rm -f data/kaist
mkdir -p data/kaist
ln -s ~/Downloads/kaist_all/subset1/*.csv data/kaist/
ln -s ~/Downloads/kaist_all/subset2 data/kaist/subset2
ln -s ~/Downloads/kaist_all/subset3 data/kaist/subset3
```

Confirm the tool sees all of it — this prints a census before it scores
anything:

```bash
python tools/eval_kaist.py --data data/kaist --bearing KAIST6205U \
    --targets inner_0 --seconds 20 --min-samples 8 | head -20
```

## 5. Run it on everything

Substitute the actual replicate names from the census. Assuming three healthy
and three of each fault per subset:

```bash
python tools/eval_kaist.py --data data/kaist --bearing KAIST6205U \
    --seconds 290 \
    --healthy normal_0 normal_1 normal_2 \
              normal_subset2-0 normal_subset2-1 normal_subset2-2 \
              normal_subset3-0 normal_subset3-1 normal_subset3-2 \
    --targets inner_0 inner_1 inner_2 outer_0 outer_1 outer_2 \
              inner_subset2-0 outer_subset2-0 \
              inner_subset3-0 outer_subset3-0 \
    --json runs/kaist_all.json
```

`--seconds 290` is the change that uses the full recordings. Nine healthy
recordings means nine leave-one-out folds, so this run is roughly thirty times
the work of the current one — expect an hour or two rather than four minutes.
Start it and walk away.

**If you only want the n fix and not the length fix**, drop `--seconds 290` and
it stays at 74 s per record and runs in about fifteen minutes. The n = 3 → n = 9
improvement is the bigger of the two for the volume.

**If space is tight**, you can delete every `current_*.csv` as you go. Nothing
in the vibration result touches them, and until the authors give us the current
sampling rate they cannot be used anyway.

## 6. What to expect

If the method holds, the numbers that should barely move are the detection rate
and the false-alarm count; the ones that should improve are the abstention rate
and the window at which faults are named. If the false-alarm count stops being
zero once n is nine rather than three, **that is the result** and it goes in the
volume as the result. Finding that out before a reviewer does is the entire
reason for doing this.
