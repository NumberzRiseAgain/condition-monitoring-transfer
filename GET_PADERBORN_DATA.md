# Finishing the Paderborn evidence base

Part 1 §3.1 currently reads *25 of 25 outer race named, 0 wrong-part calls,
0 of 80 false alarms.* Those are real numbers and the code reproduces them
exactly. What the volume does not say loudly enough is that they come from
**one** outer-race bearing, **one** inner-race bearing and **one** healthy
bearing. §5.6 admits the remaining bearings were not run.

The archive holds 32. Finishing it turns a single-bearing claim into a fleet
claim, and turns the false-alarm number from *unseen segments of a bearing the
baseline learned on* into *a healthy bearing the baseline has never met* — which
is the question a reviewer actually asks.

The container cannot reach `groups.uni-paderborn.de` (HTTP 403 from the egress
proxy), so this download is yours. It is not large.

---

## 1. What to download

`https://groups.uni-paderborn.de/kat/BearingDataCenter/` — a plain directory
index, one `.rar` per bearing, about 160 MB each.

**You already have K001, KA04 and KI21.** Fetch these fourteen:

| | codes | why |
|---|---|---|
| Healthy | `K002 K003 K004 K005 K006` | five more healthy bearings — this is what makes leave-one-bearing-out possible at all |
| Outer race, real damage | `KA15 KA16 KA22 KA30` | accelerated-lifetime damage, not machined |
| Inner race, real damage | `KI04 KI14 KI16 KI17 KI18` | same |

About **2.2 GB**. Skip `KA01/03/05/06/07/08/09` and `KI01/03/05/07/08` — those
are machined defects, and the volume's argument is explicitly about real
accelerated-lifetime damage, so including them would weaken the claim rather
than strengthen it. Skip `KB23/24/27` too; they are combined faults and the
evaluation excludes that class.

```bash
cd ~/Downloads
mkdir -p paderborn_rar && cd paderborn_rar
for C in K002 K003 K004 K005 K006 KA15 KA16 KA22 KA30 KI04 KI14 KI16 KI17 KI18; do
  curl -L -O "https://groups.uni-paderborn.de/kat/BearingDataCenter/$C.rar"
done
ls -lh
```

Fourteen files, each 150–170 MB. If any comes back a few kilobytes, it failed —
re-run that one.

## 2. Extract

Each archive expands to a folder of `.mat` files named
`N15_M07_F10_KA15_1.mat` and so on. **Keep those filenames.** The loader refuses
to guess labels from anything else, on purpose.

On macOS, `unar` handles RAR and installs cleanly from Homebrew:

```bash
brew install unar
cd ~/Downloads/paderborn_rar
for f in *.rar; do unar -q -o ~/Downloads/paderborn "$f"; done
```

If you would rather not install anything, The Unarchiver from the App Store does
the same job by double-click; extract everything into `~/Downloads/paderborn`.

Then flatten so each code has its own directory of `.mat` files:

```bash
cd ~/Downloads/paderborn
for d in */; do echo "$d $(ls "$d"/*.mat 2>/dev/null | wc -l) mat files"; done
```

You should see 14 directories with 80 `.mat` files each — 4 operating
conditions × 20 runs.

> **A trap worth knowing.** A half-extracted archive does not error politely. It
> gives `OSError: could not read bytes` from deep inside scipy. If you see that,
> the archive did not finish; re-extract it rather than debugging the loader.

## 3. Point the repository at it

```bash
cd /path/to/cbmx
for d in ~/Downloads/paderborn/*/; do
  ln -s "$d"/*.mat data/paderborn/ 2>/dev/null
done
ls data/paderborn/*.mat | wc -l
```

Expect roughly 1,200 files once the existing three codes are counted in. The
loader reads the bearing code out of each filename, so they can all live in one
directory.

## 4. Run it

**The fleet run, with the false-alarm question asked properly:**

```bash
python tools/eval_dataset.py --dataset paderborn --data data/paderborn \
    --holdout bearing --real-damage-only \
    --out runs/paderborn_fleet.json
```

Each of the six healthy bearings is held out in turn while the baseline is
fitted on the other five and frozen. It then scores the held-out healthy
bearing — those are the false alarms — and every damaged bearing, in every fold.

Expect an hour or two; 1,200 records at 64 kHz is real work.

**The comparison run**, which is the number currently in the volume:

```bash
python tools/eval_dataset.py --dataset paderborn --data data/paderborn \
    --holdout record --real-damage-only \
    --out runs/paderborn_record.json
```

Same data, weaker hold-out. Printing both is how we show the fleet number is not
an artefact of the protocol.

**And the current-channel version**, which is the one that matters for AAG,
since the deployed gear has motor current and no accelerometers:

```bash
python tools/eval_dataset.py --dataset paderborn --data data/paderborn \
    --channel current --holdout bearing --real-damage-only \
    --out runs/paderborn_fleet_current.json
```

## 5. What to expect, and what would be worth knowing

The numbers that should hold if the method is real: **0 wrong-part calls** and a
detection rate on outer-race bearings near the current 25/25.

The numbers that will probably get worse, and should: **inner race**. §5.1
already records 0 of 15 at the mildest grade, and five more inner-race bearings
will very likely confirm that the inner-race model is weaker, not stronger.
That is worth having in the volume before a reviewer finds it.

The number that matters most: **false alarms on the five healthy bearings the
baseline has never seen.** If that stays at zero, §3.1 becomes a much stronger
sentence than it is today. If it does not, we need to know now, and the volume
says so instead.

Whatever comes back goes in as it comes back.
