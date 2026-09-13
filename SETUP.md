# Running cbmx from your terminal

Every command below was executed on a clean checkout with an empty `data/`
directory on 29 August 2026, and the output quoted is what it actually printed.
Python 3.11.15, numpy 2.4.6, scipy 1.17.1, pandas 3.0.5.

---

## 1. Prerequisites

You need **Python 3.9 or newer** and nothing else. No compiler, no CUDA, no
Docker. Check what you have:

```bash
python3 --version
```

macOS ships a usable Python 3; on Windows use the python.org installer and tick
"Add python.exe to PATH", or use WSL, where these instructions work unchanged.

---

## 2. Install

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

On Windows the activate line is `.venv\Scripts\activate` instead. The pip
upgrade is not optional — see below.

> **Nothing in this guide has a `#` comment on the end of a command line.**
> macOS uses zsh, and interactive zsh does not treat `#` as a comment unless
> `interactive_comments` is set. Pasting `pytest tests/ -q  # 65 passed` gives
> you `ERROR: file or directory not found: #`, which looks like a broken test
> suite and is not one. Every command below is safe to paste whole.

> ### The upgrade line is the one that breaks people
>
> If you skip it on macOS you get:
>
> ```
> ERROR: File "setup.py" or "setup.cfg" not found.
> Directory cannot be installed in editable mode
> (A "pyproject.toml" file was found, but editable mode currently
>  requires a setuptools-based build.)
> ```
>
> Nothing is wrong with the repository. **pip gained PEP 660 — editable
> installs driven from `pyproject.toml` alone — in version 21.3, and the
> Python that ships with macOS carries pip 21.2.4.** One release below the
> line. Upgrading pip inside the venv fixes it outright; verified against
> pip 21.2.4, which reproduces the error exactly, and again after the
> upgrade, which does not.
>
> `python tools/preflight.py` now checks for this and names it.

`-e` installs in editable mode, so the source stays where it is and your edits
take effect immediately. The dependencies are numpy, scipy, pyyaml and pandas.

### If `pip install -e` still refuses, skip it entirely

Editable installation is a convenience, not a requirement. This works with no
install of `cbmx` at all:

```bash
python -m pip install numpy scipy pyyaml pandas pytest
PYTHONPATH=src python tools/preflight.py
PYTHONPATH=src python -m pytest tests/ -q
PYTHONPATH=src python -m cbmx.cli demo
```

That last command should print five numbered sections ending in `THE THREE
CONTRACT NUMBERS` and a line saying the signals are synthetic. If you see that,
the install is sound.

Prefix every command in this guide with `PYTHONPATH=src` and everything behaves
identically — tests included, 65 passed. The only thing you lose is the bare
`cbmx` console command; use `python -m cbmx.cli` instead.

### A note on Python 3.9

3.9 works. It is what macOS ships and the code supports it. But current numpy,
scipy and pandas wheels need 3.10 or newer, so pip will resolve older pinned
versions for you. Nothing here depends on a new feature of any of them. If you
have a newer Python — `brew install python@3.12`, then
`python3.12 -m venv .venv` — prefer it, because you will get the versions this
was tested against.

> **If `python3 -m venv .venv` fails on Debian or Ubuntu** with a message about
> `ensurepip`, install `python3-venv` first:
> `sudo apt install python3-venv python3-pip`.

> **Creating the venv takes 30–60 seconds** the first time because it bootstraps
> pip. It looks like it has hung. It has not.

---

## 3. Check the install before running anything

```bash
python tools/preflight.py
```

This prints your Python version, every dependency and its version, whether
`cbmx` imports, which datasets are present, and — the useful part — **exactly
which tools you can run right now** and which are waiting on a dataset. It
exits non-zero if a prerequisite is genuinely missing.

On a fresh install with no datasets it ends:

```
  RUN     cbmx demo / physics
  RUN     tools/eval_semantic
  skip    tools/eval_latency       needs data/paderborn
  skip    tools/eval_dataset       needs data/paderborn
  ...
Prerequisites satisfied. The no-dataset commands will run.
```

---

## 4. Run the tests

```bash
python -m pytest tests/ -q
```

Expected on a clean checkout with no datasets:

```
65 passed, 2 skipped in 2.6s
```

The 2 skips are the tests that need the large public datasets. They skip
cleanly and say why. **If you see failures rather than skips, stop** — something
is wrong with the install, not with the data.

---

## 5. The three things that run with no data at all

Start here. None of these needs a download.

### `cbmx physics` — the geometry, before any data exists

```bash
python -m cbmx.cli physics
```

Prints the characteristic defect orders for each bearing in the catalogue,
derived from four caliper dimensions, with the kinematic identities checked.
This is the claim that the search is one line wide *before* the machine has been
observed. Note it flags the uOttawa dimensions as **UNVERIFIED against the
dataset paper** — that is deliberate.

### `cbmx demo` — commission, learn normal, then plant faults

```bash
python -m cbmx.cli demo
```

Runs the whole loop end to end on synthesised signals: chooses a demodulation
band once and freezes it, learns normal from six minutes of unlabelled healthy
running, then plants each fault in turn. Prints latency, confidence and MB/hour
per fault, and dumps one full evidence record so you can see what actually
crosses the link.

The row that matters is the first one: planted `none` reports **nothing**.

### `tools/eval_semantic.py` — the gate, and what it throws away

```bash
python tools/eval_semantic.py
```

Runs the asset-graph traversal, retrieval over mock work orders, part resolution
and the nine deterministic checks, then feeds it four deliberately malformed
hypotheses. Ends with `gate rejected 4 of 4 malformed hypotheses`, each with the
specific assertion that failed.

---

## 6. The tools that need a dataset

None of these datasets ship with the repository — together they are about
1.9 GB and each is licensed individually. Put them where the loaders expect:

| Put it here | Dataset | Licence |
|---|---|---|
| `data/paderborn/<CODE>/*.mat` | Paderborn KAT bearing | **CC BY-NC 4.0 — non-commercial** |
| `data/hydraulic/*.txt` + `profile.txt` | ZeMA hydraulic rig via UCI | UCI ML Repository terms |
| `data/cwru/*.mat` | Case Western bearing | CWRU Bearing Data Center terms |
| `data/ottawa/*.mat` | uOttawa variable speed | Mendeley Data terms |
| `data/kaist/*.csv` | KAIST varying-speed bearing, subset 1 | Mendeley Data terms (CC BY 4.0) |

Then:

```bash
python tools/eval_dataset.py --dataset paderborn --data data/paderborn
python tools/eval_hydraulic.py
python tools/eval_hydraulic.py --no-freeze
python tools/eval_hydraulic.py --time-split
python tools/eval_mcsa.py
python tools/eval_current_band.py
python tools/eval_transient.py
python tools/eval_prognosis.py
python tools/bench_edge.py
python tools/eval_latency.py
```

`eval_latency.py` prints the three latencies — L1 feature computation, L2
sample-arrival to anomaly indication, L3 to a confirmed diagnosis. L2 is the
number the topic's `<1 s` threshold asks for. Without Paderborn it exits
quietly with `no records for KA04`, so an empty-looking run means the dataset,
not a broken install.

A missing dataset gives a plain message — `no records under data/paderborn/KA04`
— not a stack trace. A **half-extracted** one gives a scipy `OSError: could not
read bytes`. If you see that, the archive did not extract fully; re-extract it.

**Paderborn extraction.** The archives are RAR. Debian's `7zip-rar` codec
segfaults on them and `unar` will not install on current Debian or Ubuntu. Use
`bsdtar` from `libarchive-tools`, which handles them in about nine seconds each:

```bash
sudo apt install libarchive-tools
bsdtar -xf N09_M07_F10_KA04.rar -C data/paderborn/KA04
```

---

## 7. The actuator experiment — the one you can run today

This is the only dataset-backed experiment whose data is a plain `git clone`,
because it is MIT licensed and hosted on GitHub. About 156 MB.

```bash
git clone https://github.com/vtnsi/actuator_dataset
python tools/eval_actuator.py --data actuator_dataset/Data
```

Takes a couple of minutes. It prints, in order: the **portability screen**
(fitted on one healthy actuator, scored on a second healthy actuator, using no
fault label), then the two matched-load damage experiments, then the
healthy-versus-healthy control, which must never fire.

Add `--json runs/actuator.json` to keep the numbers.

Read `RUNLOG_2026-08-29_actuator.md` beside it — it records what this
establishes, and the four things it does not.

---

## 7a. The varying-speed experiment

This is the one that removes the fixed-speed caveat: real bearing damage under
a randomly varying 613–2480 rpm, with a tachometer to check the answer against.

Download subset 1 from `data.mendeley.com/datasets/vxkj334rzv` — it is a large
nested archive; put the `vibration_*.csv`, `current_*.csv` and `rpm_*.csv`
files directly in `data/kaist`. The vibration files are around 600 MB of ASCII
each; the tool reads only the first `--seconds` of each, so you do not need
them all to see a result.

```bash
python tools/eval_kaist.py --data data/kaist --bearing KAIST6205U \
    --targets inner_0 inner_1 inner_2 outer_0 outer_1 outer_2 \
    --json runs/kaist_full.json
```

About a minute for six recordings. Two flags worth knowing:

`--no-align` reproduces the failure the run log opens with. The tachometer and
signal files do not start together, and without the correction the run detects
nothing while reporting a perfect false-alarm record — which is what a broken
monitor looks like from the outside.

`--bearing` is required and has no default, because **the dataset publishes no
bearing geometry at all**. `KAIST6205U` is an assumption, flagged UNVERIFIED in
the catalogue, and the tool prints a banner saying so. If you have the four
dimensions from the Data in Brief paper, add a catalogue entry with them and
use that instead.

Read `RUNLOG_2026-08-29_kaist.md` beside it — the inner race is found on 3 of 3
recordings and the outer race on 0 of 3, and the log says what that does and
does not establish.

---

## 8. If something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `File "setup.py" or "setup.cfg" not found` | pip older than 21.3 (macOS ships 21.2.4) | `python -m pip install --upgrade pip setuptools wheel`, then reinstall |
| `ModuleNotFoundError: No module named 'cbmx'` | venv not activated, or `pip install -e` not run | `source .venv/bin/activate`, then reinstall — or use the `PYTHONPATH=src` route in section 2 |
| `No module named pytest` | the `[dev]` extra never installed because the editable install failed | fix the pip version first; the two failures are the same failure |
| `ERROR: file or directory not found: #` | you pasted a command with a trailing `# comment` into zsh | re-run without the comment; zsh does not strip `#` interactively |
| `no tests ran in 0.00s` | same as above — pytest got `#` as a path | re-run the bare command |
| `ModuleNotFoundError: No module named 'pandas'` | installed before pandas was declared a dependency | `python -m pip install -e ".[dev]"` again |
| `does not appear to be a Python project` | you are not in the repository root | `cd` to the directory holding `pyproject.toml` |
| `FileNotFoundError: data/hydraulic/profile.txt` | dataset absent | Section 6, or just skip that tool |
| `OSError: could not read bytes` | archive extracted partially | re-extract with `bsdtar` |
| tests **fail** rather than skip | broken install | delete `.venv` and redo section 2 |

`python tools/preflight.py` answers most of these before you hit them.

---

## 9. Two things worth knowing before you trust a result

Both are in `RUNLOG_2026-08-25.md`, and both were silent.

**A run that never speaks passes a false-alarm test perfectly.** A decimation
helper once reset the bearing geometry, so every fault line landed on empty
spectrum and the run reported 0 detections and 0 false alarms across 122
records. Never read a false-alarm rate without the detection rate beside it.

**A result that does not move when the inputs move is not a result.** Two
functions with the same name returned opposite pairs, so an order of 3.05 was
read as 3.05 dB and every condition returned the same number.

Both now have regression tests in `tests/test_v2_modules.py`.
