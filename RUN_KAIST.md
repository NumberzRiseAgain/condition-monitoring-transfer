# Running the KAIST experiment yourself

Paste-safe. No inline `#` comments anywhere — interactive zsh does not strip
them, and `pytest tests/ -q  # 65 passed` gives you `ERROR: file or directory
not found: #`, which looks like a broken suite and is not one.

## 0. Once

```bash
cd /path/to/cbmx
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
```

If `pip install -e` refuses, skip it and prefix every command below with
`PYTHONPATH=src`. That route is fully supported; see `SETUP.md` section 2.

## 1. The data

Download subset 1 from `data.mendeley.com/datasets/vxkj334rzv` and put the
`vibration_*.csv`, `current_*.csv` and `rpm_*.csv` files directly in
`data/kaist`. You already have these under `~/Downloads/kaist`, so:

```bash
mkdir -p data
ln -s ~/Downloads/kaist data/kaist
```

The tool reads only the first `--seconds` of each file, so the truncated copies
work as they are.

## 2. Check the install before running anything

```bash
python tools/preflight.py
python -m pytest tests/test_kaist.py -q
```

Expect `7 passed`. Two of those seven are data-backed: one re-derives the
25.6 kHz sampling rate from the data, the other checks the tachometer really is
misaligned. If they skip rather than pass, `data/kaist` is not where the tool
looks.

## 3. The main run

```bash
python tools/eval_kaist.py --data data/kaist --bearing KAIST6205U \
    --targets inner_0 inner_1 inner_2 outer_0 outer_1 outer_2 \
    --json runs/kaist_loo.json
```

About four minutes. Read section 5 first: detection and false alarm are on the
same line and neither means anything without the other.

What you should see: inner race named in 9 of 9 record-folds, outer race in 0
of 9, zero alarms on 3 of 3 held-out healthy recordings, and the held-out
healthy record's accumulator never rising above −0.11 against a +3.81 boundary.

## 4. The three runs that make the main one believable

**The tachometer ablation.** Everything else identical; the speed reference is
left as shipped.

```bash
python tools/eval_kaist.py --data data/kaist --bearing KAIST6205U \
    --targets inner_0 inner_1 inner_2 outer_0 outer_1 outer_2 --no-align
```

Detection drops to 0 of 6 while the false-alarm record stays perfect. That is
what a broken monitor looks like from the outside, and it is the reason the two
numbers are never printed apart.

**Band sensitivity.** The kurtosis vote was 9–8–7, close enough that a reviewer
will ask whether the band is doing the work. Run any of these; the conclusion
does not move.

```bash
python tools/eval_kaist.py --data data/kaist --bearing KAIST6205U \
    --targets inner_0 inner_1 inner_2 outer_0 outer_1 outer_2 --band 10400 11200
python tools/eval_kaist.py --data data/kaist --bearing KAIST6205U \
    --targets inner_0 inner_1 inner_2 outer_0 outer_1 outer_2 --band 500 12800
```

The second is the interesting one: 0.5–12.8 kHz is the whole usable spectrum,
which is to say no band selection at all. Inner 9/9, outer 0/9, 0 alarms.

**The rest of the evidence base**, unchanged by any of this:

```bash
python -m pytest tests/ -q
python -m cbmx.cli physics
python -m cbmx.cli demo
python tools/eval_semantic.py
python tools/eval_actuator.py --data actuator_dataset/Data
```

## 5. Flags worth knowing

| flag | what it does |
|---|---|
| `--bearing` | required, no default, because the dataset publishes no geometry |
| `--no-align` | leaves the tachometer as shipped; the ablation above |
| `--band LO HI` | overrides the commissioned demodulation band |
| `--seconds N` | how much of each recording to read; default 74 |
| `--window N` | window length in seconds; default 1.0 |
| `--min-samples N` | observations before a line may be scored; default 25 |
| `--healthy` | which recordings are rotated leave-one-out |
| `--channels` | default is all four accelerometers |
| `--json PATH` | writes every number the console prints |

## 6. If something looks wrong

`no rpm_<cond>_<rep>.csv` — the speed reference is not optional; the tool will
not silently proceed without one.

Every line reads `no_baseline` — the baseline has fewer than `--min-samples`
observations in that regime. Raise `--seconds` or lower `--min-samples`, and
say which you did.

Detection 0 and false alarms 0 — check whether `--no-align` is set. That
combination is the failure signature, not a clean result.
