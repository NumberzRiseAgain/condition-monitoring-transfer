# Working on cbmx

## The two gates

Run both before every commit. CI runs exactly these, on Python 3.9 and 3.12.

```bash
pip install -e ".[dev]"

flake8 src/cbmx tools tests scripts     # 0 findings
python3 scripts/tidy.py --check         # 0 findings; drop --check to autofix
PYTHONPATH=src pytest tests/ -q         # 108 passed, 4 skipped
```

`scripts/tidy.py` removes unused imports and pointless `f` prefixes in place.
Both are mechanical, so they are fixed automatically instead of argued about in
review. It honours `# noqa`, so it agrees with flake8 rather than contradicting
it.

## What the lint config does and does not enforce

`setup.cfg` disables **E241/E272** — multiple spaces after `,` or `:`. Column
alignment inside dict and table literals is deliberate here: bearing geometry,
channel maps and published census counts are aligned so a wrong digit is visible
to a reader. The checker that objects to that is turned off; the alignment
stays.

Line length is 100, not 79.

## Writing a test

Every test names the defect it guards and says why that defect was silent.
This is not decoration. Three times in this project a run produced zero
detections with a perfect false-alarm record, and each time the tempting move
was to adjust something until it spoke. Twice the cause was a real bug — a
decimation helper resetting bearing geometry, a tachometer misaligned by half a
second — and once the honest answer was zero. A test whose docstring explains
which of those it is prevents the next person from having that argument again.

**Tests must not need a downloaded dataset.** Build the file layout in
`tmp_path` instead. `tests/test_loaders.py` and `tests/test_ims.py` show the
pattern: a handful of synthetic captures with the exact structure the loader
parses, including the malformed cases. Tests that do touch a real dataset skip
cleanly and say what is missing.

## Adding a dataset loader

1. Normalise to `io/base.Record`; keep the dataset's own vocabulary out of the
   rest of the codebase.
2. **Never invent bearing geometry or fault labels.** If the publisher does not
   state them, leave the table empty and let the evaluation decline to report
   the number that depends on it. `io/ims.py` and `io/xjtu.py` both do this, and
   both explain why in their module docstring.
3. Sniff the header row rather than assuming it. Mirrors differ, and a header
   read as data shifts every index by one without erroring.
4. Sort by the real ordering key. Capture filenames are timestamps or integers;
   sorted as text, `2003.11.01` precedes `2003.10.22` and `10` precedes `2`.
5. Add the data directory to `.gitignore`.

## Before claiming a result

Read `PROGNOSIS_STOPPING_RULE.md`. It was written and dated before the
run-to-failure data was downloaded, and it is the reason the prognosis claim in
the proposal is a withdrawal rather than a number. Write the rule down first;
that is the cheapest defence there is against your own hindsight.

**Detection and false alarms are always reported together.** A monitor that
never speaks passes a false-alarm test perfectly.
