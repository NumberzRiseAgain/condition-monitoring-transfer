#!/usr/bin/env bash
# Paderborn fleet sweep — every operating condition, both channels.
#
#   bash run_fleet.sh
#
# Runs from the repository root. Bootstraps its own Python environment if the
# one on PATH does not have numpy and scipy, then writes one log and one JSON
# per run into runs/ and prints a compact digest at the end. Paste the digest.
#
# First run takes a few extra minutes to build the venv. After that, ~25 min.

set -u
cd "$(dirname "$0")"
mkdir -p runs

# ── interpreter ─────────────────────────────────────────────────────────────
# Three cases, in order of preference: a venv we already built, a system Python
# that already has what we need, or build the venv now. Announcing which one is
# in use matters — "no module named numpy" from a script that looked like it
# worked yesterday is almost always this and nothing else.
PY=""
if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import numpy,scipy" 2>/dev/null; then
  PY=".venv/bin/python"
elif python3 -c "import numpy,scipy" 2>/dev/null; then
  PY="python3"
else
  echo "numpy/scipy not importable from python3 — building .venv"
  echo "(this takes 1-3 minutes and needs the network, once)"
  python3 -m venv .venv || { echo "venv creation failed; see SETUP.md section 2"; exit 1; }
  # macOS ships pip 21.2.4 and PEP 660 landed in 21.3. Upgrading first is not
  # optional; without it the install fails with a message about setup.py that
  # has nothing to do with this repository.
  .venv/bin/python -m pip install --quiet --upgrade pip setuptools wheel
  .venv/bin/python -m pip install --quiet numpy scipy pyyaml pandas
  .venv/bin/python -c "import numpy,scipy" 2>/dev/null || {
    echo "dependencies still missing after install; stopping"; exit 1; }
  PY=".venv/bin/python"
fi

echo "=============================================================="
echo "Paderborn fleet sweep — started $(date '+%H:%M:%S')"
echo "python: $PY  ($($PY -c 'import numpy,scipy,sys;print("py",sys.version.split()[0],"numpy",numpy.__version__,"scipy",scipy.__version__)'))"
echo "=============================================================="

BAND="--band 500 2000"
COMMON="--dataset paderborn --holdout bearing --real-damage-only"
CONDS="N15_M07_F10 N09_M07_F10 N15_M01_F10 N15_M07_F04"
FAILED=0

runone () {   # label, outstem, extra args...
  local label="$1"; local stem="$2"; shift 2
  echo
  echo ">>> $label"
  PYTHONPATH=src "$PY" tools/eval_dataset.py $COMMON "$@" \
      --out "runs/${stem}.json" > "runs/${stem}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    FAILED=$((FAILED+1))
    echo "    FAILED (exit $rc) -> runs/${stem}.log"
    tail -3 "runs/${stem}.log" | sed 's/^/      /'
  else
    echo "    ok -> runs/${stem}.log"
  fi
}

for C in $CONDS; do
  D="data/pu_$C"
  if [ ! -d "$D" ]; then
    echo; echo "skip $C  (no $D)"
    continue
  fi
  runone "$C  vibration  ($(ls "$D"/*.mat 2>/dev/null | wc -l | tr -d ' ') records)" \
         "sweep_${C}_vib" --data "$D" $BAND
done

runone "N15_M07_F10  motor current  (the channel AAG actually has)" \
       "sweep_N15_M07_F10_cur" --data data/pu_N15_M07_F10 --channel current $BAND

runone "N15_M07_F10  vibration, band auto-commissioned  (the ablation)" \
       "sweep_N15_M07_F10_autoband" --data data/pu_N15_M07_F10

echo
echo "=============================================================="
echo "DIGEST — paste this part back"
echo "=============================================================="
for f in runs/sweep_*.log; do
  [ -e "$f" ] || continue
  echo
  echo "--- $(basename "$f")"
  grep -E "^BAND|healthy [0-9]+ +faults|FALSE ALARMS|DETECTION|ModuleNotFound|Error" "$f" | head -6
  sed -n '/bearing     fault/,/DETECTION/p' "$f" | head -12
done
echo
echo "runs that failed: $FAILED"
echo "finished $(date '+%H:%M:%S')"
