#!/usr/bin/env bash
# One script. Runs every dataset you have, prints what it got next to what it
# should get, and says whether they match.
#
#   bash verify.sh
#
# Datasets you do not have are skipped and named, so a short scorecard means
# missing data, never a hidden failure. Nothing here needs the network.
#
# The Paderborn views under data/pu_* are symlink farms pointing at one shared
# extraction. If that extraction moves or is deleted the links survive as
# dangling names, so every dataset guard below counts files that RESOLVE
# (`find -L ... -type f`) and not names that merely exist. Counting names lets
# the run start against an empty set and report a false alarm rate of zero on
# nothing at all, which is the exact failure this script exists to catch.
#
#   PADERBORN=/path/to/paderborn bash verify.sh    # if not at data/paderborn

set -u
cd "$(dirname "$0")"
mkdir -p runs

# Where the shared Paderborn extraction lives (<CODE>/*.mat, one dir per
# bearing). Overridable, because it is not the same place on every machine.
PADERBORN="${PADERBORN:-}"
if [ -z "$PADERBORN" ]; then
  for cand in data/paderborn ../paderborn ~/Downloads/paderborn; do
    [ -d "$cand" ] && { PADERBORN="$cand"; break; }
  done
fi

# Count the .mat files under $1 that actually resolve.
n_mat () { find -L "$1" -maxdepth 1 -name '*.mat' -type f 2>/dev/null | wc -l; }

# ── interpreter ─────────────────────────────────────────────────────────────
PY=""
if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import numpy,scipy" 2>/dev/null; then
  PY=".venv/bin/python"
elif python3 -c "import numpy,scipy" 2>/dev/null; then
  PY="python3"
else
  echo "Building .venv (once, needs the network)..."
  python3 -m venv .venv && .venv/bin/python -m pip install -q --upgrade pip setuptools wheel \
    && .venv/bin/python -m pip install -q numpy scipy pyyaml pandas || {
      echo "could not build the environment; see SETUP.md"; exit 1; }
  PY=".venv/bin/python"
fi

GOT=(); WANT=(); NAME=(); NOTE=()
record () { NAME+=("$1"); GOT+=("$2"); WANT+=("$3"); NOTE+=("$4"); }

banner () { echo; echo "── $1"; }

echo "=============================================================="
echo "cbmx verification — $(date '+%Y-%m-%d %H:%M')"
echo "python: $PY"
echo "=============================================================="

# ── 1. the code itself ──────────────────────────────────────────────────────
banner "Unit tests"
T=$(PYTHONPATH=src "$PY" -m pytest tests/ -q 2>&1 | tail -1)
echo "   $T"
# Some tests are data-gated: 108 pass on a clone with no datasets, 110 once the
# hydraulic and Paderborn sets are present.  A hardcoded 108 therefore prints
# DIFFERENT on a perfectly good run, which teaches the reader to ignore the
# scorecard.  The real pass condition is: at least 108 passed and nothing failed.
T_PASS=$(echo "$T" | grep -oE '[0-9]+ passed' | grep -oE '[0-9]+' | head -1)
T_FAIL=$(echo "$T" | grep -oE '[0-9]+ (failed|error)' | grep -oE '[0-9]+' | head -1)
T_FAIL=${T_FAIL:-0}
T_PASS=${T_PASS:-0}
if [ "$T_PASS" -ge 108 ] && [ "$T_FAIL" -eq 0 ]; then
  record "unit tests" "108+ passed, 0 failed" "108+ passed, 0 failed" \
         "the regression suite, including every bug we have found"
else
  record "unit tests" "$T_PASS passed, $T_FAIL failed" "108+ passed, 0 failed" \
         "the regression suite, including every bug we have found"
fi

# ── 2. no data needed ───────────────────────────────────────────────────────
banner "Runs with no dataset at all"
S=$(PYTHONPATH=src "$PY" tools/eval_semantic.py 2>&1 | grep -oE "gate rejected [0-9]+ of [0-9]+" | tail -1)
echo "   semantic gate: $S"
record "semantic gate" "$S" "gate rejected 4 of 4" \
       "four deliberately malformed hypotheses, all discarded"

# ── 3. Paderborn, the main evidence base ────────────────────────────────────
if [ -d data/pu_N15_M07_F10 ] && [ "$(n_mat data/pu_N15_M07_F10)" -gt 100 ]; then
  banner "Paderborn — real bearing damage, 4 operating conditions"
  for C in N15_M07_F10 N09_M07_F10 N15_M01_F10 N15_M07_F04; do
    [ -d "data/pu_$C" ] || continue
    if [ "$(n_mat "data/pu_$C")" -lt 100 ]; then
      echo "   $C   SKIPPED — $(ls "data/pu_$C" | wc -l) names present, none resolve"
      continue
    fi
    PYTHONPATH=src "$PY" tools/eval_dataset.py --dataset paderborn --data "data/pu_$C" \
        --holdout bearing --real-damage-only --band 500 2000 \
        --out "runs/v_pu_$C.json" > "runs/v_pu_$C.log" 2>&1
    D=$(grep -oE "DETECTION [0-9]+/[0-9]+" "runs/v_pu_$C.log" | head -1)
    F=$(grep -oE "FALSE ALARMS [0-9]+ of [0-9]+" "runs/v_pu_$C.log" | head -1)
    echo "   $C   $D   $F"
    record "Paderborn $C" "$F" "FALSE ALARMS 0 of" \
           "no alarm on any healthy bearing the baseline had never met"
  done
  banner "Paderborn — the band ablation (this one SHOULD fail)"
  PYTHONPATH=src "$PY" tools/eval_dataset.py --dataset paderborn \
      --data data/pu_N15_M07_F10 --holdout bearing --real-damage-only \
      --out runs/v_pu_autoband.json > runs/v_pu_autoband.log 2>&1
  A=$(grep -oE "DETECTION [0-9]+/[0-9]+" runs/v_pu_autoband.log | head -1)
  echo "   letting the band be chosen automatically: $A"
  record "band ablation" "$A" "DETECTION 0/" \
         "with the band chosen automatically it detects NOTHING — that is the point"
else
  banner "Paderborn — SKIPPED, no data"
  if [ -d data/pu_N15_M07_F10 ]; then
    echo "   $(ls data/pu_N15_M07_F10 | wc -l) linked names are present but none of them"
    echo "   resolve: the shared extraction they point at has moved or been deleted."
    echo "   Re-extract it, then rebuild the views. See DOWNLOADS.md section 1."
  else
    echo "   see DOWNLOADS.md section 1"
  fi
fi

# ── 4. Paderborn, vibration against motor current ───────────────────────────
# Reads the shared extraction directly rather than a per-condition view,
# because it needs both channels of the same capture.
if [ -n "$PADERBORN" ] && [ -d "$PADERBORN" ]; then
  banner "Vibration vs motor current, same bearings, same captures"
  PYTHONPATH=src "$PY" tools/eval_mcsa.py --data "$PADERBORN" \
      --healthy K002 --json runs/v_mcsa.json > runs/v_mcsa.log 2>&1
  V=$(sed -n '/══ vibration/,$p' runs/v_mcsa.log | grep -c "OK ")
  C=$(sed -n '/══ current/,/══ vibration/p' runs/v_mcsa.log | grep -c "OK ")
  echo "   vibration named the damaged part on $V bearing-conditions"
  echo "   motor current named it on $C"
  record "current vs vibration" "current $C" "current 0" \
         "the sensors the gear has today see nothing; that is the case for adding one"
else
  banner "Vibration vs motor current — SKIPPED, no Paderborn extraction"
  echo "   looked in data/paderborn, ../paderborn, ~/Downloads/paderborn"
  echo "   set PADERBORN=/path/to/paderborn to point it somewhere else"
fi

# ── 5. KAIST, changing speed ────────────────────────────────────────────────
if [ -d data/kaist ] && \
   [ "$(find -L data/kaist -maxdepth 1 -name 'vibration_*.csv' -type f 2>/dev/null | wc -l)" -gt 3 ]
then
  banner "KAIST — speed changing constantly, like an arrestment"
  PYTHONPATH=src "$PY" tools/eval_kaist.py --data data/kaist --bearing KAIST6205U \
      --targets inner_0 inner_1 inner_2 outer_0 outer_1 outer_2 \
      --json runs/v_kaist.json > runs/v_kaist.log 2>&1
  K=$(grep -oE "FALSE ALARM [0-9]+ of [0-9]+" runs/v_kaist.log | tail -1)
  echo "   $K held-out healthy recordings"
  sed -n '/record  *fit/,/^$/p' runs/v_kaist.log | tail -8
  record "KAIST" "$K" "FALSE ALARM 0 of" "no alarm on any held-out healthy recording"
else
  banner "KAIST — SKIPPED, no data"; echo "   see DOWNLOADS.md section 4"
fi

# ── 6. uOttawa, the honest negative ─────────────────────────────────────────
if [ -d data/ottawa ] && [ "$(n_mat data/ottawa)" -gt 10 ]; then
  banner "uOttawa — second changing-speed rig (expected to find NOTHING)"
  PYTHONPATH=src "$PY" tools/eval_dataset.py --dataset ottawa --data data/ottawa \
      --ppr 1024 --band 2000 6000 --out runs/v_ottawa.json > runs/v_ottawa.log 2>&1
  O=$(grep -E "^   correct" runs/v_ottawa.log | head -1 | tr -s ' ')
  echo "  $O"
  record "uOttawa" "$(echo $O | grep -oE '[0-9]+/[0-9]+')" "0/33" \
         "finds nothing — the one dataset whose bearing dimensions we could not verify"
else
  banner "uOttawa — SKIPPED, no data"; echo "   see DOWNLOADS.md"
fi

# ── 7. actuator ─────────────────────────────────────────────────────────────
if [ -d actuator_dataset/Data ]; then
  banner "VT/NSWC hydraulic actuator — Navy-published, pressure sensors only"
  PYTHONPATH=src "$PY" tools/eval_actuator.py --data actuator_dataset/Data \
      > runs/v_actuator.log 2>&1
  grep -E "cycle [0-9]|never" runs/v_actuator.log | head -4
else
  banner "VT/NSWC hydraulic actuator — SKIPPED, no data"; echo "   see DOWNLOADS.md"
fi

# ── 8. NASA/IMS run to failure ──────────────────────────────────────────────
# Not run here. It reads 7,588 captures and takes about fifteen minutes, which
# is longer than everything above put together, so it has its own script. What
# this section does is read back the result if that script has already been run,
# so the scorecard carries the withdrawn prognosis claim instead of omitting it.
if [ -f runs/ims_readme.json ]; then
  banner "NASA/IMS — run to failure (read back from a previous run_ims.sh)"
  P=$(PYTHONPATH=src "$PY" - <<'PY'
import json
import statistics
units = json.load(open("runs/ims_readme.json"))["units"]
called = [u for u in units if u["is_failure"] and u.get("first_call") is not None]
fails = [u for u in units if u["is_failure"]]
left = sorted(u["pct_life_left"] for u in called)
med = statistics.median(left) if left else float("nan")
print(f"spoke on {len(called)}/{len(fails)} failures, median life left {med:.0f}%")
PY
)
  echo "   $P"
  record "IMS prognosis" "$P" "median life left 1%" \
         "1% against the 10% threshold fixed before download — the claim is WITHDRAWN"
elif [ -d data/ims ] && [ -n "$(ls data/ims 2>/dev/null)" ]; then
  banner "NASA/IMS — present but not run"
  echo "   run it separately, about fifteen minutes:  bash run_ims.sh"
else
  banner "NASA/IMS — SKIPPED, no data"; echo "   see DOWNLOADS.md"
fi

# ── scorecard ───────────────────────────────────────────────────────────────
echo
echo "=============================================================="
echo "SCORECARD — paste this back"
echo "=============================================================="
printf "%-26s %-26s %s\n" "CHECK" "GOT" "MATCHES EXPECTED?"
for i in "${!NAME[@]}"; do
  g="${GOT[$i]}"; w="${WANT[$i]}"
  case "$g" in *"$w"*) v="yes" ;; *) v="DIFFERENT — expected $w" ;; esac
  printf "%-26s %-26s %s\n" "${NAME[$i]}" "$g" "$v"
done
echo
echo "What each line means:"
for i in "${!NAME[@]}"; do printf "  %-26s %s\n" "${NAME[$i]}" "${NOTE[$i]}"; done
echo
echo "Full output for every run is in runs/v_*.log"
echo "finished $(date '+%H:%M:%S')"
