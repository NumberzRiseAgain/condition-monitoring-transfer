#!/usr/bin/env bash
# reproduce.sh -- regenerate every number the ARRESTLINE volume takes from cbmx, in one command.
#
#   cd 08_Analysis/code && ./reproduce.sh            (this is what scripts/evidence_pack.sh runs)
#
# Order: the tests first (a failing test stops the study before it produces numbers), then
# the experiments that need no dataset, then every dataset experiment whose data is present
# under ../data. A dataset that is absent is SKIPPED AND NAMED -- never silently passed, and
# never counted as a result. The datasets are about 1.9 GB, each under its own licence, and
# are not stored in the pursuit folder; DOWNLOADS.md says where each one comes from and
# where it goes. Results go to ../results/e*.json; the transcript of each step to
# ../results/transcripts/. No network, no GPU, no model weights.
#
# Exit code: 0 only when the tests pass and every experiment that RAN exited 0. Skips do not
# fail the run, but they are listed in ../results/_run_status.json and in manifest.json so a
# reader can see exactly what this run does and does not cover.
set -u
cd "$(dirname "$0")"
export PYTHONPATH=src
export PYTHONHASHSEED=0
SEED="${SEED:-101}"
DATA=../data
RES=../results
TR=$RES/transcripts
mkdir -p "$RES" "$TR"

PY=python3
$PY -c "import numpy, scipy, pandas, yaml, pytest" 2>/dev/null || {
  echo "missing a library; pip install -r requirements.txt"; exit 1; }

RUN=(); SKIP=(); PRESENT=(); ABSENT=(); FAIL=0
step_ok () { RUN+=("$1"); echo "   ok      $1"; }
step_fail () { RUN+=("$1 (FAILED)"); FAIL=$((FAIL+1)); echo "   FAILED  $1  -- see $TR/$2"; }
skip () { SKIP+=("$1 -- $2"); echo "   SKIPPED $1 -- $2"; }
have_dir_with () { [ -d "$1" ] && [ "$(find -L "$1" -maxdepth 1 -name "$2" -type f 2>/dev/null | wc -l)" -ge "${3:-1}" ]; }

echo "=============================================================="
echo "cbmx reproduce -- $(date -u '+%Y-%m-%dT%H:%M:%SZ')   seed $SEED"
echo "python: $($PY -V 2>&1)   numpy $($PY -c 'import numpy;print(numpy.__version__)')   scipy $($PY -c 'import scipy;print(scipy.__version__)')"
echo "=============================================================="

# -- dataset census ------------------------------------------------------------------------
for d in paderborn hydraulic kaist ottawa cwru ims; do
  case $d in
    hydraulic) t="$DATA/hydraulic/profile.txt"; [ -f "$t" ] && PRESENT+=("$d") || ABSENT+=("$d");;
    paderborn) have_dir_with "$DATA/paderborn" '*' 1 && PRESENT+=("$d") || ABSENT+=("$d");;
    kaist)     have_dir_with "$DATA/kaist" 'vibration_*.csv' 4 && PRESENT+=("$d") || ABSENT+=("$d");;
    ottawa)    have_dir_with "$DATA/ottawa" '*.mat' 10 && PRESENT+=("$d") || ABSENT+=("$d");;
    cwru)      have_dir_with "$DATA/cwru" '*.mat' 1 && PRESENT+=("$d") || ABSENT+=("$d");;
    ims)       [ -d "$DATA/ims" ] && [ -n "$(ls "$DATA/ims" 2>/dev/null)" ] && PRESENT+=("$d") || ABSENT+=("$d");;
  esac
done
[ -d "$DATA/actuator_dataset/Data" ] && PRESENT+=("actuator") || ABSENT+=("actuator")
echo "datasets present: ${PRESENT[*]:-none}"
echo "datasets absent:  ${ABSENT[*]:-none}"

# -- E0 the tests ----------------------------------------------------------------------------
echo; echo "-- E0 unit tests"
$PY -m pytest tests/ -q -rs > "$TR/e0_tests.txt" 2>&1; rc=$?
SUMMARY=$(tail -1 "$TR/e0_tests.txt")
echo "   $SUMMARY"
$PY - "$SUMMARY" "$rc" "$RES/e0_tests.json" <<'PY'
import json, re, sys
s, rc, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
g = lambda k: int(m.group(1)) if (m := re.search(r"(\d+) " + k, s)) else 0
json.dump({"experiment": "E0 unit tests (pytest)", "summary": s, "exit_code": rc,
           "passed": g("passed"), "skipped": g("skipped"), "failed": g("failed"), "errors": g("error"),
           "note": "skips are data-gated tests naming the dataset they need; a failure stops the study"},
          open(out, "w"), indent=2)
PY
if [ $rc -ne 0 ]; then echo "   tests failed; stopping before any number is produced"; exit 1; fi
step_ok "E0 unit tests"

# -- E1 physics catalogue: geometry -> fault orders, identities, published multipliers -----
echo; echo "-- E1 physics catalogue"
$PY -m cbmx.cli physics > "$TR/e1_physics.txt" 2>&1; rc=$?
$PY - "$TR/e1_physics.txt" "$rc" "$RES/e1_physics.json" <<'PY'
import json, re, sys
t = open(sys.argv[1]).read(); rc = int(sys.argv[2])
bearings = re.findall(r"^(\S.*?)\s+n=(\d+)\s+d=", t, re.M)
devs = [float(x) for x in re.findall(r"max deviation ([0-9.]+)", t)]
json.dump({"experiment": "E1 bearing physics catalogue (cbmx physics)", "exit_code": rc,
           "bearings": len(bearings), "identities_all_hold": t.count("identities: all hold"),
           "max_deviation_vs_published": max(devs) if devs else None,
           "note": "fault orders from geometry, checked against the published multipliers; no data needed"},
          open(sys.argv[3], "w"), indent=2)
PY
[ $rc -eq 0 ] && step_ok "E1 physics catalogue" || step_fail "E1 physics catalogue" e1_physics.txt

# -- E2 the synthetic end-to-end demo ------------------------------------------------------
echo; echo "-- E2 synthetic end-to-end demo (seed $SEED)"
$PY run_e2_demo.py --seed "$SEED" --json "$RES/e2_demo.json" --transcript "$TR/e2_demo.txt" > /dev/null 2>&1; rc=$?
grep -E "latency +worst|link +peak|LRU isolation|quiet machine" "$TR/e2_demo.txt" | sed 's/^/   /'
[ $rc -eq 0 ] && step_ok "E2 synthetic demo" || step_fail "E2 synthetic demo" e2_demo.txt

# -- E3 the semantic layer and the gate that discards its output ---------------------------
echo; echo "-- E3 semantic layer and validation gate"
if [ -f "$DATA/semantic/asset_card.json" ]; then
  $PY tools/eval_semantic.py --root "$DATA/semantic" --json "$RES/e3_semantic.json" > "$TR/e3_semantic.txt" 2>&1; rc=$?
  grep -oE "gate rejected [0-9]+ of [0-9]+" "$TR/e3_semantic.txt" | tail -1 | sed 's/^/   /'
  [ $rc -eq 0 ] && step_ok "E3 semantic gate" || step_fail "E3 semantic gate" e3_semantic.txt
else
  skip "E3 semantic gate" "data/semantic fixtures absent"
fi

# -- E17 the same gate, on a published maintenance corpus -----------------------------------
# Added 12 Sep 2026. E3 runs the gate on an invented asset card; this runs the identical
# validate() against the S1000D Issue 6 sample data set: a real parts catalogue and real
# procedures somebody else published. data/s1000d_public is a symlink to the one copy of that
# corpus on this tree (it is shared with DON26BZ05-NV072); absent, the step is skipped by name.
echo; echo "-- E17 semantic gate on a published corpus"
if [ -d "$DATA/s1000d_public" ] && [ -n "$(ls "$DATA/s1000d_public" 2>/dev/null)" ]; then
  $PY tools/eval_semantic_public.py --csdb "$DATA/s1000d_public" \
      --json "$RES/e17_semantic_public.json" > "$TR/e17_semantic_public.txt" 2>&1; rc=$?
  grep -E "TRUE HYPOTHESES|rejected " "$TR/e17_semantic_public.txt" | sed 's/^/   /'
  [ $rc -eq 0 ] && step_ok "E17 semantic gate, published corpus" \
                || step_fail "E17 semantic gate, published corpus" e17_semantic_public.txt
else
  skip "E17 semantic gate, published corpus" "data/s1000d_public absent (S1000D Issue 6 sample set)"
fi

# -- E4, E5 Paderborn: real damage, four conditions, and the band ablation -----------------
echo; echo "-- E4/E5 Paderborn (real bearing damage)"
ANYPU=0
for C in N15_M07_F10 N09_M07_F10 N15_M01_F10 N15_M07_F04; do
  if have_dir_with "$DATA/pu_$C" '*.mat' 100; then
    ANYPU=1
    $PY tools/eval_dataset.py --dataset paderborn --data "$DATA/pu_$C" --holdout bearing \
        --real-damage-only --band 500 2000 --out "$RES/e4_pu_$C.json" > "$TR/e4_pu_$C.txt" 2>&1; rc=$?
    echo "   $C  $(grep -oE 'DETECTION [0-9]+/[0-9]+' "$TR/e4_pu_$C.txt" | head -1)  $(grep -oE 'FALSE ALARMS [0-9]+ of [0-9]+' "$TR/e4_pu_$C.txt" | head -1)"
    [ $rc -eq 0 ] && step_ok "E4 Paderborn $C" || step_fail "E4 Paderborn $C" "e4_pu_$C.txt"
  else
    skip "E4 Paderborn $C" "data/pu_$C absent (see DOWNLOADS.md section 1)"
  fi
done
if [ $ANYPU -eq 1 ] && have_dir_with "$DATA/pu_N15_M07_F10" '*.mat' 100; then
  $PY tools/eval_dataset.py --dataset paderborn --data "$DATA/pu_N15_M07_F10" --holdout bearing \
      --real-damage-only --out "$RES/e5_pu_autoband.json" > "$TR/e5_pu_autoband.txt" 2>&1; rc=$?
  echo "   autoband  $(grep -oE 'DETECTION [0-9]+/[0-9]+' "$TR/e5_pu_autoband.txt" | head -1)"
  [ $rc -eq 0 ] && step_ok "E5 band ablation" || step_fail "E5 band ablation" e5_pu_autoband.txt
else
  skip "E5 band ablation" "data/pu_N15_M07_F10 absent"
fi

# -- E6..E10 Paderborn shared extraction: current vs vibration, torsional band, transient, latency, edge
echo; echo "-- E6..E10 Paderborn shared extraction"
if have_dir_with "$DATA/paderborn" '*' 1; then
  $PY tools/eval_mcsa.py --data "$DATA/paderborn" --healthy K002 --json "$RES/e6_mcsa.json" > "$TR/e6_mcsa.txt" 2>&1 \
     && step_ok "E6 vibration vs motor current" || step_fail "E6 vibration vs motor current" e6_mcsa.txt
  $PY tools/eval_current_band.py --data "$DATA/paderborn" --json "$RES/e7_current_band.json" > "$TR/e7_current_band.txt" 2>&1 \
     && step_ok "E7 current torsional band" || step_fail "E7 current torsional band" e7_current_band.txt
  $PY tools/eval_transient.py --data "$DATA/paderborn" --json "$RES/e8_transient.json" > "$TR/e8_transient.txt" 2>&1 \
     && step_ok "E8 speed transient" || step_fail "E8 speed transient" e8_transient.txt
  $PY tools/eval_latency.py --data "$DATA/paderborn" --json "$RES/e9_latency.json" > "$TR/e9_latency.txt" 2>&1 \
     && step_ok "E9 latency" || step_fail "E9 latency" e9_latency.txt
  $PY tools/bench_edge.py --data "$DATA/paderborn/KA04" --json "$RES/e10_bench_edge.json" > "$TR/e10_bench_edge.txt" 2>&1 \
     && step_ok "E10 edge bench" || step_fail "E10 edge bench" e10_bench_edge.txt
else
  for e in "E6 vibration vs motor current" "E7 current torsional band" "E8 speed transient" "E9 latency" "E10 edge bench"; do
    skip "$e" "data/paderborn absent (CC BY-NC 4.0; see GET_PADERBORN_DATA.md)"; done
fi

# -- E11, E12 ZeMA hydraulic: the governance layer off the bearing; prognosis precondition --
echo; echo "-- E11/E12 ZeMA hydraulic rig"
if [ -f "$DATA/hydraulic/profile.txt" ]; then
  $PY tools/eval_hydraulic.py --root "$DATA/hydraulic" --json "$RES/e11_hydraulic.json" > "$TR/e11_hydraulic.txt" 2>&1 \
     && step_ok "E11 hydraulic" || step_fail "E11 hydraulic" e11_hydraulic.txt
  $PY tools/eval_hydraulic.py --root "$DATA/hydraulic" --no-freeze --json "$RES/e11_hydraulic_nofreeze.json" > "$TR/e11_hydraulic_nofreeze.txt" 2>&1 \
     && step_ok "E11 hydraulic, freeze ablation" || step_fail "E11 hydraulic, freeze ablation" e11_hydraulic_nofreeze.txt
  $PY tools/eval_hydraulic.py --root "$DATA/hydraulic" --time-split --json "$RES/e11_hydraulic_timesplit.json" > "$TR/e11_hydraulic_timesplit.txt" 2>&1 \
     && step_ok "E11 hydraulic, unseen regimes" || step_fail "E11 hydraulic, unseen regimes" e11_hydraulic_timesplit.txt
  $PY tools/eval_prognosis.py --root "$DATA/hydraulic" --json "$RES/e12_prognosis.json" > "$TR/e12_prognosis.txt" 2>&1 \
     && step_ok "E12 prognosis precondition" || step_fail "E12 prognosis precondition" e12_prognosis.txt
else
  for e in "E11 hydraulic" "E11 hydraulic, freeze ablation" "E11 hydraulic, unseen regimes" "E12 prognosis precondition"; do
    skip "$e" "data/hydraulic absent (UCI; see DOWNLOADS.md section 2)"; done
fi

# -- E13 KAIST varying speed, E14 uOttawa, E15 actuator, E16 NASA/IMS ----------------------
echo; echo "-- E13..E16 varying speed, second rig, actuator, run to failure"
if have_dir_with "$DATA/kaist" 'vibration_*.csv' 4; then
  $PY tools/eval_kaist.py --data "$DATA/kaist" --bearing KAIST6205U \
      --targets inner_0 inner_1 inner_2 outer_0 outer_1 outer_2 --json "$RES/e13_kaist.json" > "$TR/e13_kaist.txt" 2>&1 \
     && step_ok "E13 KAIST varying speed" || step_fail "E13 KAIST varying speed" e13_kaist.txt
else skip "E13 KAIST varying speed" "data/kaist absent (Mendeley; see GET_KAIST_DATA.md)"; fi
if have_dir_with "$DATA/ottawa" '*.mat' 10; then
  $PY tools/eval_dataset.py --dataset ottawa --data "$DATA/ottawa" --ppr 1024 --band 2000 6000 \
      --out "$RES/e14_ottawa.json" > "$TR/e14_ottawa.txt" 2>&1 \
     && step_ok "E14 uOttawa" || step_fail "E14 uOttawa" e14_ottawa.txt
else skip "E14 uOttawa" "data/ottawa absent"; fi
if [ -d "$DATA/actuator_dataset/Data" ]; then
  $PY tools/eval_actuator.py --data "$DATA/actuator_dataset/Data" --json "$RES/e15_actuator.json" > "$TR/e15_actuator.txt" 2>&1 \
     && step_ok "E15 hydraulic actuator" || step_fail "E15 hydraulic actuator" e15_actuator.txt
else skip "E15 hydraulic actuator" "data/actuator_dataset absent"; fi
if [ -d "$DATA/ims" ] && [ -n "$(ls "$DATA/ims" 2>/dev/null)" ]; then
  $PY tools/eval_runtofailure.py --dataset ims --data "$DATA/ims" --scope readme \
      --json "$RES/e16_ims_readme.json" > "$TR/e16_ims_readme.txt" 2>&1 \
     && step_ok "E16 NASA/IMS run to failure (readme scope)" || step_fail "E16 NASA/IMS run to failure" e16_ims_readme.txt
  $PY tools/eval_runtofailure.py --dataset ims --data "$DATA/ims" --scope full \
      --json "$RES/e16_ims_full.json" > "$TR/e16_ims_full.txt" 2>&1 \
     && step_ok "E16 NASA/IMS run to failure (full folder)" || step_fail "E16 NASA/IMS run to failure (full)" e16_ims_full.txt
  if [ -f "$RES/e16_ims_readme.json" ] && [ -f "$RES/e16_ims_full.json" ]; then
    $PY tools/eval_rtf_discrimination.py --json "$RES/e16_ims_readme.json" \
        --json "$RES/e16_ims_full.json" --out "$RES/e16_ims_discrimination.json" \
        > "$TR/e16_ims_discrimination.txt" 2>&1 \
       && step_ok "E16b IMS bearing-level discrimination" || step_fail "E16b IMS discrimination" e16_ims_discrimination.txt
  else skip "E16b IMS bearing-level discrimination" "needs both E16 scopes"; fi
else skip "E16 NASA/IMS run to failure" "data/ims absent (NASA PCoE)"; fi

# -- status, manifest, digests of the carried-forward August runs --------------------------
$PY - "$RES/_run_status.json" "${#RUN[@]}" "${RUN[@]}" -- "${#SKIP[@]}" "${SKIP[@]}" -- "${PRESENT[@]:-}" -- "${ABSENT[@]:-}" <<'PY'
import json, sys
argv = sys.argv[1:]; out = argv[0]; rest = argv[1:]
n = int(rest[0]); run = rest[1:1+n]; rest = rest[1+n:]
assert rest[0] == "--"; rest = rest[1:]
m = int(rest[0]); skipped = rest[1:1+m]; rest = rest[1+m:]
assert rest[0] == "--"; rest = rest[1:]
i = rest.index("--"); present = [x for x in rest[:i] if x]; absent = [x for x in rest[i+1:] if x]
json.dump({"run": run, "skipped": skipped, "datasets_present": present, "datasets_absent": absent},
          open(out, "w"), indent=2)
PY
$PY manifest.py --seed "$SEED" --results "$RES" --status "$RES/_run_status.json"
# Runs carried forward: outputs produced by this same code on a machine that had the
# datasets, on a date other than today. They are hashed here so a volume can cite them
# by digest, and they are kept in their own file so nobody mistakes them for this run.
# 12 Sep 2026: generalised from the single August folder to every prior_runs_* and
# fresh_* directory, so the 9 September runs stop being uncitable for want of a hash.
: > "$RES/_prior_run_digests.txt"
CARRIED=0
for d in "$RES"/prior_runs_* "$RES"/fresh_*; do
  [ -d "$d" ] || continue
  label="$(basename "$d")"
  for f in "$d"/*; do
    [ -f "$f" ] || continue
    printf '%s  %s\n' "$(shasum -a 256 "$f" | awk '{print substr($1,1,16)}')" "$label/$(basename "$f")" >> "$RES/_prior_run_digests.txt"
    CARRIED=$((CARRIED+1))
  done
done
if [ "$CARRIED" -gt 0 ]; then
  echo "carried-forward runs hashed: $CARRIED files -> results/_prior_run_digests.txt (not this run)"
else
  rm -f "$RES/_prior_run_digests.txt"
fi

echo
echo "=============================================================="
echo "ran ${#RUN[@]} step(s), skipped ${#SKIP[@]} for missing data, failed $FAIL"
echo "=============================================================="
exit $FAIL
