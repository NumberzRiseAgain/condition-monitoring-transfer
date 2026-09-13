#!/usr/bin/env bash
# Fetch and evaluate every public dataset, unattended.
#
#   nohup bash tools/run_all.sh > runs/all.log 2>&1 &
#   tail -f runs/all.log
#
# Only CWRU downloads automatically. The other three require accepting terms or
# a manual download, so this script tells you exactly where to put them and then
# skips them cleanly rather than failing the whole run. Come back to
# runs/SUMMARY.txt.
set -u
mkdir -p runs data

PY=python
command -v python >/dev/null 2>&1 || PY=python3

log() { echo; echo "############ $* ############"; echo; }

log "1/5  CWRU  (auto-download)"
if [ ! -f data/cwru/manifest.json ]; then
  $PY tools/fetch_cwru.py --out data/cwru || echo "CWRU fetch failed — continuing"
fi
if ls data/cwru/*.mat >/dev/null 2>&1; then
  $PY tools/eval_dataset.py --dataset cwru --data data/cwru --attribution line   > runs/cwru_line.txt   2>&1
  $PY tools/eval_dataset.py --dataset cwru --data data/cwru --attribution family > runs/cwru_family.txt 2>&1
  echo "CWRU done"
else
  echo "CWRU SKIPPED — no .mat files in data/cwru"
fi

log "2/5  MFPT"
if ls data/mfpt/**/*.mat >/dev/null 2>&1 || ls data/mfpt/*.mat >/dev/null 2>&1; then
  $PY tools/eval_dataset.py --dataset mfpt --data data/mfpt --attribution family > runs/mfpt_family.txt 2>&1
  $PY tools/eval_dataset.py --dataset mfpt --data data/mfpt --attribution line   > runs/mfpt_line.txt   2>&1
  echo "MFPT done"
else
  cat <<'MSG'
MFPT SKIPPED. Download the fault data sets from
    https://www.mfpt.org/fault-data-sets/
unzip into  data/mfpt/  KEEPING the numbered folder names — the condition is in
the directory name, not the file name.
MSG
fi

log "3/5  Ottawa variable speed"
if ls data/ottawa/*.mat >/dev/null 2>&1; then
  $PY tools/eval_dataset.py --dataset ottawa --data data/ottawa --attribution family --window 1.0 > runs/ottawa_family.txt 2>&1
  echo "Ottawa done"
else
  cat <<'MSG'
OTTAWA SKIPPED. Search Mendeley Data for
    "Bearing vibration data under time-varying rotational speed conditions"
unzip the .mat files into  data/ottawa/  keeping names like H-A-1.mat.
This is the one that tests variable speed, which is what an arrestment is.
MSG
fi

log "4/5  Paderborn (real damage + motor current)"
if ls data/paderborn/**/*.mat >/dev/null 2>&1 || ls data/paderborn/*.mat >/dev/null 2>&1; then
  $PY tools/eval_dataset.py --dataset paderborn --data data/paderborn --channel vibration --attribution family > runs/paderborn_vib.txt 2>&1
  $PY tools/eval_dataset.py --dataset paderborn --data data/paderborn --channel current   --attribution family > runs/paderborn_cur.txt 2>&1
  echo "Paderborn done"
else
  cat <<'MSG'
PADERBORN SKIPPED. From the KAT chair's Bearing DataCenter (Universitaet
Paderborn), download at least the healthy codes (K001-K005) and the REAL-damage
codes (KA04 KA15 KA16 KA22 KA30 KI04 KI14 KI16 KI17 KI18 KI21).
Unzip into  data/paderborn/  keeping names like N15_M07_F10_KA04_1.mat.
Real damage matters more than volume here — 5 healthy + 5 real-damage codes is
worth more than every artificial record in the set.
MSG
fi

log "5/5  SUMMARY"
$PY tools/summarise.py | tee runs/SUMMARY.txt
