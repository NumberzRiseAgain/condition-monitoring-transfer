#!/usr/bin/env bash
# NASA/IMS run-to-failure, both scopes, one command.
#
#   bash run_ims.sh
#
# Takes roughly fifteen minutes: 7,588 captures in the documented scope plus
# 9,464 in the full one, each read once and sliced four ways. Reads only; it
# writes nothing outside runs/.
#
# Two scopes are run because set 3 ships 6,324 captures for an experiment its
# own readme documents as 4,448. The documented window is PRIMARY — it is the
# readme that tells us bearing 3 failed at all, and it is also the shorter life,
# so every percentage it reports is the more conservative one. The full folder
# is run as a sensitivity check, so that the choice is visible rather than
# quietly baked in.
#
# Nothing here is swept. The alarm threshold, the baseline fraction and the
# persistence requirement are the values declared in PROGNOSIS_STOPPING_RULE.md
# before the data was downloaded. If you find yourself editing them because the
# answer came out disappointing, read that file first.

set -u
cd "$(dirname "$0")"
mkdir -p runs

PY=""
if [ -x ".venv/bin/python" ] && .venv/bin/python -c "import numpy,scipy,pandas" 2>/dev/null; then
  PY=".venv/bin/python"
elif python3 -c "import numpy,scipy,pandas" 2>/dev/null; then
  PY="python3"
else
  echo "No interpreter with numpy/scipy/pandas. See SETUP.md."; exit 1
fi

if [ ! -d data/ims ] || [ -z "$(ls data/ims 2>/dev/null)" ]; then
  echo "data/ims is empty — see DOWNLOADS.md."; exit 1
fi

echo "=============================================================="
echo "NASA/IMS run-to-failure — $(date '+%Y-%m-%d %H:%M')"
echo "python: $PY"
echo "=============================================================="

echo
echo "── census"
PYTHONPATH=src "$PY" tools/eval_runtofailure.py --dataset ims --data data/ims \
    --probe 2>&1 | tee runs/ims_census.log

for SCOPE in readme full; do
  echo
  echo "── scope: $SCOPE  $([ "$SCOPE" = readme ] && echo '(PRIMARY — the documented experiment)' || echo '(sensitivity check — every capture on disk)')"
  PYTHONPATH=src "$PY" tools/eval_runtofailure.py --dataset ims --data data/ims \
      --scope "$SCOPE" --json "runs/ims_$SCOPE.json" \
      > "runs/ims_$SCOPE.log" 2>&1
  sed -n '/unit /,$p' "runs/ims_$SCOPE.log"
done

echo
echo "=============================================================="
echo "Full output in runs/ims_readme.log and runs/ims_full.log"
echo "finished $(date '+%H:%M:%S')"
