#!/bin/bash
# Serve the pi0.5 Stretch policy for on-robot inference.
# Pair: on the robot, ~/lerobot_recorder/run_inference.py connects here (port 8000).
#
#   ./serve_stretch.sh                 newest checkpoint of the most recent run
#   ./serve_stretch.sh run2            newest checkpoint of run2
#   ./serve_stretch.sh run2 15000      a specific step
#
# The run is resolved at call time rather than hardcoded: serving a checkpoint
# from an older schema under the current transforms is a SILENT wrong answer, not
# an error, so the run name must never be a stale constant in this file.
set -euo pipefail
BASE=~/openpi/checkpoints/pi05_stretch

[ -d "$BASE" ] || { echo "no runs under $BASE — nothing trained yet"; exit 1; }

RUN="${1:-}"
if [ -z "$RUN" ]; then
    RUN=$(ls -1dt "$BASE"/*/ 2>/dev/null | head -1 | xargs -r basename)
    [ -n "$RUN" ] || { echo "no run directories under $BASE"; exit 1; }
    echo "no run given; using the most recently modified: $RUN"
    echo "  (available: $(ls -1 "$BASE" | tr '\n' ' '))"
fi
RUN_DIR="$BASE/$RUN"
[ -d "$RUN_DIR" ] || { echo "no run '$RUN' in $BASE (have: $(ls -1 $BASE | tr '\n' ' '))"; exit 1; }

STEP="${2:-$(ls "$RUN_DIR" | grep -E "^[0-9]+$" | sort -n | tail -1)}"
[ -d "$RUN_DIR/$STEP" ] || { echo "no checkpoint $STEP in $RUN_DIR (have: $(ls $RUN_DIR | grep -E "^[0-9]+$" | sort -n | tr '\n' ' '))"; exit 1; }

echo "serving checkpoint: $RUN_DIR/$STEP"
echo "the checkpoint's own norm stats are used, but the TRANSFORMS come from the"
echo "current config.py — make sure this run was trained on the same schema."
cd ~/openpi
exec .venv/bin/python scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_stretch \
    --policy.dir="$RUN_DIR/$STEP"
