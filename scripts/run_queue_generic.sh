#!/bin/bash
# usage: bash scripts/run_queue_generic.sh <queue-name> <config-dir under configs/train> <arm> [<arm> ...]
# markers/logs follow the beauty/toys queues: results/explor_<dir>_<arm>.done, results/logs/explor_<dir>_<arm>.log
NAME=$1; DIR=$2; shift 2
cd "$(dirname "$0")/.."  # repository root
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p results results/logs
for cfg in "$@"; do
  if [ -f "results/explor_${DIR}_${cfg}.done" ]; then echo "SKIP $cfg (already done)"; continue; fi
  echo "=== START $cfg | $(date '+%d.%m %H:%M') ==="
  python -m irec.train --params configs/train/$DIR/$cfg.json > results/logs/explor_${DIR}_${cfg}.log 2>&1 \
    && { touch results/explor_${DIR}_${cfg}.done; echo "=== OK    $cfg | $(date '+%d.%m %H:%M') ==="; } \
    || echo "=== FAIL  $cfg | $(date '+%d.%m %H:%M') — see results/logs/explor_${DIR}_${cfg}.log ==="
done
echo "QUEUE_${NAME}_ALL_DONE $(date '+%d.%m %H:%M')"
