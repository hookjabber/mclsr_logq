#!/bin/bash
# usage: bash scripts/run_queue_seeds.sh <queue-name> <config-dir> <seeds, comma-separated> <arm> [<arm> ...]
# multi-seed runs through the confirmatory runner: test callback stripped, test opened once per
# validation-selected checkpoint (ndcg@20 and recall@1000), JSON report with sha256 of everything.
NAME=$1; DIR=$2; SEEDS=$3; shift 3
cd "$(dirname "$0")/.."  # repository root
[ -f .venv/bin/activate ] && source .venv/bin/activate
mkdir -p results/confirm results/logs
for cfg in "$@"; do
  for seed in ${SEEDS//,/ }; do
    m="results/confirm_${DIR}_${cfg}_seed${seed}.done"
    if [ -f "$m" ]; then echo "SKIP $cfg seed $seed (already done)"; continue; fi
    echo "=== START $cfg seed $seed | $(date '+%d.%m %H:%M') ==="
    python scripts/train_confirmatory.py --params configs/train/$DIR/$cfg.json --seed $seed \
      --output results/confirm/${DIR}_${cfg}_seed${seed}.json > results/logs/confirm_${DIR}_${cfg}_seed${seed}.log 2>&1 \
      && { touch "$m"; echo "=== OK    $cfg seed $seed | $(date '+%d.%m %H:%M') ==="; } \
      || echo "=== FAIL  $cfg seed $seed | $(date '+%d.%m %H:%M') — see results/logs/confirm_${DIR}_${cfg}_seed${seed}.log ==="
  done
done
echo "QUEUE_${NAME}_ALL_DONE $(date '+%d.%m %H:%M')"
