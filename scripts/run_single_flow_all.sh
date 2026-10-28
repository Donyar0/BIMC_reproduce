#!/usr/bin/env bash
# Run from any directory; dataset ROOT paths must already be configured.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
log_dir="output/single_flow/logs/$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$log_dir"
for dataset in cifar100 cub200 miniimagenet; do
    echo "===== single_flow: $dataset ====="
    python -u main.py \
        --data_cfg "configs/datasets/${dataset}.yaml" \
        --train_cfg configs/trainers/single_flow.yaml "$@" \
        2>&1 | tee "$log_dir/${dataset}.log"
done
echo "All three datasets finished. Console logs: $log_dir"
