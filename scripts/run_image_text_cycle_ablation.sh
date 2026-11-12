#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_CFG="${1:-configs/datasets/cifar100.yaml}"
if [ "$#" -gt 0 ]; then shift; fi
LOG_DIR="logs/image_text_cycle_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
for VARIANT in unified_flow unified_flow_cycle unified_flow_cycle_incremental; do
  python -u main.py --data_cfg "$DATA_CFG" \
    --train_cfg "configs/trainers/${VARIANT}.yaml" "$@" \
    2>&1 | tee "$LOG_DIR/${VARIANT}.log"
done
