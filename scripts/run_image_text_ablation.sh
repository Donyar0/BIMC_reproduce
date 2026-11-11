#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_CFG="${1:-configs/datasets/cifar100.yaml}"
if [ "$#" -gt 0 ]; then shift; fi
LOG_DIR="logs/image_text_ablation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
# Run sequentially on the same GPU. Both YAMLs share seed/augmentation/settings.
for MODE in velocity endpoint; do
  TRAIN_CFG="configs/trainers/unified_flow.yaml"
  if [ "$MODE" = endpoint ]; then TRAIN_CFG="configs/trainers/unified_flow_endpoint.yaml"; fi
  python -u main.py --data_cfg "$DATA_CFG" --train_cfg "$TRAIN_CFG" "$@" \
    2>&1 | tee "$LOG_DIR/$MODE.log"
done
