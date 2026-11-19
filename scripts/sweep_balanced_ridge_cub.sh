#!/usr/bin/env bash
set -euo pipefail

for ridge_lambda in 10.0 50.0 100.0 500.0; do
  python -u main.py \
    --data_cfg configs/datasets/cub200.yaml \
    --train_cfg configs/trainers/balanced_ridge_mlp.yaml \
    --opts RIDGE.LAMBDA "$ridge_lambda" \
           FLOW.OUTPUT_DIR "output/balanced_ridge_mlp_lambda${ridge_lambda}"
done
