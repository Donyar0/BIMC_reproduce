# Frozen CLIP, Base Flow, and incremental random-feature ridge regression

This experiment starts from `7de9a53` and replaces Base Vision LoRA with an
analytic classifier. The CLIP image and text encoders remain frozen. Only the
Base Flow is trained; later sessions do not update Flow or replay old images.

## Run

Install `requirements.txt`, prepare the dataset and description JSON, and set
`DATASET.ROOT` in the dataset YAML. From the repository root:

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/frozen_random_ridge_flow.yaml
```

Replace `cub200.yaml` with `miniimagenet.yaml` or `cifar100.yaml` to run those
datasets. All sessions are evaluated. Add `--use_swanlab` for SwanLab logging.

## Method

1. Session 0 trains the eight-block text-to-visual Flow once from frozen CLIP
   features. It is then permanently frozen. Base training images initialize
   the random-feature ridge classifier.
2. A fixed, seeded Gaussian matrix maps each normalized 512-dimensional CLIP
   image feature to `ReLU(z @ W_rand)` in 8192 dimensions. Only
   `G = sum(phi.T @ phi)` and `C = sum(phi.T @ one_hot(label))` are accumulated
   across sessions. A Cholesky solve obtains `(G + lambda I)^-1 C` with
   `lambda = 1.0` after each session.
3. Every later session adds only its five real support images per class and
   virtual visual features obtained by transporting **each available new-class
   LLM description** through the frozen Flow. Description counts differ among
   the supplied datasets; they are not always 25. No old images, Flow replay,
   Flow distillation, or Flow adaptation are used.
4. The existing visual Bank and reverse-text Flow branch are retained as a
   geometric prior. Their cosine scores and the ridge scores are standardized
   separately across seen classes for each query, then fused equally. This
   avoids combining raw scores with different scales.

Single-view feature extraction and original visual prototypes are used:
`VISION_CALIBRATION`, `TTA`, `TTA_PROTOTYPES`, and `DIAGONAL_WHITEN` are all
disabled. The `VIS_WHITEN_POWER` setting is absent. The 8192 x 8192 float32
Gram matrix occupies 256 MiB, and solving it at every session can be expensive.

Each run writes configuration, Base Flow, session banks, ridge heads, final
ridge statistics, and CSV/JSON accuracy metrics beneath
`output/frozen_random_ridge_flow8192/<dataset>/seed_<seed>_<timestamp>/`.
