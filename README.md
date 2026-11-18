# Tangent Flow with Base Vision LoRA (rank 4)

This branch runs the method from commit `52a2e91`. Its only trainer configuration is
`configs/trainers/tangent_flow_zca_base_vision_lora.yaml`. The three dataset YAMLs
remain because they define the CUB200, miniImageNet, and CIFAR100 splits and paths.

## Run

Install `requirements.txt` and adjust `DATASET.ROOT` and `DATASET.GPT_PATH` for
your machine. From the repository root, choose a dataset configuration:

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/tangent_flow_zca_base_vision_lora.yaml
```

Replace `cub200.yaml` with `miniimagenet.yaml` or `cifar100.yaml` for those
datasets. The runner evaluates every session. Add `--use_swanlab` if experiment
logging is needed.

## Method in this configuration

- Base CLIP ViT-B/16 visual attention uses Q/V LoRA in its last four blocks,
  rank 4, for five epochs. The frozen original visual encoder supplies an
  anchor loss of weight 0.1. LoRA is merged and the encoder is frozen before
  incremental sessions. Text LoRA is not used.
- Class-name prompts and GPT descriptions form text prototypes. Training
  images form visual prototypes. Base image statistics provide the visual
  covariance and a regularized within-class ZCA matrix. Support images are
  used to estimate new-class prototypes and covariances.
- An eight-block Flow is trained from sampled text features toward visual
  features. Incremental sessions adapt it using current support features,
  generated old-class feature replay, and previous-session velocity
  distillation. No old training images are retained for replay.
- The visual Bank mixes calibrated visual prototypes with Flow-generated
  prototypes: 20% Flow for Base and 40% for new classes. Classification adds
  ZCA-scaled visual cosine and raw reverse-Flow text cosine. The configured
  `cosine_sum` route does not use Cov/KNN probability fusion or Hub correction.
- LoRA training uses one image view. Prototype construction and evaluation
  use the mean of ten views (five crops and their horizontal flips).

Results are saved under
`output/dual_funnel_zca075_base_vision_lora_rank4/<dataset>/seed_<seed>_<timestamp>/`.
