# Tangent Flow with Base Vision LoRA (rank 4)

This branch builds on commit `b16cf3b`. Its only trainer configuration is
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
  MSE anchor loss of weight 0.1 and a same-view cosine semantic loss of weight
  0.01. LoRA is merged and the encoder is frozen before incremental sessions.
  Text LoRA is not used. Since both features are normalized, the cosine and
  MSE terms constrain the same image direction at different scales.
- Class-name prompts and GPT descriptions form text prototypes. Training
  images form visual prototypes. Per-class visual covariances support old-class
  feature replay. A Base within-class covariance with fixed 0.2 shrinkage
  supplies the visual ZCA transform. Support images are used to estimate
  new-class prototypes and covariances.
- An eight-block Flow is trained from sampled text features toward visual
  features. Incremental sessions adapt it using current support features,
  generated old-class feature replay, and previous-session velocity
  distillation. No old training images are retained for replay.
- The visual Bank mixes image-mean visual prototypes with Flow-generated
  prototypes: 20% Flow for Base and 40% for new classes. BiMC visual
  `soft_calibration` is disabled (`VISION_CALIBRATION: False`). Classification adds
  ZCA-scaled visual cosine and raw reverse-Flow text cosine. The configured
  `cosine_sum` route does not use Cov/KNN probability fusion or Hub correction.
- LoRA training, prototype construction, and evaluation use one image view;
  ten-view test-time augmentation is disabled.

Results are saved under
`output/dual_funnel_zca02_singleview_cosreg001_rank4/<dataset>/seed_<seed>_<timestamp>/`.
