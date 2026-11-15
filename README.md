# Tangent Flow: ten-view, visual whitening, raw text cosine, distillation

This branch contains one experiment: `configs/trainers/tangent_flow_vis_whiten075_text_raw_distill1.yaml`. It starts from `biflow-no-lora` and retains the BiMC dataset/CLIP feature extraction needed by this experiment. Other Flow runners, ablation configs and remote scripts were removed.

## Run

Install `requirements.txt`, prepare CUB-200 and its GPT descriptions, and set `DATASET.ROOT` in `configs/datasets/cub200.yaml` to the local dataset path. Then run from the repository root:

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/tangent_flow_vis_whiten075_text_raw_distill1.yaml
```

Optional SwanLab logging: add `--use_swanlab`.

## Method

CLIP ViT-B/16 is frozen. Training/support images and test images use the normalized mean of ten CLIP views: original, four corner crops, and their horizontal flips. BiMC combines template and description text with equal weights. New-class visual prototypes receive BiMC soft calibration.

For each class, description features determine an isotropic text noise scale and image features determine a visual mean and covariance. Base covariances are shrunk toward a pooled estimate. Incremental covariances blend empirical few-shot estimates with text-similarity-weighted Base covariances. The Flow predicts velocity from text to visual features. Dual-funnel training samples one half of each batch in the early time interval toward visual centers and the other half in the late interval toward real current-session support features or sampled old-class visual features.

Base training uses 100 epochs × 100 updates. Each incremental session uses 100 updates, with half of the batch drawn from old-class distributions. From the second incremental session onward, the 32 old-class rows split into 16 Base and 16 historical incremental rows. A frozen copy of the previous Flow supplies old-class velocity distillation with weight 1.0. No old image exemplars are retained for replay.

The visual bank uses `normalize(0.8 * BiMC-calibrated visual prototype + 0.2 * Flow-generated prototype)`. Old bank rows are retained. Inference adds (1) visual cosine after scaling the query and bank by Base within-class standard deviation to the power `-0.75` and (2) raw text cosine after reverse Flow integration. This `cosine_sum` route does not use probability temperatures, covariance/KNN priors, gamma weights, or cycle gating.

Results are written beneath `output/dual_funnel_vis_whiten075_text_raw_distill1/<dataset>/seed_<seed>_<timestamp>/`, including config, checkpoints, session banks, and CSV/JSON metrics. Real-dataset accuracy is not established by this repository; run on the prepared data to measure it.
