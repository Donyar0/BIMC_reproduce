# BiMC reproduction: support-margin adaptive Flow experiment

This branch retains one runnable trainer configuration:
`configs/trainers/tangent_flow_support_margin_adaptive_text.yaml`.
Dataset configurations remain in `configs/datasets/` for CIFAR-100, CUB-200, and miniImageNet.
Set `DATASET.ROOT` in the dataset YAML to your local dataset location before running.

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/tangent_flow_support_margin_adaptive_text.yaml
```

The method uses frozen CLIP features, Base and incremental text-to-visual Flow training,
16 Base + 16 historical-increment distribution replay, prior-stage velocity distillation,
ten-view feature averaging, visual whitening, support-derived new-class score margins,
and query-adaptive description pooling. See `single_flow/README.md` for details.

The repository still contains historical model, script, and test implementations. Legacy
scripts and tests that name other trainer YAML files are not entry points for this
trimmed branch. The source paper is **Enhancing Few-Shot Class-Incremental Learning via
Training-Free Bi-Level Modality Calibration (CVPR 2025)**.
