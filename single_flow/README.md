# Support-margin adaptive text-to-visual Flow

Run from the repository root with one dataset YAML and the sole trainer YAML:

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/tangent_flow_support_margin_adaptive_text.yaml
```

Base images and each incremental support set are encoded by a frozen CLIP. The Flow
learns a text-to-visual velocity field using sampled class text origins and visual
endpoints. Incremental adaptation uses 16 Base and 16 historical incremental class
distribution replay examples per default batch and a previous-stage velocity
distillation weight of 1.0. Old image exemplars are not retained for replay.

The visual Bank combines calibrated visual and generated prototypes. A test image is
also transported backward to text space. For each class, its descriptions receive
query-dependent softmax weights with temperature 10; their weighted feature is mixed
with the original text prototype at weight 0.5. The final raw score sums visual and
text cosines and adds the support-derived margin for incremental classes. The margin
for a new session is the difference between Base and that session's correct-class
visual cosine means, computed only from training/support features. Ten-view CLIP
averaging and visual whitening power 0.75 are enabled by the trainer config.

Each run saves its configuration, per-session Flow and Bank checkpoints, and metrics
under the configured `FLOW.OUTPUT_DIR`. The repository does not include dataset
images or CLIP weights. Reported accuracy must be measured on your local data.
