# Frozen CLIP + Flow Matching (FM-only)

This cleaned-up branch is based on `unified-bidirectional-flow` and retains
the single-flow and shared bidirectional-flow experiments. Training is
**only** the Flow Matching velocity regression loss.

## Training objective

CLIP image/text encoders are frozen. Using existing BiMC text prototypes and
support visual embeddings, construct straight interpolation paths:

```python
t ~ Uniform(0,1)
x_t = (1-t) * text_proto + t * visual_feature
target_velocity = visual_feature - text_proto
L_FM = mean(sum((velocity_net(x_t,t)-target_velocity)**2, dim=-1))
```

`TARGET: sample` pairs a text class prototype with individual base-class
image features. `TARGET: prototype` pairs each class text prototype with its
visual mean. Both modes use the same FM loss.

**Removed:** Cycle Loss, classification CE, CE temperature, margin correction,
per-session Flow replay/adaptation, and the BiMC covariance/KNN score ensemble.
The original non-FM BiMC statistics required to build class prototypes still
exist; the BiMC baseline method itself is not deleted.

## Dual-direction inference

The same frozen, base-trained Flow is used throughout all incremental stages:

1. **Text to visual:** Euler integrate each class text prototype to obtain a
   generated visual prototype, then mix with that class's support visual mean
   using `BETA_BASE` or `BETA_INC`.
2. **Visual to text:** Apply the shared field in the reverse direction to each
   test image feature using `ALPHA`.
3. **Classification:** Average/interpolate visual cosine and semantic cosine
   scores using `GAMMA`, then take the predicted class with the highest score.

Every new session adds prototypes for the new support classes and does not
update or replace old prototypes or the base-trained FM parameters.

## Run

```bash
git fetch origin
git switch refactor/unified-flow-fm-only
python -m unittest discover -s single_flow -p 'test_*.py' -v

python main.py \
  --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/unified_flow.yaml \
  --use_swanlab
```

Other datasets: `configs/datasets/cifar100.yaml` and
`configs/datasets/miniimagenet.yaml`. The original BiMC baseline still runs
with `configs/trainers/bimc.yaml`.

The new trainer stores outputs under
`output/unified_flow_fm_only/<dataset>/seed_<seed>_<timestamp>/`.
Expect `base_flow.pt`, `flow_session*.pt`, `bank_session*.pt`,
`config.yaml`, `metrics.csv` and `metrics.json`, plus
`Final acc` / `Task-wise acc` outputs.

`--eval_run` accepts a **training run directory**, never a direct
`base_flow.pt` path. It refuses sessions where Flow parameters changed after
base training, or where the checkpoint declares nonzero CE/cycle weights.
Only add `--trusted_checkpoints` for checkpoints whose origin you trust.

## Experimental interpretation

This is a different protocol from the original Unified Flow, which performed
incremental replay updates and optionally mixed BiMC covariance/KNN scores.
Consequently, any accuracy difference cannot be attributed to removing cycle
loss alone. Compare it with other methods using identical CLIP/data splits and
state clearly that incremental Flow training is disabled.

The removed code is still preserved in the original branch's Git history.
