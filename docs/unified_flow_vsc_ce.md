# Unified Flow + VSC visual-guided text calibration + visual classification CE

Based on `unified-bidirectional-flow` (not `main`). Original Flow Matching
velocity regression, optional cycle loss, BiMC visual-prototype construction,
prototype bank, incremental support/replay adaptation, and evaluation remain.

## Modifications

**1. Visual-guided text prototype calibration (inference only):**

For every seen class `c`, use its STORED real support visual mean `v_c`
(not the generated visual prototype):

```
projected_visual_to_text = normalize(v_c - velocity(v_c, t=1))
calibrated_text_c = normalize((1-lambda)*text_c + lambda*projected_visual_to_text)
```

This is a **one-step approximate inverse**, consistent with the existing
query visual-to-text correction. The support-prototype correction is unscaled;
the separate mixing coefficient `TEXT_PROTO_CALIBRATION` controls how much
information enters the final text prototype. The refined text prototype
replaces `bank['text']` only in semantic classification scores.
It is calculated once per evaluated session; neither visual logits nor the
stored bank are rewritten. CLIP remains frozen.

**2. Visual classification CE (base training only):**

For differentiably generated class pseudo-visual prototypes `g_c`:

```
g_c = normalize(Euler_Flow(text_c, steps=VISUAL_CE_STEPS))
L_visual_CE = CE(normalize(real_image_feature) @ g.T / TEMPERATURE, label)
L_train = L_FM + CE_WEIGHT*L_old_endpoint_CE
          + CYCLE_WEIGHT*L_cycle
          + VISUAL_CE_WEIGHT*L_visual_CE
```

CE is class-discriminative, comparing real visual images with generated visual
class prototypes exactly as in visual-space inference; unlike the legacy
`CE_WEIGHT`, it does NOT classify predicted single-example flow endpoints
against real class means. It is intentionally limited to base training to
avoid silently changing incremental adaptation/replay. Base training loss
logs `visual_ce_loss` and the checkpoint stores `visual_ce_history`.

## Settings and ablations

In `configs/trainers/unified_flow.yaml`:

- `VISUAL_CE_WEIGHT: 0.1`: set 0 to disable the new CE.
- `VISUAL_CE_STEPS: 1`: use one differentiable Euler step during training.
  Inference remains `STEPS: 8`; integration mismatch is an experiment variable.
- `TEXT_PROTO_CALIBRATION: 0.1`: set 0 for original text prototypes.
- `CE_WEIGHT: 0.0`: original endpoint CE is kept separate and off.

The exact lambda and CE weights are starting points, **not validated optima**.
Use held-out base validation data rather than incremental test data to tune.

| Experiment | VISUAL_CE_WEIGHT | TEXT_PROTO_CALIBRATION |
|---|---:|---:|
| Original unified Flow (control) | 0 | 0 |
| Only visual-guided text prototype calibration | 0 | 0.1 |
| Only visual CE | 0.1 | 0 |
| Both modifications | 0.1 | 0.1 |

Changing `VISUAL_CE_WEIGHT` requires retraining the base model.
Changing `TEXT_PROTO_CALIBRATION` only changes evaluation; saved-run
re-evaluation supports this while checking CE training settings.

## Commands

```bash
git fetch origin
git switch feature/unified-flow-vsc-ce
python -m unittest single_flow.test_unified_flow single_flow.test_vsc_ce_flow -v

python main.py \
  --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/unified_flow.yaml \
  --use_swanlab
```

Other datasets: `configs/datasets/cifar100.yaml` and
`configs/datasets/miniimagenet.yaml`. Preserve the same seed and dataset
files for a fair comparison with the parent branch.
Check `Final acc`, `Task-wise acc`, base/novel mean accuracies.

**Scope and limitations:** These are VSC-inspired modifications of the
existing shared Flow, not a direct implementation of the SAR paper's local
classification model. One-step reverse integration is an approximation.
Historical prototypes are still kept frozen while the Flow can be adapted
incrementally (existing behavior), which may cause version misalignment.
No full CUB GPU benchmark or accuracy improvement is claimed.
