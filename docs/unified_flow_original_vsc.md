# Unified Flow × original-direction VSC semantic-guided visual calibration

Based on the original `unified-bidirectional-flow` branch. This is a **small
experimental adaptation** of the original (non-swapped) VSC-Net visual
classification module; it is NOT a reproduction of the paper's learned
MLC-Net, MLP projection or all four VSC-Net losses.

The previous swapped-direction experiment modified text prototypes and added
visual CE; on CUB the user observed reduced performance. This branch does
neither.

## What changed

The parent Unified Flow already implements both directions:

- **Text → Visual:** learn one shared Flow, generate pseudo-visual class
  prototypes, and blend them with real support visual means.
- **Visual → Text:** apply a single approximate inverse velocity step to
  query image embeddings, then score against the original CLIP text class
  prototypes. Keep this semantic branch unchanged.

New: borrow the **original VSC-Net mapping direction** by using *text-origin
pseudo-visual representations to guide real visual class prototypes*.
Use only the previously stored support bank for classes seen so far:

```
similarity[c,j] = dot(normalize(real_visual[c]),
                      normalize(flow_generated_visual[j]))
weights[c,:] = softmax(temperature * similarity[c,:])
context[c]   = sum_j weights[c,j] * flow_generated_visual[j]
delta[c]     = context[c] - flow_generated_visual[c]
visual_vsc[c] = normalize(original_fused_visual[c] + strength * delta[c])
```

`topk` limits attention to the same class and its strongest competing
classes. Subtracting the own-class generated vector avoids double-counting it,
since the original Flow bank has already fused real/Flow visual information.
This is a conservative parameter-free replacement for VSC-Net's learnable
fusion MLP, designed for a clean first experiment.

The modified visual prototypes are used both by the cosine visual classifier
and by the BiMC-style Mahalanobis ensemble, as the latter precomputes its
precision terms from the visual bank. Original text-space classification,
Flow FM training and incremental adaptation are unchanged. The original
saved `bank_session*.pt` are not overwritten by this optional inference
calibration, so it is safely reversible.

## Run

```bash
git fetch origin
git switch feature/unified-flow-original-vsc

python -m unittest single_flow.test_unified_flow single_flow.test_original_vsc -v

python main.py \
  --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/unified_flow.yaml \
  --use_swanlab
```

The resulting `Final acc`, `Task-wise acc`, local metrics and SwanLab
logging are kept. Session logs additionally show original-VSC diagnostics:
self-attention, same-class top-1 semantic alignment and average cosine
prototype shift.

## Ablations

In `configs/trainers/unified_flow.yaml`:

- `FLOW.VSC_VISUAL_STRENGTH: 0.0`: exact original Unified Flow prediction path
  (only diagnostic bookkeeping added). It is the controlled baseline.
- `FLOW.VSC_VISUAL_STRENGTH: 0.1`: experimental weak residual correction.
- `FLOW.VSC_TOPK: 5`: same class and four closest generated semantic classes.
  `1` forces the same-class context, hence exactly zero residual correction.
- `FLOW.VSC_ATTENTION_TEMPERATURE: 16.0`: sharpness of support-visual to
  pseudo-visual attention. The original BiMC TAU is separate.

No extra CE, trainable MLP or change to the FM objective. The parent
`FLOW.CE_WEIGHT: 0` and `FLOW.CYCLE_WEIGHT: 0` remain.

**Best first ablation:** same trained checkpoint, compare strength 0 and
0.1 using `--eval_run <original-run-dir>` with matching saved data/model
configuration. Evaluation-only strengths may vary without retraining; use
base validation data for choosing weights rather than selecting the
best CUB test-set score.

**Limitations:** This is a parameter-free surrogate for paper's
learnable visual prototype calibration, not an exact architectural copy.
No CUB GPU results are claimed; small-class attention can fail when visual
prototype estimates are unreliable.
