# CLIP-A-self / BiMC: reproducibility and transfer verification

This branch replaces BiMC's text calibration with a **base-trained** Self-Attention adapter on class-conditioned GPT descriptions. The original BiMC text classifier is available as a matched control with `VDT_MODE=original`, while `VDT_MODE=mean` directly averages description embeddings without training.

## Single command: full verification

```bash
python scripts/validate_vdt_fscil.py
```

This launches **10 independent processes** using **Ensemble=False** and the same data/config/seed within each original-vs-adapter pair:

- CIFAR100: seed 1, 2, 3 (3 paired comparisons, 6 runs).
- CUB200: seed 1 (one pair).
- miniImageNet: seed 1 (one pair).

All use `VDT_RATIO=0.2`, `VDT_EPOCHS=20`, `VDT_LR=1e-4`; the `original` control uses the original `LAMBDA_T=0.5` combination and trains no adapter. The two configurations in a pair have equal other BiMC settings.

For a shorter smoke test:

```bash
python scripts/validate_vdt_fscil.py --datasets cifar100 --seeds 1
```

For SwanLab logging:

```bash
python scripts/validate_vdt_fscil.py --swanlab
```

For Ensemble=True comparisons:

```bash
python scripts/validate_vdt_fscil.py --train-cfg configs/trainers/bimc_ensemble.yaml
```

## Outputs

`output/vdt_validation/<timestamp>/` contains:

- `manifest.json`: branch, commit, dirty-state flag, Python version, active YAML file SHA256, run plan and arguments.
- `<dataset>_seed<seed>_<original|attention>.log`: complete training and evaluation console logs, including `EXPERIMENT_CONFIG_JSON`.
- `results.json`, `runs.csv`: full Session accuracy list, Task-wise accuracy list, final base accuracy, macro mean of prior incremental sessions, final harmonic, runtimes and status.
- `paired_comparisons.csv`: per-seed original-vs-CLIP-A-self differences.
- `aggregate.json`: across-seed mean differences and **sample** standard deviation; std is null for one seed.

Results are written after each process even if a later process fails. The outer script returns nonzero if any run fails.

The final novel metric is a **macro average across incremental tasks** in the last session, not a sample-weighted pooled novel accuracy. Its interpretation depends on dataset test counts; `Final Acc` always comes from the evaluator.

## Reproducibility notes

- Baseline and adapter run in separate new processes; seeds are reset before data/model creation, with deterministic CuDNN settings.
- The baseline control uses `VDT_MODE=original` on this same branch rather than using historical result screenshots. This is necessary to isolate effects of code/config changes.
- Re-running seed=1 tests deterministic repeatability; multi-seed dispersion measures seed sensitivity, not same-seed nondeterminism. If exact same-seed repeatability is in doubt, manually launch the short command twice and compare the resulting manifests and logs.
- The base-only trained adapter is frozen at incremental sessions. The original BiMC class descriptions are still used by the KNN branch.
- Test on the actual GPU/data environment; unit tests check scripting but cannot establish dataset accuracy.

## Unit tests

```bash
python -m unittest discover -s tests -v
```


## Follow-up: class-group bias, matched adapters, cross-dataset seeds

Run the new follow-up suite **without replacing the historical validation files**:

```bash
python -m unittest discover -s tests -v
python scripts/run_vdt_followup.py
```

Default is **12 independent runs** under `output/vdt_followup/<timestamp>/`:
- CIFAR100 seed 1, modes `original`, `attention`, `linear`, `mlp` (four runs).
- CUB200 seeds 2 and 3, `original` vs `attention` (four runs).
- miniImageNet seeds 2 and 3, `original` vs `attention` (four runs).

The CIFAR adapter study intentionally uses a **single seed initially** to assess whether attention is preferable to supervised feature adaptation. To verify across three seeds, use:

```bash
python scripts/run_vdt_followup.py --adapter-seeds 1,2,3 --no-transfer
```

To run just the cross-dataset missing seeds:

```bash
python scripts/run_vdt_followup.py --no-adapters
```

To upload logs to SwanLab, append `--swanlab`.

### Controlled comparison and parameter counts

- `original`: original BiMC text/description fusion, no VDT training.
- `attention`: original CLIP-A-self attention over descriptions; base-only CE.
- `linear`: same per-description residual ratio, dropout, LayerNorm, masked mean, base-only CE, but with a **single linear mapping**, no description interaction.
- `mlp`: same training/aggregation but an independent 2-layer MLP per description, no description interaction.

All trainable variants use the same learning rate, weight decay, epochs, frozen CLIP features and base CE objective. For ViT-B/16, reduction=4 (single-head), dimension 512, all three have exactly **263,680 trainable adapter parameters** (`D^2+3D`). The script prints and records the count per run. The baseline `original` constructs but does not train an unused adapter; its count is **not** a parameter-cost claim.

### Bias diagnostics

Each session now logs a `BIAS_METRICS_JSON` record with:
- `novel_to_base_pct` = 100 * (number of **novel-ground-truth test examples predicted as a base class**) / (number of novel test examples)
- `base_to_novel_pct` = 100 * (number of **base-ground-truth test examples predicted as a novel class**) / (number of base test examples)
- Corresponding counts, denominators, and within-group errors.

The base session has no novel examples, so `novel_to_base_pct` is **null**, not zero. Both rates are computed from the exact **final fused logits**, not the base-only training logits; they do not require sample-weighting assumptions.

Output files: `manifest.json`, per-run logs, `results.json`, `runs.csv`, `paired_comparisons.json/csv`, and `aggregate.json`. Reports include Final and task-wise accuracy, base/novel accuracy, confusion rates, parameter counts, and runtime. Across-seed sample standard deviations are reported only if more than one paired seed is present.

### Interpretation

A positive Final Acc difference together with a stable/increased `novel_to_base_pct` may indicate gains come largely from improved base discrimination rather than novel-class generalization. A linear/MLP adapter matching attention suggests that supervised adaptation or LayerNorm/residual mixing is sufficient; attention-specific gains require attention outperforming these controlled baselines.

This suite does not silently tune per-dataset hyperparameters. **No end-to-end GPU results are claimed until the suite is executed on the dataset server.**


## Extended focused ablation: CIFAR100 + CUB200

To answer whether a simple CE-trained adapter generalizes as well as
self-attention, especially on fine-grained CUB200:

```bash
python -m unittest discover -s tests -v
python scripts/run_vdt_finegrained.py
```

This runs **20 matched and isolated experiments** under
`Ensemble=False`, `VDT_RATIO=0.2`, `VDT_EPOCHS=20`:

- CIFAR100 seeds **2, 3** × `original/attention/linear/mlp` (8 runs);
  previous seed 1 controls remain available in the earlier follow-up directory.
- CUB200 seeds **1, 2, 3** × `original/attention/linear/mlp` (12 runs).
- miniImageNet is intentionally **not repeated** here: all three seeds of
  `original/attention` have already been measured. To expand MLP/linear
  onto miniImageNet later, update the study plan separately.

Quick CUB-only validation:
```bash
python scripts/run_vdt_finegrained.py --no-cifar --cub-seeds 1
```

Run CIFAR-only:
```bash
python scripts/run_vdt_finegrained.py --no-cub
```

Optional `--swanlab` publishes the usual per-session metrics. The new runner
writes to `output/vdt_finegrained/<timestamp>/` and does not alter previous
experiment outputs:

- `results.json`, `runs.csv`, `paired_comparisons.csv/json`,
  `aggregate.json`, `manifest.json`, and per-run console logs
- `diagnostics/cub200_seed<seed>_<mode>.json`: final-session
  class-level accuracy, sorted hard classes and directed confusion pairs
- `cub_class_accuracy_deltas.csv`: same-seed changes in the accuracy of
  each class relative to the original BiMC control (positive = improvement)
- `cub_confusion_pair_deltas.csv`: directed true label → predicted label
  mistakes that increased or decreased relative to the same-seed original
  control (positive = **more errors** on that pair)

Class names are taken from the dataset's own `classes` array, and classes
are matched by integer ID within each same-seed pair. **These reports do not
claim which species belong to the same biological family**; to establish that
kind of relationship a separate, explicit taxonomy mapping would be needed.
Rates are based on actual test counts. The diagnostic uses only predictions
and ground truth at final evaluation, not training inputs or gradients.

Before interpreting deltas, check that paired original and adapter records
both completed successfully and use the same config/seed. This experiment is
a comparison of mechanisms, **not** a claim that any architecture is new or
that CUB200 accuracy has improved prior to running these trials.


## New: support-guided description selection and prototype geometry

Targeted research question: Does a 5-shot *training-support* visual prototype
help distinguish near-identical bird classes, or does base-only supervised
text adaptation collapse their textual margins?

```bash
python -m unittest discover -s tests -v
python scripts/run_vdt_visual_diagnostics.py
```

Default **20 matched runs** (same dataset, seed, CLIP backbone, BiMC classifier,
Ensemble=False, text descriptions, 20 base training epochs for attention):
- CUB200 seeds 1,2,3 × `original`, `mean`, `attention`, `visual_weighted` (12).
- CIFAR100 seed 1 × four modes (4).
- miniImageNet seed 1 × four modes (4).

Fast CUB-only test:

```bash
python scripts/run_vdt_visual_diagnostics.py --datasets cub200 --cub-seeds 1
```

Additional temperature/blend experiments (explicit configurations, **not**
automatically tuned on test set):

```bash
python scripts/run_vdt_visual_diagnostics.py --datasets cub200 --cub-seeds 1 --temp 5 --blend 0.5
```

### What `visual_weighted` actually does

For every class `c`, construct the mean visual prototype `v_c` **only**
from its own training/support images: all base-session training images for base
classes, and the session's five labeled training shots for each novel class.
Frozen CLIP encodes name-conditioned description texts `d_{c,j}`.

`w_{c,j} = softmax_j(temp * cosine(v_c, d_{c,j}))`

`t_c^weighted = normalize((1-blend) * mean_j(normalize(d_{c,j})) + blend * normalize(sum_j w_{c,j} * normalize(d_{c,j})))`

No CLIP updates, attention training, or added trainable parameters are
required for this mode. The original BiMC visual prototype calibration,
fusion `beta`, task splits and accuracy reporting stay unchanged.
`temp=0` or `blend=0` produces the no-training mean baseline up to numerical
precision. `visual_weighted` is a *diagnostic alternative*, not an established
new method.

### Geometry diagnostics (CUB200 only; final seen-class session)

For non-original modes, the script compares the **raw mean description
prototype** with the selected calibrated text prototype and the uncalibrated
training/support visual prototype. Saved geometry includes:
- `support_alignment`: base/novel mean text↔support-visual cosine and delta
- `classes`: per-class support cosine, and the nearest-other-text class
  before/after adaptation
- `most_similar_raw_pairs`: 40 closest raw text pairs and their changed
  separation `delta_separation = raw_cos - candidate_cos`. Positive = pair
  moved further apart
- `most_collapsed_pairs`, `most_separated_pairs`: largest changes in
  cosine separation. These are *geometric diagnostics*, not guarantees of
  classification gain

Files under `output/vdt_visual_diagnostics/<timestamp>/`:
`manifest.json`, complete per-run `.log`, `results.json`, `runs.csv`,
`paired_comparisons.csv/json`, `aggregate.json`,
`diagnostics/cub200_seed*_*.json`,
`geometry/cub200_seed*_*.json`,
`geometry_summary.csv`,
`cub_class_accuracy_deltas.csv`, `cub_confusion_pair_deltas.csv`.

No test labels are used in description selection or geometric calibration.
Test labels are used **only** for ordinary post-inference accuracy/confusion
evaluation. A class's genuine image prototype is saved before BiMC's visual
calibration so the weighting metric is not computed from a recursively
calibrated prototype. Interpret paired seed means before drawing conclusions;
CUB changes of a few tenths of a percentage point can be small.
