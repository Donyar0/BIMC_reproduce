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
