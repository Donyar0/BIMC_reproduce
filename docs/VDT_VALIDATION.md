# Minimal BiMC text-prototype calibration

This branch deliberately keeps **one** learned text calibration module:
a residual **LinearTextAdapter** trained on base-class image CE.

## Method

- Frozen CLIP ViT-B/16 image and text encoders.
- Each class description is encoded as `a photo of {class}.` followed by its
  natural-language visual description.
- A shared linear residual projection, LayerNorm and masked average produce
  one L2-normalized text prototype per class. Same architecture, dropout,
  residual ratio and base CE objective as the earlier `VDT_MODE=linear`
  ablation. Default ratio=0.2, epochs=20, learning rate=1e-4.
- Train **once** using base-session image features and base labels.
  The adapter is frozen at every incremental session.
- BiMC's original image prototype construction and novel-class visual
  calibration are preserved.
- Prediction uses the existing BiMC fused text + visual prototype classifier.
  `USING_ENSEMBLE: False` in `configs/trainers/bimc.yaml` means the
  covariance and KNN probabilities are not used in the final mixture.
- `VDT_MODE=original` is retained only as a comparison against the genuine
  original BiMC text/description interpolation; no adapter is constructed
  or trained.

**Removed:** Attention, MLP, uniform-mean experimental mode,
visual-conditioned weighting, contrastive fixed/CV weighting,
pseudo-incremental parameter selection, weight-entropy/geometry code,
and the multi-method experiment runners.

## Run

```bash
git switch feature/bimc-vdt-self-attention
git pull origin feature/bimc-vdt-self-attention
python -m unittest discover -s tests -v

# Default: single CIFAR100 seed-1 linear run
python scripts/run_linear_baseline.py

# Run other datasets one at a time
python scripts/run_linear_baseline.py --dataset cub200 --seed 1
python scripts/run_linear_baseline.py --dataset miniimagenet --seed 1

# Optional original BiMC control on exactly the same split
python scripts/run_linear_baseline.py --dataset cub200 --seed 1 --mode original
```

Alternatively the usual entrypoint remains supported:

```bash
python main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/bimc.yaml \
  --opts TRAINER.BiMC.VDT_MODE linear SEED 1
```

One invocation runs **one experiment**, with no hidden parameter sweeps.
The runner saves `manifest.json`, `run.log` and `result.json`
under `output/vdt_linear/<timestamp>/`. Full Final/Task-wise
accuracy and base/novel confusion metrics are kept. The final CUB
session also prints class-level confusion diagnostics; this is
evaluation-only and has no influence on prototypes or training.

## Why Linear?

On the previously supplied, matched three-seed CIFAR100 experiments,
Linear achieved the strongest mean final accuracy (~75.24%) relative
to Attention (~74.74%) and MLP (~75.07%). On CUB200, all these
adapters showed very small improvements, and Linear was slightly
better. The choice of Linear favors **simplicity and overall accuracy**,
not a claim of improved novel-class generalization: its gains
are mainly driven by base classes. These numbers are historical
observations, not newly rerun results.

To reproduce historical comparisons, use the archives from prior
experiments or commit history rather than restoring obsolete modules
in this clean branch.
