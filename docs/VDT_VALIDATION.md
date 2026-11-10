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

## Optional incremental-session Linear Adapter fine-tuning

The default setup remains the **frozen-Adapter control**:
`VDT_INC_EPOCHS=0`. For an incremental-learning ablation, use a
**separate** low learning rate and 3 epochs on each incremental
session's 5-shot labeled support images:

```bash
# Control: base CE training only; adapter frozen for the 8/10 new sessions
python scripts/run_linear_baseline.py --dataset cifar100 --seed 1

# Incremental experiment: 3 AdamW epochs per new session, lr=1e-5
python scripts/run_linear_baseline.py --dataset cifar100 --seed 1 \
    --inc-epochs 3 --inc-lr 1e-5

# Focus on fine-grained dataset, one run only:
python scripts/run_linear_baseline.py --dataset cub200 --seed 1 \
    --inc-epochs 3 --inc-lr 1e-5
```

The underlying `main.py` flags are:
`TRAINER.BiMC.VDT_INC_EPOCHS` (default 0) and
`TRAINER.BiMC.VDT_INC_LR` (default 1e-5).
The base-stage epochs/lr are unchanged. The original BiMC mode does
not update any adapter.

**Protocol:** at the start of each incremental session, build that
session's statistics from its **training/support images only**.
Fine-tune the *same base-trained Linear Adapter*, with CE on current
session images and their original global class labels. The softmax
classifier consists of **all seen text prototypes**, meaning the new
shots compete against base and previously seen novel classes. No old
image replay, distillation, additional auxiliary losses, old image
features, or test data are used in the update. The CLIP encoders stay
frozen. After training, freeze the adapter, recompute all seen text
prototypes with the updated weights, and evaluate.

Each incremental session prints `VDT-Adapter incremental task=... CE=...`
and `VDT_INCREMENTAL_UPDATE_JSON` with new image count, seen-class
count, LR, epochs, parameter delta norm, and the mean cosine similarity
of **old/new text prototypes before vs. after** each update.
`result.json` stores these in `metrics.incremental_updates`; 0 updates
in the frozen run is expected. Smaller old-prototype cosine indicates
more drift. This metric does not, by itself, prove forgetting. The original Final Acc,
Task-wise Acc and base/novel bias summaries stay unchanged.

This is an intentionally minimal ablation, **not** a claim that
few-shot updating will help. Updating a globally shared adapter with
only new-class CE may drift older text prototypes and harm base-class
accuracy. Compare Base/Novel/Final accuracies against the frozen
control before trying any replay or regularization.
