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


## 2026-11-10: Inference-aligned training objective ablation

**Motivation:** the old `text_ce` trains on image-versus-text-only
scores, but the actual BiMC inference classifies using
`normalize(beta * adapted_text + (1-beta) * support_visual)`.
For CUB200 beta=0.30 (visual weight=0.70), while CIFAR100 and
miniImageNet have beta=0.65. Consequently, the training and prediction
objectives can be substantially different, especially on CUB200.
This is a hypothesis, *not* evidence that CE necessarily caused the
poor CUB200 results.

Keep the same frozen CLIP, Linear Adapter, prompt+description encoding,
BiMC fusion and default Ensemble=False. Only change the **base training
loss**, controlled by the new `VDT_TRAIN_OBJECTIVE` flag:

* `text_ce` (default): existing CE over image-to-text-only logits.
* `fusion_ce`: CE over the same L2-normalized text+visual fused
  classifier used at inference, scaled by 100. Visual prototypes
  are calculated **only from training/support images**.

To avoid a self-match shortcut in training, the **positive** class
visual prototype for each training query is calculated with the current
training image removed (leave-one-out, or LOO). Negative-class visual
prototypes remain unchanged. For the optional real incremental phase,
the 4-shot LOO support prototype is also BiMC-calibrated with the frozen
base visual prototypes before forming the positive fused classifier.
Only the current session's 5-shot images enter incremental CE;
historical image features are *not* replayed. No query/test images
are accessed during adapter optimization.

Suggested **two runs only** on CUB200 Seed 1:

```bash
git pull origin feature/bimc-vdt-self-attention
python -m unittest tests.test_vdt_objective -v
python -m unittest tests.test_vdt_linear -v

# Old loss -- base-only text CE, no incremental updates.
python scripts/run_linear_baseline.py --dataset cub200 --seed 1 --objective text_ce

# New loss -- change only the base objective.
python scripts/run_linear_baseline.py --dataset cub200 --seed 1 --objective fusion_ce
```

The default remains `text_ce`; existing command invocations are
unchanged. Do NOT compare `fusion_ce` with results using different
beta, seeds or ensemble settings.

To separately test optional incremental-session loss, use
`--inc-epochs 3 --inc-lr 1e-5 --inc-objective fusion_ce`.
Keep incremental updates **off** for the first A/B evaluation to
isolate the base training objective.

The run manifest stores both `base_objective` and
`incremental_objective`. The base stage prints
`VDT_BASE_OBJECTIVE_JSON`; per-session incremental logs retain
old/new text prototype drift diagnostics.

**Interpretation:** improvement in CUB200 base accuracy alone does
not establish fine-grained FSCIL transfer. Evaluate final accuracy,
new-class mean, and novel-to-novel confusions. Since the fusion loss
also trains on visual class prototypes, consider whether its advantage
carries over to unseen classes in future multi-seed runs.


## New: visual intra-class Self-Attention Adapter ablation

**Question:** can CLIP-A-self-style aggregation of *image features from
the same class* build more discriminative visual prototypes than an
unweighted mean, especially on CUB200?

### Components

- **Text Adapter:** original base-only Linear Adapter on prompt+visual
  descriptions. Same CE objective as before, frozen in FSCIL increments.
- **Visual Adapter:** one-head reduced-QKV self-attention on
  `[class, support-images, CLIP-feature-dim]`; residual mixing +
  LayerNorm + masked class-wise mean + L2 normalization. CLIP image
  encoder is frozen.
- **Visual training:** on base TRAIN image features, each optimizer step
  creates fresh, class-balanced episodes containing **5 support
  images and 5 disjoint query images per base class**. Supports build
  visual prototypes, while queries incur classification CE over base
  visual prototypes: `CE(100 * image_query @ visual_proto.T, class)`.
  Query images are never included in their own support aggregation.
  Default 20 epochs × 5 episodes/epoch, AdamW LR 1e-4.
- **Incremental inference:** the trained Visual Adapter remains frozen.
  Reaggregate each newly introduced class's **five available support
  images** through that same attention network. Apply original BiMC
  novel visual calibration *after* this aggregation. Base visual
  reference prototypes are the base-adapted prototypes, and remain
  fixed; previous session prototypes are cached, never re-created.
- **Full base inference:** by default the Visual Adapter aggregates
  all available base training images per class, one class at a time.
  For memory-constrained environments optionally specify
  `--visual-max-support 128`; this selects deterministic class
  subsets and is *not* identical to the default full-support setting.
- **Text+Visual fusion:** unchanged BiMC
  `normalize(beta * text_proto + (1-beta) * image_proto)`.
  Ensemble=False is maintained for the controlled comparisons.

This is **instance-level visual self-attention within a class**, not
spatial patch/region attention, and does not claim to solve patch-level
fine-grained perception by itself.

### Three-arm controlled experiment (CUB200, seed 1)

```bash
git fetch github
git switch feature/bimc-vdt-self-attention
git pull --ff-only github feature/bimc-vdt-self-attention
python -m unittest tests.test_vdt_visual -v
python -m unittest tests.test_vdt_linear -v
python -m unittest tests.test_vdt_objective -v

# Each command launches ONE experiment:
python scripts/run_linear_baseline.py --dataset cub200 --seed 1 --ablation text
python scripts/run_linear_baseline.py --dataset cub200 --seed 1 --ablation visual
python scripts/run_linear_baseline.py --dataset cub200 --seed 1 --ablation both

# Or run precisely these three arms and write a summary CSV:
python scripts/run_adapter_triplet.py --dataset cub200 --seed 1
```

Optional *fourth* original BiMC control:
`python scripts/run_adapter_triplet.py --dataset cub200 --seed 1 --include-original`.

| ablation | text classifier | visual classifier | adapter training |
| --- | --- | --- | --- |
| text | base-trained Linear | original BiMC support mean | text CE |
| visual | original BiMC prompt+description text | visual self-attention support aggregation | episodic visual CE |
| both | base-trained Linear | visual self-attention support aggregation | independent text CE + visual CE |
| original (optional) | original BiMC | original support mean | neither |

For a fair controlled first comparison, **all three have base-only
training**; do not use incremental text fine-tuning. Visual training
uses the same deliberately isolated initialization seed, support/query
episode generator, and dropout RNG in `visual` and `both` arms.
The text initialization and optimizer RNG are preserved with separate
RNG contexts, giving a cleaner paired architectural comparison.

For the initial three-arm sweep use `--objective text_ce` (default).
The earlier `fusion_ce` for text-only experiments is still supported,
but **both+fusion_ce** is explicitly disallowed until it can compute
attention-aware leave-one-out visual prototypes. This prevents
silently mixing a mean-LOO loss with an attention-based classifier.

The scripts write `output/adapter_triplet/<timestamp>/summary.csv`,
individual run logs and a `results.json`. Each run has its own
`output/vdt_linear/<timestamp>/result.json`, class-wise metrics and
source/config SHA256 manifest. Assess Base/Novel/Final and the CUB
novel-to-novel directed confusion counts, not only Final Acc.

**Known limitation:** the training episodes have 5 supports per base
class, while default base inference uses all support images. This
is intentional for 5-shot novel generalization, but a support-count
distribution shift remains and should be reported. No results are
claimed until actual GPU runs are complete.


## New: Visual Adapter CE versus multimodal fused CE

The pure visual attention module and FSCIL inference stay unchanged.
Only the **base-session visual attention training loss** is switched:

- `VDT_VISUAL_OBJECTIVE=visual_ce` (default): support images produce a
  visual attention prototype; disjoint query images are classified with
  `CE(100 * normalized_query @ normalized_visual_prototypes.T, y)`.
- `VDT_VISUAL_OBJECTIVE=fusion_ce`: take those **same** visual
  attention support prototypes and fuse with the class text classifier
  exactly as BiMC inference does:
  `P_c = normalize(beta*T_c + (1-beta)*V_c)`,
  `L = CE(100 * normalized_query @ P.T, y)`.

Text classifier `T` is **frozen and detached** throughout the visual
episodic training. In `visual` mode it is the original BiMC interpolation
`(1-VDT_ORIGINAL_LAMBDA_T)*prompt_proto +
VDT_ORIGINAL_LAMBDA_T*description_proto`, without an extra
pre-fusion normalization. In `both` mode the Linear Text Adapter is
trained first on base images using `text_ce`, then frozen and reused for
the visual `fusion_ce` loss.

Visual support and query episodes still use disjoint base TRAIN samples
(5 support + 5 separate query images **per class**). The two losses
use the *same* visual attention architecture, episode sampler, fixed
class order, optimizer, learning rate, epochs, seed, and inference
fusion. CLIP remains frozen. The original BiMC visual soft calibration
still runs **after visual aggregation** at incremental evaluation.

### Minimal paired experiment — CUB200, seed 1

```bash
git fetch github
git switch feature/bimc-vdt-self-attention
git pull --ff-only github feature/bimc-vdt-self-attention
python -m unittest tests.test_vdt_visual -v
python -m unittest tests.test_vdt_linear -v
python -m unittest tests.test_vdt_objective -v

# Exactly TWO visual-only experiments (pure CE vs fused CE).
python scripts/run_visual_objective_ablation.py --dataset cub200 --seed 1
```

Or execute individually:

```bash
python scripts/run_linear_baseline.py --dataset cub200 --seed 1 \
  --ablation visual --visual-objective visual_ce
python scripts/run_linear_baseline.py --dataset cub200 --seed 1 \
  --ablation visual --visual-objective fusion_ce
```

Optional dual-Adapter paired comparison (two more independent runs):

```bash
python scripts/run_visual_objective_ablation.py \
  --dataset cub200 --seed 1 --ablation both
```

Outputs: `output/visual_objective_ablation/<timestamp>/summary.csv`
and `summary.json`, plus complete individual logs and run manifests.
The `result.json` for each run reports `visual_objective`; training
logs emit `VDT_VISUAL_TRAIN_JSON` with objective name and fusion beta.

**Experimental interpretation:** no improvement in final CUB accuracy
does not mean the visual attention fails to optimize: check episodic
CE and class-wise errors. If `fusion_ce` improves Base but not novel,
the multimodal training objective may not generalize. The experiment
is a loss ablation, not new evidence that text/vision fusion inherently
improves fine-grained recognition.

**Note:** earlier *text* `--objective fusion_ce` still cannot be
combined with the visual Adapter: text-side leave-one-out correction
assumes visual mean prototypes, not attention aggregation.
The new flag `--visual-objective fusion_ce` is separate, and supports
both `--ablation visual` and `--ablation both`.
