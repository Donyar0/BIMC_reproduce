# Frozen CLIP: balanced ridge, kernel transport, and local features

This branch adds three **separate experiments** on top of the original
`frozen_random_ridge_flow.yaml` baseline (`424550d`). Image and text CLIP
encoders are frozen. The new experiments use one image view and no ZCA,
incremental Flow training, old-image replay, or velocity distillation.

## Run the three stages

Set the dataset root in `configs/datasets/*.yaml`, install `requirements.txt`,
then run from the repository root. Replace `cub200.yaml` with
`miniimagenet.yaml` or `cifar100.yaml` as needed.

```bash
# 1. Class-balanced ridge with the Base-trained, frozen MLP Flow.
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/balanced_ridge_mlp.yaml

# 2. Replace the MLP Flow with the class-anchor kernel velocity field.
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/balanced_ridge_kernel.yaml

# 3. Add four-layer CLS aggregation and local attribute patch pooling.
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/balanced_ridge_kernel_local.yaml
```

To run the suggested CUB200 ridge regularization values `10, 50, 100, 500`
for stage 1, use `bash scripts/sweep_balanced_ridge_cub.sh`. Each value is a
separate full experiment. A single override is also available:

```bash
python -u main.py --data_cfg configs/datasets/cub200.yaml \
  --train_cfg configs/trainers/balanced_ridge_mlp.yaml \
  --opts RIDGE.LAMBDA 50.0 FLOW.OUTPUT_DIR output/balanced_ridge_mlp_lambda50
```

## What changes between stages

- **Stage 1:** A fixed, seeded random ReLU map projects normalized CLIP
  features into 8192 dimensions. Each class contributes
  `mean_i(phi(z_i) phi(z_i)^T)` to cumulative `G`; no translated LLM
  descriptions enter `G`. Its sole target column is
  `M_c = phi(unit(n_c/(n_c+kappa) * V_c + kappa/(n_c+kappa) * Flow(T_c)))`.
  The head solves `(G + lambda I) W = M` after each session. Base Flow is
  trained once and frozen. `kappa=5` and `lambda=10` are starting values.
- **Stage 2:** The trainable Flow is replaced by a Gaussian kernel average of
  class displacement vectors `V_c - T_c` along their linear text-to-visual
  paths. Current-session anchors are appended; old anchors stay fixed. A
  class's own anchor is excluded when generating its visual prior, so the
  prior can borrow offsets from other classes. Test images use the full set
  of seen anchors for reverse transport. No SGD occurs in this stage.
- **Stage 3:** A single ViT forward returns the last four projected CLS
  features and last-layer projected Patch tokens. Normalized CLS features are
  averaged. Each Patch takes its best cosine match against **Base-class LLM
  descriptions**, then a softmax over Patch scores pools a local feature.
  The local feature is mixed into the global feature with weight `0.25`.
  Keeping the Base description dictionary fixed preserves one feature space
  across sessions and avoids using unseen-class text at Base time.

At inference, the ridge visual score and reverse-transport text cosine score
are each standardized across seen classes **for that query** and fused with
equal weight. Outputs include per-session fused, ridge-only, and text-only
accuracies under each configuration's `FLOW.OUTPUT_DIR`.

## Mathematical and experimental limits

The exact cross term for class-balanced one-hot least squares is the
**per-class mean** `mean_i(phi(z_i))`, not `phi(tilde V_c)`. The default
`RIDGE.TARGET_MODE: prototype` follows the proposed prototype target, but it
is a distinct quadratic objective. Set `RIDGE.TARGET_MODE: class_mean` to
test the exact weighted least-squares objective.

The kernel method freezes old anchors, but adding new anchors changes the
velocity field and solving a wider ridge head changes old predictions; zero
forgetting is **not guaranteed**. Stage 3 saves encoder passes but Patch
matching adds matrix multiplication work. Accuracy gains stated in a proposal
must be measured; the repository does not assume any threshold is reached.
The 8192-by-8192 float32 Gram matrix alone uses 256 MiB, and its Cholesky
solve is repeated at every session.

Run small CPU checks with:

```bash
python -m unittest discover -s single_flow -p 'test_*.py' -v
```
