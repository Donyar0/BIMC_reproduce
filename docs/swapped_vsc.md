# Swapped VSC: visual-guided text prototypes + text-generated visual prototypes

This experiment adapts the *direction-swapped* design discussed from:
Wang et al., **Visual-Semantic Cooperative Learning for Few-Shot SAR Target Classification**, IEEE JSTARS 2025.

It is **not** a faithful reproduction of the original SAR meta-learning training procedure.
It is implemented as a new FSCIL method on the original BiMC data splits.

## Design

- Frozen CLIP ViT-B/16 extracts normalized image and text embeddings.
- Text-class vectors optionally blend CLIP class-name prompts with existing BiMC LLM descriptions.
- The **visual-to-text** projector maps K-shot class visual means into the text embedding space.
  An attention-weighted visual context updates the text classifier with a learnable small residual.
  Queries also pass through the visual-to-text projector to produce semantic classification scores.
- The **text-to-visual** projector maps class text embeddings to pseudo-visual prototypes.
  Original query visual embeddings are classified against these pseudo-visual prototypes.
- Two branch probabilities are averaged (fusion weight configurable).
- Base-stage loss: text classification CE + visual classification CE + visual-to-text
  prototype consistency CE + text-to-visual indication CE.
  The last two terms are contrastive surrogates, NOT the paper's exact auxiliary losses.

**Incremental protocol:** Train directional heads using only base-class train features.
Freeze CLIP and the learned projection heads after base training. At each new session,
compute visual/text class prototypes only from the newly introduced class support set.
Keep previous prototypes and concatenate to classify every seen class. No future test
examples or old train images are accessed during increment evaluation.

## Run

On the same GPU environment and dataset paths used by the original BiMC:

```bash
git fetch origin
git switch feature/swapped-vsc-dual-classifier
python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/swapped_vsc.yaml
python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/swapped_vsc.yaml
python main.py --data_cfg configs/datasets/miniimagenet.yaml --train_cfg configs/trainers/swapped_vsc.yaml
```

The runner prints `Final acc:[...]` and `Task-wise acc:` for every session.
Projection checkpoint: `output/swapped_vsc/<dataset>/seed_<seed>/base_projection.pt`.

For SwanLab, append `--use_swanlab`. A CPU-only module-level smoke test:
```bash
python -m unittest discover -s tests -p 'test_swapped_vsc.py'
```

## Fair comparison and precautions

- The original BiMC (main trainer configuration) remains untouched.
- This branch uses **trainable projection heads at the base stage**, unlike training-free BiMC.
- The standard BiMC Mahalanobis / description-KNN ensemble is **excluded** in this
  independent experiment to isolate the dual-branch method.
- CLIP image and text embeddings share an embedding dimension. A mapping between them
  learns task-specific transformations, not a change to a different-dimensional space.
- The proposed two-direction framework should be treated as a testable hypothesis;
  no CIFAR100/CUB/miniImageNet accuracy results are claimed until GPU training is run.
- The implementation assumes the repository's contiguous FSCIL class ordering and a CUDA device,
  as in the original BiMC code.
