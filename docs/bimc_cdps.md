# BiMC + CDPS: training-free patch memory

This branch adapts the ECCV 2026 CDPS idea to the original BiMC FSCIL pipeline. No model weights are trained. The original BiMC global score is retained.

## Design

For each session, the already sampled support loader supplies the **current session's images only**. A hook on the final frozen CLIP ViT block captures patch tokens and recomputes CLS-to-patch attention from Q/K projections. Top-attention connected patches are scored by within-class consistency minus hard-negative inter-class similarity. Select top 20 patches per support image; store class-specific selected patches permanently.

For base classes, cap support images at MAX_SUPPORT=8 per class to bound memory. Incremental classes use up to 5 images. Later sessions never overwrite old-class patch memories. New-class patch selection may use frozen old-class CDPS features as negatives, without revisiting old-class raw images.

Each query gets a CDPS set-to-set similarity over seen classes. Local patch scores become softmax probabilities. BiMC scores are normalized to sum to one before fusion because BiMC already returns probability-like outputs.

Final score = GLOBAL_WEIGHT * BiMC_global + (1 - GLOBAL_WEIGHT) * CDPS_local.

## Run

From repository root:

~~~bash
git fetch origin
git switch feature/bimc-cdps-patch-memory

python main.py --data_cfg configs/datasets/cub200.yaml --train_cfg configs/trainers/bimc_cdps.yaml

python main.py --data_cfg configs/datasets/cifar100.yaml --train_cfg configs/trainers/bimc_cdps.yaml

python main.py --data_cfg configs/datasets/miniimagenet.yaml --train_cfg configs/trainers/bimc_cdps.yaml

python -m unittest tests.test_cdps_memory
~~~

The standard Final acc and Task-wise acc outputs remain. Each session also prints BiMC-only, Patch-only and Hybrid accuracy. Optionally pass --use_swanlab.

Memories are saved at output/cdps_patch_memory/<dataset>/seed_<seed>/session_<id>.pt. Each file contains class-specific tensors and metadata only. They are rebuilt upon every fresh run in this first implementation.

Ablation: change GLOBAL_WEIGHT in configs/trainers/bimc_cdps.yaml: 1.0 for original BiMC only, 0.0 for CDPS alone, and 0.5 for the initial hybrid. Sweep 0.3, 0.5, 0.7, 0.9 using a validation protocol.

## Caveats

- This is a new FSCIL adaptation, **not** a faithful reproduction of CDPS's original joint few-shot protocol or its reported numbers.
- The original paper uses OpenCLIP; this code reuses BiMC's OpenAI CLIP.
- Query patch count is capped at 64 for speed; top 20 of the matched query patches are averaged, unlike original untruncated set-to-set averaging.
- Frozen old memories may preserve historical information but are not proof of reduced forgetting.
- Query-to-memory scoring can be substantially slower than vanilla BiMC; no accuracy or GPU performance has been validated here.
- The paper's original patch mining uses potentially all few-shot support images. This version limits stored base support to eight images per class.
