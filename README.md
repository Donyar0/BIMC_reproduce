# BiMC

This is the official implementation of paper **Enhancing Few-Shot Class-Incremental Learning via Training-Free Bi-Level Modality Calibration (CVPR 2025)**.

## Abstract

Few-shot Class-Incremental Learning (FSCIL) challenges models to adapt to new classes with limited samples, presenting greater difficulties than traditional class-incremental learning. While existing approaches rely heavily on visual models and require additional training during base or incremental phases, we propose a training-free framework that leverages pre-trained visual-language models like CLIP. At the core of our approach is a novel Bi-level Modality Calibration (BiMC) strategy. Our framework initially performs intra-modal calibration, combining LLM-generated fine-grained category descriptions with visual prototypes from the base session to achieve precise classifier estimation. This is further complemented by inter-modal calibration that fuses pre-trained linguistic knowledge with task-specific visual priors to mitigate modality-specific biases. To enhance prediction robustness, we introduce additional metrics and strategies that maximize the utilization of limited data. Extensive experimental results demonstrate that our approach significantly outperforms existing methods.

## Installation

### Dataset

Please follow [CEC](https://github.com/icoz69/CEC-CVPR2021) to download *mini*-ImageNet, CUB-200 and CIFAR-100.

### Requirement

- `torch==1.13.1`
- `torchvision==0.14.1`
- `yacs==0.1.8` 
- `tqdm==4.66.1`
- `ftfy==6.1.1`
- `regex==2023.10.3`
- `scikit-learn==1.3.2`

## Experiments

First, remember to modify the data path `ROOT` in the `dataset` configuration file.

~~~BASH
# CIFAR BIMC
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc.yaml

# CIFAR BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/cifar100.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml

# MiniImagenet BIMC
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc.yaml

# MiniImagenet BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/miniimagenet.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml

# CUB200 BIMC
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc.yaml

# CUB200 BIMC_Ensemble
python main.py --data_cfg ./configs/datasets/cub200.yaml --train_cfg ./configs/trainers/bimc_ensemble.yaml
~~~

## Acknowledgment

In this repository, we build our code based on the following excellent open-source projects. We sincerely thank all the authors for sharing their great work:

- [LP-DiF](https://github.com/1170300714/LP-DiF)
- [TEEN](https://github.com/wangkiw/TEEN)
- [FeCAM](https://github.com/dipamgoswami/FeCAM)
- [CuPL](https://github.com/sarahpratt/CuPL)
- [AdaptCLIPZS](https://github.com/cvl-umass/AdaptCLIPZS)
- [LibContinual](https://github.com/RL-VIG/LibContinual)
- [LibFewShot](https://github.com/RL-VIG/LibFewShot)




## Training-image caption text calibration (experiment)

Default `DATASET.DESCRIPTION_SOURCE: "label"` keeps the original label-generated
description prototype. Set `"image"` to replace **only the description prototype
used in BIMC text calibration**. Label text templates, vision prototypes,
covariance, PGA, and the label-description matching ensemble branch remain the
same. `TRAINER.BiMC.TEXT_CALIBRATION: True` is required for this replacement
to affect the fused classifier.

The implementation uses captions **only for the support images actually sampled
from the current training session**, to prevent using non-support or test images
in 5-shot incremental sessions. It raises an error if those captions are missing.
The default caption budget is five support images per class (also for base
classes), configurable using `IMAGE_DESCRIPTION_MAX_PER_CLASS` (0 = all).
Captions are truncated to CLIP's 77-token context size if necessary.

**CUB200**: This repository already includes AREA-style captions at
`description/CUB/001.Black_footed_Albatross_descriptions.txt`, etc.,
formatted as `image_filename.jpg<TAB>caption`. Use
`configs/datasets/cub200_image_descriptions.yaml`.

**CIFAR100**: This repository ALREADY includes image-derived descriptions
in `description/cifar100/`, e.g. `apple_descriptions.txt` with
`train_00002.png<TAB>caption` lines. Each `train_NNNNN.png` is matched
against the original torchvision CIFAR100 training index. Use
`configs/datasets/cifar100_image_descriptions.yaml`; no JSON conversion or
additional caption generation is required **for the caption-covered samples**.

**Important FSCIL sampling difference:** Not every CIFAR100 class has captions
for all 500 training images (some files contain only 20 entries). A randomly
sampled novel 5-shot set may therefore contain no corresponding captions.
For immediate execution, the CIFAR100 image-caption YAML explicitly enables
`IMAGE_DESCRIPTION_CAPTION_AWARE_SAMPLING: True`: when sampling each *novel*
class's 5 support images, the data manager selects randomly among TRAIN
images with captions. Base training image selection remains unchanged.
For each class the text prototype searches ALL current-session support
images for up to five available matching captions (rather than checking only
the first five support images). No unused or test image captions contribute.
If fewer than five matches exist, the code raises a clear error.

**Fair comparison:** Caption-aware selection changes the novel support split.
To compare with label descriptions, use the same CIFAR100 image-caption YAML
but set `DESCRIPTION_SOURCE: "label"` while keeping
`IMAGE_DESCRIPTION_CAPTION_AWARE_SAMPLING: True` and the same random seed.
Do NOT compare results with a prior label-description experiment using the
original unrestricted 5-shot split as if the support sets were identical.

Alternatively, to preserve your original 5-shot split set
`IMAGE_DESCRIPTION_CAPTION_AWARE_SAMPLING: False` and generate missing
captions for those exact sampled training images. The code will refuse
to substitute descriptions of non-support images.

A JSON mapping of original training indices to captions is also supported.

For an isolated ablation, use `USE_PGA: False`, keep the same seed,
`LAMBDA_T`, `USING_ENSEMBLE`, backbone and dataset split, and compare
`DESCRIPTION_SOURCE: "label"` against `"image"`.
SwanLab reports caption counts as `text_calibration/image_caption_count_*`.
