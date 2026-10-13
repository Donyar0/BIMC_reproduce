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

**CIFAR100**: Provide a JSON file at
`description/cifar100_image_descriptions.json`, mapping the **original
torchvision training dataset indices** to captions. Example *format only*:

```json
{
  "0": "Description of training image 0",
  "1": "Description of training image 1"
}
```

The example captions are placeholders, not valid annotations. No CIFAR100
image-generated captions are bundled or automatically generated. Use
`configs/datasets/cifar100_image_descriptions.yaml` once you have
generated captions for actual training images. Each sampled support image
must be represented in the caption JSON.

For an isolated ablation, use `USE_PGA: False`, keep the same seed,
`LAMBDA_T`, `USING_ENSEMBLE`, backbone and dataset split, and compare
`DESCRIPTION_SOURCE: "label"` against `"image"`.
SwanLab reports caption counts as `text_calibration/image_caption_count_*`.
