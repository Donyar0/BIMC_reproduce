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




## Label + training-image description fusion

Set `DATASET.DESCRIPTION_SOURCE: "fusion"` and
`DATASET.DESCRIPTION_FUSION_ALPHA: 0.5` to combine the original BIMC
description prototype and the image-caption prototype:

```
d_fused = normalize((1 - alpha) * d_label + alpha * d_image)
t_calibrated = (1 - lambda_t) * t_template + lambda_t * d_fused
```

The original `label` mode is preserved as the baseline. The old
`image`-only mode has been removed; `alpha=1` provides the image-only
endpoint for controlled ablations within the fusion formulation.
`TEXT_CALIBRATION=True` is required for the fused prototype to affect
the final class scores. The original label descriptions still support
the separate description-matching ensemble branch.

For CIFAR100, use
`configs/datasets/cifar100_description_fusion.yaml` with existing
`description/cifar100/*_descriptions.txt`. For CUB200, use
`configs/datasets/cub200_description_fusion.yaml` with existing
`description/CUB/*_descriptions.txt`. The text encoder remains frozen.

Only captions of current-session training support images are used.
CIFAR100 contains some classes with partial training-caption coverage.
The provided fusion YAML selects novel 5-shot examples from
caption-covered training images. This changes the sampling population:
for a fair comparison, keep this option and the seed fixed in both
`DESCRIPTION_SOURCE: label` and `fusion` runs.

SwanLab records caption counts, fusion weight and label/image
description-prototype cosine similarity.
