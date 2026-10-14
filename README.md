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




## Training-free dual consistency calibration (original BiMC 5-shot protocol)

Run CIFAR100 with the original **random** 5-shot selection:

```bash
python main.py --data_cfg configs/datasets/cifar100_dual_consistency.yaml --train_cfg configs/trainers/bimc.yaml --use_swanlab
```

The new `DESCRIPTION_SOURCE: "dual"` implements two CLIP consistency gates,
without training weights or changing BiMC's `np.random.choice` sampling:

- **TT**: cosine between the frozen CLIP representation of original
  class-level fine-grained BIMC descriptions and the mean frozen-CLIP
  representation of image captions belonging **only to sampled training
  support images**. `alpha_c = DUAL_ALPHA_MAX * clamp((TT_c -
  DUAL_TT_LOW)/(DUAL_TT_HIGH-DUAL_TT_LOW), 0, 1)`.
- **TV**: cosine between original fine-grained BIMC description prototype
  and the **uncalibrated** mean CLIP visual prototype built from the same
  sampled support images. `beta_c = clamp(BETA + DUAL_BETA_SHIFT *
  (TV_c - DUAL_TV_CENTER), 0, 1)`.

The final prototype per class is:

```
text_bimc = (1 - lambda_t)*text_template + lambda_t*label_description
text_dual = (1 - alpha_c)*text_bimc + alpha_c*image_caption_prototype
proto = normalize(beta_c*text_dual + (1 - beta_c)*calibrated_visual)
```

Only the **prototype classifier** is changed. The original label-description
KNN ensemble, covariance branch, original BiMC visual calibration and optional
PGA remain unchanged.

### Missing annotations with the original shot split

The CIFAR100 image-caption directory contains only 20 annotated TRAIN
images for most classes, versus 500 possible TRAIN images per class.
**The new branch never resamples training supports to match annotations**.
If none of a class's 5 original random novel support images has a caption,
`alpha_c=0` and the text component exactly falls back to original BiMC.
TV is still computable because all five sampled images are available. If
only some of the five supports have captions, TT uses only those matches;
coverage is logged and should be reported. This ensures no unlabeled or
unselected extra training images and no test image descriptions leak into
the novel 5-shot classifier, but it also means TT often has no influence
until captions for the **actual sampled supports** have been generated.

For high-quality strict protocol experiments, generate the missing captions
for the exact five support images chosen by the original seeded BiMC split.
Do not replace the 5-shot data and do not average the other 15 annotated
training images into the new class prototype.

The original `DESCRIPTION_SOURCE: "label"` baseline uses the same
unchanged BiMC shot sampling. Compare fixed `BETA`, TT-only
(`DUAL_BETA_SHIFT: 0`), TV-only (`DUAL_ALPHA_MAX: 0`), and full dual
consistency, with the same seed and support indices. SwanLab logs selected
caption counts, class-level coverage, TT/TV means and alpha/beta means and
ranges by session.

### Earlier fixed-alpha fusion

The older `DESCRIPTION_SOURCE: "fusion"` config remains an experimental
ablation. It requires captions for every requested sampled support image
and may fail when no annotation exists. The new dual mode is the recommended
implementation for the original random-shot protocol.


## Orthogonal semantic residual calibration (training-free)

Experimental branch: `experiment/orthogonal-semantic-residual`.

```bash
python main.py --data_cfg configs/datasets/cifar100_orthogonal_residual.yaml --train_cfg configs/trainers/bimc.yaml --use_swanlab
# Optional CUB configuration:
python main.py --data_cfg configs/datasets/cub200_orthogonal_residual.yaml --train_cfg configs/trainers/bimc.yaml --use_swanlab
# Geometry-only tests (CPU):
python -m unittest discover -s tests -p 'test_semantic_residual.py'
```

The new `DESCRIPTION_SOURCE: "residual"` keeps the **original BiMC random-shot
selection** and its original `BETA` (e.g. 0.65 for CIFAR100). It never
resamples by caption availability, reads test captions, or uses captions of
unselected training images. It does not change calibrated vision prototypes,
covariance matching or the original description-KNN ensemble.

For each class, `a` is the normalized prototype of original BiMC
LLM-generated fine-grained category descriptions. Each selected training
support image and *its own* available caption are encoded through frozen CLIP.
After normalizing each feature, compute:

```
r_text   = text_caption - dot(text_caption, a)*a
r_visual = image_feature - dot(image_feature, a)*a
w_i      = max(0, cosine(r_text, r_visual))
delta_c  = sum_i (w_i * r_text_i) / number_of_matched_support_captions
text_bimc = (1 - lambda_t)*prompt_proto + lambda_t*label_description_proto
text_new = text_bimc + RESIDUAL_ETA*delta_c
final_proto = normalize(BETA*text_new + (1-BETA)*calibrated_visual_proto)
```

Near-zero residual norms are gated to zero using `RESIDUAL_MIN_NORM`.
Negative residual alignment is also gated to zero. Corrections are **not**
normalized before adding them to BiMC text, retaining the original
text--visual fusion scale; `RESIDUAL_ETA: 0` exactly recovers the BiMC
classifier under the same shot selection and otherwise unchanged configuration.

Missing support captions simply yield a zero correction; the support images
are unchanged. Partial caption coverage uses only matching selected images.
On CIFAR100 most Novel classes currently have 20 annotated train examples
out of 500, so few random 5-shots match the available captions. This means
the residual experiment may behave nearly identically to BiMC on many
Novel classes. Generate descriptions for the **five actually sampled**
training supports to test this idea fully, without modifying the dataset split.

SwanLab records caption support coverage, matched counts, positive residual
alignment and average correction norm for each session. Use the same seed for
the BiMC baseline, an unweighted-residual ablation and the default
visual-evidence residual ablation. This is a research hypothesis, not a
validated description hallucination detector; shared image/background bias
may still yield positive residual alignment.
