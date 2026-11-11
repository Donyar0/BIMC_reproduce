"""Pixel-space augmentation, used exclusively by base-session Flow training."""
import torch
from torchvision import transforms


class GaussianNoise:
    def __init__(self, std=0.03, probability=0.5):
        self.std = float(std)
        self.probability = float(probability)

    def __call__(self, image):
        if torch.rand(()) < self.probability:
            image = image + torch.randn_like(image) * self.std
        return image.clamp(0, 1)


def build_flow_transform(cfg):
    if not 0 <= cfg.ROTATION <= 180 or cfg.NOISE_STD < 0:
        raise ValueError('Invalid Flow rotation/noise strength')
    if not 0 <= cfg.ERASE_P <= 1 or not 0 <= cfg.NOISE_P <= 1:
        raise ValueError('Augmentation probabilities must be in [0,1]')
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.RandomRotation(cfg.ROTATION, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.RandomErasing(p=cfg.ERASE_P, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0),
        GaussianNoise(cfg.NOISE_STD, cfg.NOISE_P),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073],
                             [0.26862954, 0.26130258, 0.27577711]),
    ])
