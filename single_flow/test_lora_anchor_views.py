"""Check that the cached teacher view matches the LoRA training view."""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from engine.lora_base import AlignedAnchorDataset, cache_base_anchor_features


class _Images(Dataset):
    def __init__(self):
        self.images = torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 4) + 1
        self.transform = lambda image: image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {'idx': index, 'image': self.transform(self.images[index].clone()), 'label': 0}


class _Teacher(nn.Module):
    def forward(self, images):
        return torch.stack((images.mean(dim=(1, 2, 3)),
                            images[..., 0].mean(dim=(1, 2)),
                            images[..., -1].mean(dim=(1, 2))), dim=1)


class AlignedAnchorTest(unittest.TestCase):
    def test_cached_view_matches_centered_and_flipped_training_image(self):
        source = _Images()
        center = lambda image: image + 1
        manager = SimpleNamespace(test_transform=center,
                                  loader_kwargs=lambda: {'num_workers': 0})
        cfg = SimpleNamespace(LORA=SimpleNamespace(ALIGN_ANCHOR_VIEW=True),
                              DATALOADER=SimpleNamespace(
                                  TRAIN=SimpleNamespace(BATCH_SIZE_BASE=2)))
        teacher = _Teacher()
        cache = cache_base_anchor_features(teacher, source, manager, cfg, 'cpu', 3)
        self.assertEqual(cache.shape, (2, 2, 3))
        train_dataset = AlignedAnchorDataset(source, center)
        torch.manual_seed(1)
        for index in range(len(source)):
            for _ in range(12):
                sample = train_dataset[index]
                expected_image = center(source.images[index])
                if sample['anchor_view']:
                    expected_image = expected_image.flip(-1)
                self.assertTrue(torch.equal(sample['image'], expected_image))
                expected_feature = F.normalize(
                    teacher(expected_image.unsqueeze(0)).float(), dim=-1)[0]
                actual_feature = cache[index, sample['anchor_view']].float()
                self.assertTrue(torch.allclose(actual_feature, expected_feature,
                                               atol=3e-3, rtol=1e-3))


if __name__ == '__main__':
    unittest.main()
