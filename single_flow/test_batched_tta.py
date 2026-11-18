"""Check that packing TTA views preserves the ten-view feature average."""
import unittest

import torch
from torch import nn

from models.bimc import BiMC


class _ToyClip(nn.Module):
    def encode_image(self, images):
        middle = images.shape[-1] // 2
        whole = images.mean(dim=(2, 3))
        left = images[..., :middle].mean(dim=(2, 3))
        right = images[..., middle:].mean(dim=(2, 3))
        return torch.cat((whole, left - right, images.square().mean(dim=(2, 3))), dim=-1)


class BatchedTTATest(unittest.TestCase):
    def test_chunk_sizes_match_serial_ten_view_features(self):
        model = BiMC.__new__(BiMC)
        nn.Module.__init__(model)
        model.device = 'cpu'
        model.clip_model = _ToyClip()
        torch.manual_seed(7)
        images = torch.randn(3, 3, 32, 32)
        serial = model.extract_img_feature(images, tta=True, tta_view_chunk=1)
        for chunk_size in (2, 3, 5, 10):
            batched = model.extract_img_feature(images, tta=True,
                                                tta_view_chunk=chunk_size)
            self.assertTrue(torch.allclose(serial, batched, atol=1e-6, rtol=1e-6))


if __name__ == '__main__':
    unittest.main()
