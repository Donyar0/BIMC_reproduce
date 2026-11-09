"""CPU checks: python -m unittest discover -s tests -v"""
import unittest

import torch
import torch.nn.functional as F

from models.vdt_self_attention import SelfAttentionTextAdapter, pack_description_bank


class TestVDTSelfAttention(unittest.TestCase):
    def test_variable_description_counts(self):
        features = torch.randn(7, 8)
        labels = torch.tensor([0, 0, 0, 1, 2, 2, 2])
        packed, mask, classes = pack_description_bank(features, labels)
        self.assertEqual(tuple(packed.shape), (3, 3, 8))
        self.assertEqual(mask.sum(dim=1).tolist(), [3, 1, 3])
        self.assertEqual(classes.tolist(), [0, 1, 2])
        model = SelfAttentionTextAdapter(8, reduction=2, ratio=0.2, dropout=0.)
        with torch.no_grad():
            prototypes = model(packed, mask)
        self.assertEqual(tuple(prototypes.shape), (3, 8))
        torch.testing.assert_close(prototypes.norm(dim=-1), torch.ones(3), atol=1e-5, rtol=0)

    def test_padding_does_not_change_result(self):
        torch.manual_seed(0)
        model = SelfAttentionTextAdapter(8, reduction=2, ratio=0.2, dropout=0.).eval()
        x = torch.randn(1, 2, 8)
        short_mask = torch.ones(1, 2, dtype=torch.bool)
        padded = torch.cat([x, torch.randn(1, 5, 8) * 100], dim=1)
        long_mask = torch.tensor([[True, True, False, False, False, False, False]])
        with torch.no_grad():
            a = model(x, short_mask)
            b = model(padded, long_mask)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)

    def test_training_reaches_parameters(self):
        torch.manual_seed(1)
        model = SelfAttentionTextAdapter(8, reduction=2, ratio=0.3, dropout=0.)
        packed = torch.randn(3, 4, 8)
        mask = torch.ones(3, 4, dtype=torch.bool)
        images = F.normalize(torch.randn(10, 8), dim=-1)
        labels = torch.randint(0, 3, (10,))
        logits = images @ model(packed, mask).T
        F.cross_entropy(logits * 10, labels).backward()
        self.assertIsNotNone(model.q_proj.weight.grad)
        self.assertTrue(torch.isfinite(model.q_proj.weight.grad).all())
        self.assertGreater(float(model.q_proj.weight.grad.abs().sum()), 0)

    def test_half_precision_inputs(self):
        model = SelfAttentionTextAdapter(8, reduction=2)
        packed = torch.randn(2, 3, 8, dtype=torch.float16)
        mask = torch.ones(2, 3, dtype=torch.bool)
        got = model(packed, mask)
        self.assertEqual(got.dtype, torch.float32)
        self.assertEqual(tuple(got.shape), (2, 8))

    def test_classwise_attention_independent(self):
        model = SelfAttentionTextAdapter(8, reduction=2, dropout=0.).eval()
        packed = torch.randn(2, 3, 8)
        mask = torch.ones(2, 3, dtype=torch.bool)
        with torch.no_grad():
            original = model(packed, mask)[0]
            packed[1] *= 100
            changed = model(packed, mask)[0]
        torch.testing.assert_close(original, changed)

    def test_invalid_mask(self):
        model = SelfAttentionTextAdapter(8, reduction=2)
        with self.assertRaises(ValueError):
            model(torch.randn(2, 3, 8), torch.zeros(2, 3, dtype=torch.bool))


if __name__ == '__main__':
    unittest.main()
