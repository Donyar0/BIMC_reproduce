"""Run with: python -m unittest discover -s tests -v"""
import random
import unittest

import torch
import torch.nn.functional as F

from models.neighbor_semantics import calibrate_neighbor_features, sample_description_index


class TestNeighborSemantics(unittest.TestCase):
    def test_per_class_sampling_is_reproducible(self):
        expected = sample_description_index(40, seed=1, class_id=7)
        random.seed(4321)
        _ = [random.random() for _ in range(20)]
        self.assertEqual(sample_description_index(40, 1, 7), expected)
        self.assertGreaterEqual(expected, 0)
        self.assertLess(expected, 40)
        with self.assertRaises(ValueError):
            sample_description_index(0, 1, 7)

    def test_single_seen_class_and_clipped_k(self):
        joint = torch.tensor([[1., 0., 0., 1.]])
        expected = F.normalize(torch.tensor([[1., 1.]]), dim=-1)
        torch.testing.assert_close(calibrate_neighbor_features(joint, 2, 1), expected)
        self.assertEqual(calibrate_neighbor_features(torch.cat([joint, joint]), 200, 0.1).shape, (2, 2))

    def test_topk_excludes_self(self):
        joint = torch.tensor([[1., 0., 1., 0.], [0.9, 0.1, 1., 0.], [0., 1., 0., 1.]])
        result = calibrate_neighbor_features(joint, topk=1, tau=0.8)
        similarities = F.normalize(joint, dim=-1) @ F.normalize(joint, dim=-1).T
        similarities.fill_diagonal_(float('-inf'))
        nearest = similarities.argmax(dim=1)
        self.assertEqual(nearest[0].item(), 1)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.allclose(result.norm(dim=-1), torch.ones(3), atol=1e-6))

    def test_no_neighbors_reduces_to_prompt_description_fusion(self):
        joint = torch.randn(5, 8, dtype=torch.float32)
        out = calibrate_neighbor_features(joint, 0, 1.0)
        left, right = joint.chunk(2, dim=-1)
        expected = F.normalize(F.normalize(left, dim=-1) + F.normalize(right, dim=-1), dim=-1)
        torch.testing.assert_close(out, expected)

    def test_reject_invalid_shape(self):
        with self.assertRaises(ValueError):
            calibrate_neighbor_features(torch.zeros(3, 5), 2, 0.1)
        with self.assertRaises(ValueError):
            calibrate_neighbor_features(torch.zeros(0, 4), 2, 0.1)
        with self.assertRaises(ValueError):
            calibrate_neighbor_features(torch.zeros(3, 4), -1, 0.1)


if __name__ == '__main__':
    unittest.main()
