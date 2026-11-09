"""Run with: python -m unittest discover -s tests -v"""
import unittest

import torch
import torch.nn.functional as F

from models.neighbor_semantics import calibrate_neighbor_features


class TestNeighborSemantics(unittest.TestCase):
    def test_multiple_pairs_per_class_aggregate(self):
        pairs = torch.tensor([
            [1., 0., 1., 0.], [1., 0., 0., 1.],
            [0., 1., 0., 1.],
            [0., 1., 1., 0.], [0., 1., 0.6, 0.8], [0., 1., 0.8, 0.6],
        ])
        labels = torch.tensor([0, 0, 1, 2, 2, 2])
        got = calibrate_neighbor_features(pairs, labels, 0, 0.1)
        expected = []
        for cid in (0, 1, 2):
            a, b = pairs[labels == cid].chunk(2, -1)
            per_pair = F.normalize(F.normalize(a, dim=-1) + F.normalize(b, dim=-1), dim=-1)
            expected.append(F.normalize(per_pair.mean(0), dim=-1))
        torch.testing.assert_close(got, torch.stack(expected))

    def test_all_descriptions_affect_output(self):
        pairs = torch.tensor([[1., 0., 1., 0.], [1., 0., 0., 1.], [0., 1., 0., 1.]])
        labels = torch.tensor([0, 0, 1])
        baseline = calibrate_neighbor_features(pairs, labels, 1, 0.3)
        modified = pairs.clone()
        modified[1] = torch.tensor([1., 0., -1., 0.])
        changed = calibrate_neighbor_features(modified, labels, 1, 0.3)
        self.assertFalse(torch.allclose(baseline[0], changed[0]))

    def test_cross_class_neighbor_selection(self):
        pairs = torch.tensor([[1., 0., 1., 0.], [1., 0., 1., 0.], [1., 0., 0., 1.]])
        labels = torch.tensor([0, 0, 1])
        no_neighbor = calibrate_neighbor_features(pairs, labels, 0, 1.)
        with_neighbor = calibrate_neighbor_features(pairs, labels, 1, 1.)
        self.assertFalse(torch.allclose(no_neighbor[0], with_neighbor[0]))
        self.assertEqual(with_neighbor.shape, (2, 2))

    def test_chunking_does_not_change_result(self):
        generator = torch.Generator().manual_seed(123)
        pairs = torch.randn(19, 8, generator=generator)
        labels = torch.tensor([0]*2 + [1]*5 + [2]*3 + [3]*9)
        one = calibrate_neighbor_features(pairs, labels, 2, 0.1, chunk_size=1)
        many = calibrate_neighbor_features(pairs, labels, 2, 0.1, chunk_size=32)
        torch.testing.assert_close(one, many)

    def test_single_class_and_large_k(self):
        pairs = torch.randn(3, 6)
        labels = torch.zeros(3, dtype=torch.long)
        out = calibrate_neighbor_features(pairs, labels, topk=100, tau=0.1)
        self.assertEqual(out.shape, (1, 3))
        self.assertTrue(torch.isfinite(out).all())

    def test_many_neighbors_and_float16(self):
        pairs = torch.randn(7, 8, dtype=torch.float16)
        labels = torch.tensor([0, 0, 1, 1, 1, 2, 2])
        out = calibrate_neighbor_features(pairs, labels, topk=500, tau=0.1)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(out.shape, (3, 4))
        self.assertTrue(torch.isfinite(out).all())

    def test_invalid_inputs(self):
        for p, l in [
            (torch.empty(0, 4), torch.tensor([], dtype=torch.long)),
            (torch.randn(2, 5), torch.tensor([0, 1])),
            (torch.randn(3, 4), torch.tensor([0, 1])),
            (torch.randn(3, 4), torch.tensor([0, 0, 2])),
        ]:
            with self.assertRaises(ValueError):
                calibrate_neighbor_features(p, l, 1, 0.1)
        with self.assertRaises(ValueError):
            calibrate_neighbor_features(torch.randn(2, 4), torch.tensor([0, 1]), -1, 0.1)


if __name__ == '__main__':
    unittest.main()
