"""CPU-only sanity checks for the deterministic semantic-residual geometry."""
import unittest
import torch

from models.semantic_residual import paired_orthogonal_residual


class TestOrthogonalResidual(unittest.TestCase):
    def test_perfect_agreement(self):
        a = torch.tensor([1., 0., 0.])
        e = torch.tensor([[0., 1., 0.]])
        v = torch.tensor([[0., 1., 0.]])
        delta, w, cos, tnorm, vnorm = paired_orthogonal_residual(a, e, v)
        torch.testing.assert_close(delta, torch.tensor([0., 1., 0.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(w, torch.tensor([1.]), atol=1e-6, rtol=0)
        self.assertAlmostEqual(cos.item(), 1., places=6)
        self.assertGreater(tnorm.item(), 0)
        self.assertGreater(vnorm.item(), 0)
        self.assertAlmostEqual(torch.dot(a, delta).item(), 0., places=6)

    def test_opposed_residual_has_zero_weight(self):
        a = torch.tensor([1., 0., 0.])
        e = torch.tensor([[0., 1., 0.]])
        v = torch.tensor([[0., -1., 0.]])
        delta, w, cos, _, _ = paired_orthogonal_residual(a, e, v)
        self.assertLess(cos.item(), 0)
        torch.testing.assert_close(delta, torch.zeros(3), atol=1e-6, rtol=0)
        self.assertEqual(w.item(), 0)

    def test_near_zero_residual_suppressed(self):
        a = torch.tensor([1., 0., 0.])
        e = a[None].clone()
        v = torch.tensor([[0., 1., 0.]])
        delta, w, cos, _, _ = paired_orthogonal_residual(a, e, v)
        self.assertEqual(w.item(), 0)
        self.assertEqual(cos.item(), 0)
        self.assertEqual(delta.abs().sum().item(), 0)

    def test_mean_over_matched_captions_not_positive_weights(self):
        a = torch.tensor([1., 0., 0.])
        e = torch.tensor([[0., 1., 0.], [0., 0., 1.]])
        v = torch.tensor([[0., 1., 0.], [0., 0., -1.]])
        delta, weights, _, _, _ = paired_orthogonal_residual(a, e, v)
        torch.testing.assert_close(weights, torch.tensor([1., 0.]), atol=1e-6, rtol=0)
        torch.testing.assert_close(delta, torch.tensor([0., 0.5, 0.]), atol=1e-6, rtol=0)


if __name__ == "__main__":
    unittest.main()
