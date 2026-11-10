"""CPU-only unit tests for the revised Linear Adapter training objective."""
import unittest

import torch
from torch.nn import functional as F

from models.vdt_objective import linear_adapter_ce, support_feature_moments


class TestLinearAdapterObjective(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(29)
        self.support = F.normalize(torch.randn(12, 8), dim=-1)
        self.labels = torch.arange(3).repeat_interleave(4)
        self.text = F.normalize(torch.randn(3, 8), dim=-1).requires_grad_(True)
        self.sums, self.counts = support_feature_moments(
            self.support, self.labels, num_classes=3
        )
        self.visual = F.normalize(self.sums, dim=-1)

    def test_text_ce_is_original_logits(self):
        loss = linear_adapter_ce(self.support[:4], self.labels[:4],
                                 self.text, mode="text_ce")
        expected = F.cross_entropy(100.0 * self.support[:4] @ self.text.t(),
                                   self.labels[:4])
        torch.testing.assert_close(loss, expected, atol=1e-5, rtol=1e-5)

    def test_fusion_ce_backpropagates_to_text_only(self):
        visual = self.visual.detach().clone().requires_grad_(False)
        loss = linear_adapter_ce(
            self.support, self.labels, self.text,
            visual_prototypes=visual, beta=0.3, mode="fusion_ce",
            support_sums=self.sums, support_counts=self.counts,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(self.text.grad)
        self.assertGreater(self.text.grad.abs().sum().item(), 0)
        self.assertIsNone(visual.grad)

    def test_loo_excludes_query_from_own_visual_prototype(self):
        # Query [1,0] has a misleading support-class self-match.
        # With one other support [0,1] LOO correct-class visual is [0,1].
        supports = F.normalize(torch.tensor([[1., 0.], [0., 1.],
                                              [-1., 0.], [-1., 0.]]), dim=-1)
        labels = torch.tensor([0, 0, 1, 1])
        sums, counts = support_feature_moments(supports, labels, num_classes=2)
        visual = F.normalize(sums, dim=-1)
        text = F.normalize(torch.tensor([[1., 0.], [-1., 0.]]), dim=-1)
        beta = 0.2
        actual = linear_adapter_ce(supports[:1], labels[:1], text,
                                   visual_prototypes=visual, beta=beta,
                                   support_sums=sums, support_counts=counts,
                                   mode="fusion_ce")
        logits = 100 * (supports[:1] @
                        F.normalize(beta * text + (1-beta) * visual, dim=-1).t())
        loo_class_0 = F.normalize(beta * text[0] +
                                  (1-beta) * supports[1], dim=-1)
        logits[0, 0] = 100 * (supports[0] * loo_class_0).sum()
        expected = F.cross_entropy(logits, labels[:1])
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_rejects_single_support_and_bad_beta(self):
        with self.assertRaises(ValueError):
            linear_adapter_ce(
                self.support[:1], self.labels[:1], self.text,
                visual_prototypes=self.visual, beta=0.3,
                mode="fusion_ce",
                support_sums=self.sums, support_counts=torch.ones_like(self.counts)
            )
        with self.assertRaises(ValueError):
            linear_adapter_ce(
                self.support, self.labels, self.text,
                visual_prototypes=self.visual, beta=0.0,
                mode="fusion_ce", support_sums=self.sums,
                support_counts=self.counts
            )

    def test_nonquery_support_class_moments_are_ok(self):
        # Real incremental mode only has support images for current classes.
        new_classes = self.labels >= 2
        sums, counts = support_feature_moments(
            self.support[new_classes], self.labels[new_classes], 3
        )
        self.assertEqual(counts.tolist(), [0, 0, 4])
        loss = linear_adapter_ce(
            self.support[new_classes], self.labels[new_classes], self.text,
            visual_prototypes=self.visual, beta=.3,
            mode="fusion_ce", support_sums=sums, support_counts=counts
        )
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
