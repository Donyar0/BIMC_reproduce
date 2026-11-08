"""CPU-only unit tests for the swapped VSC projection head."""
import unittest
import torch
from torch.nn import functional as F
from models.swapped_vsc import SwappedVSCHead


class SwappedVSCHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.head = SwappedVSCHead(dim=16, hidden_dim=32)
        self.text = F.normalize(torch.randn(4, 16), dim=-1)
        self.visual = F.normalize(torch.randn(4, 16), dim=-1)
        self.query = F.normalize(torch.randn(9, 16), dim=-1)
        self.labels = torch.arange(9) % 4

    def test_shapes_and_probability_normalization(self):
        result = self.head(self.query, self.text, self.visual)
        self.assertEqual(tuple(result["logits_text"].shape), (9, 4))
        self.assertEqual(tuple(result["logits_visual"].shape), (9, 4))
        self.assertEqual(tuple(result["calibrated_text"].shape), (4, 16))
        self.assertEqual(tuple(result["pseudo_visual"].shape), (4, 16))
        self.assertTrue(torch.allclose(
            result["probabilities"].sum(-1), torch.ones(9), atol=1e-6
        ))

    def test_base_training_updates_head_but_not_input(self):
        x = self.query.detach().clone().requires_grad_(False)
        loss, components = self.head.losses(
            x, self.labels, self.text, self.visual
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(len(components), 4)
        loss.backward()
        self.assertTrue(any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.head.parameters()
        ))
        self.assertIsNone(x.grad)

    def test_prototype_bank_expands_without_retraining(self):
        self.head.eval()
        with torch.no_grad():
            before = self.head(self.query, self.text, self.visual)
            added_t = F.normalize(torch.randn(2, 16), dim=-1)
            added_v = F.normalize(torch.randn(2, 16), dim=-1)
            after = self.head(
                self.query,
                torch.cat([self.text, added_t]),
                torch.cat([self.visual, added_v]),
            )
        self.assertEqual(tuple(before["probabilities"].shape), (9, 4))
        self.assertEqual(tuple(after["probabilities"].shape), (9, 6))

    def test_mismatched_prototypes_rejected(self):
        with self.assertRaises(ValueError):
            self.head.prototype_bank(self.text, self.visual[:2])


if __name__ == "__main__":
    unittest.main()
