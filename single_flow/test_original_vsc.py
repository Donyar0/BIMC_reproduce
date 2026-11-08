"""CPU-only checks for original-direction VSC + Unified Flow."""
import unittest
import torch
import torch.nn.functional as F

from single_flow.unified_flow import (
    vsc_calibrate_visual_prototypes, classify_bidirectional, prepare_ensemble)


class ZeroFlow(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class OriginalVSCIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.eye = torch.eye(3)
        self.bank = {
            'class_ids': [0, 1, 2],
            'real': self.eye,
            'generated': F.normalize(self.eye + 0.2, dim=-1),
            'prototypes': self.eye.clone(),
            'text': self.eye.clone(),
        }

    def test_zero_strength_returns_exact_baseline_prototypes(self):
        original = self.bank['prototypes'].clone()
        result = vsc_calibrate_visual_prototypes(
            self.bank, strength=0., temperature=16., topk=2)
        self.assertTrue(torch.equal(result, original))
        self.assertTrue(torch.equal(self.bank['prototypes'], original))
        model = ZeroFlow()
        _, old_scores = classify_bidirectional(
            model, self.eye, self.bank, alpha=0.15, gamma=0.5)
        bank_copy = dict(self.bank, prototypes=result)
        _, new_scores = classify_bidirectional(
            model, self.eye, bank_copy, alpha=0.15, gamma=0.5)
        self.assertTrue(torch.equal(old_scores, new_scores))

    def test_nonzero_strength_uses_visual_guided_semantic_context(self):
        calibrated, stats = vsc_calibrate_visual_prototypes(
            self.bank, strength=0.2, temperature=8., topk=3,
            return_stats=True)
        self.assertEqual(calibrated.shape, self.eye.shape)
        self.assertTrue(torch.isfinite(calibrated).all())
        self.assertTrue(torch.allclose(
            calibrated.norm(dim=-1), torch.ones(3), atol=1e-6))
        self.assertFalse(torch.allclose(calibrated, self.eye))
        self.assertIn('vsc/self_attention_mean', stats)
        self.assertIn('vsc/mean_cosine_shift', stats)
        self.assertTrue(torch.equal(self.bank['prototypes'], self.eye))
        # More semantic context is present when self is not the only key.
        identity = vsc_calibrate_visual_prototypes(
            self.bank, strength=0.2, temperature=8., topk=1)
        self.assertTrue(torch.allclose(identity, self.eye, atol=1e-6))

    def test_topk_and_invalid_arguments(self):
        a = vsc_calibrate_visual_prototypes(
            self.bank, strength=.1, temperature=16, topk=3)
        b = vsc_calibrate_visual_prototypes(
            self.bank, strength=.1, temperature=16, topk=8)
        self.assertTrue(torch.allclose(a, b, atol=1e-6))
        for bad in (-0.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                vsc_calibrate_visual_prototypes(
                    self.bank, strength=bad, topk=2)
        with self.assertRaises(ValueError):
            vsc_calibrate_visual_prototypes(
                self.bank, strength=.1, temperature=0, topk=2)
        with self.assertRaises(ValueError):
            vsc_calibrate_visual_prototypes(
                self.bank, strength=.1, topk=0)

    def test_ensemble_receives_calibrated_visual_prototypes(self):
        calibrated = vsc_calibrate_visual_prototypes(
            self.bank, strength=0.2, temperature=8, topk=3)
        prepped = prepare_ensemble(
            dict(self.bank, prototypes=calibrated),
            torch.eye(3), self.eye, torch.arange(3), 'cpu')
        expected = (calibrated @ torch.linalg.pinv(torch.eye(3))).T
        self.assertTrue(torch.allclose(
            prepped['weighted_prototypes'], expected))
        self.assertTrue(torch.equal(self.bank['prototypes'], self.eye))


if __name__ == '__main__':
    unittest.main()
