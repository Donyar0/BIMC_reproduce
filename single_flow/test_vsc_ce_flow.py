"""Tests for two opt-in VSC-inspired extensions of unified Flow."""
import unittest

import torch
import torch.nn.functional as F

from single_flow.fm import DeepFlowMatchingNet
from single_flow.s1_flow import fit, visual_prototype_classification_ce
from single_flow.unified_flow import calibrate_text_bank, classify_bidirectional


class ZeroVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, x, t):
        return x * self.scale


class ConstantVelocity(torch.nn.Module):
    def __init__(self, delta):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(delta, dtype=torch.float32),
                                        requires_grad=False)

    def forward(self, x, t):
        return self.delta.unsqueeze(0).expand_as(x)


class VSCFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_calibration_zero_matches_original_and_nonzero_uses_real_visual(self):
        model = ZeroVelocity()
        text = F.normalize(torch.tensor([[1., 0., 0.], [0., 1., 0.]]), dim=-1)
        visual = F.normalize(torch.tensor([[1., 1., 0.], [0., 1., 1.]]), dim=-1)
        bank = dict(class_ids=[0, 1], text=text, real=visual,
                    prototypes=visual)
        self.assertTrue(torch.allclose(calibrate_text_bank(model, bank, strength=0), text))
        calibrated = calibrate_text_bank(model, bank, strength=0.25)
        expected = F.normalize(0.75 * text + 0.25 * visual, dim=-1)
        self.assertTrue(torch.allclose(calibrated, expected, atol=1e-6))
        self.assertFalse(torch.allclose(calibrated, text))
        query = F.normalize(torch.tensor([[1., 0.3, 0.], [0., 1., 0.3]]), dim=-1)
        _, baseline = classify_bidirectional(model, query, bank, alpha=0.15, gamma=0.5)
        _, disabled = classify_bidirectional(model, query, bank, alpha=0.15,
                                            gamma=0.5, text_calibration=0)
        _, enabled = classify_bidirectional(model, query, bank, alpha=0.15,
                                           gamma=0.5, text_calibration=0.25)
        self.assertTrue(torch.allclose(disabled, baseline))
        self.assertFalse(torch.allclose(enabled, baseline))
        with self.assertRaises(ValueError):
            calibrate_text_bank(model, bank, strength=1.1)

    def test_visual_ce_matches_generated_prototype_classifier_and_gradients(self):
        torch.manual_seed(13)
        model = DeepFlowMatchingNet(in_channels=4, model_channels=8,
                                    out_channels=4, num_res_blocks=1)
        text = F.normalize(torch.randn(3, 4), dim=-1)
        visual = F.normalize(torch.randn(8, 4), dim=-1)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        loss = visual_prototype_classification_ce(
            model, text, visual, labels, temperature=0.07, steps=2)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(
            sum(p.grad.abs().sum().item() for p in model.parameters()
                if p.grad is not None), 0.)
        with self.assertRaises(ValueError):
            visual_prototype_classification_ce(
                model, text, visual, labels, temperature=0.07, steps=0)

    def test_zero_visual_ce_keeps_old_training_weights(self):
        eye = torch.eye(3)
        cache = dict(stage='base', split='train', encoder_id='unit',
                     class_ids=[0, 1, 2], text_prototypes=eye,
                     visual_prototypes=eye,
                     image_features=eye, image_labels=torch.arange(3))
        settings = dict(epochs=1, batch_size=3, blocks=1, hidden=8,
                        target='sample', seed=11, device='cpu')
        original = fit(cache, **settings)
        disabled = fit(cache, visual_ce_weight=0, **settings)
        enabled = fit(cache, visual_ce_weight=0.1, visual_ce_steps=1, **settings)
        self.assertTrue(all(torch.equal(original['state_dict'][k],
                                           disabled['state_dict'][k])
                            for k in original['state_dict']))
        self.assertTrue(any(not torch.equal(original['state_dict'][k],
                                           enabled['state_dict'][k])
                            for k in original['state_dict']))
        self.assertEqual(enabled['training']['visual_ce_weight'], 0.1)
        self.assertEqual(len(enabled['visual_ce_history']), 1)
        self.assertGreater(enabled['visual_ce_history'][0], 0.)

    def test_calibrated_text_cache_matches_uncached(self):
        model = ConstantVelocity([0.1, 0., 0.])
        eye = torch.eye(3)
        bank = dict(class_ids=[0, 1, 2], text=eye,
                    real=F.normalize(eye + 0.1, dim=-1),
                    prototypes=eye)
        cached = calibrate_text_bank(model, bank, strength=0.2)
        _, a = classify_bidirectional(model, eye, bank, alpha=0.1,
                                      gamma=0.5, text_calibration=0.2)
        _, b = classify_bidirectional(model, eye, bank, alpha=0.1,
                                      gamma=0.5, calibrated_text=cached)
        self.assertTrue(torch.allclose(a, b))


if __name__ == "__main__":
    unittest.main()
