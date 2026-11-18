"""Check that optional calibration starts from the original Flow scores."""
import unittest

import torch
import torch.nn.functional as F

from single_flow.fm import DeepFlowMatchingNet
from single_flow.prototype_calibrator import (ResidualPrototypeCalibrator,
                                              fit_prototype_calibrator)


class PrototypeCalibratorTest(unittest.TestCase):
    def test_initial_scores_are_unchanged_and_correction_has_gradients(self):
        prototypes = F.normalize(torch.tensor([[1., 0., 0.], [0., 1., 0.],
                                               [0., 0., 1.]]), dim=-1)
        features = F.normalize(torch.tensor([[0.2, 0.8, 0.1], [0.8, 0.1, 0.2]]), dim=-1)
        raw_scores = torch.tensor([[0.1, 0.3, 0.2], [0.3, 0.1, 0.2]])
        calibrator = ResidualPrototypeCalibrator(prototypes, num_base=2)
        self.assertTrue(torch.allclose(calibrator(features, raw_scores), raw_scores))
        labels = torch.tensor([1, 2])
        F.cross_entropy(20.0 * calibrator(features, raw_scores), labels).backward()
        self.assertIsNotNone(calibrator.prototype_delta.grad)
        self.assertIsNotNone(calibrator.novel_bias.grad)
        self.assertGreater(calibrator.prototype_delta.grad.norm().item(), 0.0)
        self.assertGreater(calibrator.novel_bias.grad.abs().item(), 0.0)

    def test_fit_uses_flow_replay_and_current_support(self):
        architecture = dict(in_channels=4, model_channels=8,
                            out_channels=4, num_res_blocks=1)
        flow = DeepFlowMatchingNet(**architecture)
        checkpoint = dict(flow_direction='text_to_visual',
                          method='gaussian-text-visual-v2', architecture=architecture,
                          state_dict=flow.state_dict())
        prototypes = F.normalize(torch.eye(4)[:3], dim=-1)
        bank = dict(class_ids=[0, 1, 2], prototypes=prototypes, text=prototypes)
        memory = dict(session_lengths=[2, 1], text=prototypes,
                      sigma=torch.full((3,), 0.05),
                      support_features=F.normalize(
                          torch.tensor([[0., 0., 1., 0.], [0.1, 0., 0.9, 0.]]), dim=-1),
                      support_labels=torch.tensor([2, 2]))
        scoring = dict(alpha=0.15, gamma=0.5, steps=2, score_mode='cosine_sum',
                       text_score_weight=1.2, hub_weight=0.,
                       whitening_matrix=torch.eye(4), text_whitening_mode='none')
        model, history = fit_prototype_calibrator(
            checkpoint, memory, bank, device='cpu', num_base=2,
            pseudo_per_class=2, noise_scale=0.25, flow_steps=2, solver='euler',
            scoring_kwargs=scoring, steps=3, batch_size=4, lr=0.01,
            temperature=20., reg_weight=0.1, whitening_matrix=torch.eye(4))
        self.assertEqual(len(history), 3)
        self.assertTrue(torch.isfinite(torch.tensor(history)).all())
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))


if __name__ == '__main__':
    unittest.main()
