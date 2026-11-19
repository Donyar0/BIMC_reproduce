"""Check cumulative ridge statistics and the closed-form classifier."""
import unittest

import torch

from single_flow.frozen_random_ridge import (IncrementalRandomFeatureRidge,
                                             fuse_flow_ridge)
from single_flow.text_distribution_flow import distributions, fit, load_model


class IncrementalRandomFeatureRidgeTests(unittest.TestCase):
    def test_cumulative_statistics_match_a_joint_closed_form_fit(self):
        base = torch.tensor([[1., 0., 0.], [0., 1., 0.],
                             [0.8, 0.2, 0.], [0.1, 0.9, 0.]])
        base_labels = torch.tensor([0, 1, 0, 1])
        real_new = torch.tensor([[0., 0., 1.]])
        virtual_new = torch.tensor([[0.1, 0.2, 0.9], [0.2, 0., 0.8]])
        ridge = IncrementalRandomFeatureRidge(3, 12, 0.7, 19, 'cpu')
        frozen_projection = ridge.projection.clone()

        ridge.add_session([0, 1], [(base, base_labels)], batch_size=2)
        base_cross = ridge.cross.clone()
        ridge.solve()
        ridge.add_session([2], [(real_new, torch.tensor([2])),
                                (virtual_new, torch.tensor([2, 2]))], batch_size=2)
        ridge.solve()

        all_features = torch.cat((base, real_new, virtual_new))
        all_labels = torch.tensor([0, 1, 0, 1, 2, 2, 2])
        phi = ridge.project(all_features)
        labels = torch.nn.functional.one_hot(all_labels, 3).float()
        gram = phi.T @ phi
        cross = phi.T @ labels
        expected = torch.linalg.solve(gram + 0.7 * torch.eye(12), cross)
        self.assertTrue(torch.allclose(ridge.projection, frozen_projection))
        self.assertFalse(ridge.projection.requires_grad)
        self.assertTrue(torch.allclose(ridge.gram, gram, atol=1e-6))
        self.assertTrue(torch.allclose(ridge.cross, cross, atol=1e-6))
        self.assertTrue(torch.equal(ridge.cross[:, :2], base_cross))
        self.assertTrue(torch.allclose(ridge.weights, expected, atol=1e-6))
        self.assertEqual(ridge.class_ids, [0, 1, 2])
        self.assertEqual(ridge.observation_count, 7)

    def test_fusion_matches_branch_scales_per_query(self):
        flow = torch.tensor([[1., 2., 3.]])
        ridge = torch.tensor([[100., 200., 300.]])
        fused = fuse_flow_ridge(flow, ridge, 0.5)
        self.assertEqual(int(fused.argmax(-1)), 2)
        self.assertTrue(torch.allclose(fused, fused.mean() +
                                       (flow - flow.mean()) / flow.std(unbiased=False)))

    def test_base_funnel_trains_without_replay_covariance(self):
        stats = {
            'class_index': [0, 1],
            'text_proto': torch.tensor([[1., 0., 0.], [0., 1., 0.]]),
            'description_features': torch.tensor([[1., .1, 0.], [1., 0., .1],
                                                  [.1, 1., 0.], [0., 1., .1]]),
            'description_targets': torch.tensor([0, 0, 1, 1]),
            'images_features': torch.tensor([[1., 0., .1], [1., .1, 0.],
                                             [0., 1., .1], [.1, 1., 0.]]),
            'images_targets': torch.tensor([0, 0, 1, 1]),
            'image_proto': torch.tensor([[1., .05, .05], [.05, 1., .05]]),
        }
        memory = distributions(stats, include_covariance=False)
        self.assertNotIn('cov', memory)
        self.assertEqual(distributions(stats)['cov'].shape, (2, 3, 3))
        checkpoint = fit(memory, epochs=1, steps_per_epoch=1, batch_size=4,
                         lr=0.001, weight_decay=0., blocks=1, hidden=8,
                         seed=7, device='cpu', funnel_mode=True)
        frozen = load_model(checkpoint, 'cpu').requires_grad_(False)
        self.assertEqual(len(checkpoint['loss_history']), 1)
        self.assertTrue(all(not param.requires_grad for param in frozen.parameters()))


if __name__ == '__main__':
    unittest.main()
