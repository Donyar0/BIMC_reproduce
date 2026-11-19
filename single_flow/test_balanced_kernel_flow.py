"""Check class weighting, prototype targets, and local kernel transport."""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.clip.model import VisionTransformer
from models.bimc import BiMC
from single_flow.balanced_random_ridge import (BalancedRandomFeatureRidge,
                                                shrink_visual_prototype)
from single_flow.kernel_velocity import KernelVelocityField, kernel_transport


class BalancedKernelFlowTests(unittest.TestCase):
    def test_balanced_statistics_and_prototype_target(self):
        features = torch.tensor([[1., 0.], [1., .1], [1., -.1], [1., .2], [0., 1.]])
        labels = torch.tensor([0, 0, 0, 0, 1])
        prototypes = F.normalize(torch.tensor([[1., 0.], [.2, .8]]), dim=-1)
        head = BalancedRandomFeatureRidge(2, 16, 0.2, 3, 'cpu')
        self.assertEqual(head.add_session([0, 1], features, labels, prototypes, 2), [4, 1])
        phi = head.project(features)
        expected_gram = phi[:4].T @ phi[:4] / 4 + phi[4:].T @ phi[4:]
        expected_target = head.project(prototypes).T
        self.assertTrue(torch.allclose(head.gram, expected_gram, atol=1e-6))
        self.assertTrue(torch.allclose(head.cross, expected_target, atol=1e-6))
        self.assertTrue(torch.allclose(head.solve(), torch.linalg.solve(
            expected_gram + .2 * torch.eye(16), expected_target), atol=1e-5))

    def test_class_mean_target_is_exact_weighted_least_squares(self):
        features = torch.tensor([[1., 0.], [1., .1], [0., 1.]])
        labels = torch.tensor([0, 0, 1])
        head = BalancedRandomFeatureRidge(2, 12, 1., 4, 'cpu', 'class_mean')
        head.add_session([0, 1], features, labels, F.normalize(features[:2], dim=-1))
        phi = head.project(features)
        expected = torch.stack((phi[:2].mean(0), phi[2]), dim=1)
        self.assertTrue(torch.allclose(head.cross, expected, atol=1e-6))
        before = head.gram.clone()
        head.add_session([2], torch.tensor([[-1., 0.]]), torch.tensor([2]),
                         torch.tensor([[-1., 0.]]))
        self.assertTrue(torch.allclose(head.gram - before,
                                       head.project(torch.tensor([[-1., 0.]])).T
                                       @ head.project(torch.tensor([[-1., 0.]])), atol=1e-6))

    def test_shrinkage_depends_on_real_shot_count(self):
        real = F.normalize(torch.tensor([[1., 0.], [1., 0.]]), dim=-1)
        generated = F.normalize(torch.tensor([[0., 1.], [0., 1.]]), dim=-1)
        result = shrink_visual_prototype(real, generated, torch.tensor([5, 500]), 5.)
        self.assertGreater(result[1, 0], result[0, 0])

    def test_kernel_leave_one_out_and_anchor_stability(self):
        field = KernelVelocityField(.2)
        text = torch.tensor([[1., 0.], [0., 1.]])
        visual = F.normalize(torch.tensor([[1., 1.], [-1., 1.]]), dim=-1)
        field.append(text, visual)
        before = field.text.clone(), field.visual.clone()
        own = field(text[:1], torch.zeros(1, 1))
        other = field(text[:1], torch.zeros(1, 1), exclude=torch.tensor([0]))
        self.assertTrue(torch.allclose(own, visual[:1] - text[:1], atol=1e-4))
        self.assertTrue(torch.allclose(other, visual[1:] - text[1:], atol=1e-6))
        self.assertEqual(kernel_transport(field, text, 4, exclude=torch.tensor([0, 1])).shape,
                         text.shape)
        self.assertTrue(torch.equal(field.text, before[0]))
        self.assertTrue(torch.equal(field.visual, before[1]))

    def test_one_vit_forward_exposes_last_layer_and_patches(self):
        visual = VisionTransformer(32, 16, 64, 4, 4, 16).eval()
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            final = visual(images)
            layers, patches = visual(images, all_layer_outputs=True,
                                     return_patch_tokens=True)
        self.assertEqual(len(layers), 4)
        self.assertEqual(patches.shape, (2, 4, 16))
        self.assertTrue(torch.allclose(final, layers[-1], atol=1e-5))

    def test_single_view_multilayer_patch_feature_is_finite(self):
        visual = VisionTransformer(32, 16, 64, 4, 4, 16).eval()

        class TinyClip:
            def __init__(self, visual_model):
                self.visual = visual_model

            def encode_image(self, images, **kwargs):
                return self.visual(images, **kwargs)

        feature_cfg = SimpleNamespace(MULTILAYER=True, LAST_LAYERS=2,
                                      PATCH_POOLING=True, DESCRIPTION_CHUNK=3,
                                      PATCH_TEMPERATURE=10., PATCH_WEIGHT=.25)
        mock = SimpleNamespace(cfg=SimpleNamespace(FEATURE=feature_cfg),
                               clip_model=TinyClip(visual),
                               _patch_description_bank=F.normalize(torch.randn(7, 16), dim=-1))
        images = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            output = BiMC._encode_single_view(mock, images)
        self.assertEqual(output.shape, (2, 16))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.allclose(output.norm(dim=-1), torch.ones(2), atol=1e-5))


if __name__ == '__main__':
    unittest.main()
