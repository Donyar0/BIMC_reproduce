"""FM-only shared bidirectional Flow tests (no replay, Cycle, CE, ensemble)."""
import unittest
from unittest.mock import patch
import torch
from single_flow.s1_flow import fit
from single_flow.unified_flow import build_bank, classify_bidirectional, load_model, real_means


class UnifiedFlowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_fm_only_training_and_bidirectional_inference(self):
        eye = torch.eye(3)
        cache = dict(stage='base', split='train', encoder_id='synthetic',
                     class_ids=[0, 1, 2], text_prototypes=eye,
                     visual_prototypes=eye, image_features=eye,
                     image_labels=torch.arange(3))
        checkpoint = fit(cache, epochs=2, batch_size=3, blocks=1,
                         hidden=8, target='sample', seed=1)
        self.assertEqual(len(checkpoint['loss_history']), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(checkpoint['loss_history']))))
        self.assertEqual(checkpoint['training']['target'], 'sample')
        self.assertFalse(any(k in checkpoint['training']
                             for k in ('cycle_weight', 'ce_weight', 'temperature')))
        model = load_model(checkpoint, 'cpu').eval()
        memory = dict(class_ids=[0, 1, 2], text=eye,
                      real=real_means(cache), encoder_id='synthetic')
        bank = build_bank(checkpoint, memory, steps=2, num_base=3,
                          beta_base=0, beta_inc=0, device='cpu')
        self.assertTrue(torch.allclose(bank['prototypes'], eye))
        predictions, scores = classify_bidirectional(model, eye, bank, alpha=0, gamma=1)
        self.assertEqual(predictions.tolist(), [0, 1, 2])
        self.assertTrue(torch.allclose(scores, eye))
        predictions, scores = classify_bidirectional(
            model, eye, bank, alpha=0.15, gamma=0.5)
        self.assertEqual(tuple(scores.shape), (3, 3))
        self.assertTrue(torch.isfinite(scores).all())

    def test_old_prototypes_are_not_overwritten_by_new_sessions(self):
        class Velocity:
            def __init__(self, vector):
                self.vector = torch.tensor(vector, dtype=torch.float32)
                self.batch_sizes = []
            def eval(self):
                return self
            def __call__(self, x, t):
                self.batch_sizes.append(len(x))
                return self.vector.expand_as(x)

        checkpoint = {'encoder_id': 'synthetic'}
        eye = torch.eye(3)
        first = dict(class_ids=[0, 1], text=eye[:2],
                     real=eye[:2], encoder_id='synthetic')
        novel = dict(class_ids=[2], text=eye[2:],
                     real=eye[2:], encoder_id='synthetic')
        velocity = Velocity([.1, 0., 0.])
        with patch('single_flow.unified_flow.load_model', return_value=velocity):
            base = build_bank(checkpoint, first, steps=2, num_base=2,
                              beta_base=.2, beta_inc=.5, device='cpu')
            merged = build_bank(checkpoint, novel, steps=2, num_base=0,
                                beta_base=.2, beta_inc=.5, device='cpu',
                                previous_bank=base)
        self.assertEqual(merged['class_ids'], [0, 1, 2])
        self.assertTrue(torch.equal(
            merged['prototypes'][:2], base['prototypes']))
        self.assertEqual(velocity.batch_sizes, [2, 2, 1, 1])
        with patch('single_flow.unified_flow.load_model', return_value=velocity):
            with self.assertRaises(ValueError):
                build_bank(checkpoint, novel, steps=2, num_base=0,
                           beta_base=.2, beta_inc=.5, device='cpu',
                           previous_bank=merged)

    def test_base_training_rejects_incremental_and_test_features(self):
        eye = torch.eye(3)
        cache = dict(stage='base', split='train', encoder_id='synthetic',
                     class_ids=[0, 1, 2], text_prototypes=eye,
                     visual_prototypes=eye, image_features=eye,
                     image_labels=torch.arange(3))
        with self.assertRaises(ValueError):
            fit(dict(cache, stage='incremental'), epochs=1, blocks=1)
        with self.assertRaises(ValueError):
            fit(dict(cache, split='test'), epochs=1, blocks=1)


if __name__ == '__main__':
    unittest.main()
