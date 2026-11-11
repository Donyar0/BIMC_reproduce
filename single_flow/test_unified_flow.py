import unittest
import torch
from single_flow.s1_flow import fit
from single_flow.unified_flow import adapt, build_bank, classify_bidirectional, load_model, real_means


class UnifiedFlowTests(unittest.TestCase):
    def test_adaptation_bank_and_scores(self):
        torch.set_num_threads(1)
        eye = torch.eye(3)
        base = dict(stage='base', split='train', encoder_id='test', class_ids=[0, 1],
                    text_prototypes=eye[:2], visual_prototypes=eye[:2],
                    image_features=eye[:2], image_labels=torch.tensor([0, 1]))
        ckpt = fit(base, epochs=1, batch_size=2, blocks=1, hidden=8, target='sample')
        memory = dict(class_ids=[0, 1], text=eye[:2], real=real_means(base))
        novel = dict(encoder_id='test', class_ids=[2], text_prototypes=eye[2:],
                     image_features=torch.tensor([[0.2, 0., 1.]]), image_labels=torch.tensor([2]))
        updated = adapt(ckpt, novel, memory, steps=2, batch_size=3, lr=0.001,
                        weight_decay=0, device='cpu')
        self.assertTrue(any(not torch.equal(ckpt['state_dict'][k], updated['state_dict'][k])
                            for k in ckpt['state_dict']))
        memory = dict(class_ids=[0, 1, 2], text=eye, real=eye, encoder_id='test')
        bank = build_bank(updated, memory, steps=2, num_base=2,
                          beta_base=0, beta_inc=0, device='cpu')
        self.assertTrue(torch.allclose(bank['prototypes'], eye))
        model = load_model(updated, 'cpu').eval()
        pred, scores = classify_bidirectional(model, eye, bank, alpha=0, gamma=1)
        self.assertTrue(torch.equal(pred, torch.arange(3)))
        self.assertTrue(torch.allclose(scores, eye))
        _, scores = classify_bidirectional(model, eye, bank, alpha=0.15, gamma=0.5)
        self.assertEqual(scores.shape, (3, 3))
        self.assertTrue(torch.isfinite(scores).all())

    def test_fm_only_training_is_deterministic(self):
        torch.set_num_threads(1)
        eye = torch.eye(3)
        base = dict(stage='base', split='train', encoder_id='test', class_ids=[0, 1],
                    text_prototypes=eye[:2], visual_prototypes=eye[:2],
                    image_features=eye[:2], image_labels=torch.tensor([0, 1]))
        kwargs = dict(epochs=2, batch_size=2, blocks=1, hidden=8,
                      target='sample', seed=9)
        first = fit(base, **kwargs)
        repeat = fit(base, **kwargs)
        self.assertTrue(all(torch.equal(first['state_dict'][k], repeat['state_dict'][k])
                            for k in first['state_dict']))
        self.assertTrue(all(torch.isfinite(torch.tensor(first['loss_history']))))
        self.assertEqual(first['training']['target'], 'sample')

    def test_bank_appends_only_new_classes_and_freezes_old_prototypes(self):
        from unittest.mock import patch

        class Velocity:
            def __init__(self, vector):
                self.vector = torch.tensor(vector, dtype=torch.float32)
                self.batch_sizes = []

            def eval(self):
                return self

            def __call__(self, x, t):
                self.batch_sizes.append(len(x))
                return self.vector.expand_as(x)

        ckpt = {'encoder_id': 'test'}
        first = dict(class_ids=[0, 1], text=torch.eye(3)[:2],
                     real=torch.eye(3)[:2], encoder_id='test')
        new = dict(class_ids=[2], text=torch.eye(3)[2:],
                   real=torch.eye(3)[2:], encoder_id='test')
        v0, v1 = Velocity([0., 0., 0.]), Velocity([0.5, 0., 0.])
        with patch('single_flow.unified_flow.load_model', return_value=v0):
            old = build_bank(ckpt, first, steps=2, num_base=2,
                             beta_base=0.25, beta_inc=0.5, device='cpu')
        old_generated = old['generated'].clone()
        old_prototypes = old['prototypes'].clone()
        with patch('single_flow.unified_flow.load_model', return_value=v1):
            bank = build_bank(ckpt, new, steps=2, num_base=0,
                              beta_base=0.25, beta_inc=0.5, device='cpu',
                              previous_bank=old)
        self.assertEqual(bank['class_ids'], [0, 1, 2])
        self.assertEqual(v1.batch_sizes, [1, 1])
        self.assertTrue(torch.equal(bank['generated'][:2], old_generated))
        self.assertTrue(torch.equal(bank['prototypes'][:2], old_prototypes))
        self.assertFalse(torch.equal(bank['generated'][2], new['text'][0]))
        self.assertTrue(torch.equal(old['generated'], old_generated))
        self.assertTrue(torch.equal(old['prototypes'], old_prototypes))
        with patch('single_flow.unified_flow.load_model', return_value=v1):
            with self.assertRaises(ValueError):
                build_bank(ckpt, new, steps=2, num_base=0, beta_base=0.25,
                           beta_inc=0.5, device='cpu', previous_bank=bank)

    def test_only_flow_scores_remain(self):
        class NoOpFlow:
            def __call__(self, x, t):
                return torch.zeros_like(x)

        query = torch.nn.functional.normalize(torch.tensor([[1.0, 0.5, 0.0]]), dim=-1)
        bank = dict(class_ids=[0, 1], prototypes=torch.eye(3)[:2],
                    text=torch.nn.functional.normalize(torch.tensor(
                        [[1.0, 0.0, 0.0], [0.5, 0.5, 0.7]]), dim=-1))
        _, score = classify_bidirectional(NoOpFlow(), query, bank, alpha=0.2, gamma=0.6)
        expected = 0.6*(query @ bank['prototypes'].T) + 0.4*(query @ bank['text'].T)
        self.assertTrue(torch.allclose(score, expected))


if __name__ == '__main__':
    unittest.main()
