"""Synthetic episodic tests; no CIFAR images or GPU required."""
import contextlib
import io
import unittest
import torch
from single_flow.meta_visual_to_text import fit_meta, sample_episode


class MetaVisualToTextTests(unittest.TestCase):
    def test_disjoint_support_query_and_pseudo_base(self):
        ids = [0, 1, 2, 3]
        labels = torch.arange(4).repeat_interleave(12)
        e = sample_episode(ids, labels, pseudo_new=2, shot=3, query=4,
                           generator=torch.Generator().manual_seed(12))
        self.assertFalse(set(e['support'].tolist()) & set(e['novel_query'].tolist()))
        self.assertFalse(set(e['pseudo_new_ids']) & set(e['pseudo_base_ids']))
        self.assertEqual(len(e['support']), 6)
        self.assertEqual(len(e['novel_query']), 8)
        self.assertTrue(set(labels[e['base_fm']].tolist()).isdisjoint(e['pseudo_new_ids']))
        self.assertTrue(set(labels[e['old_query']].tolist()).isdisjoint(e['pseudo_new_ids']))

    def test_meta_query_gradient_changes_flow(self):
        torch.set_num_threads(1)
        ids = [0, 1, 2, 3]
        labels = torch.arange(4).repeat_interleave(12)
        targets = torch.eye(4)
        base = dict(stage='base', split='train', encoder_id='synthetic',
                    class_ids=ids, text_prototypes=targets,
                    image_features=(targets[labels] + 0.2).float(),
                    image_labels=labels.long())
        kwargs = dict(epochs=2, episodes_per_epoch=3, pseudo_new=2,
                      shot=3, query=3, base_batch=6, old_query_batch=6,
                      inner_steps=1, inner_lr=1e-3, meta_lr=1e-3,
                      blocks=1, hidden=8, seed=1, device='cpu')
        with contextlib.redirect_stdout(io.StringIO()):
            learned = fit_meta(base, **kwargs)
        self.assertEqual(learned['direction'], 'visual_to_text')
        self.assertEqual(learned['training']['mode'], 'episodic_first_order_maml')
        self.assertEqual(len(learned['meta_history']), 2)
        self.assertTrue(all(torch.isfinite(v).all() for v in learned['state_dict'].values()))
        self.assertGreater(learned['meta_history'][0]['query_novel_ce'], 0.)
        with self.assertRaises(ValueError):
            fit_meta(dict(base, split='test'), **kwargs)
        with self.assertRaises(ValueError):
            fit_meta(base, **dict(kwargs, shot=10, query=4))


if __name__ == '__main__':
    unittest.main()
