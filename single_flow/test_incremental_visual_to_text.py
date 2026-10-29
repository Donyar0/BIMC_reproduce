"""Synthetic regression checks for optional session-wise Flow adaptation."""
import contextlib
import io
import unittest
from unittest.mock import patch
import torch
from single_flow import visual_to_text as vtt


class IncrementalFlowTests(unittest.TestCase):
    def test_seen_ce_uses_all_classes_and_old_checkpoint_is_unchanged(self):
        torch.set_num_threads(1)
        base = dict(stage='base', split='train', encoder_id='synthetic',
                    class_ids=[0, 1], text_prototypes=torch.eye(4)[:2],
                    image_features=torch.eye(4)[:2],
                    image_labels=torch.tensor([0, 1], dtype=torch.long))
        with contextlib.redirect_stdout(io.StringIO()):
            ck = vtt.fit(base, epochs=2, blocks=1, hidden=8, ce_weight=0.)
        inc = dict(stage='incremental', split='train', encoder_id='synthetic',
                   class_ids=[2, 3], text_prototypes=torch.eye(4)[2:],
                   image_features=torch.eye(4)[2:],
                   image_labels=torch.tensor([2, 3], dtype=torch.long))
        bank = vtt.build_text_bank(ck, base)
        bank = vtt.build_text_bank(ck, inc, bank)
        snapshot = {k: value.clone() for k, value in ck['state_dict'].items()}
        calls = []
        cross_entropy = vtt.F.cross_entropy

        def track(logits, targets):
            calls.append((logits.shape[1], targets.tolist()))
            return cross_entropy(logits, targets)

        with patch.object(vtt.F, 'cross_entropy', side_effect=track):
            with contextlib.redirect_stdout(io.StringIO()):
                result = vtt.finetune_incremental(
                    ck, inc, bank, session_id=1, epochs=2,
                    batch_size=2, lr=1e-4)
        self.assertEqual(calls, [(4, [2, 3]), (4, [2, 3])])
        self.assertEqual(result['trained_through_session'], 1)
        self.assertEqual(result['incremental_updates'][0]['num_samples'], 2)
        self.assertTrue(all(torch.equal(v, ck['state_dict'][k]) for k, v in snapshot.items()))
        self.assertTrue(any(not torch.equal(v, result['state_dict'][k]) for k, v in snapshot.items()))
        with self.assertRaises(ValueError):
            vtt.finetune_incremental(ck, dict(inc, split='test'), bank, session_id=1)
        with self.assertRaises(ValueError):
            vtt.finetune_incremental(result, inc, bank, session_id=2)


if __name__ == '__main__':
    unittest.main()
