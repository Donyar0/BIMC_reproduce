"""FM-only visual-to-text training and full-Flow inference smoke tests."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from single_flow import visual_to_text as vtt


class VisualToTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        text = torch.eye(4)
        cls.cache = dict(stage='base', split='train', encoder_id='synthetic',
                         class_ids=[10, 20, 30, 40], text_prototypes=text,
                         image_features=text.roll(1, dims=1),
                         image_labels=torch.tensor([10, 20, 30, 40]))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.ck = vtt.fit(cls.cache, epochs=150, blocks=1, hidden=32, lr=0.005)

    def test_fm_training_and_full_flow_classification(self):
        model = vtt.load_flow(self.ck)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        raw = self.cache['image_features'] @ self.cache['text_prototypes'].T
        self.assertFalse(torch.equal(raw.argmax(-1), torch.arange(4)))
        for steps in (1, 8):
            pred, scores = vtt.classify(
                model, self.cache['image_features'], self.cache['text_prototypes'],
                steps=steps, candidate_class_ids=self.cache['class_ids'])
            self.assertEqual(pred.tolist(), self.cache['class_ids'])
            torch.testing.assert_close(scores, vtt.transform_images(
                model, self.cache['image_features'], steps) @ self.cache['text_prototypes'].T)
        self.assertLess(sum(self.ck['loss_history'][-10:]), sum(self.ck['loss_history'][:10]))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_no_ce_and_no_partial_alpha(self):
        model = vtt.load_flow(self.ck)
        with patch.object(vtt.F, 'cross_entropy', side_effect=AssertionError('CE called')):
            with contextlib.redirect_stdout(io.StringIO()):
                vtt.fit(self.cache, epochs=1, blocks=1, hidden=8, ce_weight=0)
        with self.assertRaisesRegex(ValueError, 'CE'):
            vtt.fit(self.cache, epochs=1, blocks=1, hidden=8, ce_weight=0.1)
        with self.assertRaisesRegex(ValueError, 'alpha'):
            vtt.classify_alphas(model, self.cache['image_features'],
                                self.cache['text_prototypes'], alphas=[0.2])
        with self.assertRaises(ValueError):
            vtt.classify(model, self.cache['image_features'], torch.ones(3, 5))

    def test_checkpoint_and_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'base_flow.pt'
            torch.save(self.ck, path)
            self.assertEqual(vtt.load_checkpoint(path)['direction'], 'visual_to_text')
        for changes in [dict(stage='incremental'), dict(split='test'),
                        dict(image_labels=torch.tensor([10, 20, 30, 99]))]:
            with self.assertRaises(ValueError):
                vtt.fit(dict(self.cache, **changes), epochs=1, blocks=1)
        with self.assertRaises(ValueError):
            vtt.load_flow(dict(self.ck, direction='text_to_visual'))


if __name__ == '__main__':
    unittest.main()
