import contextlib
import io
import unittest
from unittest.mock import patch
import torch
from single_flow import visual_to_text as vtt


class VisualToTextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        text = torch.eye(4)
        cls.cache = dict(stage='base', split='train', encoder_id='synthetic',
                         class_ids=[10,20,30,40], text_prototypes=text,
                         image_features=text.roll(1, dims=1),
                         image_labels=torch.tensor([10,20,30,40]))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.ck = vtt.fit(cls.cache, epochs=150, blocks=1, hidden=32, lr=0.005)

    def test_actual_image_to_text_learning(self):
        bank = vtt.build_text_bank(self.ck, self.cache)
        model = vtt.load_flow(self.ck)
        before = {k:v.clone() for k,v in model.state_dict().items()}
        # Raw images deliberately predict the wrong classes; trained Flow must fix them.
        baseline = self.cache['image_features'] @ self.cache['text_prototypes'].T
        self.assertFalse(torch.equal(baseline.argmax(-1), torch.arange(4)))
        for steps in [1,8]:
            pred, scores = vtt.classify(model, self.cache['image_features'], bank, steps)
            self.assertEqual(pred.tolist(), self.cache['class_ids'])
            self.assertTrue(torch.isfinite(scores).all())
        self.assertLess(sum(self.ck['loss_history'][-10:]),sum(self.ck['loss_history'][:10]))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        self.assertTrue(all(torch.equal(before[k],v) for k,v in model.state_dict().items()))

    def test_incremental_text_only_and_old_rows(self):
        base = vtt.build_text_bank(self.ck,self.cache)
        new = dict(encoder_id='synthetic', class_ids=[99], text_prototypes=torch.ones(1,4))
        bank = vtt.build_text_bank(self.ck,new,base)
        self.assertEqual(bank['class_ids'],[10,20,30,40,99])
        self.assertTrue(torch.equal(bank['prototypes'][:4],base['prototypes']))
        with self.assertRaises(ValueError):
            vtt.build_text_bank(self.ck,self.cache,base)
        with self.assertRaises(ValueError):
            vtt.build_text_bank(self.ck,dict(new,encoder_id='other'))
        with self.assertRaises(ValueError):
            vtt.load_flow(dict(self.ck,direction='text_to_visual'))

    def test_training_pair_direction_and_ce_gradient(self):
        captured = []
        original = vtt.TensorDataset
        ce_inputs = []
        original_ce = vtt.F.cross_entropy
        def capture_ce(logits, labels):
            logits.retain_grad()
            ce_inputs.append(logits)
            return original_ce(logits, labels)
        def capture(*tensors):
            captured.append(tensors)
            return original(*tensors)
        with patch.object(vtt,'TensorDataset',capture), patch.object(vtt.F,'cross_entropy',capture_ce), contextlib.redirect_stdout(io.StringIO()):
            vtt.fit(self.cache,epochs=1,blocks=1,hidden=8,ce_weight=0.1)
            vtt.fit(dict(self.cache,visual_prototypes=self.cache['image_features']),
                    source='prototype',epochs=1,blocks=1,hidden=8,ce_weight=0.)
        for x0,x1,y in captured:
            self.assertTrue(torch.equal(x0,self.cache['image_features']))
            self.assertTrue(torch.equal(x1,self.cache['text_prototypes'][y]))
        self.assertEqual(len(ce_inputs),1)
        self.assertIsNotNone(ce_inputs[0].grad)
        self.assertGreater(ce_inputs[0].grad.norm().item(),0.)

    def test_reject_leakage_and_invalid_inputs(self):
        for changes in [dict(stage='incremental'), dict(split='test'),
                        dict(image_labels=torch.tensor([10,20,30,99]))]:
            with self.assertRaises(ValueError):
                vtt.fit(dict(self.cache,**changes),epochs=1,blocks=1)
        model=vtt.load_flow(self.ck)
        with self.assertRaises(ValueError):
            vtt.transform_images(model,torch.eye(4),steps=0)
        with self.assertRaises(ValueError):
            vtt.classify(model,torch.eye(4),dict(class_ids=[1],prototypes=torch.ones(1,4)))


if __name__=='__main__':
    unittest.main()
