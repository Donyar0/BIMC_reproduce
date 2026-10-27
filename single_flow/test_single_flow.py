import contextlib
import io
import unittest
import torch
from s1_flow import fit, generate, classify, fuse_text

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        x = torch.eye(4)
        cls.cache = dict(stage='base', split='train', encoder_id='synthetic',
                         class_ids=[10,20,30,40], template_features=x,
                         description_features=x, image_features=x.roll(1, dims=1),
                         image_labels=torch.tensor([10,20,30,40]))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.ck = fit(cls.cache, epochs=120, blocks=1, hidden=16, lr=0.01)

    def test_learning_and_freeze(self):
        self.assertLess(sum(self.ck['loss_history'][-10:]), sum(self.ck['loss_history'][:10]))
        before = {k:v.clone() for k,v in self.ck['state_dict'].items()}
        bank = generate(self.ck, self.cache, steps=8)
        self.assertTrue(torch.allclose(bank['prototypes'].norm(dim=-1), torch.ones(4)))
        pred, _ = classify(bank['prototypes'], bank)
        self.assertEqual(pred.tolist(), self.cache['class_ids'])
        for k,v in before.items():
            self.assertTrue(torch.equal(v,self.ck['state_dict'][k]))
        new = dict(self.cache, class_ids=[50], template_features=torch.ones(1,4), description_features=torch.ones(1,4))
        merged = generate(self.ck,new,existing=bank)
        self.assertTrue(torch.equal(merged['prototypes'][:4],bank['prototypes']))
        with self.assertRaises(ValueError):
            generate(self.ck,self.cache,existing=bank)

    def test_reject_leakage(self):
        for changes in [dict(stage='incremental'),dict(split='test'),dict(image_labels=torch.tensor([10,20,30,99]))]:
            with self.assertRaises(ValueError):
                fit(dict(self.cache,**changes),epochs=1,blocks=1)

    def test_sample_and_ce(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result=fit(self.cache,epochs=2,blocks=1,hidden=8,target='sample',ce_weight=0.1)
        self.assertTrue(all(torch.isfinite(torch.tensor(result['loss_history']))))
        self.assertTrue(torch.equal(fuse_text(torch.eye(4),torch.eye(4)),torch.eye(4)))

if __name__ == '__main__':
    unittest.main()

