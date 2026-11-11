import contextlib
import io
import unittest
import torch
from s1_flow import fit, generate, classify

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        x = torch.eye(4)
        cls.cache = dict(stage='base', split='train', encoder_id='synthetic',
                         class_ids=[10,20,30,40], text_prototypes=x, visual_prototypes=x.roll(1, dims=1), image_features=x.roll(1, dims=1),
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
        new = dict(self.cache, class_ids=[50], text_prototypes=torch.ones(1,4))
        merged = generate(self.ck,new,existing=bank)
        self.assertTrue(torch.equal(merged['prototypes'][:4],bank['prototypes']))
        with self.assertRaises(ValueError):
            generate(self.ck,self.cache,existing=bank)

    def test_reject_leakage(self):
        for changes in [dict(stage='incremental'),dict(split='test')]:
            with self.assertRaises(ValueError):
                fit(dict(self.cache,**changes),epochs=1,blocks=1)

    def test_sample_training(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result=fit(self.cache,epochs=2,blocks=1,hidden=8,target='sample')
        self.assertTrue(all(torch.isfinite(torch.tensor(result['loss_history']))))
        with self.assertRaises(ValueError):
            fit(dict(self.cache, image_labels=torch.tensor([10,20,30,99])),
                epochs=1, blocks=1, target='sample')

    def test_direct_prototypes_and_export(self):
        from bimc_cache import from_task_statistics
        stats = dict(class_index=[10,20,30,40], text_proto=self.cache['text_prototypes'],
                     image_proto=self.cache['visual_prototypes'],
                     images_features=self.cache['image_features'],
                     images_targets=self.cache['image_labels'])
        cache = from_task_statistics(stats, 'synthetic', 0)
        self.assertTrue(torch.equal(cache['visual_prototypes'], stats['image_proto']))
        self.assertTrue(torch.equal(cache['text_prototypes'], stats['text_proto']))
        # Prototype mode must work without any sample features/labels.
        del cache['image_features'], cache['image_labels']
        with contextlib.redirect_stdout(io.StringIO()):
            ck = fit(cache, epochs=2, blocks=1, hidden=8)
        text_only = {k:cache[k] for k in ('encoder_id','class_ids','text_prototypes')}
        self.assertEqual(generate(ck,text_only)['prototypes'].shape, (4,4))
        with self.assertRaises(ValueError):
            fit(from_task_statistics(stats,'synthetic',1),epochs=1,blocks=1)
        with self.assertRaises(ValueError):
            from_task_statistics(dict(stats,class_index=[20,10,30,40]),'synthetic',0)

    def test_bimc_shared_text_formula(self):
        # Load just the pure method via AST; no CLIP weights or CUDA needed.
        import ast
        from pathlib import Path
        from types import SimpleNamespace
        source = ast.parse((Path(__file__).resolve().parents[1]/'models/bimc.py').read_text())
        cls = next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='BiMC')
        method = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='fused_text_prototype')
        namespace = {}
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<method>','exec'), namespace)
        cfg = SimpleNamespace(TEXT_CALIBRATION=True,LAMBDA_T=0.3)
        obj = SimpleNamespace(cfg=SimpleNamespace(TRAINER=SimpleNamespace(BiMC=cfg)))
        text,desc,image = torch.randn(3,4),torch.randn(3,4),torch.randn(3,4)
        mixed = namespace['fused_text_prototype'](obj,text,desc)
        self.assertTrue(torch.equal(mixed,0.7*text+0.3*desc))
        cfg.TEXT_CALIBRATION=False
        self.assertTrue(torch.equal(namespace['fused_text_prototype'](obj,text,desc),text))

if __name__ == '__main__':
    unittest.main()


