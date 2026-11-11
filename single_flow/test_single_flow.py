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
                         class_ids=[10,20,30,40], text_prototypes=x,
                         visual_prototypes=x.roll(2, dims=1), fused_prototypes=x.roll(1, dims=1),
                         image_features=x.roll(2, dims=1),
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

    def test_fused_target_is_not_visual_only(self):
        # Changing the visual-only prototype cannot change the Flow training.
        alternate = dict(self.cache, visual_prototypes=self.cache['visual_prototypes'].flip(0))
        with contextlib.redirect_stdout(io.StringIO()):
            ck_a = fit(self.cache, epochs=2, blocks=1, hidden=8, seed=7)
            ck_b = fit(alternate, epochs=2, blocks=1, hidden=8, seed=7)
            ck_c = fit(dict(self.cache, fused_prototypes=self.cache['visual_prototypes']),
                       epochs=2, blocks=1, hidden=8, seed=7)
        self.assertEqual(ck_a['flow_target'], 'bimc_fused')
        self.assertEqual(ck_a['loss_history'], ck_b['loss_history'])
        for key in ck_a['state_dict']:
            self.assertTrue(torch.equal(ck_a['state_dict'][key], ck_b['state_dict'][key]))
        self.assertTrue(any(not torch.equal(ck_a['state_dict'][k], ck_c['state_dict'][k])
                            for k in ck_a['state_dict']))

    def test_endpoint_ablation_uses_correct_visual_targets(self):
        # The visual-only modes must ignore the BiMC fused prototype.
        base = dict(self.cache)
        changed_fused = dict(base, fused_prototypes=base['fused_prototypes'].flip(0))
        with contextlib.redirect_stdout(io.StringIO()):
            a = fit(base, epochs=2, blocks=1, hidden=8, seed=4,
                    target='prototype', endpoint='visual_prototype')
            b = fit(changed_fused, epochs=2, blocks=1, hidden=8, seed=4,
                    target='prototype', endpoint='visual_prototype')
            c = fit(base, epochs=2, blocks=1, hidden=8, seed=4,
                    target='sample', endpoint='visual_sample')
            # Moving individual images without changing visual mean prototypes
            # must affect individual-sample FM but not class-prototype FM.
            changed_images = dict(base, image_features=base['image_features'].roll(1, dims=1))
            d = fit(changed_images, epochs=2, blocks=1, hidden=8, seed=4,
                    target='sample', endpoint='visual_sample')
        self.assertEqual(a['flow_target'], 'visual_prototype')
        self.assertEqual(c['flow_target'], 'visual_sample')
        self.assertEqual(a['training']['target_space'], 'visual_prototype')
        self.assertEqual(c['training']['target_space'], 'visual_sample')
        self.assertEqual(a['loss_history'], b['loss_history'])
        self.assertNotEqual(c['loss_history'], d['loss_history'])
        for key in a['state_dict']:
            self.assertTrue(torch.equal(a['state_dict'][key], b['state_dict'][key]))
        with self.assertRaisesRegex(ValueError, 'requires target=sample'):
            fit(base, epochs=1, blocks=1, endpoint='visual_sample')
        with self.assertRaisesRegex(ValueError, 'requires target=prototype'):
            fit(base, epochs=1, blocks=1, target='sample', endpoint='visual_prototype')
        with self.assertRaisesRegex(ValueError, 'endpoint must be'):
            fit(base, epochs=1, blocks=1, endpoint='unknown')
        with contextlib.redirect_stdout(io.StringIO()):
            sample_ce = fit(base, epochs=1, blocks=1, hidden=8, seed=4,
                            target='sample', endpoint='visual_sample', ce_weight=.1)
        self.assertTrue(torch.isfinite(torch.tensor(sample_ce['loss_history'])).all())

    def test_reject_leakage(self):
        for changes in [dict(stage='incremental'),dict(split='test')]:
            with self.assertRaises(ValueError):
                fit(dict(self.cache,**changes),epochs=1,blocks=1)

    def test_sample_and_ce(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result=fit(self.cache,epochs=2,blocks=1,hidden=8,target='sample',ce_weight=0.1)
        self.assertTrue(all(torch.isfinite(torch.tensor(result['loss_history']))))
        with self.assertRaises(ValueError):
            fit(dict(self.cache, image_labels=torch.tensor([10,20,30,99])),
                epochs=1, blocks=1, target='sample')

    def test_direct_prototypes_and_export(self):
        from bimc_cache import from_task_statistics
        stats = dict(class_index=[10,20,30,40], text_proto=self.cache['text_prototypes'],
                     image_proto=self.cache['visual_prototypes'],
                     fused_proto=self.cache['fused_prototypes'],
                     images_features=self.cache['image_features'],
                     images_targets=self.cache['image_labels'])
        cache = from_task_statistics(stats, 'synthetic', 0)
        self.assertTrue(torch.equal(cache['visual_prototypes'], stats['image_proto']))
        self.assertTrue(torch.equal(cache['text_prototypes'], stats['text_proto']))
        self.assertTrue(torch.equal(cache['fused_prototypes'], stats['fused_proto']))
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
        with self.assertRaisesRegex(ValueError, 'fused_proto'):
            from_task_statistics({k:v for k,v in stats.items() if k != 'fused_proto'}, 'synthetic', 0)

    def test_bimc_shared_text_formula(self):
        # Load just the pure method via AST; no CLIP weights or CUDA needed.
        import ast
        from pathlib import Path
        from types import SimpleNamespace
        source = ast.parse((Path(__file__).resolve().parents[1]/'models/bimc.py').read_text())
        cls = next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='BiMC')
        methods = [n for n in cls.body if isinstance(n,ast.FunctionDef) and
                   n.name in ('fused_text_prototype', 'fused_visual_text_prototype')]
        namespace = {'F': torch.nn.functional}
        exec(compile(ast.Module(body=methods,type_ignores=[]),'<methods>','exec'), namespace)
        cfg = SimpleNamespace(TEXT_CALIBRATION=True,LAMBDA_T=0.3)
        obj = SimpleNamespace(cfg=SimpleNamespace(TRAINER=SimpleNamespace(BiMC=cfg)))
        text,desc,image = torch.randn(3,4),torch.randn(3,4),torch.randn(3,4)
        mixed = namespace['fused_text_prototype'](obj,text,desc)
        self.assertTrue(torch.equal(mixed,0.7*text+0.3*desc))
        beta = 0.65
        obj.fused_text_prototype = lambda t,d: namespace['fused_text_prototype'](obj,t,d)
        actual = namespace['fused_visual_text_prototype'](obj,text,desc,image,beta)
        expected = torch.nn.functional.normalize(beta*mixed+(1-beta)*image, dim=-1)
        self.assertTrue(torch.allclose(actual,expected))
        cfg.TEXT_CALIBRATION=False
        self.assertTrue(torch.equal(namespace['fused_text_prototype'](obj,text,desc),text))
        actual = namespace['fused_visual_text_prototype'](obj,text,desc,image,beta)
        expected = torch.nn.functional.normalize(beta*text+(1-beta)*image, dim=-1)
        self.assertTrue(torch.allclose(actual,expected))

if __name__ == '__main__':
    unittest.main()


