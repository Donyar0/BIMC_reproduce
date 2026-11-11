import contextlib
import io
import unittest
import torch
from s1_flow import fit, generate, classify, output_to_velocity

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

    def test_endpoint_velocity_identity(self):
        x0 = torch.tensor([[1., 0., 0., 0.]])
        x1 = torch.tensor([[0., 1., 0., 0.]])
        for t_value in (0.0, 0.2, 0.875):
            t = torch.tensor([[t_value]])
            xt = (1-t)*x0 + t*x1
            predicted_v = output_to_velocity(x1, xt, t, 'endpoint')
            self.assertTrue(torch.allclose(predicted_v, x1-x0, atol=1e-6))
            self.assertTrue(torch.allclose(output_to_velocity(predicted_v, xt, t),
                                           predicted_v))
        with self.assertRaises(ValueError):
            output_to_velocity(x1, x0, torch.zeros(1, 1), 'invalid')

    def test_endpoint_training_and_generation(self):
        with contextlib.redirect_stdout(io.StringIO()):
            endpoint_ck = fit(self.cache, epochs=12, blocks=1, hidden=16,
                              lr=0.003, prediction='endpoint', ce_weight=0.1,
                              endpoint_t_max=0.875, seed=13)
        self.assertEqual(endpoint_ck['prediction'], 'endpoint')
        self.assertEqual(endpoint_ck['training']['target_space'], 'bimc_fused')
        self.assertTrue(torch.isfinite(torch.tensor(endpoint_ck['loss_history'])).all())
        end_bank = generate(endpoint_ck, self.cache, steps=8)
        self.assertEqual(end_bank['prediction'], 'endpoint')
        self.assertTrue(torch.isfinite(end_bank['prototypes']).all())
        self.assertTrue(torch.allclose(end_bank['prototypes'].norm(dim=-1),
                                       torch.ones(4), atol=1e-5))
        next_cache = dict(self.cache, class_ids=[50], text_prototypes=torch.ones(1, 4))
        combined = generate(endpoint_ck, next_cache, existing=end_bank)
        self.assertTrue(torch.equal(combined['prototypes'][:4], end_bank['prototypes']))
        with self.assertRaisesRegex(ValueError, 'different prediction/steps'):
            generate(self.ck, next_cache, existing=end_bank)
        with self.assertRaisesRegex(ValueError, 'different prediction/steps'):
            generate(endpoint_ck, next_cache, steps=4, existing=end_bank)
        with self.assertRaises(ValueError):
            fit(self.cache, epochs=1, blocks=1, prediction='endpoint', endpoint_t_max=1.0)
        with self.assertRaises(ValueError):
            fit(self.cache, epochs=1, blocks=1, prediction='unknown')

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


