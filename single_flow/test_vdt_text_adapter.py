"""Synthetic CPU tests: T2V Flow uses VDT text both at x0 and in fused x1."""
import contextlib
import io
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from models.vdt_linear import LinearTextAdapter
from single_flow.vdt_text_adapter import (
    train_base_text_adapter, adapted_prototypes,
    replace_flow_cache_text, fingerprint,
)
from single_flow.s1_flow import fit, generate


class TextToVisualAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.desc = F.normalize(torch.randn(8, 8), dim=-1)
        self.desc_ids = torch.tensor([0,0,0,1,1,2,2,2])
        self.images = F.normalize(torch.randn(12, 8), dim=-1)
        self.labels = torch.tensor([0]*4+[1]*4+[2]*4)
        self.cfg = SimpleNamespace(
            RATIO=0.2,DROPOUT=0.1,EPOCHS=2,BATCH_SIZE=4,
            LR=1e-3,WEIGHT_DECAY=1e-4)

    def test_full_adapted_source_target_and_new_class(self):
        with contextlib.redirect_stdout(io.StringIO()):
            adapter = train_base_text_adapter(self.desc, self.desc_ids,
                self.images,self.labels,[0,1,2],self.cfg,1,'cpu')
        self.assertIsInstance(adapter, LinearTextAdapter)
        self.assertFalse(adapter.training)
        self.assertTrue(all(not p.requires_grad for p in adapter.parameters()))
        text = adapted_prototypes(adapter,self.desc,self.desc_ids,[0,1,2])
        cache=dict(stage='base',split='train',encoder_id='synthetic',
                   class_ids=[0,1,2],
                   visual_prototypes=F.normalize(torch.randn(3,8),dim=-1))
        updated=replace_flow_cache_text(cache,text,beta=0.65)
        torch.testing.assert_close(updated['text_prototypes'],text)
        torch.testing.assert_close(updated['fused_prototypes'],
            F.normalize(.65*text+.35*cache['visual_prototypes'],dim=-1))
        self.assertNotIn('text_prototypes',cache)
        updated['text_adapter_sha256']=fingerprint(adapter)
        with contextlib.redirect_stdout(io.StringIO()):
            ck=fit(updated,epochs=2,blocks=1,hidden=8,seed=1)
        ck['text_adapter_sha256']=fingerprint(adapter)
        self.assertEqual(ck['flow_target'],'bimc_fused')
        self.assertNotIn('prediction',ck)  # original velocity-network predictor
        bank=generate(ck,updated,steps=2)
        novel_desc=F.normalize(torch.randn(3,8),dim=-1)
        novel_text=adapted_prototypes(adapter,novel_desc,
                                      torch.tensor([3,3,3]),[3])
        novel=replace_flow_cache_text(dict(encoder_id='synthetic',class_ids=[3],
            visual_prototypes=F.normalize(torch.randn(1,8),dim=-1)),novel_text,.65)
        novel['text_adapter_sha256']=fingerprint(adapter)
        merged=generate(ck,novel,steps=2,existing=bank)
        torch.testing.assert_close(bank['prototypes'],merged['prototypes'][:3])
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            generate(ck,dict(novel,text_adapter_sha256='wrong'),steps=2)
        with self.assertRaisesRegex(ValueError,'fingerprint'):
            generate(ck,dict(novel,text_adapter_sha256='wrong'),steps=2,existing=bank)

    def test_visual_endpoints_keep_adapted_source(self):
        from single_flow.s1_flow import text_cache
        with contextlib.redirect_stdout(io.StringIO()):
            adapter = train_base_text_adapter(self.desc, self.desc_ids,
                self.images, self.labels, [0,1,2], self.cfg, 1, 'cpu')
        text=adapted_prototypes(adapter,self.desc,self.desc_ids,[0,1,2])
        base=dict(stage='base',split='train',encoder_id='synthetic',
                  class_ids=[0,1,2],visual_prototypes=F.normalize(torch.randn(3,8),dim=-1),
                  image_features=self.images,image_labels=self.labels)
        cache=replace_flow_cache_text(base,text,.65)
        cache['text_adapter_sha256']=fingerprint(adapter)
        torch.testing.assert_close(text_cache(cache)[1],text.cpu())
        for target,endpoint in [('prototype','visual_prototype'),('sample','visual_sample')]:
            with contextlib.redirect_stdout(io.StringIO()):
                ck=fit(cache,epochs=2,blocks=1,hidden=8,seed=3,
                       target=target,endpoint=endpoint)
            self.assertEqual(ck['flow_target'],endpoint)
            self.assertNotIn('prediction',ck)  # unchanged velocity prediction

    def test_invalid_beta_and_class_ids(self):
        cache=dict(class_ids=[0],visual_prototypes=torch.ones(1,8))
        with self.assertRaises(ValueError):
            replace_flow_cache_text(cache,torch.ones(1,8),1.2)
        with self.assertRaises(ValueError):
            replace_flow_cache_text(cache,torch.ones(2,8),.5)


if __name__=='__main__':
    unittest.main()
