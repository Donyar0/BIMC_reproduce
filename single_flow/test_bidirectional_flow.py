"""Synthetic checks for bidirectional prototype consistency and both sessions."""
import ast
import contextlib
import csv
import io
import json
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import torch
import torch.nn.functional as F

from single_flow.bimc_cache import from_task_statistics
from single_flow.bidirectional_flow import (
    SCORE_NAMES, integrate, prototype_contrast, fit_base, adapt,
    build_bank, load_model, score_three,
)

ROOT = Path(__file__).resolve().parents[1]


def runner_class(namespace):
    source = ast.parse((ROOT/'engine/bidirectional_flow_engine.py').read_text())
    node = next(n for n in source.body if isinstance(n,ast.ClassDef)
                and n.name=='BidirectionalFlowRunner')
    exec(compile(ast.Module(body=[node],type_ignores=[]),'bidirectional_flow_engine.py','exec'),namespace)
    return namespace['BidirectionalFlowRunner']


class BidirectionalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.eye = torch.eye(4)
        self.cache = dict(stage='base',split='train',encoder_id='synthetic',
                          class_ids=[0,1],text_prototypes=self.eye[:2],
                          image_features=self.eye[:2].roll(1,1),
                          image_labels=torch.tensor([0,1]))

    def test_reverse_integral_and_contrastive_pairs(self):
        class Constant(torch.nn.Module):
            def forward(self,x,t):
                return torch.tensor([[.2, .1, 0., 0.]],device=x.device).expand_as(x)
        model=Constant()
        source=self.eye[:2]
        target=integrate(model,source,steps=4)
        self.assertTrue(torch.allclose(integrate(model,target,4,reverse=True),source,atol=1e-6))
        matching=prototype_contrast(model,source,source,4,.07)
        mismatched=prototype_contrast(model,source,source.flip(0),4,.07)
        self.assertLess(matching,mismatched)

    def test_base_incremental_bank_and_three_scores(self):
        with contextlib.redirect_stdout(io.StringIO()):
            checkpoint=fit_base(self.cache,epochs=2,batch_size=2,lr=.01,
                weight_decay=0,blocks=1,hidden=8,contrast_weight=.2,
                temperature=.07,contrast_steps=2,seed=3,device='cpu')
        self.assertEqual(len(checkpoint['loss_history']),2)
        self.assertTrue(all(np.isfinite(row['contrast']) for row in checkpoint['loss_history']))
        first=build_bank(checkpoint,self.cache,steps=2,device='cpu')
        novel=dict(self.cache,stage='incremental',class_ids=[2,3],
                   text_prototypes=self.eye[2:],
                   image_features=self.eye[2:].roll(1,1),
                   image_labels=torch.tensor([2,3]))
        frozen=build_bank(checkpoint,novel,steps=2,device='cpu',existing=first)
        self.assertTrue(torch.equal(first['generated'],frozen['generated'][:2]))
        with contextlib.redirect_stdout(io.StringIO()):
            updated=adapt(checkpoint,novel,epochs=2,batch_size=2,lr=.01,
                          weight_decay=0,seed=4,device='cpu')
        self.assertEqual(updated['adaptation_history'][0]['class_ids'],[2,3])
        self.assertTrue(any(not torch.equal(v,updated['state_dict'][k])
                            for k,v in checkpoint['state_dict'].items()))
        adapted_bank=build_bank(updated,novel,steps=2,device='cpu',existing=first,
                                regenerate_existing=True)
        expected=F.normalize(integrate(load_model(updated,'cpu'),
                                       adapted_bank['text'],2),dim=-1)
        self.assertTrue(torch.allclose(adapted_bank['generated'],expected,atol=1e-6))
        model=load_model(updated,'cpu')
        scores=score_three(model,self.eye[:1].half(),adapted_bank,steps=2)
        self.assertEqual(set(scores),set(SCORE_NAMES)|{'fused'})
        self.assertTrue(torch.allclose(scores['fused'],sum(scores[n] for n in SCORE_NAMES)/3))
        reverse=F.normalize(integrate(model,self.eye[:1],2,reverse=True),dim=-1)
        self.assertTrue(torch.allclose(scores['reverse_to_text'],
                                       reverse@adapted_bank['text'].T,atol=1e-6))
        with self.assertRaisesRegex(ValueError,'incremental'):
            adapt(checkpoint,self.cache,epochs=1,batch_size=2,lr=.01,
                  weight_decay=0,seed=1,device='cpu')

    def test_runner_frozen_and_incremental(self):
        calls=[]
        def checked_fit(*args,**kwargs):
            calls.append('base')
            self.assertTrue(torch.is_grad_enabled())
            return fit_base(*args,**kwargs)
        def checked_adapt(*args,**kwargs):
            calls.append('adapt')
            self.assertTrue(torch.is_grad_enabled())
            return adapt(*args,**kwargs)
        namespace=dict(Runner=object,torch=torch,np=np,Path=Path,datetime=datetime,
                       time=time,csv=csv,json=json,tqdm=lambda x:x,
                       from_task_statistics=from_task_statistics,
                       SCORE_NAMES=SCORE_NAMES,fit_base=checked_fit,adapt=checked_adapt,
                       build_bank=build_bank,load_model=load_model,score_three=score_three,
                       log_metrics=lambda *a,**kw:None)
        RunnerType=runner_class(namespace)
        eye=self.eye
        class Model(torch.nn.Module):
            def build_task_statistics(self,names,loader,class_index,**kwargs):
                ids=torch.as_tensor(class_index)
                return dict(class_index=class_index,text_proto=eye[ids],
                            image_proto=eye[ids].roll(1,1),fused_proto=eye[ids],
                            images_features=eye[ids].roll(1,1),images_targets=ids)
            def extract_img_feature(self,images):return images
        class Data:
            class_index_in_task=[np.array([0,1]),np.array([2,3])]
            class_names=['a','b','c','d'];num_tasks=2
            def get_dataloader(self,task_id,source,**kwargs):
                ids=torch.arange(2 if task_id==0 else 4)
                return [dict(image=eye[ids],label=ids)]
        class Eval:
            def calc_accuracy(self,values,labels,task):
                return dict(mean_acc=float((values.argmax(-1)==labels).float().mean()*100),
                            task_acc=[],base_avg_acc=0.,inc_avg_acc=0.,harmonic_acc=0.)
        with tempfile.TemporaryDirectory() as temp:
            for adapt_enabled in (False,True):
                cfg=NS(SEED=1,DATASET=NS(NAME='synthetic'),MODEL=NS(BACKBONE=NS(NAME='fake')),
                    TRAINER=NS(BiMC=NS(PREC='fp32',TEXT_CALIBRATION=True,
                                        VISION_CALIBRATION=False)),
                    FLOW=NS(OUTPUT_DIR=temp,EPOCHS=1,BATCH_SIZE=2,LR=.01,
                            WEIGHT_DECAY=0.,BLOCKS=1,HIDDEN=8,STEPS=2,
                            CONTRAST_WEIGHT=.1,CONTRAST_TEMPERATURE=.07,
                            CONTRAST_STEPS=2,ADAPT=adapt_enabled,ADAPT_EPOCHS=1,
                            ADAPT_BATCH_SIZE=2,ADAPT_LR=.01,SCORE_MODE='fused',
                            SCORE_WEIGHTS=[1.,1.,1.]),dump=lambda:'fake: true\n')
                runner=RunnerType();runner.cfg=cfg;runner.model=Model();runner.device='cpu'
                runner.is_distributed=False;runner.data_manager=Data();runner.evaluator=Eval()
                runner.parse_batch=lambda batch:(batch['image'],batch['label'])
                runner.acc_list=[];runner.task_acc_list=[]
                with contextlib.redirect_stdout(io.StringIO()):runner.run()
                results=[p for p in Path(temp).rglob('metrics.json')]
                run_dir=max(results,key=lambda p:p.stat().st_mtime_ns).parent
                metrics=json.loads((run_dir/'metrics.json').read_text())
                self.assertEqual(metrics['adapt'],adapt_enabled)
                self.assertEqual(len(metrics['sessions']),2)
                self.assertIn('reverse_to_generated_acc',metrics['sessions'][1])
                bank=torch.load(run_dir/'bank_session1.pt',weights_only=True)
                self.assertEqual(bank['class_ids'],[0,1,2,3])
                self.assertEqual((run_dir/'flow_session1.pt').exists(),adapt_enabled)
            self.assertEqual(calls,['base','base','adapt'])

    def test_both_experiment_configs(self):
        from yacs.config import CfgNode
        source=ast.parse((ROOT/'main.py').read_text())
        nodes=[n for n in source.body if isinstance(n,ast.FunctionDef)
               and n.name in ('extend_cfg','setup_cfg')]
        namespace={'CN':CfgNode}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'main.py','exec'),namespace)
        for mode,enabled in [('frozen',False),('incremental',True)]:
            cfg=namespace['setup_cfg'](str(ROOT/'configs/datasets/cub200.yaml'),
                str(ROOT/f'configs/trainers/bidirectional_flow_{mode}.yaml'))
            self.assertEqual(cfg.METHOD,'bidirectional_single_flow')
            self.assertEqual(cfg.FLOW.ADAPT,enabled)
            self.assertEqual(cfg.FLOW.SCORE_WEIGHTS,[1.,1.,1.])
            self.assertTrue(cfg.TRAINER.BiMC.TEXT_CALIBRATION)


if __name__=='__main__':unittest.main()
