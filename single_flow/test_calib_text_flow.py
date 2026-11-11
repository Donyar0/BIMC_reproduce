"""Synthetic integration checks without downloading CLIP or datasets."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import torch
import torch.nn.functional as F
from models.vdt_linear import LinearTextAdapter, pack_description_bank
from models.vdt_objective import linear_adapter_ce, support_feature_moments
from single_flow.bimc_cache import from_task_statistics
from single_flow.s1_flow import fit
from single_flow.unified_flow import adapt, build_bank, classify_bidirectional, load_model, real_means
import json


class CalibTextFlowTests(unittest.TestCase):
    def test_base_training_freezing_and_all_flow_consumers(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        path = Path(__file__).resolve().parents[1] / 'models/bimc.py'
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        ns = dict(torch=torch, nn=torch.nn, F=F, json=json,
                  LinearTextAdapter=LinearTextAdapter, pack_description_bank=pack_description_bank,
                  linear_adapter_ce=linear_adapter_ce, support_feature_moments=support_feature_moments)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), ns)
        model = ns['BiMC'].__new__(ns['BiMC'])
        torch.nn.Module.__init__(model)
        model.clip_model = torch.nn.Linear(4, 4)
        model.text_adapter = LinearTextAdapter(4, dropout=0)
        model.text_adapter_trained = False
        model.visual_adapter = None
        model.cfg = NS(DATASET=NS(GPT_PATH='', BETA=0.5), TRAINER=NS(BiMC=NS(
            VDT_EPOCHS=2, VDT_LR=0.01, VDT_WEIGHT_DECAY=0.,
            VDT_TRAIN_OBJECTIVE='text_ce', VDT_BATCH_SIZE=2)))
        eye = torch.eye(4)
        def descriptions(names, path, start):
            ids = torch.arange(start, start+len(names))
            return eye[ids].repeat_interleave(2, 0), ids.repeat_interleave(2)
        model.inference_vdt_features = descriptions
        def stats(ids):
            return dict(class_index=ids, text_proto=-eye[ids], image_proto=eye[ids],
                        images_features=eye[ids].roll(1, dims=-1), images_targets=torch.tensor(ids))
        model.eval().requires_grad_(False)
        before = {k:v.clone() for k,v in model.text_adapter.state_dict().items()}
        with torch.no_grad():
            base = model.calibrate_flow_text(stats([0,1]), ['a','b'], 0)
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in model.text_adapter.state_dict().items()))
        self.assertTrue(all(not p.requires_grad for p in model.text_adapter.parameters()))
        frozen = {k:v.clone() for k,v in model.text_adapter.state_dict().items()}
        novel = model.calibrate_flow_text(stats([2]), ['c'], 1)
        self.assertTrue(all(torch.equal(frozen[k],v) for k,v in model.text_adapter.state_dict().items()))
        for state in (base, novel):
            ids = torch.tensor(state['class_index'])
            packed, mask, _ = pack_description_bank(eye[ids].repeat_interleave(2,0), ids.repeat_interleave(2))
            self.assertTrue(torch.allclose(state['text_proto'], model.text_adapter(packed,mask)))
        cache = from_task_statistics(base, 'calibText-test', 0)
        ckpt = fit(cache, epochs=1, batch_size=2, blocks=1, hidden=8, target='sample')
        memory = dict(class_ids=cache['class_ids'], text=cache['text_prototypes'],
                      real=real_means(cache), encoder_id=ckpt['encoder_id'])
        bank = build_bank(ckpt, memory, steps=2, num_base=2, beta_base=.1, beta_inc=.5, device='cpu')
        new = from_task_statistics(novel, ckpt['encoder_id'], 1)
        ckpt = adapt(ckpt,new,memory,steps=1,batch_size=2,lr=.001,weight_decay=0,device='cpu')
        current = dict(class_ids=[2], text=new['text_prototypes'], real=real_means(new), encoder_id=ckpt['encoder_id'])
        bank = build_bank(ckpt,current,steps=2,num_base=0,beta_base=.1,beta_inc=.5,device='cpu',previous_bank=bank)
        expected = torch.cat([base['text_proto'],novel['text_proto']])
        self.assertTrue(torch.equal(bank['text'],expected))
        _, score = classify_bidirectional(load_model(ckpt,'cpu'),eye[:3],bank,alpha=0,gamma=0)
        self.assertTrue(torch.allclose(score,eye[:3] @ expected.T))

class UnifiedCalibRunnerTests(unittest.TestCase):
    def test_run_and_saved_evaluation_use_calibrated_bank(self):
        import contextlib
        import csv
        import io
        import tempfile
        import time
        from datetime import datetime
        import numpy as np
        from yacs.config import CfgNode
        root = Path(__file__).resolve().parents[1]
        def load_nodes(path, select, ns):
            nodes = [n for n in ast.parse(path.read_text()).body if select(n)]
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
        ns = {'CN':CfgNode}
        load_nodes(root/'main.py', lambda n:isinstance(n,ast.FunctionDef) and n.name=='extend_cfg', ns)
        cfg=CfgNode(); ns['extend_cfg'](cfg)
        cfg.merge_from_file(str(root/'configs/datasets/cifar100.yaml'))
        cfg.merge_from_file(str(root/'configs/trainers/unified_flow.yaml'))
        cfg.MODEL.BACKBONE.NAME='synthetic'; cfg.DEVICE.DEVICE_NAME='cpu'
        for key,value in dict(EPOCHS=1,BATCH_SIZE=2,BLOCKS=1,HIDDEN=8,STEPS=2,ADAPT_STEPS=1).items():
            setattr(cfg.FLOW,key,value)
        ns=dict(Runner=object,torch=torch,np=np,Path=Path,datetime=datetime,time=time,csv=csv,json=json,
                tqdm=lambda x:x,from_task_statistics=from_task_statistics,fit=fit,adapt=adapt,
                build_bank=build_bank,real_means=real_means,load_model=load_model,
                classify_bidirectional=classify_bidirectional,log_metrics=lambda *a,**kw:None)
        load_nodes(root/'engine/unified_flow_engine.py',lambda n:isinstance(n,ast.ClassDef),ns)
        eye=torch.eye(4); calls=[]
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.text_adapter=torch.nn.Linear(4,4)
            def build_task_statistics(self,names,loader,class_index,**kw):
                ids=torch.as_tensor(class_index)
                return dict(class_index=class_index,text_proto=-eye[ids],image_proto=eye[ids],
                            images_features=eye[ids],images_targets=ids)
            def calibrate_flow_text(self,stats,names,task_id):
                calls.append(task_id)
                return dict(stats,text_proto=eye[torch.as_tensor(stats['class_index'])])
            def extract_img_feature(self,images): return images
        class Data:
            class_index_in_task=[np.array([0,1]),np.array([2])]
            class_names=['a','b','c']; num_tasks=2
            def get_dataloader(self,task_id,source,**kw):
                ids=torch.arange(2 if task_id==0 else 3)
                return [dict(image=eye[ids],label=ids)]
        class Evaluator:
            def calc_accuracy(self,scores,targets,task_id):
                return dict(mean_acc=float((scores.argmax(-1)==targets).float().mean()*100),
                            task_acc=[],base_avg_acc=0.,inc_avg_acc=0.,harmonic_acc=0.)
        with tempfile.TemporaryDirectory() as temp:
            cfg.FLOW.OUTPUT_DIR=temp
            runner=ns['UnifiedFlowRunner'](); runner.cfg=cfg; runner.model=Model()
            runner.device='cpu'; runner.is_distributed=False; runner.data_manager=Data()
            runner.evaluator=Evaluator(); runner.parse_batch=lambda b:(b['image'],b['label'])
            runner.acc_list=[]; runner.task_acc_list=[]
            with contextlib.redirect_stdout(io.StringIO()): runner.run()
            self.assertEqual(calls,[0,1])
            run_dir=next(Path(temp).rglob('base_flow.pt')).parent
            self.assertTrue((run_dir/'calib_text.pt').exists())
            bank=torch.load(run_dir/'bank_session1.pt',weights_only=True)
            self.assertTrue(torch.equal(bank['text'],eye[:3]))
            ckpt=torch.load(run_dir/'flow_session1.pt',weights_only=True)
            self.assertEqual(ckpt['text_prototype_method'],'calibText-linear-v1')
            before=runner.acc_list.copy(); runner.acc_list=[]; runner.task_acc_list=[]
            with contextlib.redirect_stdout(io.StringIO()): runner.evaluate_saved(run_dir)
            self.assertEqual(calls,[0,1])
            self.assertEqual(runner.acc_list,before)
            ckpt.pop('text_prototype_method'); torch.save(ckpt,run_dir/'flow_session1.pt')
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError,'legacy text'):
                runner.evaluate_saved(run_dir)


if __name__ == '__main__':
    unittest.main()
