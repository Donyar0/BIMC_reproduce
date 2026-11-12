"""Numerical and two-session clean-feature integration tests without real CLIP."""
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
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from single_flow.bimc_cache import from_task_statistics
from single_flow.unified_flow import real_means
from single_flow.image_text_flow import (fit, load_model, velocity_from_output, transport,
    build_bank, classify_bidirectional, METHOD, DIRECTION)

ROOT = Path(__file__).resolve().parents[1]


def load_nodes(path, predicate, namespace):
    nodes = [n for n in ast.parse(path.read_text()).body if predicate(n)]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)


def config():
    from yacs.config import CfgNode
    ns = {'CN':CfgNode}
    load_nodes(ROOT/'main.py',lambda n:isinstance(n,ast.FunctionDef) and n.name=='extend_cfg',ns)
    cfg=CfgNode(); ns['extend_cfg'](cfg)
    cfg.merge_from_file(str(ROOT/'configs/datasets/cifar100.yaml'))
    cfg.merge_from_file(str(ROOT/'configs/trainers/unified_flow.yaml'))
    return cfg


class NumericalTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(1)

    def test_endpoint_conversion_and_gradient_are_velocity_loss(self):
        x0=torch.tensor([[1.,0.,0.]])
        x1=torch.tensor([[0.,1.,0.]])
        t=torch.tensor([[.2]])
        xt=(1-t)*x0+t*x1
        self.assertTrue(torch.allclose(velocity_from_output(x1,xt,t,'endpoint',.05),x1-x0))
        prediction=(x1+torch.tensor([[.1,.2,.3]])).requires_grad_()
        loss=(velocity_from_output(prediction,xt,t,'endpoint',.05)-(x1-x0)).square().sum()
        expected=((prediction-x1)/(1-t)).square().sum()
        self.assertTrue(torch.allclose(loss,expected))
        loss.backward()
        self.assertTrue(torch.allclose(prediction.grad,2*(prediction.detach()-x1)/(1-t).square()))

    def test_forward_and_reverse_directions(self):
        class Constant:
            def __call__(self,x,t): return torch.tensor([[-1.,1.,0.]]).expand_as(x)
        x0=torch.tensor([[1.,0.,0.]])
        x1=torch.tensor([[0.,1.,0.]])
        for solver in ('euler', 'midpoint', 'heun', 'rk4'):
            self.assertTrue(torch.allclose(transport(Constant(),x0,8,solver=solver),x1,atol=1e-6))
            self.assertTrue(torch.allclose(transport(Constant(),x1,8,reverse=True,solver=solver),x0,atol=1e-6))
        bank=dict(class_ids=[0,1],text=torch.eye(3)[:2],prototypes=torch.eye(3)[:2])
        _,score=classify_bidirectional(Constant(),x0,bank,alpha=1,gamma=0,steps=8)
        self.assertTrue(torch.allclose(score,torch.tensor([[0.,1.]]),atol=1e-6))

    def test_rk4_accuracy_and_solver_cost(self):
        class TimeField:
            def __init__(self): self.calls=0
            def __call__(self,x,t):
                self.calls+=1
                return torch.cat((t,torch.zeros_like(t)),dim=1)
        source=torch.tensor([[0.,1.]])
        exact=F.normalize(torch.tensor([[.5,1.]]),dim=-1)
        costs={'euler':1,'midpoint':2,'heun':2,'rk4':4}
        for solver,cost in costs.items():
            field=TimeField()
            forward=transport(field,source,2,solver=solver)
            self.assertEqual(field.calls,2*cost)
            if solver!='euler':
                self.assertTrue(torch.allclose(forward,exact,atol=1e-6))
            field=TimeField()
            backward=transport(field,torch.tensor([[.5,1.]]),2,reverse=True,solver=solver)
            self.assertEqual(field.calls,2*cost)
            if solver!='euler':
                self.assertTrue(torch.allclose(backward,source,atol=1e-6))
        with self.assertRaisesRegex(ValueError,'SOLVER'):
            transport(TimeField(),source,2,solver='bad')

    def test_original_bimc_fusion(self):
        ns=dict(torch=torch,nn=torch.nn,F=F)
        load_nodes(ROOT/'models/bimc.py',lambda n:isinstance(n,ast.ClassDef),ns)
        model=ns['BiMC'].__new__(ns['BiMC']);torch.nn.Module.__init__(model)
        model.cfg=NS(TRAINER=NS(BiMC=NS(TEXT_CALIBRATION=True,LAMBDA_T=.5)))
        a,b=torch.eye(3)[:2],torch.eye(3)[1:]
        self.assertTrue(torch.equal(model.fused_text_prototype(a,b),.5*(a+b)))

    def test_modes_train_base_only_on_clean_features_and_stay_finite(self):
        text=F.normalize(torch.tensor([[1.,1.,0.,0.],[0.,0.,1.,1.]]),dim=-1)
        cache=dict(stage='base',split='train',encoder_id='test',class_ids=[0,1],
                   text_prototypes=text,image_features=torch.eye(4)[:2],
                   image_labels=torch.tensor([0,1]))
        for mode in ('velocity','endpoint'):
            with torch.enable_grad():
                ckpt=fit(cache,epochs=2,batch_size=2,lr=.001,weight_decay=0,blocks=1,
                    hidden=8,prediction=mode,time_eps=.05,seed=7,device='cpu')
            self.assertEqual(ckpt['training']['input_features'],'clean_base_images')
            self.assertTrue(torch.isfinite(torch.tensor(ckpt['loss_history'])).all())
            field=load_model(ckpt,'cpu').eval()
            self.assertTrue(torch.isfinite(field(torch.eye(4)[:2],torch.ones(2,1))).all())
            memory=dict(class_ids=[0,1],text=text,real=torch.eye(4)[:2],encoder_id='test')
            bank=build_bank(ckpt,memory,steps=8,num_base=2,beta_base=.1,beta_inc=.5,device='cpu')
            self.assertTrue(torch.isfinite(bank['prototypes']).all())
            invalid=dict(cache,stage='incremental')
            with self.assertRaisesRegex(ValueError,'base training'):
                fit(invalid,epochs=1,batch_size=2,lr=.001,weight_decay=0,blocks=1,
                    hidden=8,prediction=mode,time_eps=.05,seed=7,device='cpu')
        with self.assertRaisesRegex(ValueError,'Incompatible'):
            load_model(dict(ckpt,flow_direction='text_to_image'),'cpu')


class RunnerTests(unittest.TestCase):
    def test_solver_ablation_script_writes_comparable_summary(self):
        import sys
        from yacs.config import CfgNode
        from scripts import run_ode_solver_ablation
        with tempfile.TemporaryDirectory() as temp:
            run_dir=Path(temp)
            commands=[]
            cfg=CfgNode({'DATASET':{'NAME':'synthetic'},'FLOW':{
                'PREDICTION':'velocity','SOLVER':'euler','OUTPUT_DIR':'unused',
                'STEPS':4,'AUGMENT':{'ROTATION':15.}}})
            (run_dir/'config.yaml').write_text(cfg.dump())
            def fake_run(command, **kwargs):
                commands.append(command)
                with open(command[command.index('--train_cfg')+1]) as handle:
                    selected=CfgNode.load_cfg(handle)
                self.assertNotIn('AUGMENT', selected.FLOW)
                target=Path(selected.FLOW.OUTPUT_DIR)/'synthetic'/'eval_fake'
                target.mkdir(parents=True)
                (target/'metrics.json').write_text(json.dumps({
                    'final_acc':60.,'avg_acc':70.,'sessions':[{'elapsed_time':1.25}]}))
            argv=['script',str(run_dir),'--solvers','euler','rk4','--trusted_checkpoints']
            with patch.object(sys,'argv',argv), patch.object(run_ode_solver_ablation.subprocess,'run',side_effect=fake_run):
                run_ode_solver_ablation.main()
            with (run_dir/'ode_solver_ablation'/'summary.csv').open() as handle:
                rows=list(csv.DictReader(handle))
            self.assertEqual([row['solver'] for row in rows],['euler','rk4'])
            self.assertTrue(all(row['final_acc']=='60.0' for row in rows))
            self.assertTrue(all('--trusted_checkpoints' in command for command in commands))

    def test_solver_ablation_reports_child_failure_from_log(self):
        import sys
        from yacs.config import CfgNode
        from scripts import run_ode_solver_ablation
        with tempfile.TemporaryDirectory() as temp:
            run_dir=Path(temp)
            cfg=CfgNode({'DATASET':{'NAME':'synthetic'},'FLOW':{
                'PREDICTION':'velocity','SOLVER':'euler','OUTPUT_DIR':'unused','STEPS':4}})
            (run_dir/'config.yaml').write_text(cfg.dump())
            def fail(command, **kwargs):
                kwargs['stdout'].write('original checkpoint error\n')
                raise run_ode_solver_ablation.subprocess.CalledProcessError(1,command)
            argv=['script',str(run_dir),'--solvers','rk4']
            with patch.object(sys,'argv',argv), patch.object(run_ode_solver_ablation.subprocess,'run',side_effect=fail):
                with contextlib.redirect_stderr(io.StringIO()) as error_output:
                    with self.assertRaisesRegex(RuntimeError,'rk4 evaluation failed'):
                        run_ode_solver_ablation.main()
            self.assertIn('original checkpoint error',error_output.getvalue())

    def test_two_session_training_once_and_saved_evaluation_both_modes(self):
        torch.set_num_threads(1)
        eye=torch.eye(4)
        training_calls=[]
        def checked_fit(cache,**kwargs):
            training_calls.append(cache['stage'])
            self.assertTrue(torch.equal(cache['image_features'],eye[:2]))
            self.assertTrue(torch.is_grad_enabled())
            return fit(cache,**kwargs)
        ns=dict(Runner=object,torch=torch,np=np,Path=Path,datetime=datetime,time=time,
            csv=csv,json=json,
            tqdm=lambda x:x,from_task_statistics=from_task_statistics,fit=checked_fit,
            build_bank=build_bank,real_means=real_means,load_model=load_model,
            classify_bidirectional=classify_bidirectional,METHOD=METHOD,DIRECTION=DIRECTION,
            log_metrics=lambda *a,**kw:None)
        load_nodes(ROOT/'engine/unified_flow_engine.py',lambda n:isinstance(n,ast.ClassDef),ns)
        class Model(torch.nn.Module):
            def build_task_statistics(self,names,loader,class_index,**kw):
                ids=torch.as_tensor(class_index)
                return dict(class_index=class_index,text_proto=F.normalize(eye[ids]+.5,dim=-1),
                            image_proto=eye[ids],images_features=eye[ids],images_targets=ids)
            def extract_img_feature(self,images):
                if images.ndim==2: return images
                return torch.cat([images.mean((2,3)),torch.ones(len(images),1)],dim=-1)
        class Data:
            class_index_in_task=[np.array([0,1]),np.array([2])]
            class_names=['a','b','c']; num_tasks=2
            def get_dataloader(self,task_id,source,**kw):
                if source=='train':
                    ids=torch.as_tensor(self.class_index_in_task[task_id])
                    return [dict(image=eye[ids],label=ids)]
                ids=torch.arange(2 if task_id==0 else 3)
                return [dict(image=eye[ids],label=ids)]
        class Evaluator:
            def calc_accuracy(self,scores,targets,task_id):
                return dict(mean_acc=float((scores.argmax(-1)==targets).float().mean()*100),
                            task_acc=[],base_avg_acc=0.,inc_avg_acc=0.,harmonic_acc=0.)
        for mode in ('velocity','endpoint'):
            with tempfile.TemporaryDirectory() as temp:
                cfg=config();cfg.FLOW.PREDICTION=mode;cfg.FLOW.OUTPUT_DIR=temp
                cfg.DEVICE.DEVICE_NAME='cpu';cfg.DATALOADER.NUM_WORKERS=0
                for key,value in dict(EPOCHS=2,BATCH_SIZE=2,BLOCKS=1,HIDDEN=8,STEPS=4).items():
                    setattr(cfg.FLOW,key,value)
                runner=ns['UnifiedFlowRunner'](); runner.cfg=cfg;runner.model=Model()
                runner.device='cpu';runner.is_distributed=False;runner.data_manager=Data()
                runner.evaluator=Evaluator();runner.parse_batch=lambda b:(b['image'],b['label'])
                runner.acc_list=[];runner.task_acc_list=[]
                with contextlib.redirect_stdout(io.StringIO()):runner.run()
                run_dir=next(Path(temp).rglob('base_flow.pt')).parent
                base=torch.load(run_dir/'flow_session0.pt',weights_only=True)
                inc=torch.load(run_dir/'flow_session1.pt',weights_only=True)
                self.assertTrue(all(torch.equal(base['state_dict'][k],inc['state_dict'][k]) for k in base['state_dict']))
                bank0=torch.load(run_dir/'bank_session0.pt',weights_only=True)
                bank1=torch.load(run_dir/'bank_session1.pt',weights_only=True)
                self.assertTrue(torch.equal(bank0['prototypes'],bank1['prototypes'][:2]))
                self.assertTrue(torch.allclose(bank1['text'],F.normalize(eye[:3]+.5,dim=-1)))
                before=runner.acc_list.copy(); runner.acc_list=[];runner.task_acc_list=[]
                with contextlib.redirect_stdout(io.StringIO()):runner.evaluate_saved(run_dir)
                self.assertEqual(runner.acc_list,before)
                cfg.FLOW.SOLVER='rk4'
                runner.acc_list=[];runner.task_acc_list=[]
                with contextlib.redirect_stdout(io.StringIO()):runner.evaluate_saved(run_dir)
                self.assertEqual(len(runner.acc_list),2)
                cfg.FLOW.STEPS=8
                runner.acc_list=[];runner.task_acc_list=[]
                with contextlib.redirect_stdout(io.StringIO()):runner.evaluate_saved(run_dir)
                self.assertEqual(len(runner.acc_list),2)
                cfg.FLOW.PREDICTION='endpoint' if mode=='velocity' else 'velocity'
                with self.assertRaisesRegex(ValueError,'PREDICTION'):
                    runner.evaluate_saved(run_dir)
        self.assertEqual(training_calls,['base','base'])


if __name__=='__main__':unittest.main()
