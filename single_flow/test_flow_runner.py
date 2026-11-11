"""Exercise actual FlowRunner orchestration with synthetic CLIP/data only."""
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

from s1_flow import fit, generate, classify
from bimc_cache import from_task_statistics

ROOT = Path(__file__).resolve().parents[1]


def load_class(path, name, namespace):
    module = ast.parse(path.read_text())
    cls = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == name)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


class RunnerTests(unittest.TestCase):
    def test_full_two_session_run(self):
        torch.set_num_threads(1)
        evaluator = load_class(ROOT/'utils/evaluator.py', 'AccuracyEvaluator',
                               {'torch':torch, 'np':np})
        training_calls = []
        def checked_fit(cache, **kwargs):
            training_calls.append(cache['stage'])
            self.assertTrue(torch.is_grad_enabled())
            return fit(cache, **kwargs)
        namespace = dict(Runner=object, torch=torch, np=np, Path=Path, datetime=datetime,
                         time=time, csv=csv, json=json, tqdm=lambda x:x,
                         from_task_statistics=from_task_statistics, fit=checked_fit,
                         generate=generate, classify=classify, log_metrics=lambda *a,**k:None)
        runner_type = load_class(ROOT/'engine/flow_engine.py','FlowRunner',namespace)
        eye = torch.eye(3)
        class Model(torch.nn.Module):
            def build_task_statistics(self,names,loader,class_index,**kwargs):
                ids = torch.as_tensor(class_index)
                return dict(class_index=class_index,text_proto=eye[ids],image_proto=eye[ids],
                            fused_proto=eye[ids],
                            images_features=eye[ids],images_targets=ids)
            def extract_img_feature(self,images):
                return images
        class Data:
            class_index_in_task = [np.array([0,1]),np.array([2])]
            class_names = ['a','b','c']
            num_tasks = 2
            def get_dataloader(self,task_id,source,**kwargs):
                ids = torch.arange(2 if task_id == 0 else 3)
                return [dict(image=eye[ids],label=ids)]
        with tempfile.TemporaryDirectory() as directory:
            runner = runner_type()
            runner.cfg = NS(SEED=1,DATASET=NS(NAME='synthetic'),MODEL=NS(BACKBONE=NS(NAME='synthetic')),
                TRAINER=NS(BiMC=NS(PREC='fp32',VISION_CALIBRATION=True,PROTOTYPE_CACHE_DIR='')),
                FLOW=NS(OUTPUT_DIR=directory,EPOCHS=2,BATCH_SIZE=4,LR=0.001,
                        WEIGHT_DECAY=0.,BLOCKS=1,HIDDEN=8,TARGET='prototype',
                        PREDICTION='velocity',ENDPOINT_T_MAX=0.875,CE_WEIGHT=0.,
                        TEMPERATURE=0.07,STEPS=2),dump=lambda:'synthetic: true\n')
            runner.model=Model(); runner.device='cpu'; runner.is_distributed=False
            runner.data_manager=Data(); runner.evaluator=evaluator(Data.class_index_in_task)
            runner.acc_list=[]; runner.task_acc_list=[]
            runner.parse_batch=lambda batch:(batch['image'],batch['label'])
            with contextlib.redirect_stdout(io.StringIO()) as stream:
                runner.run()
            self.assertEqual(training_calls,['base'])
            self.assertIn('Final acc:',stream.getvalue())
            self.assertIn('Task-wise acc:',stream.getvalue())
            result_path=next(Path(directory).rglob('metrics.json'))
            summary=json.loads(result_path.read_text())
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['completed_sessions'],2)
            self.assertEqual(summary['final_acc'],100.)
            with (result_path.parent/'metrics.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),2)
            bank0=torch.load(result_path.parent/'bank_session0.pt',weights_only=True)
            bank1=torch.load(result_path.parent/'bank_session1.pt',weights_only=True)
            self.assertTrue(torch.equal(bank0['prototypes'],bank1['prototypes'][:2]))
            self.assertEqual(bank0['prediction'], 'velocity')
            ck = torch.load(result_path.parent/'base_flow.pt',weights_only=True)
            self.assertEqual(ck['prediction'], 'velocity')

    def test_old_classes_compete_with_new_classes(self):
        evaluator=load_class(ROOT/'utils/evaluator.py','AccuracyEvaluator',{'torch':torch,'np':np})
        # A base example is predicted as the novel class: it must count as wrong.
        metrics=evaluator([[0,1],[2]]).calc_accuracy(
            torch.tensor([[0.,0.,1.],[0.,1.,0.],[0.,0.,1.]]),torch.tensor([0,1,2]),1)
        self.assertEqual(metrics['mean_acc'],66.67)
        self.assertEqual(metrics['base_avg_acc'],50.)
        self.assertEqual(metrics['inc_avg_acc'],100.)

    def test_all_dataset_configs(self):
        source=ast.parse((ROOT/'main.py').read_text())
        functions=[n for n in source.body if isinstance(n,ast.FunctionDef)
                   and n.name in ('extend_cfg','setup_cfg')]
        # Use real YACS validation, without importing optional logging/vision libs.
        from yacs.config import CfgNode
        namespace={'CN':CfgNode}
        exec(compile(ast.Module(body=functions,type_ignores=[]),'main.py','exec'),namespace)
        for dataset, sessions in [('cifar100',9),('cub200',11),('miniimagenet',9)]:
            for trainer in ['single_flow','single_flow_endpoint','bimc']:
                cfg=namespace['setup_cfg'](str(ROOT/f'configs/datasets/{dataset}.yaml'),
                                          str(ROOT/f'configs/trainers/{trainer}.yaml'))
                self.assertEqual(1+(cfg.DATASET.NUM_CLASSES-cfg.DATASET.NUM_INIT_CLS)//cfg.DATASET.NUM_INC_CLS,sessions)
                if trainer.startswith('single_flow'):
                    self.assertEqual(cfg.METHOD,'single_flow')
                    self.assertEqual(cfg.FLOW.PREDICTION,
                                     'endpoint' if trainer == 'single_flow_endpoint' else 'velocity')


if __name__ == '__main__':
    unittest.main()
