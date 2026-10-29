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
from unittest.mock import patch

import numpy as np
import torch

from single_flow.visual_to_text import (
    fit, build_text_bank, load_flow, load_checkpoint, classify_alphas, validate_alphas,
)
from engine.visual_to_text_engine import validate_logit_gammas, subtract_base_logit_bias

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
                         build_text_bank=build_text_bank, load_flow=load_flow,
                         load_checkpoint=load_checkpoint, classify_alphas=classify_alphas,
                         validate_alphas=validate_alphas,
                         validate_logit_gammas=validate_logit_gammas,
                         subtract_base_logit_bias=subtract_base_logit_bias,
                         log_metrics=lambda *a,**k:None)
        runner_type = load_class(ROOT/'engine/visual_to_text_engine.py','VisualToTextRunner',namespace)
        eye = torch.eye(3)
        class Model(torch.nn.Module):
            template = ['a photo of {}']
            def inference_text_feature(self,names,template,begin):
                ids=torch.arange(begin,begin+len(names))
                return eye[ids],ids
            def inference_all_img_feature(self,loader,begin):
                batch=loader[0]
                return batch['image'],batch['label'],batch['image']
            def extract_img_feature(self,images):
                return images
        class Data:
            read_only = False
            class_index_in_task = [np.array([0,1]),np.array([2])]
            class_names = ['a','b','c']
            num_tasks = 2
            def get_dataloader(self,task_id,source,**kwargs):
                if self.read_only and source == 'train':
                    raise AssertionError('Checkpoint evaluation must skip training data')
                if source == 'train' and task_id > 0:
                    raise AssertionError('Incremental support images must not be read')
                ids = torch.arange(2 if task_id == 0 else 3)
                return [dict(image=eye[ids],label=ids)]
        with tempfile.TemporaryDirectory() as directory:
            runner = runner_type()
            runner.cfg = NS(SEED=1,DATASET=NS(NAME='synthetic'),MODEL=NS(BACKBONE=NS(NAME='synthetic')),
                TRAINER=NS(BiMC=NS(PREC='fp32',TEXT_CALIBRATION=False,PROTOTYPE_CACHE_DIR='')),
                FLOW=NS(OUTPUT_DIR=directory,EPOCHS=2,BATCH_SIZE=4,LR=0.001,
                        WEIGHT_DECAY=0.,BLOCKS=1,HIDDEN=8,SOURCE='sample',CE_WEIGHT=0.,
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

            # Reuse exactly that checkpoint and scan all strengths in one pass.
            runner.cfg.FLOW.CHECKPOINT = str(result_path.parent/'base_flow.pt')
            runner.cfg.FLOW.ALPHAS = [0., 0.1, 0.2, 0.5, 1.]
            runner.data_manager.read_only = True
            before = load_checkpoint(runner.cfg.FLOW.CHECKPOINT)
            from single_flow import visual_to_text as vtt
            with patch.object(runner.model, 'extract_img_feature', wraps=runner.model.extract_img_feature) as extract, \
                    patch.object(vtt, 'transform_images', wraps=vtt.transform_images) as transform, \
                    contextlib.redirect_stdout(io.StringIO()) as stream:
                runner.run()
            self.assertIn('Training skipped', stream.getvalue())
            self.assertEqual(training_calls, ['base'])
            self.assertEqual(extract.call_count, 2)
            self.assertEqual(transform.call_count, 2)
            after = load_checkpoint(runner.cfg.FLOW.CHECKPOINT)
            self.assertTrue(all(torch.equal(v, after['state_dict'][k]) for k, v in before['state_dict'].items()))
            scanned_path = next(p for p in Path(directory).rglob('metrics.json') if p != result_path)
            scanned = json.loads(scanned_path.read_text())
            self.assertEqual(scanned['completed_sessions'], 2)
            self.assertTrue(scanned['complete'])
            self.assertEqual(len(scanned['by_alpha']), 5)
            self.assertEqual(len(scanned['sessions']), 10)
            self.assertNotIn('final_acc', scanned)  # no ambiguous combined accuracy
            for alpha in ['0.0', '1.0']:
                self.assertEqual(scanned['by_alpha'][alpha]['final_acc'], 100.)
            with (scanned_path.parent/'metrics.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 10)
            task1 = next(r for r in scanned['sessions'] if r['task_id'] == 1 and r['alpha'] == 0)
            self.assertEqual(task1['novel_only_acc'], 100.)
            self.assertEqual(task1['current_only_acc'], 100.)
            self.assertEqual(task1['novel_to_base_pct'], 0.)
            self.assertFalse((scanned_path.parent/'base_flow.pt').exists())

            # Missing checkpoint and incompatible split must fail, never retrain.
            runner.cfg.SEED = 2
            with self.assertRaisesRegex(ValueError, 'seed'), contextlib.redirect_stdout(io.StringIO()):
                runner.run()
            runner.cfg.SEED = 1
            runner.cfg.FLOW.CHECKPOINT = str(Path(directory)/'missing.pt')
            with self.assertRaises(FileNotFoundError), contextlib.redirect_stdout(io.StringIO()):
                runner.run()
            self.assertEqual(training_calls, ['base'])

    def test_diagnostics_do_not_replace_all_seen_class_scores(self):
        runner_type = load_class(ROOT/'engine/visual_to_text_engine.py', 'VisualToTextRunner',
                                 dict(Runner=object, torch=torch))
        runner = runner_type()
        runner.data_manager = NS(class_index_in_task=[[0, 1], [2, 3], [4, 5]])
        labels = torch.tensor([2, 3, 4, 5])
        # Every novel example loses to a base score, but novel-only is perfect.
        logits = torch.eye(6)[labels] * 0.8
        logits[:, 0] = 0.95
        acc = runner.novel_diagnostics(logits, labels, 2)
        self.assertEqual(acc['novel_only_acc'], 100.)
        self.assertEqual(acc['current_only_acc'], 100.)
        self.assertEqual(acc['novel_samples'], 4)
        self.assertEqual(acc['current_samples'], 2)
        self.assertEqual(acc['novel_to_base_pct'], 100.)
        self.assertTrue((logits.argmax(1) == 0).all())
        self.assertEqual(acc['current_class_acc'], {'4': 0., '5': 0.})
        empty = runner.novel_diagnostics(torch.ones(1, 6), torch.tensor([0]), 2)
        self.assertIsNone(empty['novel_only_acc'])
        self.assertIsNone(empty['current_only_acc'])
        json.dumps(empty, allow_nan=False)

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
            for trainer in ['visual_to_text','single_flow','bimc']:
                cfg=namespace['setup_cfg'](str(ROOT/f'configs/datasets/{dataset}.yaml'),
                                          str(ROOT/f'configs/trainers/{trainer}.yaml'))
                self.assertEqual(1+(cfg.DATASET.NUM_CLASSES-cfg.DATASET.NUM_INIT_CLS)//cfg.DATASET.NUM_INC_CLS,sessions)
                if trainer=='single_flow':
                    self.assertEqual(cfg.METHOD,'single_flow')
                if trainer=='visual_to_text':
                    self.assertEqual(cfg.FLOW.ALPHAS, [0., 0.1, 0.2, 0.5, 1.])
                    self.assertEqual(cfg.FLOW.CHECKPOINT, '')
                    self.assertEqual(cfg.FLOW.LOGIT_GAMMAS, [0.0])


if __name__ == '__main__':
    unittest.main()


