"""Evaluation-only logit adjustment tests without importing heavy CLIP backbones."""
import ast
import json
import math
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import torch

ROOT = Path(__file__).resolve().parents[1]
source = ast.parse((ROOT / 'engine/visual_to_text_engine.py').read_text())
funcs = [n for n in source.body if isinstance(n, ast.FunctionDef)
         and n.name in ('validate_logit_gammas', 'subtract_base_logit_bias')]
ns = {'math': math}
exec(compile(ast.Module(body=funcs, type_ignores=[]), 'engine/visual_to_text_engine.py', 'exec'), ns)
validate = ns['validate_logit_gammas']
adjust = ns['subtract_base_logit_bias']


class LogitAdjustmentTests(unittest.TestCase):
    def test_only_base_columns_change(self):
        x = torch.tensor([[0.9, 0.8, 0.85, 0.7],
                          [0.1, 0.4, 0.2, 0.3]])
        y = adjust(x, base_count=2, gamma=0.1)
        torch.testing.assert_close(y[:, :2], x[:, :2] - 0.1)
        torch.testing.assert_close(y[:, 2:], x[:, 2:])
        torch.testing.assert_close(x[0], torch.tensor([0.9, 0.8, 0.85, 0.7]))
        self.assertEqual(int(x[0].argmax()), 0)
        self.assertEqual(int(y[0].argmax()), 2)
        self.assertIs(adjust(x, 2, 0.), x)

    def test_gamma_validation_and_no_novel_early_adjustment(self):
        self.assertEqual(validate([0, 0.005, 0.01, 0.1]), [0.0, 0.005, 0.01, 0.1])
        for vals in [[], [-0.1], [float('nan')], [float('inf')], ['0.01'], [False], [0.1, 0.1]]:
            with self.assertRaises(ValueError):
                validate(vals)
        logits = torch.tensor([[0.3, 0.2]])
        torch.testing.assert_close(adjust(logits, 2, 0.1),
                                   logits - 0.1)
        with self.assertRaises(ValueError):
            adjust(logits, 0, 0.1)

    def test_eval_config_is_separate_and_disables_training(self):
        from yacs.config import CfgNode
        tree = ast.parse((ROOT / 'main.py').read_text())
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('extend_cfg', 'setup_cfg')]
        cfg_ns = {'CN': CfgNode}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), 'main.py', 'exec'), cfg_ns)
        cfg = cfg_ns['setup_cfg'](
            str(ROOT / 'configs/datasets/cifar100.yaml'),
            str(ROOT / 'configs/trainers/visual_to_text_logit_adjust.yaml'))
        self.assertTrue(cfg.FLOW.EVAL_ONLY)
        self.assertFalse(cfg.FLOW.META.ENABLED)
        self.assertFalse(cfg.FLOW.INC.ENABLED)
        self.assertEqual(cfg.FLOW.LOGIT_GAMMAS, [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1])


if __name__ == '__main__':
    unittest.main()
