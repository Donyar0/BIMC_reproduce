"""Focused CPU tests: python -m unittest discover -s tests -v"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from argparse import Namespace

import torch

from models.vdt_self_attention import (
    SelfAttentionTextAdapter, LinearTextAdapter, MLPTextAdapter,
)
from utils.evaluator import AccuracyEvaluator
from scripts.run_vdt_followup import build_plan, parse_bias, summarize


class TestFollowup(unittest.TestCase):
    def test_adapter_parameter_matching_and_gradients(self):
        # D=8 suffices for equality, which also holds for CLIP D=512.
        torch.manual_seed(1)
        modes = [
            SelfAttentionTextAdapter(embed_dim=8, reduction=4, ratio=0.2, dropout=0.),
            LinearTextAdapter(embed_dim=8, ratio=0.2, dropout=0.),
            MLPTextAdapter(embed_dim=8, ratio=0.2, dropout=0.),
        ]
        counts = [sum(x.numel() for x in m.parameters()) for m in modes]
        self.assertEqual(counts, [8**2 + 3*8]*3)
        features = torch.randn(2, 3, 8)
        mask = torch.tensor([[True, True, False], [True, False, False]])
        for model in modes:
            model.train()
            output = model(features, mask)
            self.assertEqual(tuple(output.shape), (2, 8))
            self.assertTrue(torch.isfinite(output).all())
            output.square().sum().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_bias_rates_and_zero_novel(self):
        evaluator = AccuracyEvaluator([[0, 1], [2, 3]])
        scores = torch.tensor([
            [4., 1., 0., 0.],  # true base->base correct
            [0., 1., 3., 0.],  # true base->novel mistake
            [6., 0., 1., 0.],  # true novel->base mistake
            [0., 0., 1., 5.],  # true novel->novel correct
        ])
        y = torch.tensor([0, 1, 2, 3])
        b = evaluator.calc_base_novel_bias(scores, y, task_id=1)
        self.assertEqual(b["base_to_novel_pct"], 50.)
        self.assertEqual(b["novel_to_base_pct"], 50.)
        self.assertEqual(b["num_base_test"], 2)
        self.assertEqual(b["num_novel_test"], 2)
        only_base = evaluator.calc_base_novel_bias(scores[:2], y[:2], task_id=0)
        self.assertIsNone(only_base["novel_to_base_pct"])

    def test_followup_plan(self):
        args = Namespace(adapters=True, transfer=True,
                         adapter_seeds=[1], transfer_seeds=[2, 3])
        plan = build_plan(args)
        self.assertEqual(len(plan), 12)
        self.assertEqual(plan[:4], [
            ("cifar100", 1, "original"), ("cifar100", 1, "attention"),
            ("cifar100", 1, "linear"), ("cifar100", 1, "mlp"),
        ])
        self.assertEqual(plan[-2:], [
            ("miniimagenet", 3, "original"), ("miniimagenet", 3, "attention")
        ])

    def test_bias_parser_and_paired_deltas(self):
        event = {"session": 8, "novel_to_base_pct": 30., "base_to_novel_pct": 5.}
        bias, sessions = parse_bias("BIAS_METRICS_JSON:" + json.dumps(event))
        self.assertEqual(bias["novel_to_base_pct"], 30.)
        self.assertEqual(len(sessions), 1)
        common = {
            "dataset": "cifar100", "seed": 1, "status": "success",
            "last_acc": 73., "average_acc": 78., "final_base_acc": 80.,
            "final_novel_task_mean": 67., "novel_to_base_pct": 30.,
            "base_to_novel_pct": 5.,
        }
        baseline = dict(common, mode="original")
        model = dict(common, mode="linear", last_acc=75.,
                     novel_to_base_pct=25.)
        with TemporaryDirectory() as temp:
            pairs, aggregate = summarize([baseline, model], Path(temp))
            self.assertEqual(pairs[0]["delta_last_acc"], 2.)
            self.assertEqual(pairs[0]["delta_novel_to_base_pct"], -5.)
            self.assertEqual(aggregate[0]["paired_seeds"], 1)
            self.assertTrue((Path(temp) / "runs.csv").exists())


if __name__ == "__main__":
    unittest.main()
