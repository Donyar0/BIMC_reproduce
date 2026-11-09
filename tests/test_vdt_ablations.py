"""Tests: python -m unittest discover -s tests -v"""
import unittest
from argparse import Namespace

import torch
import torch.nn.functional as F

from models.vdt_self_attention import mean_description_prototypes
from scripts.run_vdt_ablations import EXPERIMENTS, make_command, parse_results


class TestVDTAblationSuite(unittest.TestCase):
    def test_five_experiments_are_isolated_and_correct(self):
        self.assertEqual(len(EXPERIMENTS), 5)
        self.assertEqual([(v[1], v[2], v[3]) for v in EXPERIMENTS], [
            ("mean", 0.2, 0),
            ("attention", 0.1, 20),
            ("attention", 0.3, 20),
            ("attention", 0.2, 5),
            ("attention", 0.2, 50),
        ])

    def test_no_training_mean_pooling(self):
        features = torch.tensor([
            [[1., 0.], [0., 1.], [100., 100.]],
            [[0., 1.], [100., 100.], [100., 100.]],
        ])
        mask = torch.tensor([[True, True, False], [True, False, False]])
        result = mean_description_prototypes(features, mask)
        self.assertTrue(torch.allclose(result[0], F.normalize(torch.tensor([1., 1.]), dim=0)))
        self.assertTrue(torch.allclose(result[1], torch.tensor([0., 1.])))
        self.assertTrue(torch.allclose(result.norm(dim=-1), torch.ones(2)))

    def test_command_has_independent_overrides(self):
        args = Namespace(data_cfg="configs/datasets/cifar100.yaml", train_cfg="configs/trainers/bimc.yaml",
                         swanlab=False, swanlab_project="test", swanlab_mode="offline")
        cmd = make_command(args, "A_mean_no_training", "mean", 0.2, 0)
        self.assertIn("--opts", cmd)
        self.assertEqual(cmd[cmd.index("TRAINER.BiMC.VDT_MODE")+1], "mean")
        self.assertEqual(cmd[cmd.index("TRAINER.BiMC.VDT_EPOCHS")+1], "0")
        self.assertNotIn("--use_swanlab", cmd)

    def test_parse_results(self):
        log = "some progress\nFinal acc:[79.7, 78.06, 72.61]\nTask-wise acc:\ntask  0, acc:[79.7]\ntask  1, acc:[79.18, 64.6]\n"
        final, tasks = parse_results(log)
        self.assertEqual(final, [79.7, 78.06, 72.61])
        self.assertEqual(tasks, [[79.7], [79.18, 64.6]])
        self.assertEqual(parse_results("error"), (None, []))


if __name__ == "__main__":
    unittest.main()
