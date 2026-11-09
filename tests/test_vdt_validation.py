"""CPU tests for cross-dataset, paired validation and log parsing."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.validate_vdt_fscil import (
    build_plan, parse_result, write_summary, command_for,
)
from argparse import Namespace


class TestVDTValidation(unittest.TestCase):
    def test_default_plan_has_ten_independent_runs(self):
        plan = build_plan([1, 2, 3], ["cifar100", "cub200", "miniimagenet"])
        self.assertEqual(len(plan), 10)
        self.assertEqual(plan[:4], [
            ("cifar100", 1, "original"), ("cifar100", 1, "attention"),
            ("cifar100", 2, "original"), ("cifar100", 2, "attention"),
        ])
        self.assertEqual(plan[-2:], [
            ("miniimagenet", 1, "original"), ("miniimagenet", 1, "attention"),
        ])

    def test_accuracies_parse_correctly(self):
        sample = (
            "Final acc:[79.7, 78.0, 72.5]\n"
            "Task-wise acc:\n"
            "task 0, acc:[79.7]\n"
            "task 1, acc:[78.4, 66.0]\n"
            "task 2, acc:[75.0, 55.0, 70.0]\n"
        )
        r = parse_result(sample)
        self.assertEqual(r["last_acc"], 72.5)
        self.assertEqual(r["final_base_acc"], 75.0)
        self.assertEqual(r["final_novel_task_mean"], 62.5)
        self.assertIsNone(parse_result("Final acc:[70, 71]\nTask-wise acc:\ntask 0, acc:[70]"))

    def test_paired_deltas_and_partial_persistence(self):
        base = dict(dataset="cifar100", seed=1, status="success",
                    average_acc=76, final_base_acc=80,
                    final_novel_task_mean=65, final_harmonic=71.72,
                    elapsed_seconds=20, log="base.log")
        original = dict(base, mode="original", last_acc=73)
        trained = dict(base, mode="attention", last_acc=75,
                       final_base_acc=83, final_novel_task_mean=64)
        with TemporaryDirectory() as temporary:
            folder = Path(temporary)
            pairs, aggregate = write_summary(folder, [original, trained])
            self.assertEqual(pairs[0]["delta_last_acc"], 2)
            self.assertEqual(pairs[0]["delta_final_base_acc"], 3)
            self.assertEqual(pairs[0]["delta_final_novel_task_mean"], -1)
            self.assertIsNone(aggregate[0]["delta_last_acc_std"])
            self.assertEqual(len(json.loads((folder / "results.json").read_text())), 2)

    def test_flags_include_original_mode(self):
        args = Namespace(train_cfg="configs/trainers/bimc.yaml",
                         ratio=0.2, epochs=20, original_lambda_t=0.5,
                         swanlab=False)
        c = command_for("cifar100", 2, "original", args, "run")
        self.assertEqual(c[c.index("SEED")+1], "2")
        self.assertEqual(c[c.index("TRAINER.BiMC.VDT_MODE")+1], "original")
        self.assertEqual(c[c.index("TRAINER.BiMC.VDT_ORIGINAL_LAMBDA_T")+1], "0.5")


if __name__ == "__main__":
    unittest.main()
