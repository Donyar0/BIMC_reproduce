"""CPU tests: python -m unittest discover -s tests -v"""
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from utils.finegrained_diagnostics import summarize_class_confusions
from scripts.run_vdt_finegrained import (
    build_plan, compare_classwise, extract_diagnostics,
)


class TestVDTFinegrained(unittest.TestCase):
    def test_diagnostic_counts_and_names(self):
        logits = torch.tensor([
            [9., 1., 0.],  # class0 -> class0
            [0., 2., 3.],  # class0 -> class2 wrong
            [0., 8., 1.],  # class1 -> class1
            [0., 5., 1.],  # class2 -> class1 wrong
            [0., 0., 5.],  # class2 -> class2
        ])
        targets = torch.tensor([0, 0, 1, 2, 2])
        r = summarize_class_confusions(logits, targets, ["bird A", "bird B", "bird C"],
                                       [0, 1], top_k=10)
        self.assertEqual(r["num_test_samples"], 5)
        self.assertEqual(r["per_class"][0]["accuracy_pct"], 50)
        self.assertEqual(r["per_class"][2]["accuracy_pct"], 50)
        pairs = {(p["true_id"], p["pred_id"]): p for p in r["all_confusion_pairs"]}
        self.assertEqual(pairs[(0, 2)]["count"], 1)
        self.assertEqual(pairs[(2, 1)]["count"], 1)
        self.assertEqual(pairs[(2, 1)]["true_group"], "novel")
        self.assertEqual(pairs[(2, 1)]["pred_name"], "bird B")

    def test_no_errors(self):
        data = summarize_class_confusions(np.eye(3), np.arange(3),
                                         ["a", "b", "c"], [0])
        self.assertEqual(data["all_confusion_pairs"], [])
        self.assertEqual(len(data["per_class"]), 3)

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):
            summarize_class_confusions(torch.randn(3, 2), torch.ones(2),
                                       ["a", "b"], [0])

    def test_experiment_grid(self):
        plan = build_plan()
        self.assertEqual(len(plan), 20)
        self.assertEqual(plan[:4], [
            ("cifar100", 2, "original"), ("cifar100", 2, "attention"),
            ("cifar100", 2, "linear"), ("cifar100", 2, "mlp"),
        ])
        self.assertEqual(plan[-4:], [
            ("cub200", 3, "original"), ("cub200", 3, "attention"),
            ("cub200", 3, "linear"), ("cub200", 3, "mlp"),
        ])

    def test_extract_and_compare(self):
        import json
        d1 = summarize_class_confusions(
            [[2., 1.], [2., 1.]], [0, 1], ["bird 0", "bird 1"], [0]
        )
        d2 = summarize_class_confusions(
            [[2., 1.], [0., 3.]], [0, 1], ["bird 0", "bird 1"], [0]
        )
        log = "hello\nFINEGRAINED_DIAGNOSTICS_JSON:" + json.dumps(d2) + "\n"
        self.assertEqual(extract_diagnostics(log)["per_class"], d2["per_class"])
        with TemporaryDirectory() as folder:
            records = [
                {"dataset": "cub200", "seed": 1, "mode": "original",
                 "status": "success", "diagnostics": d1},
                {"dataset": "cub200", "seed": 1, "mode": "attention",
                 "status": "success", "diagnostics": d2},
            ]
            delta, errors = compare_classwise(records, Path(folder))
            novel = [x for x in delta if x["class_id"] == 1][0]
            self.assertEqual(novel["delta_accuracy_pct"], 100.)
            pair = [x for x in errors if x["true_id"] == 1 and x["pred_id"] == 0][0]
            self.assertEqual(pair["delta_count"], -1)
            self.assertTrue((Path(folder)/"cub_confusion_pair_deltas.csv").is_file())


if __name__ == "__main__":
    unittest.main()
