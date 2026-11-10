"""CPU-only tests for contrastive and pseudo-incremental VDT verification.

python -m unittest tests.test_vdt_contrastive -v
"""
import json
from argparse import Namespace
import unittest

import torch
from torch.nn import functional as F

from models.vdt_contrastive import (
    contrastive_description_prototypes, description_weight_diagnostics
)
from models.vdt_pseudo_validation import select_base_only_contrastive_strength
from scripts.run_vdt_contrastive import (
    MODES, make_cmd, parse_named_json, CV_RE, WEIGHT_RE, plan_from_args
)


class TestContrastiveVDT(unittest.TestCase):
    def test_zero_lambda_equals_visual_weighted_and_padding_safe(self):
        from models.vdt_self_attention import visual_weighted_description_prototypes
        desc = F.normalize(torch.randn(3, 4, 8), dim=-1)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
                            dtype=torch.bool)
        visual = F.normalize(torch.randn(3, 8), dim=-1)
        new, w, _ = contrastive_description_prototypes(
            desc, mask, visual, temperature=10, blend=.5, negative_strength=0)
        old, w_old = visual_weighted_description_prototypes(
            desc, mask, visual, temperature=10, blend=.5)
        torch.testing.assert_close(new, old, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(w, w_old, atol=1e-6, rtol=1e-6)
        self.assertEqual(w[0, 2:].tolist(), [0., 0.])
        torch.testing.assert_close(w.sum(dim=1), torch.ones(3))

    def test_hard_negative_penalty_prefers_discriminative_description(self):
        # Description 0 matches the current class AND competing class;
        # description 1 is slightly less aligned but much more distinctive.
        visual = F.normalize(torch.tensor([[1., 0., 0.], [0.9, 0.436, 0.]]), dim=-1)
        desc = F.normalize(torch.tensor([[
            [0.95, 0.31, 0.], [0.95, -0.31, 0.]
        ], [
            [0.9, 0.436, 0.], [0.9, 0.436, 0.]
        ]]), dim=-1)
        mask = torch.ones(2, 2, dtype=torch.bool)
        _, own, _ = contrastive_description_prototypes(
            desc, mask, visual, negative_strength=0., temperature=20., blend=1.)
        _, contrastive, _ = contrastive_description_prototypes(
            desc, mask, visual, negative_strength=2., temperature=20., blend=1.)
        self.assertGreater(float(contrastive[0, 1]), float(own[0, 1]))
        self.assertGreater(float(contrastive[0, 1]), 0.5)

    def test_single_class_and_invalid_inputs(self):
        desc = torch.randn(1, 3, 4)
        visual = torch.randn(1, 4)
        mask = torch.ones(1, 3, dtype=torch.bool)
        p, w, _ = contrastive_description_prototypes(
            desc, mask, visual, negative_strength=2.)
        self.assertEqual(tuple(p.shape), (1, 4))
        self.assertTrue(torch.isfinite(p).all())
        torch.testing.assert_close(w.sum(-1), torch.ones(1))
        with self.assertRaises(ValueError):
            contrastive_description_prototypes(desc,mask,visual,negative_strength=-1)

    def test_entropy_diagnostics_are_normalized(self):
        desc = F.normalize(torch.randn(2,3,6),dim=-1)
        mask = torch.tensor([[1,1,1],[1,0,0]],dtype=torch.bool)
        visual = F.normalize(torch.randn(2,6),dim=-1)
        weights = torch.tensor([[1/3,1/3,1/3],[1.,0.,0.]])
        info = description_weight_diagnostics(
            desc,mask,visual,weights,["a","b"],base_count=1)
        self.assertAlmostEqual(info["per_class"][0]["normalized_entropy"],1.,places=5)
        self.assertEqual(info["per_class"][1]["normalized_entropy"],0.)
        self.assertIsNone(info["per_class"][1]["intra_description_cos"])
        self.assertEqual(info["novel"]["num_classes"],1)

    def test_base_only_cv_reproducible(self):
        # Synthetic base TRAIN data; no separate test data provided.
        c,dim=8,8
        x=F.normalize(torch.eye(c).repeat_interleave(14,dim=0)+
                      .01*torch.randn(c*14,dim),dim=-1)
        y=torch.arange(c).repeat_interleave(14)
        desc=F.normalize(torch.eye(c).repeat_interleave(2,dim=0),dim=-1)
        labels=torch.arange(c).repeat_interleave(2)
        state={
            "images_features":x, "images_targets":y,
            "vdt_features":desc, "vdt_targets":labels,
        }
        settings=dict(base_state=state,beta=.3,shift_weight=.1,tau=16,
                      seed=7,candidates=[0.,.5,1.],episodes=2,
                      query_per_class=3,novel_shots=5)
        selected_a,report_a=select_base_only_contrastive_strength(**settings)
        selected_b,report_b=select_base_only_contrastive_strength(**settings)
        self.assertEqual(selected_a,selected_b)
        self.assertEqual(report_a,report_b)
        self.assertEqual(len(report_a["candidate_results"]),3)
        self.assertEqual(report_a["selection_source"],"base_training_images_only")
        self.assertEqual(report_a["pseudo_novel_classes_per_episode"],2)

    def test_runner_modes_and_log_parsing(self):
        self.assertEqual(len(MODES),6)
        a=Namespace(datasets="cub200",cub_seeds=[1],other_seeds=[1])
        self.assertEqual(len(plan_from_args(a)),6)
        obj={"selected_strength":.5}
        self.assertEqual(parse_named_json("VDT_PSEUDO_CV_JSON:"+json.dumps(obj)+"\n",CV_RE),obj)
        self.assertIsNone(parse_named_json("no statistics",WEIGHT_RE))
        args=Namespace(train_cfg="configs/trainers/bimc.yaml",swanlab=False,
                       epochs=20,ratio=.2,temp=10,blend=.5,
                       negative_strength=1.,cv_episodes=3,cv_queries=5)
        cmd=make_cmd(args,"cub200",1,"contrastive_cv")
        self.assertEqual(cmd[cmd.index("TRAINER.BiMC.VDT_MODE")+1],"contrastive_cv")


if __name__ == "__main__":
    unittest.main()
