"""No GPU or dataset required: python -m unittest tests.test_vdt_visual_diagnostics"""
import json
from argparse import Namespace
import unittest

import torch
from torch.nn import functional as F

from models.vdt_self_attention import (
    mean_description_prototypes, visual_weighted_description_prototypes
)
from utils.prototype_geometry import prototype_geometry
from scripts.run_vdt_visual_diagnostics import (
    plan_from_args, parse_geometry, make_cmd
)


class TestVisualVDT(unittest.TestCase):
    def setUp(self):
        self.desc = F.normalize(torch.tensor([
            [[1., 0.], [0., 1.], [100., 100.]],
            [[0., 1.], [1., 0.], [1., 1.]],
        ]), dim=-1)
        self.mask = torch.tensor([[True, True, False], [True, True, True]])
        self.visual = torch.tensor([[1., 0.], [0., 1.]])

    def test_zero_blend_matches_mean(self):
        weighted, weights = visual_weighted_description_prototypes(
            self.desc, self.mask, self.visual, temperature=10, blend=0)
        mean = mean_description_prototypes(self.desc,self.mask)
        torch.testing.assert_close(weighted,mean,atol=1e-6,rtol=1e-6)
        self.assertEqual(float(weights[0,2]),0.)

    def test_visual_weighting_prefers_matching_description(self):
        weighted, weights = visual_weighted_description_prototypes(
            self.desc, self.mask, self.visual, temperature=20, blend=1)
        self.assertGreater(float(weights[0,0]),.99)
        self.assertGreater(float(weights[1,0]),.99)
        self.assertGreater(float(weighted[0] @ self.visual[0]),
                           float(self.desc[0,:2].mean(dim=0) @ self.visual[0]))
        self.assertFalse(weighted.requires_grad)

    def test_padded_values_do_not_change_output(self):
        a,_=visual_weighted_description_prototypes(
            self.desc,self.mask,self.visual,temperature=10,blend=0.5)
        changed=self.desc.clone()
        changed[0,2]=torch.tensor([-500.,-500.])
        b,_=visual_weighted_description_prototypes(
            changed,self.mask,self.visual,temperature=10,blend=0.5)
        torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)

    def test_rejects_invalid_parameters(self):
        with self.assertRaises(ValueError):
            visual_weighted_description_prototypes(
                self.desc,self.mask,self.visual,temperature=-1)
        with self.assertRaises(ValueError):
            visual_weighted_description_prototypes(
                self.desc,self.mask,self.visual,blend=2)

    def test_geometry_margin_and_support_alignment(self):
        raw=F.normalize(torch.tensor([[1.,0.],[.8,.6],[0.,1.]]),dim=-1)
        adapted=F.normalize(torch.tensor([[1.,0.],[0.,1.],[.2,.98]]),dim=-1)
        report=prototype_geometry(raw,adapted,raw,
                                  ["a","b","c"],base_count=1,top_k=3)
        self.assertEqual(len(report["classes"]),3)
        p={(x["class_a_id"],x["class_b_id"]):x
           for x in report["most_similar_raw_pairs"]}
        self.assertGreater(p[(0,1)]["delta_separation"],0)
        self.assertLess(report["support_alignment"]["delta_novel_mean"],0)

    def test_plan_and_geometry_log(self):
        args=Namespace(datasets="cub200,cifar100",cub_seeds=[1,2],
                       other_seeds=[1])
        plan=plan_from_args(args)
        self.assertEqual(len(plan),12)
        self.assertEqual(plan[:4],[
            ("cub200",1,"original"),("cub200",1,"mean"),
            ("cub200",1,"attention"),("cub200",1,"visual_weighted")])
        example={"support_alignment":{"delta_novel_mean":0.1}}
        self.assertEqual(parse_geometry(
            "VDT_GEOMETRY_JSON:"+json.dumps(example)+"\n"),example)
        cmdargs=Namespace(train_cfg="configs/trainers/bimc.yaml",
                          temp=10.,blend=.5,epochs=20,ratio=.2,swanlab=False)
        cmd=make_cmd(cmdargs,"cub200",2,"visual_weighted")
        self.assertEqual(cmd[cmd.index("TRAINER.BiMC.VDT_MODE")+1],"visual_weighted")


if __name__ == "__main__":
    unittest.main()
