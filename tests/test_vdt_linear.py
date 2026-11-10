"""CPU tests for the minimal text-prototype calibration branch.

python -m unittest discover -s tests -v
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from models.vdt_linear import LinearTextAdapter, pack_description_bank
from scripts.run_linear_baseline import parse_run


class TestCleanLinear(unittest.TestCase):
    def test_adapter_parameters_and_gradients(self):
        torch.manual_seed(123)
        model = LinearTextAdapter(embed_dim=8, ratio=.2, dropout=0.)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 8*8+3*8)
        values = torch.randn(3, 4, 8)
        mask = torch.tensor([[1,1,1,0], [1,0,0,0], [1,1,0,0]],dtype=torch.bool)
        output = model(values, mask)
        self.assertEqual(tuple(output.shape),(3,8))
        torch.testing.assert_close(output.norm(dim=-1),torch.ones(3),atol=1e-5,rtol=0)
        labels=torch.tensor([0,1,2,0])
        img=torch.nn.functional.normalize(torch.randn(4,8),dim=-1)
        loss=torch.nn.functional.cross_entropy(10*img@output.T,labels)
        loss.backward()
        self.assertIsNotNone(model.projection.weight.grad)
        self.assertTrue(torch.isfinite(model.projection.weight.grad).all())

    def test_padding_is_ignored(self):
        torch.manual_seed(1)
        model=LinearTextAdapter(embed_dim=8,dropout=0.).eval()
        base=torch.randn(1,2,8)
        mask=torch.tensor([[1,1]],dtype=torch.bool)
        extended=torch.cat((base,torch.ones(1,3,8)*100),dim=1)
        extended_mask=torch.tensor([[1,1,0,0,0]],dtype=torch.bool)
        with torch.no_grad():
            torch.testing.assert_close(model(base,mask),
                                       model(extended,extended_mask),
                                       atol=1e-6,rtol=1e-6)

    def test_packing(self):
        features=torch.randn(5,8)
        labels=torch.tensor([1,1,0,2,2])
        packed,mask,classes=pack_description_bank(features,labels)
        self.assertEqual(classes.tolist(),[0,1,2])
        self.assertEqual(mask.sum(-1).tolist(),[1,2,2])
        self.assertEqual(tuple(packed.shape),(3,2,8))

    def test_unexpected_mode_not_in_cli(self):
        from scripts.run_linear_baseline import main
        # Mode check is done by CLI argument choices and model construction.
        source=Path(__file__).resolve().parents[1]/"models"/"bimc.py"
        content=source.read_text(encoding="utf-8")
        self.assertIn('if mode not in ("linear", "original"):',content)
        self.assertNotIn("contrastive_description_prototypes",content)

    def test_all_dataset_trainer_configs_load(self):
        # Catch strict YACS keys left behind when obsolete modules are removed.
        from main import setup_cfg
        root = Path(__file__).resolve().parents[1]
        for dataset in ("cifar100", "cub200", "miniimagenet"):
            for trainer in ("bimc", "bimc_ensemble"):
                with self.subTest(dataset=dataset, trainer=trainer):
                    cfg = setup_cfg(
                        str(root / "configs" / "datasets" / (dataset + ".yaml")),
                        str(root / "configs" / "trainers" / (trainer + ".yaml")),
                    )
                    self.assertEqual(cfg.TRAINER.BiMC.VDT_MODE, "linear")
                    self.assertEqual(
                        cfg.TRAINER.BiMC.USING_ENSEMBLE,
                        trainer == "bimc_ensemble",
                    )

    def test_result_parser(self):
        log="BIAS_METRICS_JSON:{\"session\":0}\nFinal acc:[80.0,75.0]\nTask-wise acc:\ntask  0, acc:[80.0]\ntask  1, acc:[78.0,50.0]\n"
        parsed=parse_run(log)
        self.assertEqual(parsed["last_acc"],75.)
        self.assertEqual(parsed["task_wise_acc"][-1],[78.,50.])
        self.assertEqual(parsed["bias_by_session"][0]["session"],0)


if __name__=="__main__":
    unittest.main()
