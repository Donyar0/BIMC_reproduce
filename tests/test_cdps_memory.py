"""CPU smoke tests for the incremental CDPS feature bank."""
import tempfile
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from models.clip.model import VisionTransformer
from models.cdps_memory import IncrementalCDPS

class ToyCLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = VisionTransformer(input_resolution=32, patch_size=16,
                                        width=64, layers=2, heads=4, output_dim=32)
    def encode_image(self, x):
        return self.visual(x)

def config(folder):
    return SimpleNamespace(SEED=1, DATASET=SimpleNamespace(NAME="toy"),
        TRAINER=SimpleNamespace(CDPS=SimpleNamespace(
            TOP_PERCENT=0.75, NUM_REGIONS=2, TOPK=2, NUM_CONFUSE=2,
            OMEGA=0.1, MAX_SUPPORT=2, MAX_QUERY_PATCHES=4,
            QUERY_TOPK=2, TEMPERATURE=1.0, CACHE_DIR=folder)))

class CDPSMemoryTest(unittest.TestCase):
    def test_old_class_memory_is_frozen(self):
        torch.manual_seed(1)
        with tempfile.TemporaryDirectory() as folder:
            bank = IncrementalCDPS(ToyCLIP().eval(), config(folder))
            def batch(labels):
                return [{"image": torch.rand(len(labels), 3, 32, 32),
                         "label": torch.tensor(labels)}]
            bank.add_session(0, batch([0, 0, 1, 1]), [0, 1])
            old = bank.memory[0].clone()
            scores = bank.probabilities(torch.rand(3, 3, 32, 32), [0, 1])
            self.assertEqual(tuple(scores.shape), (3, 2))
            torch.testing.assert_close(scores.sum(dim=1), torch.ones(3), atol=1e-5, rtol=0)
            bank.add_session(1, batch([2, 2]), [2])
            torch.testing.assert_close(bank.memory[0], old)
            self.assertEqual(bank.session_of_class[0], 0)
            self.assertEqual(bank.session_of_class[2], 1)
            self.assertTrue((bank.output / "session_00.pt").is_file())
            self.assertTrue((bank.output / "session_01.pt").is_file())
            scores = bank.probabilities(torch.rand(2, 3, 32, 32), [0, 1, 2])
            self.assertEqual(tuple(scores.shape), (2, 3))

    def test_prevent_overwriting_old_memory(self):
        with tempfile.TemporaryDirectory() as folder:
            bank = IncrementalCDPS(ToyCLIP().eval(), config(folder))
            loader = [{"image": torch.rand(2, 3, 32, 32),
                       "label": torch.tensor([0, 0])}]
            bank.add_session(0, loader, [0])
            with self.assertRaises(RuntimeError):
                bank.add_session(1, loader, [0])

if __name__ == "__main__":
    unittest.main()
