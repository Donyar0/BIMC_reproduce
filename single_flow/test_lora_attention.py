"""Small numerical checks for CLIP's packed Q/V LoRA adapter."""
import unittest
import torch
from torch import nn
from models.clip.model import LoRAMultiheadAttention


class LoRAAttentionTests(unittest.TestCase):
    def test_zero_adapter_and_merge(self):
        torch.manual_seed(3)
        original = nn.MultiheadAttention(16, 4).eval()
        adapter = LoRAMultiheadAttention(original, r=2).eval()
        x = torch.randn(5, 2, 16)
        before = original(x, x, x, need_weights=False)[0]
        initial = adapter(x, x, x, need_weights=False)[0]
        self.assertTrue(torch.allclose(before, initial, atol=1e-6))
        with torch.no_grad():
            adapter.q_B.normal_(std=0.01)
            adapter.v_B.normal_(std=0.01)
        adapted = adapter(x, x, x, need_weights=False)[0]
        merged = adapter.merge().eval()(x, x, x, need_weights=False)[0]
        self.assertTrue(torch.allclose(adapted, merged, atol=1e-6))

    def test_only_lora_parameters_receive_gradients(self):
        torch.manual_seed(4)
        adapter = LoRAMultiheadAttention(nn.MultiheadAttention(16, 4), r=2)
        x = torch.randn(5, 2, 16)
        adapter(x, x, x, need_weights=False)[0].square().mean().backward()
        self.assertIsNotNone(adapter.q_B.grad)
        self.assertIsNotNone(adapter.v_B.grad)
        self.assertIsNone(adapter.orig_attn.in_proj_weight.grad)


if __name__ == '__main__':
    unittest.main()
