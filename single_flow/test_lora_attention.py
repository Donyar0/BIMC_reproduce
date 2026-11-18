"""Small numerical checks for CLIP's packed Q/V LoRA adapter."""
import unittest
import torch
from torch import nn
from models.clip.model import CLIP, LoRAMultiheadAttention
from engine.lora_base import encode_base_prompts


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

    def test_text_encoder_prompt_loss_reaches_text_lora(self):
        torch.manual_seed(5)
        clip_model = CLIP(embed_dim=16, image_resolution=32, vision_layers=2,
                          vision_width=64, vision_patch_size=16, context_length=8,
                          vocab_size=100, transformer_width=32, transformer_heads=4,
                          transformer_layers=2)
        block = clip_model.transformer.resblocks[-1]
        block.attn = LoRAMultiheadAttention(block.attn, r=2)
        clip_model.requires_grad_(False)
        block.attn.q_B.requires_grad_(True)
        block.attn.v_B.requires_grad_(True)
        tokens = torch.tensor([[1, 8, 99, 0, 0, 0, 0, 0],
                               [1, 9, 99, 0, 0, 0, 0, 0]])
        text = encode_base_prompts(clip_model, tokens, num_classes=2, num_templates=1)
        (text[0] - text[1]).square().sum().backward()
        self.assertIsNotNone(block.attn.q_B.grad)
        self.assertIsNotNone(block.attn.v_B.grad)
        self.assertGreater(block.attn.q_B.grad.norm().item(), 0.0)


if __name__ == '__main__':
    unittest.main()
