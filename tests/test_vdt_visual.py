"""CPU unit tests for leakage-free intra-class visual attention calibration."""
import contextlib
import io
import unittest

import torch
from torch.nn import functional as F

from models.vdt_visual import (
    VisionSelfAttentionAdapter, sample_disjoint_visual_episode,
    train_visual_aggregator, aggregate_visual_classes,
)


class TestVisualAdapter(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(43)
        self.images = F.normalize(torch.randn(6 * 15, 8), dim=-1)
        self.labels = torch.arange(6).repeat_interleave(15)

    def test_episode_support_and_query_are_disjoint(self):
        rng = torch.Generator(device="cpu").manual_seed(5)
        support, mask, query, y, support_ids, query_ids = (
            sample_disjoint_visual_episode(
                self.images, self.labels, 6, 5, 4, generator=rng
            )
        )
        self.assertEqual(tuple(support.shape), (6, 5, 8))
        self.assertEqual(tuple(query.shape), (24, 8))
        self.assertEqual(tuple(mask.shape), (6, 5))
        self.assertFalse(bool((
            support_ids.flatten().unsqueeze(1) ==
            query_ids.flatten().unsqueeze(0)
        ).any()))
        self.assertEqual(y.tolist(), [i for i in range(6) for _ in range(4)])

    def test_attention_is_permutation_invariant_in_eval(self):
        adapter = VisionSelfAttentionAdapter(
            embed_dim=8, reduction=2, dropout=0.
        ).eval()
        tokens = self.images[:5].unsqueeze(0)
        mask = torch.ones((1, 5), dtype=torch.bool)
        with torch.no_grad():
            first = adapter(tokens, mask)
            second = adapter(tokens[:, [4, 0, 3, 1, 2]], mask)
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first.norm(dim=-1), torch.ones(1))

    def test_base_episodic_train_changes_weights_and_freezes(self):
        adapter = VisionSelfAttentionAdapter(
            embed_dim=8, reduction=2, dropout=0.
        )
        params_before = [p.detach().clone() for p in adapter.parameters()]
        with contextlib.redirect_stdout(io.StringIO()), torch.no_grad():
            logs = train_visual_aggregator(
                adapter, self.images, self.labels, 6, epochs=2,
                steps_per_epoch=2, support_per_class=5, query_per_class=4,
                lr=1e-3, weight_decay=0., seed=12
            )
        self.assertEqual(len(logs), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(logs))))
        self.assertTrue(any(
            not torch.equal(old, p)
            for old, p in zip(params_before, adapter.parameters())
        ))
        self.assertFalse(adapter.training)
        self.assertTrue(all(not p.requires_grad for p in adapter.parameters()))

    def test_aggregate_uses_available_support_and_support_cap(self):
        adapter = VisionSelfAttentionAdapter(
            embed_dim=8, reduction=2, dropout=0.
        ).eval()
        classes = torch.arange(6)
        full = aggregate_visual_classes(
            adapter, self.images, self.labels, classes
        )
        capped = aggregate_visual_classes(
            adapter, self.images, self.labels, classes, max_support=5
        )
        self.assertEqual(tuple(full.shape), (6, 8))
        self.assertEqual(tuple(capped.shape), (6, 8))
        torch.testing.assert_close(full.norm(dim=-1), torch.ones(6), atol=1e-5, rtol=0)
        self.assertTrue(torch.isfinite(capped).all())
        five_shot = aggregate_visual_classes(
            adapter, self.images[:5], self.labels[:5], torch.tensor([0])
        )
        self.assertEqual(tuple(five_shot.shape), (1, 8))

    def test_bimc_visual_prototype_preserves_fp16_and_base_reference(self):
        # Integration smoke without downloading or constructing any CLIP weights.
        from types import SimpleNamespace
        from models.bimc import BiMC
        model = BiMC.__new__(BiMC)
        torch.nn.Module.__init__(model)
        model.cfg = SimpleNamespace(TRAINER=SimpleNamespace(BiMC=SimpleNamespace(
            VDT_VISUAL_MAX_SUPPORT=0,
            VISION_CALIBRATION=True, LAMBDA_I=0.1, TAU=16,
        )))
        model.visual_adapter = VisionSelfAttentionAdapter(
            embed_dim=8, reduction=2, dropout=0.
        ).eval()
        model.visual_adapter_trained = True
        for p in model.visual_adapter.parameters():
            p.requires_grad_(False)
        base = {
            "images_features": self.images.half(),
            "images_targets": self.labels,
            "image_proto": torch.zeros(6, 8, dtype=torch.float16),
            "class_index": list(range(6)),
        }
        new = {
            "images_features": F.normalize(torch.randn(5, 8), dim=-1).half(),
            "images_targets": torch.full((5,), 6, dtype=torch.long),
            "image_proto": torch.zeros(1, 8, dtype=torch.float16),
            "class_index": [6],
        }
        out_base = model.vdt_visual_prototypes(base, task_id=0)
        self.assertEqual(out_base.dtype, torch.float16)
        self.assertEqual(model.base_vision_prototype.dtype, torch.float32)
        out_new = model.vdt_visual_prototypes(new, task_id=1)
        self.assertEqual(out_new.dtype, torch.float16)
        self.assertEqual(tuple(out_new.shape), (1, 8))
        self.assertTrue(torch.isfinite(out_new).all())

    def test_rejects_underfilled_support_query_split(self):
        with self.assertRaises(ValueError):
            sample_disjoint_visual_episode(
                self.images, self.labels, 6,
                support_per_class=11, query_per_class=5
            )


if __name__ == "__main__":
    unittest.main()
