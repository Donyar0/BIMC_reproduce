"""Verify prototype hub correction is independent of query batch composition."""
import unittest

import torch
import torch.nn.functional as F

from single_flow.text_distribution_flow import classify_bidirectional


class ZeroVelocity(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)


class HubScoreTests(unittest.TestCase):
    def test_cosine_hub_offset_is_class_specific_and_batch_independent(self):
        bank = dict(class_ids=[0, 1, 2],
                    prototypes=F.normalize(torch.tensor([[1., 0.], [0.8, 0.6], [-1., 0.]]), dim=-1),
                    text=F.normalize(torch.tensor([[1., 0.], [0., 1.], [0.6, 0.8]]), dim=-1))
        queries = F.normalize(torch.tensor([[1., 0.], [0., 1.]]), dim=-1)
        kwargs = dict(alpha=1.0, gamma=0.5, steps=2, score_mode='cosine_sum',
                      whitening_matrix=torch.eye(2), text_whitening_mode='none',
                      text_score_weight=1.2, hub_weight=0.2, return_components=True)
        _, together, _, components = classify_bidirectional(ZeroVelocity(), queries, bank, **kwargs)
        separate = torch.cat([classify_bidirectional(ZeroVelocity(), query[None], bank,
                                                     **kwargs)[1] for query in queries])
        self.assertTrue(torch.allclose(together, separate, atol=1e-6))

        visual = queries @ bank['prototypes'].T
        text = queries @ bank['text'].T
        visual_hub = ((bank['prototypes'] @ bank['prototypes'].T).sum(-1) - 1) / 2
        text_hub = ((bank['text'] @ bank['text'].T).sum(-1) - 1) / 2
        expected = visual - 0.2 * visual_hub + 1.2 * (text - 0.2 * text_hub)
        self.assertTrue(torch.allclose(together, expected, atol=1e-6))
        self.assertTrue(torch.allclose(components['flow'], together))


if __name__ == '__main__':
    unittest.main()
