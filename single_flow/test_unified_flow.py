import unittest
import torch
from single_flow.s1_flow import fit
from single_flow.unified_flow import adapt, build_bank, classify_bidirectional, load_model, real_means


class UnifiedFlowTests(unittest.TestCase):
    def test_adaptation_bank_and_scores(self):
        torch.set_num_threads(1)
        eye = torch.eye(3)
        base = dict(stage='base', split='train', encoder_id='test', class_ids=[0, 1],
                    text_prototypes=eye[:2], visual_prototypes=eye[:2],
                    image_features=eye[:2], image_labels=torch.tensor([0, 1]))
        ckpt = fit(base, epochs=1, batch_size=2, blocks=1, hidden=8, target='sample')
        memory = dict(class_ids=[0, 1], text=eye[:2], real=real_means(base))
        novel = dict(encoder_id='test', class_ids=[2], text_prototypes=eye[2:],
                     image_features=torch.tensor([[0.2, 0., 1.]]), image_labels=torch.tensor([2]))
        updated = adapt(ckpt, novel, memory, steps=2, batch_size=3, lr=0.001,
                        weight_decay=0, device='cpu')
        self.assertTrue(any(not torch.equal(ckpt['state_dict'][k], updated['state_dict'][k])
                            for k in ckpt['state_dict']))
        memory = dict(class_ids=[0, 1, 2], text=eye, real=eye)
        bank = build_bank(updated, memory, steps=2, num_base=2,
                          beta_base=0, beta_inc=0, device='cpu')
        self.assertTrue(torch.allclose(bank['prototypes'], eye))
        model = load_model(updated, 'cpu').eval()
        pred, scores = classify_bidirectional(model, eye, bank, alpha=0, gamma=1)
        self.assertTrue(torch.equal(pred, torch.arange(3)))
        self.assertTrue(torch.allclose(scores, eye))
        _, scores = classify_bidirectional(model, eye, bank, alpha=0.15, gamma=0.5)
        self.assertEqual(scores.shape, (3, 3))
        self.assertTrue(torch.isfinite(scores).all())


if __name__ == '__main__':
    unittest.main()
