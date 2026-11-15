import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from datasets.data_manager import DatasetManager
from main import setup_cfg
from utils.multicrop import predict_multicrop


class TenCropTests(unittest.TestCase):
    def test_single_crop_preserves_original_output(self):
        images = torch.randn(3, 3, 4, 4)
        expected = torch.randn(3, 7)
        calls = []
        def predict(batch):
            calls.append(batch)
            return expected
        self.assertIs(predict_multicrop(images, predict), expected)
        self.assertIs(calls[0], images)
        self.assertEqual(len(calls), 1)

    def test_average_predictions_per_image_with_bounded_batch(self):
        images = torch.randn(3, 10, 3, 4, 4)
        calls = []
        def predict(batch):
            calls.append(batch.shape[0])
            return batch.mean((2, 3)).softmax(-1).half()
        expected = torch.stack([predict(images[:, i]).float() for i in range(10)]).mean(0)
        calls.clear()
        actual = predict_multicrop(images, predict)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(calls, [3] * 10)
        self.assertEqual(actual.dtype, torch.float32)
        for shape in [(2, 9, 3, 4, 4), (2, 3, 4)]:
            with self.assertRaises(ValueError):
                predict_multicrop(torch.zeros(shape), predict)

    def test_transforms_and_support_samples_are_shared(self):
        cfg = setup_cfg('configs/datasets/cifar100.yaml', 'configs/trainers/bimc.yaml', True)
        images = np.random.default_rng(1).integers(0, 256, (6, 240, 320, 3), dtype=np.uint8)
        def setup(manager, root, name):
            manager.train_data = manager.test_data = images
            manager.train_targets = manager.test_targets = np.array([0, 0, 0, 1, 1, 1])
            manager.class_names = ['a', 'b']
            manager.num_total_classes = 2
        cfg = cfg.clone()
        cfg.defrost()
        cfg.DATASET.NUM_INIT_CLS = 2
        cfg.DATASET.NUM_BASE_SHOT = 1
        cfg.DATALOADER.NUM_WORKERS = 0
        with patch.object(DatasetManager, '_setup_data', setup):
            manager = DatasetManager(cfg)
        np.random.seed(1)
        support = manager.get_dataset(0, 'train', 'test')
        baseline = manager.get_dataset(0, 'test', 'test')
        ten = manager.get_dataset(0, 'test', 'ten_crop')
        self.assertIs(support.transform, baseline.transform)
        np.testing.assert_array_equal(baseline.labels, ten.labels)
        np.testing.assert_array_equal(baseline.images, ten.images)
        self.assertEqual(len(support), 2)
        crop_tensor = ten[0]['image']
        self.assertEqual(crop_tensor.shape, (10, 3, 224, 224))
        torch.testing.assert_close(crop_tensor[4], baseline[0]['image'])
        # Match torchvision's specified ordering and flip behavior exactly.
        resized = transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)(Image.fromarray(images[0]))
        expected = manager.ten_crop_transform.transforms[-1](transforms.TenCrop(224)(resized))
        torch.testing.assert_close(crop_tensor, expected)
        with self.assertRaises(ValueError):
            manager.get_dataset(0, 'train', 'ten_crop')
        batch = next(iter(manager.get_dataloader(0, 'test', 'ten_crop')))
        self.assertEqual(batch['image'].shape, (6, 10, 3, 224, 224))
        self.assertEqual(batch['label'].shape, (6,))

    def test_comparison_reuses_task_statistics(self):
        from engine.engine import Runner
        runner = Runner.__new__(Runner)
        runner.cfg = setup_cfg('configs/datasets/cifar100.yaml', 'configs/trainers/bimc.yaml', True)
        runner.acc_list, runner.task_acc_list, runner.ten_crop_acc_list = [], [], []
        runner.data_manager = SimpleNamespace(
            num_tasks=2, class_names=['a', 'b'], class_index_in_task=[[0], [1]],
            get_dataloader=lambda *a, **k: object())
        builds, evaluations = [], []
        def build(*args, **kwargs):
            builds.append(kwargs['class_index'])
            return object()
        runner.model = SimpleNamespace(eval=lambda: None, build_task_statistics=build)
        runner.merge_dicts = lambda states: tuple(states)
        def evaluate(task_id, state, mode='test'):
            evaluations.append((task_id, state, mode))
            return dict(mean_acc=50., base_avg_acc=50., inc_avg_acc=50., harmonic_acc=50., task_acc=[50.])
        runner.inference_task_covariance = evaluate
        with patch('engine.engine.log_metrics'):
            runner.run()
        self.assertEqual(len(builds), 2)
        self.assertEqual([e[2] for e in evaluations], ['test', 'ten_crop'] * 2)
        for i in (0, 2):
            self.assertIs(evaluations[i][1], evaluations[i + 1][1])

    def test_existing_configs_default_to_single_crop(self):
        for dataset in ('cifar100', 'cub200', 'miniimagenet'):
            for trainer in ('bimc', 'bimc_ensemble'):
                cfg = setup_cfg(f'configs/datasets/{dataset}.yaml', f'configs/trainers/{trainer}.yaml')
                self.assertFalse(cfg.DATALOADER.TEST.COMPARE_TEN_CROP)


if __name__ == '__main__':
    unittest.main()
