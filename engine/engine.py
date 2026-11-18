import torch
import torch.nn as nn
import models.clip as clip
from datasets.data_manager import DatasetManager
from torch.nn import functional as F
from tqdm import tqdm
from utils.evaluator import AccuracyEvaluator
from utils.experiment_logger import log_metrics
from models.bimc import BiMC
import numpy as np
import time
from pathlib import Path


class Runner:

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_manager = DatasetManager(cfg,) 
        self.device = cfg.DEVICE.DEVICE_NAME

        self.model = BiMC(cfg, self.data_manager.template, self.device)

        # device
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)
            self.is_distributed = True
        else:
            self.is_distributed = False
            

        self.acc_list = []
        self.task_acc_list = []
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)


    def merge_dicts(self, dict_list):
        result = {}

        keys_to_merge = [
            'description_proto', 
            'description_features', 
            'description_targets',
            'text_features', 
            'text_targets',
            'image_proto', 
            'images_features', 
            'images_targets'
        ]

        for key in keys_to_merge:
            result[key] = torch.cat([d[key] for d in dict_list], dim=0)


        weights = [len(d['class_index']) for d in dict_list]


        cov_keys = [
            'cov_image',
        ]
        cov_sums = {key: torch.zeros_like(dict_list[0][key]) for key in cov_keys}
        weight_sum = sum(weights)

        for i, d in enumerate(dict_list):
            for key in cov_keys:
                cov_sums[key] += d[key] * weights[i]

        for key in cov_keys:
            if weight_sum > 0: 
                result[key] = cov_sums[key] / weight_sum

        return result



    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device, non_blocking=True)
        targets = targets.to(self.device, non_blocking=True)
        return data, targets

