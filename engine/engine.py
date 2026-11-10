import torch
import torch.nn as nn
import models.clip as clip
from datasets.data_manager import DatasetManager
from torch.nn import functional as F
from tqdm import tqdm
from utils.evaluator import AccuracyEvaluator
from utils.experiment_logger import log_metrics
from models.bimc import BiMC
from models.cdps_memory import IncrementalCDPS
import numpy as np
import time


class Runner:

    def __init__(self, cfg):
        self.cfg = cfg
        self.data_manager = DatasetManager(cfg,) 
        self.device = cfg.DEVICE.DEVICE_NAME

        self.model = BiMC(cfg, self.data_manager.template, self.device)
        self.cdps = IncrementalCDPS(self.model.clip_model, cfg) if cfg.TRAINER.CDPS.ENABLED else None

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



    @torch.no_grad()
    def run(self):
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')
        state_dict_list = []
        for i in range(self.data_manager.num_tasks):
            self.model.eval()

            current_class_name = np.array(self.data_manager.class_names)[self.data_manager.class_index_in_task[i]]
            loader = self.data_manager.get_dataloader(i, source='train', mode='test', accumulate_past=False)            



            current_state_dict = self.model.build_task_statistics(current_class_name, loader,
                                                             class_index=self.data_manager.class_index_in_task[i], 
                                                             calibrate_novel_vision_proto=self.cfg.TRAINER.BiMC.VISION_CALIBRATION,)

            if self.cdps is not None:
                # Use exactly this session's support loader; no old training images.
                self.cdps.add_session(i, loader, self.data_manager.class_index_in_task[i])

            state_dict_list.append(current_state_dict)            
            merged_state_dict = self.merge_dicts(state_dict_list)

            start_time = time.time()
            acc = self.inference_task_covariance(i, merged_state_dict)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'+++++++++++  task {i}, time: {elapsed_time} ++++++++++++++++')

            print(f'=> Task [{i}], Acc: {acc["mean_acc"]:.3f}')
            self.acc_list.append(round(acc["mean_acc"], 3))
            self.task_acc_list.append(acc['task_acc'])
            log_metrics({
                'task_id': i,
                'mean_acc': acc['mean_acc'],
                'base_avg_acc': acc['base_avg_acc'],
                'inc_avg_acc': acc['inc_avg_acc'],
                'harmonic_acc': acc['harmonic_acc'],
                'elapsed_time': elapsed_time,
            }, step=i)

        print(f'Final acc:{self.acc_list}')
        print('Task-wise acc:')
        for i, task_acc in enumerate(self.task_acc_list):
            print(f'task {i:2d}, acc:{task_acc}')
        log_metrics({
            'final_acc': self.acc_list[-1] if self.acc_list else 0.0,
            "avg_acc": sum(self.acc_list) / len(self.acc_list),
            
        })
    

    @torch.no_grad()
    def inference_task_covariance(self, task_id, state_dict):

        beta = self.cfg.DATASET.BETA

        image_proto = state_dict['image_proto']
        cov_image = state_dict['cov_image']
        text_features = state_dict['text_features']
        description_proto = state_dict['description_proto']
        description_features = state_dict['description_features']
        description_targets = state_dict['description_targets']

        num_base_class = len(self.data_manager.class_index_in_task[0])
        num_accumulated_class = max(self.data_manager.class_index_in_task[task_id]) + 1
        
        test_loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        all_logits = []
        all_global_logits = []
        all_local_logits = []
        all_targets = []

        for i, batch in enumerate(tqdm(test_loader)):
            data, targets = self.parse_batch(batch)
            logits = self.model.forward_ours(data, num_accumulated_class, num_base_class,
                                                   image_proto, 
                                                   cov_image,
                                                   description_proto,
                                                   description_features, 
                                                   description_targets,
                                                   text_features,
                                                   beta=beta)

            if self.cdps is not None:
                class_ids = list(range(num_accumulated_class))
                patch_prob = self.cdps.probabilities(data, class_ids)
                global_prob = logits.float()
                # BiMC outputs probabilities, not raw cosine logits.
                # Renormalize because its base / incremental ensemble has separate mixing.
                global_prob = global_prob / global_prob.sum(-1, keepdim=True).clamp_min(1e-12)
                w = float(self.cfg.TRAINER.CDPS.GLOBAL_WEIGHT)
                if not 0 <= w <= 1:
                    raise ValueError("CDPS GLOBAL_WEIGHT must be in [0, 1].")
                all_global_logits.append(global_prob.cpu())
                all_local_logits.append(patch_prob.cpu())
                logits = w * global_prob + (1. - w) * patch_prob
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        eval_acc = self.evaluator.calc_accuracy(all_logits, all_targets, task_id)
        if self.cdps is not None:
            global_acc = self.evaluator.calc_accuracy(torch.cat(all_global_logits), all_targets, task_id)
            local_acc = self.evaluator.calc_accuracy(torch.cat(all_local_logits), all_targets, task_id)
            print("CDPS ablation session %d: BiMC=%.2f, Patch=%.2f, Hybrid=%.2f (global_weight=%.2f)" %
                  (task_id, global_acc["mean_acc"], local_acc["mean_acc"],
                   eval_acc["mean_acc"], self.cfg.TRAINER.CDPS.GLOBAL_WEIGHT))
            log_metrics({"bimc_only_acc": global_acc["mean_acc"],
                         "cdps_only_acc": local_acc["mean_acc"],
                         "hybrid_acc": eval_acc["mean_acc"]}, step=task_id) 
        print(f"Test acc mean: {eval_acc['mean_acc']}, task-wise acc: {eval_acc['task_acc']}")
        return eval_acc
    

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets
