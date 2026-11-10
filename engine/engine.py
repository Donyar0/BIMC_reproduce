import torch
import torch.nn as nn
import models.clip as clip
from datasets.data_manager import DatasetManager
from torch.nn import functional as F
from tqdm import tqdm
from utils.evaluator import AccuracyEvaluator
from utils.finegrained_diagnostics import summarize_class_confusions
from utils.prototype_geometry import prototype_geometry
from models.vdt_self_attention import pack_description_bank, mean_description_prototypes
from utils.experiment_logger import log_metrics
from models.bimc import BiMC
import numpy as np
import time
import json


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
        self.bias_history = []
        self.evaluator = AccuracyEvaluator(self.data_manager.class_index_in_task)


    def merge_dicts(self, dict_list):
        result = {}

        keys_to_merge = [
            'description_proto', 
            'description_features', 
            'vdt_features',
            'vdt_targets',
            'description_targets',
            'text_features', 
            'text_targets',
            'image_proto',
            'support_proto',
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

            # Train attention once on session-0 images, before moving to later sessions.
            core_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            if i == 0:
                core_model.train_base_text_adapter(current_state_dict)

            state_dict_list.append(current_state_dict)
            merged_state_dict = self.merge_dicts(state_dict_list)
            # The original-BiMC control retains native prompt/description
            # embeddings; attention/mean branches replace only text_features.
            mode = self.cfg.TRAINER.BiMC.VDT_MODE
            if mode != "original":
                core_model.text_adapter.eval()
                with torch.no_grad():
                    if mode == "visual_weighted":
                        adapted_text, class_ids, _ = core_model.vdt_visual_weighted_prototypes(
                            merged_state_dict['vdt_features'],
                            merged_state_dict['vdt_targets'],
                            merged_state_dict['support_proto'],
                        )
                    else:
                        adapted_text, class_ids = core_model.vdt_prototypes(
                            merged_state_dict['vdt_features'],
                            merged_state_dict['vdt_targets']
                        )
                expected = torch.arange(len(adapted_text), device=class_ids.device)
                if not torch.equal(class_ids, expected):
                    raise ValueError("Unexpected class IDs when computing VDT text prototypes")
                merged_state_dict['text_features'] = adapted_text

                # Geometry is diagnostic only: compute at the final session
                # from frozen text descriptions and training/support prototypes.
                if (i == self.data_manager.num_tasks - 1 and
                        self.cfg.DATASET.NAME.lower() == "cub200"):
                    packed, valid, ids = pack_description_bank(
                        merged_state_dict['vdt_features'],
                        merged_state_dict['vdt_targets']
                    )
                    raw_desc_mean = mean_description_prototypes(packed, valid)
                    geometry = prototype_geometry(
                        raw_desc_mean, adapted_text,
                        merged_state_dict['support_proto'],
                        self.data_manager.class_names,
                        len(self.data_manager.class_index_in_task[0]),
                        top_k=40,
                    )
                    print("VDT_GEOMETRY_JSON:" + json.dumps(geometry, ensure_ascii=False))

            start_time = time.time()
            acc = self.inference_task_covariance(i, merged_state_dict)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f'+++++++++++  task {i}, time: {elapsed_time} ++++++++++++++++')

            print(f'=> Task [{i}], Acc: {acc["mean_acc"]:.3f}')
            self.acc_list.append(round(acc["mean_acc"], 3))
            self.task_acc_list.append(acc['task_acc'])
            self.bias_history.append(acc['bias'])
            print("BIAS_METRICS_JSON:" + json.dumps({"session": i, **acc['bias']}, sort_keys=True))
            if i == self.data_manager.num_tasks - 1:
                print("Final base acc:{:.2f} | Final novel mean acc:{:.2f} | Final harmonic acc:{:.2f}".format(
                    acc['base_avg_acc'], acc['inc_avg_acc'], acc['harmonic_acc']
                ))

            metrics = {
                'task_id': i,
                'mean_acc': acc['mean_acc'],
                'base_avg_acc': acc['base_avg_acc'],
                'inc_avg_acc': acc['inc_avg_acc'],
                'harmonic_acc': acc['harmonic_acc'],
                'elapsed_time': elapsed_time,
            }
            # None in the base session is not a valid numeric SwanLab metric.
            for metric in ('base_to_novel_pct', 'novel_to_base_pct',
                           'base_to_novel_count', 'novel_to_base_count'):
                if acc['bias'][metric] is not None:
                    metrics[metric] = acc['bias'][metric]
            log_metrics(metrics, step=i)

        print(f'Final acc:{self.acc_list}')
        print('FINAL_BIAS_JSON:' + json.dumps(self.bias_history[-1], sort_keys=True))
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

            all_logits.append(logits)
            all_targets.append(targets)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        eval_acc = self.evaluator.calc_accuracy(all_logits, all_targets, task_id)
        eval_acc['bias'] = self.evaluator.calc_base_novel_bias(all_logits, all_targets, task_id)
        # Only the final CUB200 session needs the full 200-way confusion
        # diagnostic; no test labels are used in any training path.
        if (self.cfg.DATASET.NAME.lower() == "cub200" and
                task_id == self.data_manager.num_tasks - 1):
            diagnostics = summarize_class_confusions(
                all_logits, all_targets, self.data_manager.class_names,
                self.data_manager.class_index_in_task[0], top_k=30
            )
            print("FINEGRAINED_DIAGNOSTICS_JSON:" + json.dumps(diagnostics, ensure_ascii=False))
 
        print(f"Test acc mean: {eval_acc['mean_acc']}, task-wise acc: {eval_acc['task_acc']}")
        return eval_acc
    

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets
