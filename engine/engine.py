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

        if all(d.get('residual_correction') is not None for d in dict_list):
            result['residual_correction'] = torch.cat([d['residual_correction'] for d in dict_list], dim=0)
        else:
            result['residual_correction'] = None

        # Per-class dual-consistency tensors follow task class order.
        dual_keys = ('dual_alpha', 'dual_beta', 'dual_image_proto', 'dual_tt', 'dual_tv')
        for key in dual_keys:
            if all(d.get(key) is not None for d in dict_list):
                result[key] = torch.cat([d[key] for d in dict_list], dim=0)
            else:
                result[key] = None

        # PGA tensors are class aligned: [C_task,D,K] and [C_task,D].
        # Concatenate per-task statistics into [C_all,D,K] and [C_all,D].
        if all(d.get('pga_basis') is not None for d in dict_list):
            result['pga_basis'] = torch.cat([d['pga_basis'] for d in dict_list], dim=0)
            result['pga_centers'] = torch.cat([d['pga_centers'] for d in dict_list], dim=0)
        else:
            result['pga_basis'] = None
            result['pga_centers'] = None


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

            counts = current_state_dict.get('image_description_counts')
            if counts is not None:
                diag = {
                    'text_calibration/image_caption_count_mean': sum(counts) / len(counts),
                    'text_calibration/image_caption_count_min': min(counts),
                    'text_calibration/image_caption_count_max': max(counts),
                    'text_calibration/image_caption_coverage': sum(n > 0 for n in counts) / len(counts),
                }
                if current_state_dict.get('dual_alpha') is not None:
                    for key in ('dual_alpha', 'dual_beta', 'dual_tt', 'dual_tv'):
                        vals = current_state_dict[key].float()
                        diag['dual/' + key + '_mean'] = vals.mean().item()
                        diag['dual/' + key + '_base_or_session_min'] = vals.min().item()
                        diag['dual/' + key + '_base_or_session_max'] = vals.max().item()
                else:
                    diag['text_calibration/fusion_alpha'] = self.cfg.DATASET.DESCRIPTION_FUSION_ALPHA
                    diag['text_calibration/label_image_cosine_mean'] = current_state_dict['description_fusion_cosine']
                log_metrics(diag, step=i)

            if current_state_dict.get('residual_counts') is not None:
                n = current_state_dict['residual_counts']
                possible = current_state_dict['residual_possible']
                scores = current_state_dict['residual_scores']
                log_metrics({
                    'residual/matched_support_count': sum(n),
                    'residual/support_total': sum(possible),
                    'residual/caption_coverage': sum(n) / max(sum(possible), 1),
                    'residual/classes_with_captions': sum(x > 0 for x in n),
                    'residual/positive_agreement_mean': sum(scores) / len(scores),
                    'residual/correction_norm_mean':
                        current_state_dict['residual_correction'].float().norm(dim=-1).mean().item(),
                }, step=i)
                print(f'Residual coverage task {i}: {sum(n)}/{sum(possible)} supports')

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
        all_targets = []
        pga_base_sum = pga_inc_sum = pga_true_sum = pga_margin_sum = 0.0
        pga_count = inc_total = inc_to_base = inc_wrong = 0

        for i, batch in enumerate(tqdm(test_loader)):
            data, targets = self.parse_batch(batch)
            logits, pga_scores = self.model.forward_ours(data, num_accumulated_class, num_base_class,
                                                   image_proto, 
                                                   cov_image,
                                                   description_proto,
                                                   description_features, 
                                                   description_targets,
                                                   text_features,
                                                   beta=beta,
                                                   pga_basis=state_dict['pga_basis'],  # [C_all,D,K] or None
                                                   pga_centers=state_dict['pga_centers'],  # [C_all,D] or None
                                                   return_pga_scores=True,
                                                   dual_alpha=state_dict['dual_alpha'],
                                                   dual_beta=state_dict['dual_beta'],
                                                   dual_image_proto=state_dict['dual_image_proto'],
                                                   residual_correction=state_dict['residual_correction'])


            # logits [B,C]; pga_scores [B,C] (or None); targets [B].
            preds = logits.argmax(dim=1)  # [B]
            inc_mask = targets >= num_base_class  # [B]
            if inc_mask.any():
                inc_targets = targets[inc_mask]  # [N_inc]
                inc_preds = preds[inc_mask]  # [N_inc]
                n_inc = inc_targets.numel()
                inc_total += n_inc
                inc_wrong += (inc_preds != inc_targets).sum().item()
                inc_to_base += (inc_preds < num_base_class).sum().item()
                if pga_scores is not None:
                    scores = pga_scores[inc_mask].float()  # [N_inc,C]
                    base_scores = scores[:, :num_base_class]  # [N_inc,C_base]
                    novel_scores = scores[:, num_base_class:]  # [N_inc,C_inc]
                    true_scores = scores.gather(
                        1, inc_targets.long().unsqueeze(1)
                    ).squeeze(1)  # [N_inc]
                    strongest_base = base_scores.max(dim=1).values  # [N_inc]
                    pga_base_sum += base_scores.mean(dim=1).sum().item()
                    pga_inc_sum += novel_scores.mean(dim=1).sum().item()
                    pga_true_sum += true_scores.sum().item()
                    pga_margin_sum += (true_scores - strongest_base).sum().item()
                    pga_count += n_inc

            all_logits.append(logits)
            all_targets.append(targets)

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        # Track P(predicted Base | true incremental) even with PGA disabled.
        if inc_total > 0:
            diag = {
                "bias/inc_to_base_rate": inc_to_base / inc_total,
                "bias/inc_error_rate": inc_wrong / inc_total,
                "bias/inc_to_base_among_errors": inc_to_base / max(inc_wrong, 1),
            }
            if pga_count > 0:
                diag.update({
                    "pga/base_score_mean": pga_base_sum / pga_count,
                    "pga/inc_score_mean": pga_inc_sum / pga_count,
                    "pga/true_score_mean": pga_true_sum / pga_count,
                    "pga/true_vs_base_margin": pga_margin_sum / pga_count,
                })
            log_metrics(diag, step=task_id)
            print(f"Task {task_id} PGA/bias diagnostics: {diag}")

        eval_acc = self.evaluator.calc_accuracy(all_logits, all_targets, task_id) 
        print(f"Test acc mean: {eval_acc['mean_acc']}, task-wise acc: {eval_acc['task_acc']}")
        return eval_acc
    

    def parse_batch(self, batch):
        data = batch['image']
        targets = batch['label']
        data = data.to(self.device)
        targets = targets.to(self.device)
        return data, targets
