"""Frozen CLIP with balanced ridge and either Base-trained or kernel transport."""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from engine.engine import Runner
from single_flow.balanced_random_ridge import BalancedRandomFeatureRidge, shrink_visual_prototype
from single_flow.flow_utils import transport
from single_flow.frozen_random_ridge import fuse_flow_ridge
from single_flow.kernel_velocity import KernelVelocityField, kernel_transport
from single_flow.text_distribution_flow import distributions, fit, load_model
from utils.experiment_logger import log_metrics


class BalancedKernelFlowRunner(Runner):
    @torch.no_grad()
    def run(self):
        cfg = self.cfg
        if cfg.LORA.ENABLED or cfg.FLOW.ADAPT_STEPS != 0 or cfg.FLOW.TTA or cfg.FLOW.TTA_PROTOTYPES:
            raise ValueError('Requires frozen CLIP, no Flow adapt, and a single image view')
        if cfg.FLOW.DIAGONAL_WHITEN or cfg.TRAINER.BiMC.VISION_CALIBRATION:
            raise ValueError('Disable ZCA and legacy vision calibration')
        if cfg.GEOMETRY.FLOW_TYPE not in ('mlp', 'kernel'):
            raise ValueError('GEOMETRY.FLOW_TYPE must be mlp or kernel')
        if (cfg.GEOMETRY.KAPPA < 0 or
                (cfg.FEATURE.PATCH_POOLING and not cfg.FEATURE.MULTILAYER) or
                not 0 <= cfg.FEATURE.PATCH_WEIGHT <= 1 or
                cfg.FEATURE.PATCH_TEMPERATURE <= 0 or
                cfg.FEATURE.DESCRIPTION_CHUNK < 1):
            raise ValueError('Invalid prototype shrinkage or frozen feature settings')
        if not 0 <= cfg.RIDGE.BLEND_WEIGHT <= 1:
            raise ValueError('RIDGE.BLEND_WEIGHT must be in [0, 1]')
        model = self.model.module if self.is_distributed else self.model
        model.eval().requires_grad_(False)
        output = (Path(cfg.FLOW.OUTPUT_DIR) / cfg.DATASET.NAME.lower()
                  / f'seed_{cfg.SEED}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}')
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        field = (KernelVelocityField(cfg.GEOMETRY.KERNEL_BANDWIDTH).to(self.device)
                 if cfg.GEOMETRY.FLOW_TYPE == 'kernel' else None)
        ridge = bank = None
        rows = []
        for task_id, class_ids in enumerate(self.data_manager.class_index_in_task):
            started = time.time()
            names = np.asarray(self.data_manager.class_names)[class_ids]
            loader = self.data_manager.get_dataloader(
                task_id, source='train', mode='test', accumulate_past=False)
            stats = model.build_task_statistics(
                names, loader, class_index=class_ids,
                calibrate_novel_vision_proto=False, calculate_covariance=False)
            text = F.normalize(stats['text_proto'].detach().float(), dim=-1).to(self.device)
            real = F.normalize(stats['image_proto'].detach().float(), dim=-1).to(self.device)
            features, labels = stats['images_features'], stats['images_targets']

            if task_id == 0 and cfg.GEOMETRY.FLOW_TYPE == 'mlp':
                memory = distributions(stats, include_covariance=False)
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    checkpoint = fit(
                        memory, epochs=cfg.FLOW.EPOCHS,
                        steps_per_epoch=cfg.FLOW.STEPS_PER_EPOCH,
                        batch_size=cfg.FLOW.BATCH_SIZE, lr=cfg.FLOW.LR,
                        weight_decay=cfg.FLOW.WEIGHT_DECAY, blocks=cfg.FLOW.BLOCKS,
                        hidden=cfg.FLOW.HIDDEN or None, seed=cfg.SEED,
                        device=self.device, noise_scale=cfg.FLOW.NOISE_SCALE,
                        funnel_mode=True, warmup_ratio=cfg.FLOW.WARMUP_RATIO,
                        min_lr_ratio=cfg.FLOW.MIN_LR_RATIO)
                torch.save(checkpoint, output / 'base_flow.pt')
                field = load_model(checkpoint, self.device).eval().requires_grad_(False)

            old_count = 0 if bank is None else len(bank['class_ids'])
            if cfg.GEOMETRY.FLOW_TYPE == 'kernel':
                field.append(text, real)
                excluded = torch.arange(old_count, old_count + len(text), device=self.device)
                generated = kernel_transport(field, text, cfg.FLOW.STEPS, exclude=excluded)
                torch.save({'text': field.text.cpu(), 'visual': field.visual.cpu(),
                            'bandwidth': field.bandwidth},
                           output / f'kernel_anchors_session{task_id}.pt')
            else:
                generated = transport(field, text, cfg.FLOW.STEPS, solver=cfg.FLOW.SOLVER)
            counts = torch.bincount(labels.to(self.device, torch.long) - old_count,
                                    minlength=len(class_ids))
            calibrated = shrink_visual_prototype(real, generated, counts, cfg.GEOMETRY.KAPPA)
            if ridge is None:
                ridge = BalancedRandomFeatureRidge(
                    features.shape[-1], cfg.RIDGE.FEATURE_DIM, cfg.RIDGE.LAMBDA,
                    cfg.SEED + 10007, self.device, cfg.RIDGE.TARGET_MODE)
            ridge.add_session(class_ids, features, labels, calibrated,
                              batch_size=cfg.RIDGE.BATCH_SIZE)
            ridge.solve()
            new_bank = dict(class_ids=[int(c) for c in class_ids],
                            text=text.cpu(), real=real.cpu(),
                            generated=generated.cpu(), prototypes=calibrated.cpu())
            bank = (new_bank if bank is None else
                    dict(class_ids=bank['class_ids'] + new_bank['class_ids'],
                         **{key: torch.cat((bank[key], new_bank[key]))
                            for key in ('text', 'real', 'generated', 'prototypes')}))
            torch.save(bank, output / f'bank_session{task_id}.pt')
            torch.save({'weights': ridge.weights.cpu(), 'class_ids': ridge.class_ids},
                       output / f'ridge_head_session{task_id}.pt')
            del stats, loader

            acc = self.inference_task(task_id, bank, field, ridge)
            self.acc_list.append(acc['mean_acc'])
            self.task_acc_list.append(acc['task_acc'])
            row = dict(task_id=task_id, seen_classes=len(bank['class_ids']),
                       real_support_features=ridge.observation_count,
                       **acc, elapsed_time=time.time() - started)
            rows.append(row)
            print(f'Task {task_id}: fused={acc["mean_acc"]:.2f}% '
                  f'ridge={acc["ridge_mean_acc"]:.2f}% '
                  f'text={acc["text_mean_acc"]:.2f}%', flush=True)
            log_metrics({k: v for k, v in row.items() if k != 'task_acc'}, step=task_id)
            self.save_results(output, rows)
        torch.save({'projection': ridge.projection.cpu(), 'gram': ridge.gram.cpu(),
                    'cross': ridge.cross.cpu(), 'weights': ridge.weights.cpu(),
                    'class_ids': ridge.class_ids, 'regularization': ridge.regularization,
                    'target_mode': ridge.target_mode}, output / 'ridge_final_state.pt')
        print(f'Final session Acc: {self.acc_list[-1]:.3f}')
        print(f'Average session Acc: {sum(self.acc_list) / len(self.acc_list):.3f}')
        print(f'Results directory: {output}')

    @torch.no_grad()
    def inference_task(self, task_id, bank, field, ridge):
        if bank['class_ids'] != ridge.class_ids:
            raise ValueError('Bank and ridge columns are misaligned')
        model = self.model.module if self.is_distributed else self.model
        loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        text_bank = bank['text'].to(self.device)
        fused_scores, ridge_scores, text_scores, targets = [], [], [], []
        for batch in tqdm(loader, desc=f'Test task {task_id}'):
            images, labels = self.parse_batch(batch)
            features = F.normalize(model.extract_img_feature(images, tta=False).float(), dim=-1)
            if self.cfg.GEOMETRY.FLOW_TYPE == 'kernel':
                reversed_features = kernel_transport(field, features, self.cfg.FLOW.STEPS,
                                                     reverse=True)
            else:
                reversed_features = transport(field, features, self.cfg.FLOW.STEPS,
                                              reverse=True, solver=self.cfg.FLOW.SOLVER)
            text_logits = reversed_features @ text_bank.T
            ridge_logits = ridge.scores(features)
            fused = fuse_flow_ridge(text_logits, ridge_logits,
                                    self.cfg.RIDGE.BLEND_WEIGHT)
            text_scores.append(text_logits.cpu())
            ridge_scores.append(ridge_logits.cpu())
            fused_scores.append(fused.cpu())
            targets.append(labels.cpu())
        if not targets:
            raise ValueError('Empty test loader')
        labels = torch.cat(targets)
        acc = self.evaluator.calc_accuracy(torch.cat(fused_scores), labels, task_id)
        acc['ridge_mean_acc'] = self.evaluator.calc_accuracy(
            torch.cat(ridge_scores), labels, task_id)['mean_acc']
        acc['text_mean_acc'] = self.evaluator.calc_accuracy(
            torch.cat(text_scores), labels, task_id)['mean_acc']
        return acc

    def save_results(self, output, rows):
        summary = dict(method='balanced_kernel_flow', dataset=self.cfg.DATASET.NAME,
                       flow_type=self.cfg.GEOMETRY.FLOW_TYPE, seed=self.cfg.SEED,
                       completed_sessions=len(rows), total_sessions=self.data_manager.num_tasks,
                       complete=len(rows) == self.data_manager.num_tasks,
                       final_acc=rows[-1]['mean_acc'],
                       avg_acc=sum(row['mean_acc'] for row in rows) / len(rows),
                       sessions=rows)
        (output / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
