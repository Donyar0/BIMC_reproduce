"""Frozen CLIP and Base Flow with cumulative random-feature ridge classification."""
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
from single_flow.flow_utils import transport
from single_flow.frozen_random_ridge import (IncrementalRandomFeatureRidge,
                                             fuse_flow_ridge)
from single_flow.text_distribution_flow import (build_bank, classify_bidirectional,
                                                distributions, fit, load_model)
from utils.experiment_logger import log_metrics


def new_class_bank_statistics(stats):
    """Only prototypes and description features are needed after Base Flow fit."""
    visual = F.normalize(stats['image_proto'].detach().float(), dim=-1).cpu()
    return dict(
        class_ids=[int(c) for c in stats['class_index']],
        text=F.normalize(stats['text_proto'].detach().float(), dim=-1).cpu(),
        real=visual, bank_real=visual,
        description_features=F.normalize(
            stats['description_features'].detach().float(), dim=-1).cpu(),
        description_labels=stats['description_targets'].detach().long().cpu())


class FrozenRidgeFlowRunner(Runner):
    @torch.no_grad()
    def run(self):
        cfg = self.cfg
        flow = cfg.FLOW
        if cfg.LORA.ENABLED or flow.ADAPT_STEPS != 0:
            raise ValueError('This experiment requires frozen CLIP and zero Flow adaptation')
        if (flow.SCORE_MODE != 'cosine_sum' or flow.DIAGONAL_WHITEN
                or flow.BIMC_BLEND_LAMBDA >= 0 or flow.USE_GEOMETRY_PRIORS):
            raise ValueError('Ridge fusion requires raw bidirectional cosine Flow scores')
        if not 0 <= cfg.RIDGE.BLEND_WEIGHT <= 1:
            raise ValueError('RIDGE.BLEND_WEIGHT must be in [0, 1]')
        model = self.model.module if self.is_distributed else self.model
        model.eval().requires_grad_(False)
        run_name = f'seed_{cfg.SEED}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        output = Path(flow.OUTPUT_DIR) / cfg.DATASET.NAME.lower() / run_name
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        checkpoint = frozen_flow = bank = ridge = None
        rows = []
        num_base = len(self.data_manager.class_index_in_task[0])
        print(f'Frozen CLIP + Base Flow + incremental ridge | Dataset: {cfg.DATASET.NAME}')
        print(f'Results directory: {output}')

        for task_id in range(self.data_manager.num_tasks):
            started = time.time()
            class_ids = self.data_manager.class_index_in_task[task_id]
            names = np.array(self.data_manager.class_names)[class_ids]
            loader = self.data_manager.get_dataloader(
                task_id, source='train', mode='test', accumulate_past=False)
            stats = model.build_task_statistics(
                names, loader, class_index=class_ids,
                calibrate_novel_vision_proto=cfg.TRAINER.BiMC.VISION_CALIBRATION,
                calculate_covariance=False)
            real_features = stats['images_features']
            real_labels = stats['images_targets']

            if task_id == 0:
                current = distributions(stats, include_covariance=False)
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    checkpoint = fit(
                        current, epochs=flow.EPOCHS,
                        steps_per_epoch=flow.STEPS_PER_EPOCH,
                        batch_size=flow.BATCH_SIZE, lr=flow.LR,
                        weight_decay=flow.WEIGHT_DECAY, blocks=flow.BLOCKS,
                        hidden=flow.HIDDEN or None, seed=cfg.SEED,
                        device=self.device, noise_scale=flow.NOISE_SCALE,
                        visual_scale=flow.VISUAL_SCALE,
                        max_radius=flow.MAX_RADIUS,
                        contrast_weight=flow.CONTRAST_WEIGHT,
                        contrast_temperature=flow.CONTRAST_TEMPERATURE,
                        funnel_mode=flow.FUNNEL_MODE,
                        warmup_ratio=flow.WARMUP_RATIO,
                        min_lr_ratio=flow.MIN_LR_RATIO)
                torch.save(checkpoint, output / 'base_flow.pt')
                frozen_flow = load_model(checkpoint, self.device).eval().requires_grad_(False)
                ridge = IncrementalRandomFeatureRidge(
                    real_features.shape[-1], cfg.RIDGE.FEATURE_DIM,
                    cfg.RIDGE.LAMBDA, cfg.SEED + 10007, self.device)
                ridge.add_session(class_ids, [(real_features, real_labels)],
                                  batch_size=cfg.RIDGE.BATCH_SIZE)
                synthetic_count = 0
            else:
                current = new_class_bank_statistics(stats)
                descriptions = current['description_features'].to(self.device)
                generated = torch.cat([
                    transport(frozen_flow, part, flow.STEPS, solver=flow.SOLVER)
                    for part in descriptions.split(cfg.RIDGE.BATCH_SIZE)])
                synthetic_count = len(generated)
                counts = ridge.add_session(
                    class_ids,
                    [(real_features, real_labels),
                     (generated, current['description_labels'])],
                    batch_size=cfg.RIDGE.BATCH_SIZE)
                print(f'Session {task_id}: {len(real_features)} real + '
                      f'{synthetic_count} Flow-translated descriptions; '
                      f'per-class totals={counts}', flush=True)

            ridge.solve()
            bank = build_bank(
                checkpoint, current, steps=flow.STEPS,
                num_base=num_base if task_id == 0 else 0,
                beta_base=flow.BETA_BASE, beta_inc=flow.BETA_INC,
                device=self.device, previous_bank=bank, solver=flow.SOLVER,
                bank_mode=flow.BANK_MODE)
            torch.save(bank, output / f'bank_session{task_id}.pt')
            torch.save({'weights': ridge.weights.detach().cpu(),
                        'class_ids': ridge.class_ids},
                       output / f'ridge_head_session{task_id}.pt')
            del stats, loader, current

            acc = self.inference_task(task_id, bank, frozen_flow, ridge)
            self.acc_list.append(acc['mean_acc'])
            self.task_acc_list.append(acc['task_acc'])
            row = dict(task_id=task_id, seen_classes=len(bank['class_ids']),
                       synthetic_train_features=synthetic_count,
                       ridge_train_features=ridge.observation_count,
                       **acc, elapsed_time=time.time() - started)
            rows.append(row)
            print(f'=> Task [{task_id}], Acc: {acc["mean_acc"]:.2f}% | '
                  f'Flow: {acc["flow_mean_acc"]:.2f}% | '
                  f'Ridge: {acc["ridge_mean_acc"]:.2f}%', flush=True)
            log_metrics({key: value for key, value in row.items() if key != 'task_acc'},
                        step=task_id)
            self.save_results(output, rows)

        torch.save({'projection': ridge.projection.cpu(),
                    'gram': ridge.gram.cpu(),
                    'cross': ridge.cross.cpu(),
                    'weights': ridge.weights.cpu(),
                    'class_ids': ridge.class_ids,
                    'regularization': ridge.regularization},
                   output / 'ridge_final_state.pt')
        average = sum(self.acc_list) / len(self.acc_list)
        print(f'Final session Acc: {self.acc_list[-1]:.3f}')
        print(f'Average session Acc: {average:.3f}')
        print(f'Results saved to: {output / "metrics.csv"}')
        log_metrics({'final_acc': self.acc_list[-1], 'avg_acc': average})

    @torch.no_grad()
    def inference_task(self, task_id, bank, flow_model, ridge):
        if bank['class_ids'] != ridge.class_ids:
            raise ValueError('Flow Bank and Ridge columns have different class orders')
        device_bank = dict(bank, prototypes=bank['prototypes'].to(self.device),
                           text=bank['text'].to(self.device))
        model = self.model.module if self.is_distributed else self.model
        loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        flow_scores, ridge_scores, fused_scores, targets = [], [], [], []
        for batch in tqdm(loader, desc=f'Test task {task_id}'):
            images, labels = self.parse_batch(batch)
            features = model.extract_img_feature(images, tta=self.cfg.FLOW.TTA)
            _, flow_logits, _ = classify_bidirectional(
                flow_model, features, device_bank, alpha=self.cfg.FLOW.ALPHA,
                gamma=self.cfg.FLOW.GAMMA, steps=self.cfg.FLOW.STEPS,
                solver=self.cfg.FLOW.SOLVER, score_mode='cosine_sum',
                text_whitening_mode='none')
            ridge_logits = ridge.scores(features)
            fused = fuse_flow_ridge(flow_logits, ridge_logits,
                                    self.cfg.RIDGE.BLEND_WEIGHT)
            flow_scores.append(flow_logits.cpu())
            ridge_scores.append(ridge_logits.cpu())
            fused_scores.append(fused.cpu())
            targets.append(labels.cpu())
        if not targets:
            raise ValueError('Empty test loader')
        labels = torch.cat(targets)
        acc = self.evaluator.calc_accuracy(torch.cat(fused_scores), labels, task_id)
        acc['flow_mean_acc'] = self.evaluator.calc_accuracy(
            torch.cat(flow_scores), labels, task_id)['mean_acc']
        acc['ridge_mean_acc'] = self.evaluator.calc_accuracy(
            torch.cat(ridge_scores), labels, task_id)['mean_acc']
        return acc

    def save_results(self, output, rows):
        summary = dict(method='frozen_random_ridge_flow', dataset=self.cfg.DATASET.NAME,
                       seed=self.cfg.SEED, accuracy_unit='percent',
                       completed_sessions=len(rows),
                       total_sessions=self.data_manager.num_tasks,
                       complete=len(rows) == self.data_manager.num_tasks,
                       final_acc=rows[-1]['mean_acc'],
                       avg_acc=sum(row['mean_acc'] for row in rows) / len(rows),
                       sessions=rows)
        (output / 'metrics.json').write_text(json.dumps(summary, indent=2),
                                             encoding='utf-8')
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(
                key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)
