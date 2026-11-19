"""Distributional text-to-visual Flow runner with exemplar-free replay."""
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
from single_flow.text_distribution_flow import (fit, adapt, distributions, combine, build_bank,
                                                classify_bidirectional, load_model, METHOD, DIRECTION)
from single_flow.flow_utils import transport
from utils.experiment_logger import log_metrics
from engine.lora_base import train_base_vision_lora


class TangentFlowRunner(Runner):
    @torch.no_grad()
    def extract_test_features(self, model, images):
        return model.extract_img_feature(images, tta=self.cfg.FLOW.TTA)

    @torch.no_grad()
    def run(self):
        cfg = self.cfg
        flow = cfg.FLOW
        if flow.ADAPT_STEPS < 0:
            raise ValueError('ADAPT_STEPS must be nonnegative')
        if flow.SOLVER not in ('euler', 'midpoint', 'heun', 'rk4'):
            raise ValueError('Unknown FLOW.SOLVER')
        if not cfg.TRAINER.BiMC.TEXT_CALIBRATION:
            raise ValueError('Enable original BiMC description fusion: TEXT_CALIBRATION: True')
        model = self.model.module if self.is_distributed else self.model
        model.eval().requires_grad_(False)
        run_name = f'seed_{cfg.SEED}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        output = Path(flow.OUTPUT_DIR) / cfg.DATASET.NAME.lower() / run_name
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        encoder_id = str((cfg.MODEL.BACKBONE.NAME, cfg.TRAINER.BiMC.PREC, str(cfg.DATASET),
                          METHOD, DIRECTION, str(cfg.TRAINER.BiMC), str(cfg.LORA)))
        rows, checkpoint, bank, memory = [], None, None, None
        bimc_stats = []
        num_base = len(self.data_manager.class_index_in_task[0])
        print(f'Method: tangent_flow | Dataset: {cfg.DATASET.NAME} | Seed: {cfg.SEED}')
        print(f'Results directory: {output}')
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')

        for task_id in range(self.data_manager.num_tasks):
            start = time.time()
            if task_id == 0 and cfg.LORA.ENABLED:
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    history = train_base_vision_lora(model, self.data_manager, cfg, output)
                (output / 'base_lora_metrics.json').write_text(
                    json.dumps({'loss_history': history}, indent=2), encoding='utf-8')
            class_ids = self.data_manager.class_index_in_task[task_id]
            names = np.array(self.data_manager.class_names)[class_ids]
            loader = self.data_manager.get_dataloader(
                task_id, source='train', mode='test', accumulate_past=False)
            stats = model.build_task_statistics(
                names, loader, class_index=class_ids,
                calibrate_novel_vision_proto=cfg.TRAINER.BiMC.VISION_CALIBRATION,
                calculate_covariance=flow.BIMC_BLEND_LAMBDA >= 0)
            if task_id == 0 and flow.DIAGONAL_WHITEN:
                base_features = F.normalize(stats['images_features'].detach().float().cpu(), dim=-1)
                base_labels = stats['images_targets'].detach().long().cpu()
                centered = torch.empty_like(base_features)
                for class_id in class_ids:
                    mask = base_labels == int(class_id)
                    centered[mask] = base_features[mask] - base_features[mask].mean(0)
                n_eff = max(len(centered) - len(class_ids), 1)
                cov = (centered.T @ centered) / n_eff
                diag_cov = torch.diag(cov.diagonal().clamp_min(1e-6))
                dim = cov.shape[0]
                shrink = 0.2
                print(f'Base ZCA covariance: n_eff={n_eff}, dim={dim}, shrink={shrink:.4f}')
                cov_reg = (1.0 - shrink) * cov + shrink * diag_cov
                eigvals, eigvecs = torch.linalg.eigh(cov_reg)
                eigvals = eigvals.clamp_min(1e-5)
                self._zca_matrix = ((eigvecs * eigvals.pow(-0.5 * flow.VIS_WHITEN_POWER))
                                    @ eigvecs.T).contiguous()
                torch.save(self._zca_matrix, output / 'base_zca_matrix.pt')
            if flow.BIMC_BLEND_LAMBDA >= 0:
                bimc_stats.append(stats)
            current = distributions(stats, base=memory if task_id else None,
                                    shrink=flow.COV_SHRINK,
                                    transfer_temperature=flow.COV_TRANSFER_TEMPERATURE)
            if flow.BANK_MODE == 'bimc_incremental' and task_id:
                lam = cfg.TRAINER.BiMC.LAMBDA_T
                current['bank_text'] = ((1-lam)*stats['text_features'].detach().float()
                                        +lam*stats['description_proto'].detach().float()).cpu()
            if cfg.TRAINER.BiMC.PROTOTYPE_CACHE_DIR:
                cache_dir = Path(cfg.TRAINER.BiMC.PROTOTYPE_CACHE_DIR) / cfg.DATASET.NAME.lower() / run_name
                cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(current, cache_dir / f'session_{task_id}.pt')
            memory = current if memory is None else combine(memory, current)
            devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices), torch.enable_grad():
                if task_id == 0:
                    checkpoint = fit(memory, epochs=flow.EPOCHS,
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
                    if flow.DIAGONAL_WHITEN and flow.TEXT_WHITEN_MODE == 'own':
                        inverse_model = load_model(checkpoint, self.device)
                        base_mapped = []
                        for part in base_features.split(256):
                            base_mapped.append(transport(inverse_model, part.to(self.device),
                                                         flow.STEPS, reverse=True,
                                                         solver=flow.SOLVER).cpu())
                        base_mapped = torch.cat(base_mapped)
                        text_rows = {int(c): row for row, c in enumerate(current['class_ids'])}
                        target_rows = torch.tensor([text_rows[int(c)] for c in base_labels])
                        residual = base_mapped - current['text'][target_rows]
                        self._text_whitening_sigma = residual.square().mean(0).sqrt().clamp_min(1e-4)
                        torch.save(self._text_whitening_sigma, output / 'base_text_space_std.pt')
                elif flow.ADAPT_STEPS:
                    checkpoint = adapt(checkpoint, memory, steps=flow.ADAPT_STEPS,
                                       batch_size=flow.BATCH_SIZE, lr=flow.ADAPT_LR,
                                       weight_decay=flow.WEIGHT_DECAY,
                                       seed=cfg.SEED+task_id, device=self.device,
                                       noise_scale=flow.NOISE_SCALE,
                                       visual_scale=flow.VISUAL_SCALE,
                                       max_radius=flow.MAX_RADIUS,
                                       replay_fraction=flow.REPLAY_FRACTION,
                                       replay_strategy=flow.REPLAY_STRATEGY,
                                       contrast_weight=flow.CONTRAST_WEIGHT,
                                       contrast_temperature=flow.CONTRAST_TEMPERATURE,
                                       funnel_mode=flow.FUNNEL_MODE,
                                       distill_weight=flow.DISTILL_WEIGHT,
                                       warmup_ratio=flow.WARMUP_RATIO,
                                       min_lr_ratio=flow.MIN_LR_RATIO)
            bank = build_bank(checkpoint, current, steps=flow.STEPS,
                              num_base=num_base if task_id == 0 else 0,
                              beta_base=flow.BETA_BASE, beta_inc=flow.BETA_INC,
                              device=self.device, previous_bank=bank, solver=flow.SOLVER,
                              bank_mode=flow.BANK_MODE)
            torch.save(checkpoint, output / f'flow_session{task_id}.pt')
            torch.save(bank, output / f'bank_session{task_id}.pt')
            bimc_state = self.merge_dicts(bimc_stats) if bimc_stats else None
            del stats, loader
            self._checkpoint = checkpoint
            acc = self.inference_task_flow(task_id, bank, memory, model, bimc_state)
            elapsed = time.time() - start
            self.acc_list.append(acc['mean_acc'])
            self.task_acc_list.append(acc['task_acc'])
            row = dict(task_id=task_id, seen_classes=len(bank['class_ids']),
                       **acc, elapsed_time=elapsed)
            rows.append(row)
            print(f'=> Task [{task_id}], Acc: {acc["mean_acc"]:.3f}')
            print(f'Base Acc: {acc["base_avg_acc"]:.2f} | '
                  f'Incremental Acc: {acc["inc_avg_acc"]:.2f} | '
                  f'Harmonic Acc: {acc["harmonic_acc"]:.2f}')
            print(f'+++++++++++  task {task_id}, time: {elapsed:.3f} ++++++++++++++++')
            log_metrics({k:v for k,v in row.items() if k != 'task_acc'}, step=task_id)
            self.save_results(output, rows)

        print(f'Final acc:{self.acc_list}')
        print('Task-wise acc:')
        for i, task_acc in enumerate(self.task_acc_list):
            print(f'task {i:2d}, acc:{task_acc}')
        average = sum(self.acc_list) / len(self.acc_list)
        print(f'Final session Acc: {self.acc_list[-1]:.3f}')
        print(f'Average session Acc: {average:.3f}')
        print(f'Results saved to: {output / "metrics.csv"}')
        log_metrics({'final_acc': self.acc_list[-1], 'avg_acc': average})

    def evaluate_saved(self, run_dir, trusted_checkpoints=False):
        raise NotImplementedError('Saved-run reevaluation is not implemented for tangent_flow')

    @torch.no_grad()
    def inference_task_flow(self, task_id, bank, memory, model, bimc_state=None):
        # The existing evaluator interprets score column index as global class ID.
        expected = [int(c) for group in self.data_manager.class_index_in_task[:task_id+1]
                    for c in group]
        if bank['class_ids'] != expected or expected != list(range(len(expected))):
            raise ValueError('BIMC evaluator requires ordered, contiguous seen-class IDs')
        loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        scores, targets, errors = [], [], []
        bimc_scores, flow_scores = [], []
        blend_lambda = self.cfg.FLOW.BIMC_BLEND_LAMBDA
        if blend_lambda >= 0 and not 0 <= blend_lambda <= 1:
            raise ValueError('BIMC_BLEND_LAMBDA must be in [0,1]')
        ablations = {name: [] for name in ('visual', 'text', 'flow',
                                          'visual_real_new', 'visual_flow_new')} if task_id == 1 else None
        base_errors, incremental_errors = [], []
        velocity = load_model(checkpoint=self._checkpoint, device=self.device).eval()
        use_priors = self.cfg.FLOW.USE_GEOMETRY_PRIORS
        cov_inv = (torch.linalg.pinv(memory['cov'].float().mean(0).to(self.device))
                   if use_priors else None)
        device_bank = dict(bank, prototypes=bank['prototypes'].to(self.device), text=bank['text'].to(self.device))
        if task_id == 1:
            num_base = len(self.data_manager.class_index_in_task[0])
            real_bank = device_bank['prototypes'].clone()
            flow_bank = device_bank['prototypes'].clone()
            real_bank[num_base:] = F.normalize(bank['real'][num_base:].to(self.device), dim=-1)
            flow_bank[num_base:] = F.normalize(bank['generated'][num_base:].to(self.device), dim=-1)
            if self.cfg.FLOW.DIAGONAL_WHITEN:
                zca = self._zca_matrix.to(self.device)
                real_bank = F.normalize(real_bank @ zca, dim=-1)
                flow_bank = F.normalize(flow_bank @ zca, dim=-1)
        for batch in tqdm(loader):
            images, labels = self.parse_batch(batch)
            features = self.extract_test_features(model, images)
            result = classify_bidirectional(velocity, features, device_bank,
                                               alpha=self.cfg.FLOW.ALPHA, gamma=self.cfg.FLOW.GAMMA,
                                               steps=self.cfg.FLOW.STEPS,
                                               solver=self.cfg.FLOW.SOLVER,
                                               cycle_scale=self.cfg.FLOW.CYCLE_SCALE,
                                               ensemble_alpha=(1.0 if blend_lambda >= 0 else
                                                               self.cfg.DATASET.ENSEMBLE_ALPHA if use_priors else 1.0),
                                               num_base=len(self.data_manager.class_index_in_task[0]),
                                               cov_inv=cov_inv,
                                               cycle_gate=self.cfg.FLOW.CYCLE_GATE,
                                               cosine_temperature=self.cfg.FLOW.COSINE_TEMPERATURE,
                                               knn_temperature=self.cfg.FLOW.KNN_TEMPERATURE,
                                               cov_divisor=self.cfg.FLOW.COV_DIVISOR,
                                               gamma_base=(self.cfg.FLOW.GAMMA_BASE if self.cfg.FLOW.GAMMA_BASE >= 0 else None),
                                               gamma_inc=(self.cfg.FLOW.GAMMA_INC if self.cfg.FLOW.GAMMA_INC >= 0 else None),
                                               score_mode=self.cfg.FLOW.SCORE_MODE,
                                               whitening_matrix=(self._zca_matrix
                                                                 if self.cfg.FLOW.DIAGONAL_WHITEN else None),
                                               visual_whitening_power=self.cfg.FLOW.VIS_WHITEN_POWER,
                                               text_whitening_mode=self.cfg.FLOW.TEXT_WHITEN_MODE,
                                               text_whitening_sigma=(self._text_whitening_sigma
                                                                     if self.cfg.FLOW.DIAGONAL_WHITEN and
                                                                        self.cfg.FLOW.TEXT_WHITEN_MODE == 'own' else None),
                                               return_components=task_id == 1)
            if task_id == 1:
                _, logits, cycle_error, components = result
                for name in ('visual', 'text', 'flow'):
                    ablations[name].append(components[name].cpu())
                z = F.normalize(features.float(), dim=-1)
                if self.cfg.FLOW.DIAGONAL_WHITEN:
                    z = F.normalize(z @ zca, dim=-1)
                ablations['visual_real_new'].append((z @ real_bank.T).cpu())
                ablations['visual_flow_new'].append((z @ flow_bank.T).cpu())
            else:
                _, logits, cycle_error = result
            if blend_lambda >= 0:
                prob_bimc = model.forward_ours(
                    images, len(bank['class_ids']), len(self.data_manager.class_index_in_task[0]),
                    bimc_state['image_proto'], bimc_state['cov_image'],
                    bimc_state['description_proto'], bimc_state['description_features'],
                    bimc_state['description_targets'], bimc_state['text_features'],
                    beta=self.cfg.DATASET.BETA)
                bimc_scores.append(prob_bimc.cpu())
                flow_scores.append(logits.cpu())
                logits = (1-blend_lambda)*prob_bimc + blend_lambda*logits
            errors.append(cycle_error.cpu())
            base_mask = labels.cpu() < len(self.data_manager.class_index_in_task[0])
            base_errors.append(cycle_error.cpu()[base_mask])
            incremental_errors.append(cycle_error.cpu()[~base_mask])
            scores.append(logits.cpu())
            targets.append(labels.cpu())
        if not scores:
            raise ValueError('Empty test loader')
        acc = self.evaluator.calc_accuracy(torch.cat(scores), torch.cat(targets), task_id)
        if blend_lambda >= 0:
            all_targets = torch.cat(targets)
            acc['bimc_mean_acc'] = self.evaluator.calc_accuracy(torch.cat(bimc_scores), all_targets, task_id)['mean_acc']
            acc['flow_mean_acc'] = self.evaluator.calc_accuracy(torch.cat(flow_scores), all_targets, task_id)['mean_acc']
            print(f'BiMC={acc["bimc_mean_acc"]:.2f}% Flow={acc["flow_mean_acc"]:.2f}% '
                  f'Blend={acc["mean_acc"]:.2f}% lambda={blend_lambda:.2f}')
        acc['cycle_error_mean'] = torch.cat(errors).mean().item()
        acc['cycle_error_base_mean'] = torch.cat(base_errors).mean().item()
        acc['cycle_error_incremental_mean'] = (torch.cat(incremental_errors).mean().item()
                                                if task_id > 0 else 0.0)
        if ablations is not None:
            all_targets = torch.cat(targets)
            for name, chunks in ablations.items():
                result = self.evaluator.calc_accuracy(torch.cat(chunks), all_targets, task_id)
                acc[f'{name}_task0_acc'] = result['task_acc'][0]
                acc[f'{name}_task1_acc'] = result['task_acc'][1]
                print(f'Task 1 ablation {name}: task0={result["task_acc"][0]:.2f}% '
                      f'task1={result["task_acc"][1]:.2f}%')
            print(f'Task 1 full: task0={acc["task_acc"][0]:.2f}% '
                  f'task1={acc["task_acc"][1]:.2f}%')
        print(f'Cycle error: base={acc["cycle_error_base_mean"]:.6g}, '
              f'incremental={acc["cycle_error_incremental_mean"]:.6g}, '
              f'gate={self.cfg.FLOW.CYCLE_GATE}')
        print(f'Test acc mean: {acc["mean_acc"]}, task-wise acc: {acc["task_acc"]}')
        return acc

    def save_results(self, output, rows):
        summary = dict(method='tangent_flow', dataset=self.cfg.DATASET.NAME, seed=self.cfg.SEED,
                       accuracy_unit='percent', completed_sessions=len(rows),
                       total_sessions=self.data_manager.num_tasks,
                       complete=len(rows) == self.data_manager.num_tasks,
                       final_acc=rows[-1]['mean_acc'],
                       avg_acc=sum(r['mean_acc'] for r in rows)/len(rows), sessions=rows)
        (output / 'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
            writer.writeheader()
            writer.writerows(rows)
