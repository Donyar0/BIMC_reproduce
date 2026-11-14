"""Base-only tangent text-to-visual Flow runner."""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from engine.engine import Runner
from single_flow.bimc_cache import from_task_statistics
from single_flow.text_distribution_flow import fit, build_bank, classify_bidirectional, load_model, description_std, METHOD, DIRECTION
from single_flow.unified_flow import real_means
from utils.experiment_logger import log_metrics


class TangentFlowRunner(Runner):
    @torch.no_grad()
    def run(self):
        cfg = self.cfg
        flow = cfg.FLOW
        if flow.ADAPT_STEPS != 0:
            raise ValueError('Tangent Flow is permanently frozen after Base; set ADAPT_STEPS: 0')
        if flow.SOLVER not in ('euler', 'midpoint', 'heun', 'rk4'):
            raise ValueError('Unknown FLOW.SOLVER')
        if not cfg.TRAINER.BiMC.TEXT_CALIBRATION:
            raise ValueError('Enable original BiMC description fusion: TEXT_CALIBRATION: True')
        if not np.isfinite(flow.LOGIT_SCALE) or flow.LOGIT_SCALE <= 0:
            raise ValueError('FLOW.LOGIT_SCALE must be finite and positive')
        model = self.model.module if self.is_distributed else self.model
        model.eval().requires_grad_(False)
        run_name = f'seed_{cfg.SEED}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        output = Path(flow.OUTPUT_DIR) / cfg.DATASET.NAME.lower() / run_name
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        encoder_id = str((cfg.MODEL.BACKBONE.NAME, cfg.TRAINER.BiMC.PREC, str(cfg.DATASET),
                          METHOD, DIRECTION, str(cfg.TRAINER.BiMC)))
        rows, checkpoint, bank = [], None, None
        num_base = len(self.data_manager.class_index_in_task[0])
        print(f'Method: tangent_flow | Dataset: {cfg.DATASET.NAME} | Seed: {cfg.SEED}')
        print(f'Results directory: {output}')
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')

        for task_id in range(self.data_manager.num_tasks):
            start = time.time()
            class_ids = self.data_manager.class_index_in_task[task_id]
            names = np.array(self.data_manager.class_names)[class_ids]
            loader = self.data_manager.get_dataloader(
                task_id, source='train', mode='test', accumulate_past=False)
            stats = model.build_task_statistics(
                names, loader, class_index=class_ids,
                calibrate_novel_vision_proto=cfg.TRAINER.BiMC.VISION_CALIBRATION,
                calculate_covariance=False)
            cache = from_task_statistics(stats, encoder_id, task_id)
            if cfg.TRAINER.BiMC.PROTOTYPE_CACHE_DIR:
                cache_dir = Path(cfg.TRAINER.BiMC.PROTOTYPE_CACHE_DIR) / cfg.DATASET.NAME.lower() / run_name
                cache_dir.mkdir(parents=True, exist_ok=True)
                cache['description_std'] = description_std(stats)
                torch.save(cache, cache_dir / f'session_{task_id}.pt')

            if task_id == 0:
                cache['description_std'] = description_std(stats)
                print('Training tangent text-to-visual Flow on base classes...')
                # run() is no_grad for CLIP/statistics; explicitly enable Flow training.
                # Isolate torch RNG so training does not alter subsequent data loading.
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    checkpoint = fit(
                        cache, epochs=flow.EPOCHS, batch_size=flow.BATCH_SIZE,
                        lr=flow.LR, weight_decay=flow.WEIGHT_DECAY,
                        blocks=flow.BLOCKS, hidden=flow.HIDDEN or None,
                        seed=cfg.SEED, device=self.device,
                        noise_scale=flow.NOISE_SCALE, max_angle=flow.MAX_ANGLE,
                        mix_probability=flow.MIX_PROBABILITY,
                        warmup_ratio=flow.WARMUP_RATIO,
                        min_lr_ratio=flow.MIN_LR_RATIO)
                torch.save(checkpoint, output / 'base_flow.pt')

            current = dict(class_ids=cache['class_ids'], text=cache['text_prototypes'].float(),
                           real=real_means(cache), encoder_id=checkpoint['encoder_id'])
            bank = build_bank(checkpoint, current, steps=flow.STEPS,
                              num_base=num_base if task_id == 0 else 0,
                              beta_base=flow.BETA_BASE, beta_inc=flow.BETA_INC,
                              device=self.device, previous_bank=bank, solver=flow.SOLVER)
            torch.save(checkpoint, output / f'flow_session{task_id}.pt')
            torch.save(bank, output / f'bank_session{task_id}.pt')
            del stats, cache, loader
            self._checkpoint = checkpoint
            acc = self.inference_task_flow(task_id, bank, model)
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
    def inference_task_flow(self, task_id, bank, model):
        # The existing evaluator interprets score column index as global class ID.
        expected = [int(c) for group in self.data_manager.class_index_in_task[:task_id+1]
                    for c in group]
        if bank['class_ids'] != expected or expected != list(range(len(expected))):
            raise ValueError('BIMC evaluator requires ordered, contiguous seen-class IDs')
        loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        scores, targets = [], []
        velocity = load_model(checkpoint=self._checkpoint, device=self.device).eval()
        device_bank = dict(bank, prototypes=bank['prototypes'].to(self.device), text=bank['text'].to(self.device))
        for batch in tqdm(loader):
            images, labels = self.parse_batch(batch)
            features = model.extract_img_feature(images)
            _, logits = classify_bidirectional(velocity, features, device_bank,
                                               alpha=self.cfg.FLOW.ALPHA, gamma=self.cfg.FLOW.GAMMA,
                                               steps=self.cfg.FLOW.STEPS,
                                               solver=self.cfg.FLOW.SOLVER)
            logits = torch.softmax(logits * self.cfg.FLOW.LOGIT_SCALE, dim=-1)
            scores.append(logits.cpu())
            targets.append(labels.cpu())
        if not scores:
            raise ValueError('Empty test loader')
        acc = self.evaluator.calc_accuracy(torch.cat(scores), torch.cat(targets), task_id)
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
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
