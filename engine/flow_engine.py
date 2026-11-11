"""Base-only Flow training with BIMC's dataset splits and accuracy evaluator."""
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
from single_flow.s1_flow import fit, generate, classify
from utils.experiment_logger import log_metrics


class FlowRunner(Runner):
    @torch.no_grad()
    def run(self):
        cfg = self.cfg
        flow = cfg.FLOW
        model = self.model.module if self.is_distributed else self.model
        model.eval().requires_grad_(False)
        run_name = f'seed_{cfg.SEED}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        output = Path(flow.OUTPUT_DIR) / cfg.DATASET.NAME.lower() / run_name
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        encoder_id = str((cfg.MODEL.BACKBONE.NAME, cfg.TRAINER.BiMC.PREC, str(cfg.DATASET)))
        rows, checkpoint, bank = [], None, None
        print(f'Method: single_flow | Dataset: {cfg.DATASET.NAME} | Seed: {cfg.SEED}')
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
                calibrate_novel_vision_proto=cfg.TRAINER.BiMC.VISION_CALIBRATION)
            cache = from_task_statistics(stats, encoder_id, task_id)
            if cfg.TRAINER.BiMC.PROTOTYPE_CACHE_DIR:
                cache_dir = Path(cfg.TRAINER.BiMC.PROTOTYPE_CACHE_DIR) / cfg.DATASET.NAME.lower() / run_name
                cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(cache, cache_dir / f'session_{task_id}.pt')

            if task_id == 0:
                print('Training Flow on base classes only...')
                # run() is no_grad for CLIP/statistics; explicitly enable Flow training.
                # Isolate torch RNG so training does not alter subsequent data loading.
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    checkpoint = fit(
                        cache, epochs=flow.EPOCHS, batch_size=flow.BATCH_SIZE,
                        lr=flow.LR, weight_decay=flow.WEIGHT_DECAY,
                        blocks=flow.BLOCKS, hidden=flow.HIDDEN or None,
                        target=flow.TARGET, seed=cfg.SEED, device=self.device)
                torch.save(checkpoint, output / 'base_flow.pt')

            bank = generate(checkpoint, cache, steps=flow.STEPS, device=self.device, existing=bank)
            torch.save(bank, output / f'bank_session{task_id}.pt')
            # Do not retain historical real-image features for later sessions.
            del stats, cache, loader
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

    @torch.no_grad()
    def inference_task_flow(self, task_id, bank, model):
        # The existing evaluator interprets score column index as global class ID.
        expected = [int(c) for group in self.data_manager.class_index_in_task[:task_id+1]
                    for c in group]
        if bank['class_ids'] != expected or expected != list(range(len(expected))):
            raise ValueError('BIMC evaluator requires ordered, contiguous seen-class IDs')
        loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        scores, targets = [], []
        device_bank = dict(bank, prototypes=bank['prototypes'].to(self.device))
        for batch in tqdm(loader):
            images, labels = self.parse_batch(batch)
            features = model.extract_img_feature(images)
            _, logits = classify(features, device_bank)
            scores.append(logits.cpu())
            targets.append(labels.cpu())
        if not scores:
            raise ValueError('Empty test loader')
        acc = self.evaluator.calc_accuracy(torch.cat(scores), torch.cat(targets), task_id)
        print(f'Test acc mean: {acc["mean_acc"]}, task-wise acc: {acc["task_acc"]}')
        return acc

    def save_results(self, output, rows):
        summary = dict(method='single_flow', dataset=self.cfg.DATASET.NAME, seed=self.cfg.SEED,
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
