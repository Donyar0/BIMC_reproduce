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
from single_flow.s1_flow import fit
from single_flow.unified_flow import adapt, build_bank, classify_bidirectional, load_model, real_means, ensemble_scores, prepare_ensemble
from utils.experiment_logger import log_metrics


class UnifiedFlowRunner(Runner):
    @torch.no_grad()
    def run(self):
        cfg = self.cfg
        flow = cfg.FLOW
        if not np.isfinite(flow.LOGIT_SCALE) or flow.LOGIT_SCALE <= 0:
            raise ValueError('FLOW.LOGIT_SCALE must be finite and positive')
        model = self.model.module if self.is_distributed else self.model
        model.eval().requires_grad_(False)
        run_name = f'seed_{cfg.SEED}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        output = Path(flow.OUTPUT_DIR) / cfg.DATASET.NAME.lower() / run_name
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        encoder_id = str((cfg.MODEL.BACKBONE.NAME, cfg.TRAINER.BiMC.PREC, str(cfg.DATASET)))
        rows, checkpoint, bank = [], None, None
        memory = None
        ensemble_stats = []
        num_base = len(self.data_manager.class_index_in_task[0])
        print(f'Method: unified_flow | Dataset: {cfg.DATASET.NAME} | Seed: {cfg.SEED}')
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
            if cfg.TRAINER.BiMC.USING_ENSEMBLE:
                ensemble_stats.append(dict(cov=stats['cov_image'].detach().cpu(),
                    count=len(class_ids), descriptions=stats['description_features'].detach().cpu(),
                    labels=stats['description_targets'].detach().cpu()))
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
                        target=flow.TARGET, ce_weight=flow.CE_WEIGHT,
                        temperature=flow.TEMPERATURE, seed=cfg.SEED, device=self.device)
                torch.save(checkpoint, output / 'base_flow.pt')

            if task_id > 0:
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    torch.manual_seed(cfg.SEED + task_id)
                    checkpoint = adapt(checkpoint, cache, memory, steps=flow.ADAPT_STEPS,
                                       batch_size=flow.BATCH_SIZE, lr=flow.ADAPT_LR,
                                       weight_decay=flow.WEIGHT_DECAY, device=self.device)
            current = dict(class_ids=cache['class_ids'], text=cache['text_prototypes'].float(),
                           real=real_means(cache))
            if memory is None:
                memory = current
            else:
                memory = dict(class_ids=memory['class_ids'] + current['class_ids'],
                              text=torch.cat([memory['text'], current['text']]),
                              real=torch.cat([memory['real'], current['real']]))
            bank = build_bank(checkpoint, memory, steps=flow.STEPS, num_base=num_base,
                              beta_base=flow.BETA_BASE, beta_inc=flow.BETA_INC, device=self.device)
            torch.save(checkpoint, output / f'flow_session{task_id}.pt')
            torch.save(bank, output / f'bank_session{task_id}.pt')
            del stats, cache, loader
            self._checkpoint = checkpoint
            self._ensemble_stats = ensemble_stats
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
    def evaluate_saved(self, run_dir):
        """Re-evaluate saved session weights and banks without optimizer steps."""
        from yacs.config import CfgNode
        run_dir = Path(run_dir)
        with (run_dir / 'config.yaml').open() as handle:
            original = CfgNode.load_cfg(handle)
        cfg = self.cfg
        # Refuse mismatched training/data settings; only scoring and logging may differ.
        for name in ('SEED', 'DATASET', 'MODEL', 'DATALOADER', 'DEVICE'):
            if getattr(original, name) != getattr(cfg, name):
                raise ValueError(f'Saved run {name} differs from current configuration')
        for name in ('PREC', 'VISION_CALIBRATION', 'LAMBDA_I', 'TAU', 'TEXT_CALIBRATION', 'LAMBDA_T', 'GAMMA_BASE', 'GAMMA_INC'):
            if getattr(original.TRAINER.BiMC, name) != getattr(cfg.TRAINER.BiMC, name):
                raise ValueError(f'Saved run BiMC.{name} differs from current configuration')
        if original.FLOW.STEPS != cfg.FLOW.STEPS:
            raise ValueError('FLOW.STEPS must match the saved run')
        if not np.isfinite(cfg.FLOW.LOGIT_SCALE) or cfg.FLOW.LOGIT_SCALE <= 0:
            raise ValueError('FLOW.LOGIT_SCALE must be finite and positive')
        output = Path(cfg.FLOW.OUTPUT_DIR) / cfg.DATASET.NAME.lower() / (
            'eval_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        (output / 'source_run.txt').write_text(str(run_dir.resolve()) + '\n', encoding='utf-8')
        model = self.model.module if self.is_distributed else self.model
        model.eval().requires_grad_(False)
        self._ensemble_stats = []
        rows = []
        num_base = len(self.data_manager.class_index_in_task[0])
        for task_id in range(self.data_manager.num_tasks):
            start = time.time()
            # Only load trusted checkpoints created by this project.
            checkpoint = torch.load(run_dir / f'flow_session{task_id}.pt', map_location='cpu', weights_only=True)
            saved = torch.load(run_dir / f'bank_session{task_id}.pt', map_location='cpu', weights_only=True)
            expected = [int(c) for group in self.data_manager.class_index_in_task[:task_id+1] for c in group]
            if saved['class_ids'] != expected or saved['encoder_id'] != checkpoint['encoder_id']:
                raise ValueError('Saved checkpoint and bank are incompatible with dataset')
            # Stored generated/text/real vectors allow all beta values without training.
            beta = torch.full((len(expected), 1), cfg.FLOW.BETA_INC)
            beta[:num_base] = cfg.FLOW.BETA_BASE
            if not (0 <= cfg.FLOW.BETA_BASE <= 1 and 0 <= cfg.FLOW.BETA_INC <= 1):
                raise ValueError('Beta values must be between 0 and 1')
            bank = dict(saved, prototypes=torch.nn.functional.normalize(
                (1-beta)*saved['real'].float() + beta*saved['generated'].float(), dim=-1))
            if cfg.TRAINER.BiMC.USING_ENSEMBLE:
                ids = self.data_manager.class_index_in_task[task_id]
                names = np.array(self.data_manager.class_names)[ids]
                loader = self.data_manager.get_dataloader(task_id, source='train', mode='test', accumulate_past=False)
                stats = model.build_task_statistics(names, loader, class_index=ids,
                    calibrate_novel_vision_proto=cfg.TRAINER.BiMC.VISION_CALIBRATION)
                cache = from_task_statistics(stats, checkpoint['encoder_id'], task_id)
                n = len(ids)
                if not torch.allclose(real_means(cache), saved['real'][-n:].float(), atol=1e-4, rtol=1e-4):
                    raise ValueError('Recomputed support prototypes differ from saved bank; cannot reproduce ensemble safely')
                self._ensemble_stats.append(dict(cov=stats['cov_image'].detach().cpu(), count=n,
                    descriptions=stats['description_features'].detach().cpu(), labels=stats['description_targets'].detach().cpu()))
            self._checkpoint = checkpoint
            acc = self.inference_task_flow(task_id, bank, model)
            self.acc_list.append(acc['mean_acc'])
            self.task_acc_list.append(acc['task_acc'])
            row = dict(task_id=task_id, seen_classes=len(expected), **acc, elapsed_time=time.time()-start)
            rows.append(row)
            log_metrics({k:v for k,v in row.items() if k != 'task_acc'}, step=task_id)
            self.save_results(output, rows)
        print(f'Final acc:{self.acc_list}')
        print('Task-wise acc:')
        for i, task_acc in enumerate(self.task_acc_list):
            print(f'task {i:2d}, acc:{task_acc}')
        print(f'Results saved to: {output / "metrics.csv"}')
        log_metrics({'final_acc': self.acc_list[-1], 'avg_acc': sum(self.acc_list)/len(self.acc_list)})

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
        if self.cfg.TRAINER.BiMC.USING_ENSEMBLE:
            total = sum(item['count'] for item in self._ensemble_stats)
            covariance = sum(item['cov'] * (item['count'] / total) for item in self._ensemble_stats)
            descriptions = torch.cat([item['descriptions'] for item in self._ensemble_stats])
            description_labels = torch.cat([item['labels'] for item in self._ensemble_stats])
            prepared_ensemble = prepare_ensemble(
                device_bank, covariance, descriptions, description_labels, self.device)
        for batch in tqdm(loader):
            images, labels = self.parse_batch(batch)
            features = model.extract_img_feature(images)
            _, logits = classify_bidirectional(velocity, features, device_bank,
                                               alpha=self.cfg.FLOW.ALPHA, gamma=self.cfg.FLOW.GAMMA)
            if self.cfg.TRAINER.BiMC.USING_ENSEMBLE:
                logits = ensemble_scores(
                    logits, features, prepared_ensemble,
                    len(self.data_manager.class_index_in_task[0]),
                    self.cfg.DATASET.ENSEMBLE_ALPHA, self.cfg.FLOW.LOGIT_SCALE)
            else:
                logits = torch.softmax(logits * self.cfg.FLOW.LOGIT_SCALE, dim=-1)
            scores.append(logits.cpu())
            targets.append(labels.cpu())
        if not scores:
            raise ValueError('Empty test loader')
        acc = self.evaluator.calc_accuracy(torch.cat(scores), torch.cat(targets), task_id)
        print(f'Test acc mean: {acc["mean_acc"]}, task-wise acc: {acc["task_acc"]}')
        return acc

    def save_results(self, output, rows):
        summary = dict(method='unified_flow', dataset=self.cfg.DATASET.NAME, seed=self.cfg.SEED,
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
