"""Text-to-visual Flow with reverse consistency and three inference scores."""
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
from single_flow.bidirectional_flow import (
    SCORE_NAMES, fit_base, adapt, build_bank, load_model, score_three,
)
from utils.experiment_logger import log_metrics


class BidirectionalFlowRunner(Runner):
    def _validate(self):
        flow = self.cfg.FLOW
        if not self.cfg.TRAINER.BiMC.TEXT_CALIBRATION:
            raise ValueError('BiMC fused-description text must be enabled')
        if flow.SCORE_MODE not in (*SCORE_NAMES, 'fused'):
            raise ValueError('Unknown FLOW.SCORE_MODE')
        if (len(flow.SCORE_WEIGHTS) != 3 or not np.isfinite(flow.SCORE_WEIGHTS).all()
                or any(w < 0 for w in flow.SCORE_WEIGHTS)
                or sum(flow.SCORE_WEIGHTS) <= 0):
            raise ValueError('FLOW.SCORE_WEIGHTS must be three nonnegative values')
        if flow.STEPS < 1 or flow.CONTRAST_STEPS < 1:
            raise ValueError('Flow integration steps must be positive')

    @torch.no_grad()
    def run(self):
        self._validate()
        cfg, flow = self.cfg, self.cfg.FLOW
        model = self.model.module if self.is_distributed else self.model
        model.eval().requires_grad_(False)
        run_name = f'seed_{cfg.SEED}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        output = Path(flow.OUTPUT_DIR) / cfg.DATASET.NAME.lower() / run_name
        output.mkdir(parents=True, exist_ok=False)
        (output/'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        encoder_id = str((cfg.MODEL.BACKBONE.NAME, cfg.TRAINER.BiMC.PREC,
                          str(cfg.DATASET), 'bidirectional_text_visual_v1'))
        checkpoint, bank, rows = None, None, []
        for task_id in range(self.data_manager.num_tasks):
            start = time.time()
            ids = self.data_manager.class_index_in_task[task_id]
            names = np.array(self.data_manager.class_names)[ids]
            loader = self.data_manager.get_dataloader(
                task_id, source='train', mode='test', accumulate_past=False)
            stats = model.build_task_statistics(
                names, loader, class_index=ids,
                calibrate_novel_vision_proto=cfg.TRAINER.BiMC.VISION_CALIBRATION)
            cache = from_task_statistics(stats, encoder_id, task_id)
            if task_id == 0:
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    checkpoint = fit_base(
                        cache, epochs=flow.EPOCHS, batch_size=flow.BATCH_SIZE,
                        lr=flow.LR, weight_decay=flow.WEIGHT_DECAY,
                        blocks=flow.BLOCKS, hidden=flow.HIDDEN or None,
                        contrast_weight=flow.CONTRAST_WEIGHT,
                        temperature=flow.CONTRAST_TEMPERATURE,
                        contrast_steps=flow.CONTRAST_STEPS,
                        seed=cfg.SEED, device=self.device)
                torch.save(checkpoint, output/'base_flow.pt')
            elif flow.ADAPT:
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    checkpoint = adapt(
                        checkpoint, cache, epochs=flow.ADAPT_EPOCHS,
                        batch_size=flow.ADAPT_BATCH_SIZE, lr=flow.ADAPT_LR,
                        weight_decay=flow.WEIGHT_DECAY,
                        seed=cfg.SEED+task_id, device=self.device)
                torch.save(checkpoint, output/f'flow_session{task_id}.pt')
            bank = build_bank(checkpoint, cache, steps=flow.STEPS,
                              device=self.device, existing=bank,
                              regenerate_existing=bool(flow.ADAPT and task_id > 0))
            torch.save(bank, output/f'bank_session{task_id}.pt')
            del stats, cache, loader
            result = self.inference_task(task_id, bank, model, checkpoint)
            row = dict(task_id=task_id, seen_classes=len(bank['class_ids']),
                       **result, elapsed_time=time.time()-start)
            self.acc_list.append(result['mean_acc'])
            self.task_acc_list.append(result['task_acc'])
            rows.append(row)
            log_metrics({k:v for k,v in row.items() if k != 'task_acc'}, step=task_id)
            self.save_results(output, rows)
            print(f'Bidirectional Flow task {task_id}: {result["mean_acc"]:.2f}%', flush=True)
        print(f'Results saved to: {output}', flush=True)
        log_metrics({'final_acc': rows[-1]['mean_acc'],
                     'avg_acc': sum(r['mean_acc'] for r in rows)/len(rows)})

    @torch.no_grad()
    def inference_task(self, task_id, bank, image_model, checkpoint):
        expected = [int(c) for group in self.data_manager.class_index_in_task[:task_id+1]
                    for c in group]
        if bank['class_ids'] != expected or expected != list(range(len(expected))):
            raise ValueError('Evaluator requires contiguous ordered class IDs')
        flow_model = load_model(checkpoint, self.device)
        device_bank = dict(bank, text=bank['text'].to(self.device),
                           generated=bank['generated'].to(self.device))
        all_scores = {name:[] for name in (*SCORE_NAMES, 'fused')}
        targets = []
        loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        for batch in tqdm(loader):
            images, labels = self.parse_batch(batch)
            features = image_model.extract_img_feature(images)
            scores = score_three(flow_model, features, device_bank,
                                 steps=self.cfg.FLOW.STEPS,
                                 weights=self.cfg.FLOW.SCORE_WEIGHTS)
            for name in all_scores:
                all_scores[name].append(scores[name].cpu())
            targets.append(labels.cpu())
        if not targets:
            raise ValueError('Empty test loader')
        targets = torch.cat(targets)
        accuracy = {name:self.evaluator.calc_accuracy(torch.cat(values), targets, task_id)
                    for name, values in all_scores.items()}
        selected = accuracy[self.cfg.FLOW.SCORE_MODE]
        return dict(**selected, **{name+'_acc':accuracy[name]['mean_acc']
                                  for name in all_scores})

    def save_results(self, output, rows):
        summary = dict(method='bidirectional_single_flow',
                       adapt=bool(self.cfg.FLOW.ADAPT),
                       score_mode=self.cfg.FLOW.SCORE_MODE,
                       dataset=self.cfg.DATASET.NAME, seed=self.cfg.SEED,
                       accuracy_unit='percent', completed_sessions=len(rows),
                       total_sessions=self.data_manager.num_tasks,
                       complete=len(rows)==self.data_manager.num_tasks,
                       final_acc=rows[-1]['mean_acc'],
                       avg_acc=sum(r['mean_acc'] for r in rows)/len(rows),
                       sessions=rows)
        (output/'metrics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        with (output/'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
