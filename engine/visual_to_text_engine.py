"""Base-only Flow training and shared-checkpoint residual-strength evaluation."""
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from engine.engine import Runner
from single_flow.meta_visual_to_text import fit_meta
from single_flow.visual_to_text import (
    fit, finetune_incremental, load_flow, load_checkpoint, classify_alphas, validate_alphas,
)
from utils.experiment_logger import log_metrics


class VisualToTextRunner(Runner):
    @torch.no_grad()
    def run(self):
        cfg = self.cfg
        flow = cfg.FLOW
        inc_cfg = getattr(flow, 'INC', None)
        inc_enabled = bool(getattr(inc_cfg, 'ENABLED', False))
        meta_cfg = getattr(flow, 'META', None)
        meta_enabled = bool(getattr(meta_cfg, 'ENABLED', False))
        alphas = validate_alphas(getattr(flow, 'ALPHAS', [1.0]))
        if alphas != [1.0]:
            raise ValueError('This FM-only experiment requires FLOW.ALPHAS: [1.0] (full Flow update)')
        model = self.model.module if self.is_distributed else self.model
        model.eval().requires_grad_(False)
        encoder_id = str((cfg.MODEL.BACKBONE.NAME, cfg.TRAINER.BiMC.PREC, str(cfg.DATASET)))
        checkpoint_path = getattr(flow, 'CHECKPOINT', '')
        checkpoint = None
        if meta_enabled and checkpoint_path:
            raise ValueError('Episodic meta training must start from a fresh initialization; remove --flow_checkpoint')
        if checkpoint_path:
            print(f'Loading pretrained Flow: {checkpoint_path}')
            checkpoint = load_checkpoint(
                checkpoint_path, trusted=getattr(flow, 'TRUST_CHECKPOINT', False))
            if checkpoint.get('trained_through_session', 0) != 0:
                raise ValueError('Supply a base Flow checkpoint, not a session-adapted checkpoint')
            if checkpoint['encoder_id'] != encoder_id:
                raise ValueError('CLIP encoder/data configuration mismatch')
            base_ids = [int(c) for c in self.data_manager.class_index_in_task[0]]
            if checkpoint['base_class_ids'] != base_ids:
                raise ValueError('Checkpoint base classes do not match this split')
            if checkpoint.get('training', {}).get('seed', cfg.SEED) != cfg.SEED:
                raise ValueError('Checkpoint seed does not match the current class split seed')
            velocity_model = load_flow(checkpoint, self.device)
            print('Flow loaded successfully. Training skipped.')
            print('Checkpoint training settings:', checkpoint.get('training', {}))

        run_name = f'seed_{cfg.SEED}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        output = Path(flow.OUTPUT_DIR) / cfg.DATASET.NAME.lower() / run_name
        output.mkdir(parents=True, exist_ok=False)
        (output / 'config.yaml').write_text(cfg.dump(), encoding='utf-8')
        rows = []
        # Preserve the inherited summary lists for the original full-Flow setting.
        reference_alpha = 1.0
        self.acc_list, self.task_acc_list = [], []
        print(f'Method: visual_to_text | Dataset: {cfg.DATASET.NAME} | Seed: {cfg.SEED}')
        print(f'Results directory: {output}')
        print(f'Flow alphas: {alphas} (full Flow only; alpha=1)')
        print(f'Incremental support adaptation: {inc_enabled}')
        print(f'Episodic pseudo-incremental meta training: {meta_enabled}')
        print(f'Start inferencing on all tasks: [0, {self.data_manager.num_tasks - 1}]')

        for task_id in range(self.data_manager.num_tasks):
            start = time.time()
            class_ids = self.data_manager.class_index_in_task[task_id]
            names = np.array(self.data_manager.class_names)[class_ids]
            text, _ = model.inference_text_feature(names, model.template, class_ids[0])
            if cfg.TRAINER.BiMC.TEXT_CALIBRATION:
                _, _, description = model.inference_all_description_feature(
                    names, cfg.DATASET.GPT_PATH, class_ids[0])
                text = model.fused_text_prototype(text, description)
            cache = dict(encoder_id=encoder_id, class_ids=[int(c) for c in class_ids],
                         text_prototypes=text.detach().float().cpu(),
                         stage='base' if task_id == 0 else 'incremental', split='train')
            if ((task_id == 0 and (checkpoint is None or cfg.TRAINER.BiMC.PROTOTYPE_CACHE_DIR))
                    or (task_id > 0 and inc_enabled)):
                loader = self.data_manager.get_dataloader(
                    task_id, source='train', mode='test', accumulate_past=False)
                features, labels, visual = model.inference_all_img_feature(loader, class_ids[0])
                if torch.unique(labels).cpu().tolist() != cache['class_ids']:
                    raise ValueError('Base image labels must match text row order')
                cache.update(image_features=features.detach().float().cpu(),
                             image_labels=labels.detach().long().cpu(),
                             visual_prototypes=visual.detach().float().cpu())
                del features, labels, visual, loader
            if cfg.TRAINER.BiMC.PROTOTYPE_CACHE_DIR:
                cache_dir = Path(cfg.TRAINER.BiMC.PROTOTYPE_CACHE_DIR) / cfg.DATASET.NAME.lower() / run_name
                cache_dir.mkdir(parents=True, exist_ok=True)
                torch.save(cache, cache_dir / f'session_{task_id}.pt')

            if task_id == 0 and checkpoint is None:
                print('Training Flow on base classes only...')
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    checkpoint = (fit_meta(
                        cache, epochs=meta_cfg.EPOCHS,
                        episodes_per_epoch=meta_cfg.EPISODES_PER_EPOCH,
                        pseudo_new=meta_cfg.PSEUDO_NEW, shot=meta_cfg.SHOT,
                        query=meta_cfg.QUERY, base_batch=meta_cfg.BASE_BATCH,
                        old_query_batch=meta_cfg.OLD_QUERY_BATCH,
                        inner_steps=meta_cfg.INNER_STEPS, inner_lr=meta_cfg.INNER_LR,
                        meta_lr=meta_cfg.LR, weight_decay=flow.WEIGHT_DECAY,
                        base_fm_weight=meta_cfg.BASE_FM_WEIGHT,
                        old_fm_weight=meta_cfg.OLD_FM_WEIGHT,
                        novel_fm_weight=meta_cfg.NOVEL_FM_WEIGHT, blocks=flow.BLOCKS,
                        hidden=flow.HIDDEN or None, seed=cfg.SEED,
                        device=self.device) if meta_enabled else fit(
                        cache, epochs=flow.EPOCHS, batch_size=flow.BATCH_SIZE,
                        lr=flow.LR, weight_decay=flow.WEIGHT_DECAY,
                        blocks=flow.BLOCKS, hidden=flow.HIDDEN or None,
                        source=flow.SOURCE, ce_weight=flow.CE_WEIGHT,
                        temperature=flow.TEMPERATURE, seed=cfg.SEED, device=self.device))
                torch.save(checkpoint, output / 'base_flow.pt')
                velocity_model = load_flow(checkpoint, self.device)

            if task_id == 0:
                provenance = dict(
                    checkpoint=str(Path(checkpoint_path).resolve()) if checkpoint_path
                    else str((output / 'base_flow.pt').resolve()),
                    training=checkpoint.get('training', {}), architecture=checkpoint['architecture'],
                    alphas=alphas, steps=flow.STEPS,
                    text_calibration=cfg.TRAINER.BiMC.TEXT_CALIBRATION,
                    text_lambda=getattr(cfg.TRAINER.BiMC, 'LAMBDA_T', None),
                    incremental_enabled=inc_enabled,
                    meta_enabled=meta_enabled,
                    meta_training=checkpoint.get('training', {}) if meta_enabled else None)
                (output / 'evaluation.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
            inc_record = None
            if task_id > 0 and inc_enabled:
                print(f'Fine-tuning on {len(cache["image_features"])} current-session support images')
                devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
                with torch.random.fork_rng(devices=devices), torch.enable_grad():
                    checkpoint = finetune_incremental(
                        checkpoint, cache, session_id=task_id,
                        epochs=inc_cfg.EPOCHS, batch_size=inc_cfg.BATCH_SIZE,
                        lr=inc_cfg.LR, weight_decay=inc_cfg.WEIGHT_DECAY,
                        ce_weight=inc_cfg.CE_WEIGHT, temperature=inc_cfg.TEMPERATURE,
                        seed=cfg.SEED, device=self.device)
                torch.save(checkpoint, output / f'flow_session{task_id}.pt')
                velocity_model = load_flow(checkpoint, self.device)
                inc_record = checkpoint['incremental_updates'][-1]
            # Recompute all seen-category fused text features at this session.
            # No persistent bank or bank_session*.pt is needed.
            seen_ids = [int(c) for group in self.data_manager.class_index_in_task[:task_id + 1]
                        for c in group]
            seen_names = np.array(self.data_manager.class_names)[seen_ids]
            candidate_text, _ = model.inference_text_feature(
                seen_names, model.template, seen_ids[0])
            if cfg.TRAINER.BiMC.TEXT_CALIBRATION:
                _, _, seen_desc = model.inference_all_description_feature(
                    seen_names, cfg.DATASET.GPT_PATH, seen_ids[0])
                candidate_text = model.fused_text_prototype(candidate_text, seen_desc)
            candidate_text = candidate_text.detach().float()
            del cache
            results = self.inference_task_flow(task_id, candidate_text, model, velocity_model, alphas)
            elapsed = time.time() - start
            self.acc_list.append(results[reference_alpha]['mean_acc'])
            self.task_acc_list.append(results[reference_alpha]['task_acc'])
            logged = {}
            for alpha, acc in results.items():
                rows.append(dict(task_id=task_id, alpha=alpha, seen_classes=len(seen_ids),
                                 inc_train_samples=inc_record['num_samples'] if inc_record else 0,
                                 **acc, elapsed_time=elapsed))
                print(f'=> Task [{task_id}] alpha={alpha:g} | Acc: {acc["mean_acc"]:.2f} | '
                      f'Base: {acc["base_avg_acc"]:.2f} | Incremental: {acc["inc_avg_acc"]:.2f} | '
                      f'Harmonic: {acc["harmonic_acc"]:.2f}')
                logged.update({f'alpha_{alpha:g}/{k}': v for k, v in acc.items()
                               if isinstance(v, (int, float))})
            log_metrics(logged, step=task_id)
            self.save_results(output, rows)
            print(f'+++++++++++ task {task_id}, shared time: {elapsed:.3f} ++++++++++++++++')

        for alpha in alphas:
            selected = [r for r in rows if r['alpha'] == alpha]
            print(f'alpha={alpha:g} Final acc: {[r["mean_acc"] for r in selected]}')
            print(f'alpha={alpha:g} Task-wise acc: {[r["task_acc"] for r in selected]}')
            print(f'alpha={alpha:g} Final session Acc: {selected[-1]["mean_acc"]:.3f} | '
                  f'Average session Acc: {sum(r["mean_acc"] for r in selected)/len(selected):.3f}')
        print(f'Results saved to: {output / "metrics.csv"}')

    @torch.no_grad()
    def inference_task_flow(self, task_id, candidate_text, model, velocity_model, alphas=None):
        # Score columns must remain global class IDs for the existing evaluator.
        expected = [int(c) for group in self.data_manager.class_index_in_task[:task_id+1]
                    for c in group]
        if expected != list(range(len(expected))) or len(candidate_text) != len(expected):
            raise ValueError('BIMC evaluator requires ordered, contiguous seen-class IDs')
        alphas = validate_alphas(alphas if alphas is not None else getattr(self.cfg.FLOW, 'ALPHAS', [1.0]))
        loader = self.data_manager.get_dataloader(task_id, source='test', mode='test')
        scores, targets = {alpha: [] for alpha in alphas}, []
        candidate_text = candidate_text.to(self.device)
        for batch in tqdm(loader):
            images, labels = self.parse_batch(batch)
            features = model.extract_img_feature(images)
            outputs = classify_alphas(velocity_model, features, candidate_text,
                                      steps=self.cfg.FLOW.STEPS, alphas=alphas,
                                      candidate_class_ids=expected)
            for alpha, (_, logits) in outputs.items():
                scores[alpha].append(logits.cpu())
            targets.append(labels.cpu())
        if not targets:
            raise ValueError('Empty test loader')
        labels = torch.cat(targets)
        results = {}
        for alpha in alphas:
            logits = torch.cat(scores[alpha])
            # Formal FSCIL evaluation always uses every seen class.
            acc = self.evaluator.calc_accuracy(logits, labels, task_id)
            acc.update(self.novel_diagnostics(logits, labels, task_id))
            results[alpha] = acc
            if task_id > 0:
                print(f'alpha={alpha:g} diagnostic only: novel-only={acc["novel_only_acc"]}, '
                      f'current-only={acc["current_only_acc"]}, '
                      f'novel -> base (%)={acc["novel_to_base_pct"]}')
                print(f'alpha={alpha:g} current-class accuracy (%): {acc["current_class_acc"]}')
        return results

    def novel_diagnostics(self, logits, labels, task_id):
        """Label-restricted candidates are diagnostic only, never formal scores."""
        result = dict(novel_samples=0, current_samples=0, novel_candidates=0,
                      current_candidates=0, novel_only_acc=None, current_only_acc=None,
                      novel_to_base_pct=None, current_to_base_pct=None,
                      current_base_max_sim=None, current_novel_max_sim=None,
                      current_class_acc={})
        if task_id == 0:
            return result
        num_base = len(self.data_manager.class_index_in_task[0])
        current_ids = torch.as_tensor(self.data_manager.class_index_in_task[task_id], dtype=torch.long)
        novel = labels >= num_base
        current = (labels[:, None] == current_ids[None, :]).any(dim=1)
        pred = logits.argmax(dim=1)
        result.update(novel_samples=int(novel.sum()), current_samples=int(current.sum()),
                      novel_candidates=logits.shape[1] - num_base, current_candidates=len(current_ids))
        if novel.any():
            novel_pred = logits[novel, num_base:].argmax(dim=1) + num_base
            result['novel_only_acc'] = (novel_pred == labels[novel]).float().mean().item() * 100
            result['novel_to_base_pct'] = (pred[novel] < num_base).float().mean().item() * 100
        if current.any():
            current_pred = current_ids[logits[current][:, current_ids].argmax(dim=1)]
            result['current_only_acc'] = (current_pred == labels[current]).float().mean().item() * 100
            result['current_to_base_pct'] = (pred[current] < num_base).float().mean().item() * 100
            result['current_base_max_sim'] = logits[current, :num_base].max(dim=1).values.mean().item()
            result['current_novel_max_sim'] = logits[current, num_base:].max(dim=1).values.mean().item()
        for cid in current_ids.tolist():
            mask = labels == cid
            result['current_class_acc'][str(cid)] = (
                (pred[mask] == labels[mask]).float().mean().item() * 100 if mask.any() else None)
        return result

    def save_results(self, output, rows):
        by_alpha = {}
        for alpha in dict.fromkeys(r['alpha'] for r in rows):
            selected = [r for r in rows if r['alpha'] == alpha]
            by_alpha[str(alpha)] = dict(
                final_acc=selected[-1]['mean_acc'],
                avg_acc=sum(r['mean_acc'] for r in selected) / len(selected),
                final_base_acc=selected[-1]['base_avg_acc'],
                final_inc_acc=selected[-1]['inc_avg_acc'],
                final_harmonic_acc=selected[-1]['harmonic_acc'])
        completed = len({r['task_id'] for r in rows})
        summary = dict(method='visual_to_text', dataset=self.cfg.DATASET.NAME, seed=self.cfg.SEED,
                       accuracy_unit='percent', completed_sessions=completed,
                       total_sessions=self.data_manager.num_tasks,
                       complete=completed == self.data_manager.num_tasks,
                       by_alpha=by_alpha, sessions=rows)
        if len(by_alpha) == 1:
            summary.update(next(iter(by_alpha.values())))
        (output / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
