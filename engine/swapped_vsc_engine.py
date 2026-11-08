"""Base-trained, incremental-frozen swapped VSC experiment.

The frozen CLIP encoder is reused from BiMC, but BiMC's calibration,
Mahalanobis and KNN ensemble are NOT part of this experiment.
"""
import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from datasets.data_manager import DatasetManager
from models.bimc import BiMC
from models.swapped_vsc import SwappedVSCHead
from utils.evaluator import AccuracyEvaluator
from utils.experiment_logger import log_metrics


class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dm = DatasetManager(cfg)
        self.device = torch.device(cfg.DEVICE.DEVICE_NAME)
        # Reuse BiMC's CLIP initialization and prompt/description encoders.
        self.encoder = BiMC(cfg, self.dm.template, self.device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        dim = int(self.encoder.clip_model.text_projection.shape[-1])
        settings = cfg.TRAINER.SwappedVSC
        self.head = SwappedVSCHead(
            dim=dim, hidden_dim=settings.HIDDEN_DIM,
            projection_strength=settings.PROJECTION_STRENGTH,
            text_strength=settings.TEXT_STRENGTH,
            attention_tau=settings.ATTENTION_TAU,
            logit_scale=settings.LOGIT_SCALE,
            fusion_weight=settings.FUSION_WEIGHT,
        ).to(self.device)
        self.settings = settings
        self.evaluator = AccuracyEvaluator(self.dm.class_index_in_task)
        self.acc_list, self.task_acc_list = [], []
        self.saved_text = []
        self.saved_visual = []
        self.output_dir = os.path.join(
            settings.OUTPUT_DIR, cfg.DATASET.NAME.lower(), "seed_" + str(cfg.SEED)
        )

    @torch.no_grad()
    def _encode_support(self, task_id):
        """Only CURRENT session's train examples; never reads future/test images."""
        indices = self.dm.class_index_in_task[task_id]
        names = np.asarray(self.dm.class_names)[indices]
        loader = self.dm.get_dataloader(
            task_id, source="train", mode="test", accumulate_past=False
        )
        text, _ = self.encoder.inference_text_feature(names, self.dm.template, int(indices[0]))
        description, _, description_proto = self.encoder.inference_all_description_feature(
            names, self.cfg.DATASET.GPT_PATH, int(indices[0])
        )
        weight = float(self.settings.DESCRIPTION_WEIGHT)
        text = F.normalize(
            (1.0 - weight) * text.float() + weight * description_proto.float(),
            dim=-1
        )

        all_features, all_labels = [], []
        for batch in tqdm(loader, desc="Cache support features"):
            images = batch["image"].to(self.device)
            features = F.normalize(
                self.encoder.extract_img_feature(images).float(), dim=-1
            )
            all_features.append(features.cpu())
            all_labels.append(batch["label"].cpu().long())
        features = torch.cat(all_features)
        labels = torch.cat(all_labels)
        visual_protos = []
        for cls_id in indices:
            mask = labels == int(cls_id)
            if not mask.any():
                raise RuntimeError("Missing support class: " + str(cls_id))
            visual_protos.append(features[mask].mean(0))
        visual = F.normalize(torch.stack(visual_protos), dim=-1).to(self.device)
        return text.to(self.device), visual, features, labels

    def _train_base(self, features, labels, text_proto, visual_proto):
        """Train ONLY on base support images; freeze projection for all future tasks."""
        self.head.train()
        optimizer = torch.optim.AdamW(
            self.head.parameters(), lr=float(self.settings.LR),
            weight_decay=float(self.settings.WEIGHT_DECAY)
        )
        features, labels = features.to(self.device), labels.to(self.device)
        batch_size = int(self.settings.BATCH_SIZE)
        num_samples = len(labels)
        if num_samples == 0:
            raise ValueError("Empty base set")
        for epoch in range(int(self.settings.EPOCHS)):
            permutation = torch.randperm(num_samples, device=self.device)
            total_loss, batch_count = 0.0, 0
            for indexes in permutation.split(batch_size):
                optimizer.zero_grad()
                loss, terms = self.head.losses(
                    features[indexes], labels[indexes],
                    text_proto, visual_proto,
                    consistency_weight=float(self.settings.CONSISTENCY_WEIGHT),
                    indication_weight=float(self.settings.INDICATION_WEIGHT)
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                batch_count += 1
            mean_loss = total_loss / max(1, batch_count)
            print("base epoch={} loss={:.6f}".format(epoch + 1, mean_loss))
            log_metrics({"swapped_vsc/base_loss": mean_loss}, step=epoch)
        self.head.eval()

    def _evaluate_session(self, task_id):
        self.head.eval()
        text_proto = torch.cat(self.saved_text)
        visual_proto = torch.cat(self.saved_visual)
        if text_proto.shape[0] != sum(len(x) for x in self.dm.class_index_in_task[:task_id + 1]):
            raise ValueError("Seen-class prototype bank is not aligned")
        test_loader = self.dm.get_dataloader(task_id, source="test", mode="test")
        outputs, targets = [], []
        with torch.no_grad():
            # Cache prototype banks once per session rather than per test batch.
            calibrated_text, pseudo_visual = self.head.prototype_bank(text_proto, visual_proto)
            for batch in tqdm(test_loader, desc="Evaluate task {}".format(task_id)):
                images = batch["image"].to(self.device)
                features = F.normalize(
                    self.encoder.extract_img_feature(images).float(), dim=-1
                )
                query_text = self.head.visual_to_text(features)
                logits_t = self.head.logit_scale * (query_text @ calibrated_text.T)
                logits_v = self.head.logit_scale * (features @ pseudo_visual.T)
                prob = (
                    self.head.fusion_weight * logits_t.softmax(-1)
                    + (1.0 - self.head.fusion_weight) * logits_v.softmax(-1)
                )
                outputs.append(prob.cpu())
                targets.append(batch["label"].cpu().long())
        return self.evaluator.calc_accuracy(
            torch.cat(outputs), torch.cat(targets), task_id
        )

    def run(self):
        print("Method: swapped VSC | Base train then frozen incremental projection")
        for task_id in range(self.dm.num_tasks):
            start = time.time()
            text_proto, visual_proto, features, labels = self._encode_support(task_id)
            if task_id == 0:
                # Target labels in this repository are contiguous [0, base_classes).
                expected = torch.arange(len(text_proto), dtype=torch.long)
                if not torch.equal(torch.unique(labels), expected):
                    raise ValueError("Base class IDs must be contiguous from zero")
                self._train_base(features, labels, text_proto, visual_proto)
                os.makedirs(self.output_dir, exist_ok=True)
                checkpoint = os.path.join(self.output_dir, "base_projection.pt")
                torch.save({
                    "head": self.head.state_dict(), "seed": self.cfg.SEED,
                    "dataset": self.cfg.DATASET.NAME,
                    "base_classes": len(text_proto),
                }, checkpoint)
                print("Saved base projection:", checkpoint)
            # Newly seen classes are appended without retraining either projection.
            self.saved_text.append(text_proto.detach())
            self.saved_visual.append(visual_proto.detach())
            result = self._evaluate_session(task_id)
            self.acc_list.append(result["mean_acc"])
            self.task_acc_list.append(result["task_acc"])
            print("=> Task [{}], Acc: {:.3f}, task-wise: {}".format(
                task_id, result["mean_acc"], result["task_acc"]))
            log_metrics({
                "task_id": task_id, "mean_acc": result["mean_acc"],
                "base_avg_acc": result["base_avg_acc"],
                "inc_avg_acc": result["inc_avg_acc"],
                "harmonic_acc": result["harmonic_acc"],
                "elapsed_time": time.time() - start
            }, step=task_id)
        print("Final acc:{}".format(self.acc_list))
        print("Task-wise acc:")
        for i, acc in enumerate(self.task_acc_list):
            print("task {:2d}, acc:{}".format(i, acc))
        log_metrics({
            "final_acc": self.acc_list[-1],
            "avg_acc": sum(self.acc_list) / len(self.acc_list)
        })
