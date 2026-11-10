"""Training-free class-discriminative patch memory adapted from CDPS (ECCV 2026).

Unlike the original joint few-shot setting, this bank is constructed session by
session. Historical class memories are frozen; only current-session support
images are accessed. No parameters or optimizer are introduced.
"""
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn.functional as F


class IncrementalCDPS:
    def __init__(self, clip_model, cfg):
        self.model = clip_model
        opts = cfg.TRAINER.CDPS
        self.top_percent = float(opts.TOP_PERCENT)
        self.num_regions = int(opts.NUM_REGIONS)
        self.topk = int(opts.TOPK)
        self.num_confuse = int(opts.NUM_CONFUSE)
        self.omega = float(opts.OMEGA)
        self.max_support = int(opts.MAX_SUPPORT)
        self.max_query_patches = int(opts.MAX_QUERY_PATCHES)
        self.query_topk = int(opts.QUERY_TOPK)
        self.temperature = float(opts.TEMPERATURE)
        self.memory = {}       # class_id -> selected patches, CPU
        self.centers = {}      # class_id -> global visual prototype, CPU
        self.session_of_class = {}
        self.device = next(clip_model.parameters()).device
        self.output = Path(opts.CACHE_DIR) / cfg.DATASET.NAME.lower() / ("seed_%d" % cfg.SEED)
        visual = clip_model.visual
        if not hasattr(visual, "transformer") or not hasattr(visual, "proj"):
            raise ValueError("CDPS requires a CLIP ViT backbone with patch tokens.")
        self.visual = visual

    @torch.no_grad()
    def _encode(self, images):
        """Capture final block tokens and recompute CLS attention without changing CLIP."""
        visual = self.visual
        last = visual.transformer.resblocks[-1]
        captured = {}

        def record(block, inputs, outputs):
            tokens = outputs.permute(1, 0, 2)  # B, 197, 768
            patches = visual.ln_post(tokens[:, 1:, :])
            if visual.proj is not None:
                patches = patches @ visual.proj
            captured["patches"] = F.normalize(patches.float(), dim=-1)

            normed = block.ln_1(inputs[0])  # 197, B, 768
            attn = block.attn
            width = normed.size(-1)
            heads = attn.num_heads
            dim = width // heads
            q_weight, k_weight, _ = attn.in_proj_weight.chunk(3, dim=0)
            if attn.in_proj_bias is not None:
                q_bias, k_bias, _ = attn.in_proj_bias.chunk(3)
            else:
                q_bias = k_bias = None
            q = F.linear(normed[0], q_weight, q_bias)
            k = F.linear(normed, k_weight, k_bias)
            q = q.reshape(-1, heads, dim)
            k = k.permute(1, 0, 2).reshape(-1, normed.size(0), heads, dim)
            score = torch.einsum("bhd,blhd->bhl", q.float(), k.float()) * (dim ** -0.5)
            captured["attention"] = score.softmax(dim=-1).mean(dim=1)[:, 1:]

        handle = last.register_forward_hook(record)
        try:
            global_feat = F.normalize(self.model.encode_image(images).float(), dim=-1)
        finally:
            handle.remove()
        if "patches" not in captured:
            raise RuntimeError("Final ViT block hook did not fire.")
        return global_feat, captured["patches"], captured["attention"]

    def _filter(self, patches, attention, is_query=False):
        """Top CLS attention + spatially connected regions, as in CDPS."""
        count = int(attention.numel())
        side = int(count ** .5)
        if side * side != count:
            raise ValueError("Expected a square spatial patch grid.")
        order = torch.argsort(attention.float().cpu(), descending=True).tolist()
        n = max(1, int(count * self.top_percent))
        chosen = set(order[:n])
        components = []
        while chosen:
            start = next(iter(chosen))
            chosen.remove(start)
            todo = deque([start])
            region = [start]
            while todo:
                node = todo.popleft()
                row, col = divmod(node, side)
                neighbors = []
                if row > 0:
                    neighbors.append(node - side)
                if row + 1 < side:
                    neighbors.append(node + side)
                if col > 0:
                    neighbors.append(node - 1)
                if col + 1 < side:
                    neighbors.append(node + 1)
                for other in neighbors:
                    if other in chosen:
                        chosen.remove(other)
                        todo.append(other)
                        region.append(other)
            components.append(region)
        components.sort(key=len, reverse=True)
        ids = [v for region in components[:self.num_regions] for v in region]
        # Avoid choosing fewer patches than needed for top-K selection.
        minimum = max(1, min(self.topk, count))
        if len(ids) < minimum:
            present = set(ids)
            ids += [v for v in order if v not in present][:minimum - len(ids)]
        if is_query and self.max_query_patches > 0 and len(ids) > self.max_query_patches:
            ids = sorted(ids, key=lambda j: float(attention[j]), reverse=True)[:self.max_query_patches]
        return patches[torch.as_tensor(ids, device=patches.device, dtype=torch.long)]

    @torch.no_grad()
    def add_session(self, task_id, loader, class_ids):
        """Build new classes only; never revisit or replace stored old memories."""
        classes = [int(c) for c in class_ids]
        if any(c in self.memory for c in classes):
            raise RuntimeError("Refusing to rebuild existing CDPS classes.")
        gathered = defaultdict(list)
        self.model.eval()
        for batch in loader:
            image = batch["image"].to(self.device)
            labels = batch["label"].tolist()
            if all(len(gathered[c]) >= self.max_support for c in classes):
                break
            global_feat, patches, attention = self._encode(image)
            for i, cid in enumerate(labels):
                cid = int(cid)
                if cid not in classes or len(gathered[cid]) >= self.max_support:
                    continue
                local = self._filter(patches[i], attention[i])
                gathered[cid].append((local.detach().cpu(), global_feat[i].detach().cpu()))
        if any(not gathered[c] for c in classes):
            raise RuntimeError("Missing CDPS support samples for current session.")

        cur_centers = {c: F.normalize(torch.stack([v for _, v in gathered[c]]).mean(0), dim=0)
                       for c in classes}
        all_centers = dict(self.centers)
        all_centers.update(cur_centers)

        for cid in classes:
            scores_to_classes = sorted(
                ((float(cur_centers[cid] @ center), other) for other, center in all_centers.items()
                 if other != cid), reverse=True
            )
            hard_negatives = [other for _, other in scores_to_classes[:self.num_confuse]]
            samples = [patch.to(self.device) for patch, _ in gathered[cid]]
            selected = []
            for candidate in samples:
                # Patch-to-class: average of per-image nearest-neighbor similarities.
                intra = torch.stack([(candidate @ part.T).max(dim=1).values for part in samples]).mean(0)
                negative_scores = []
                for other in hard_negatives:
                    other_samples = ([p.to(self.device) for p, _ in gathered[other]]
                                     if other in gathered else [self.memory[other].to(self.device)])
                    negative_scores.extend((candidate @ p.T).max(dim=1).values for p in other_samples)
                score = intra - self.omega * torch.stack(negative_scores).mean(0) if negative_scores else intra
                indices = torch.topk(score, min(self.topk, len(score))).indices
                selected.append(candidate[indices].detach().cpu())
            self.memory[cid] = F.normalize(torch.cat(selected).float(), dim=-1).contiguous()
            self.centers[cid] = cur_centers[cid].detach().cpu()
            self.session_of_class[cid] = int(task_id)

        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / ("session_%02d.pt" % task_id)
        # Save only tensor data and primitive metadata; no image replay.
        torch.save({"task_id": int(task_id), "new_classes": classes,
                    "memory": {c: self.memory[c] for c in classes},
                    "centers": {c: self.centers[c] for c in classes},
                    "settings": {"topk": self.topk, "max_support": self.max_support,
                                 "top_percent": self.top_percent, "omega": self.omega}},
                   path)
        print("CDPS stored %d classes, %d total; saved %s" %
              (len(classes), len(self.memory), path))

    @torch.no_grad()
    def probabilities(self, images, class_ids):
        """Return patch probabilities in exactly the requested class order."""
        class_ids = [int(c) for c in class_ids]
        if any(c not in self.memory for c in class_ids):
            raise KeyError("CDPS bank lacks a class requested by the classifier.")
        _, patches, attention = self._encode(images.to(self.device))
        queries = [self._filter(patches[i], attention[i], is_query=True)
                   for i in range(len(images))]
        # Process classes in bounded chunks, with padding masks for different
        # support / query patch counts. This is substantially faster than a
        # Python loop over every query/class pair.
        batch_size = len(queries)
        max_q = max(len(q) for q in queries)
        dim = queries[0].size(-1)
        q_pad = torch.zeros(batch_size, max_q, dim, device=self.device)
        q_len = torch.tensor([len(q) for q in queries], device=self.device)
        for row, q in enumerate(queries):
            q_pad[row, :len(q)] = q
        q_valid = torch.arange(max_q, device=self.device)[None, :] < q_len[:, None]
        local_logits = torch.empty(batch_size, len(class_ids), device=self.device)
        chunk = 8
        for start in range(0, len(class_ids), chunk):
            names = class_ids[start:start + chunk]
            memories = [self.memory[c].to(self.device) for c in names]
            max_m = max(len(m) for m in memories)
            bank = torch.zeros(len(names), max_m, dim, device=self.device)
            m_valid = torch.zeros(len(names), max_m, dtype=torch.bool, device=self.device)
            for j, mem in enumerate(memories):
                bank[j, :len(mem)] = mem
                m_valid[j, :len(mem)] = True
            sim = torch.einsum("bqd,cmd->bcqm", q_pad, bank)
            sim = sim.masked_fill(~m_valid[None, :, None, :], -1e4)
            nearest = sim.max(dim=-1).values
            nearest = nearest.masked_fill(~q_valid[:, None, :], -1e4)
            if self.query_topk > 0:
                ranks = min(self.query_topk, max_q)
                values = nearest.topk(ranks, dim=-1).values
                n = q_len.clamp(max=ranks)
                selection = torch.arange(ranks, device=self.device)[None, None, :] < n[:, None, None]
                score = (values * selection).sum(dim=-1) / n[:, None]
            else:
                score = nearest.masked_fill(~q_valid[:, None, :], 0).sum(dim=-1) / q_len[:, None]
            local_logits[:, start:start + len(names)] = score
        return F.softmax(local_logits / self.temperature, dim=-1)
