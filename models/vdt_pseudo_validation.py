"""Pseudo-incremental validation using base TRAIN images only.

Used solely to select the nonnegative hard-negative strength of a
training-free description reweighting method. Real test images and real
incremental shots NEVER enter validation or hyperparameter selection.

Protocol per pseudo episode:
  - split base TRAIN classes into 75% pseudo-base / 25% pseudo-novel
  - withhold query images from EVERY class;
  - use other pseudo-base training images for old prototypes;
  - use exactly five support images for each pseudo-novel prototype;
  - apply the same BiMC vision calibration to pseudo-novel prototypes;
  - classify withheld query features against the fused text+visual prototypes.
The score is the harmonic mean of pseudo-base and pseudo-novel accuracy,
averaged over fixed episodes (classes/samples seeded once, reused for ALL
candidate strengths).
"""
import math

import torch
from torch.nn import functional as F

from models.vdt_self_attention import pack_description_bank
from models.vdt_contrastive import contrastive_description_prototypes


@torch.no_grad()
def select_base_only_contrastive_strength(
    base_state, beta, shift_weight, tau, seed, candidates=(0.0, 0.5, 1.0, 2.0),
    episodes=3, query_per_class=5, novel_shots=5, temperature=10.0, blend=0.5,
):
    """Return (best_strength, report) without access to FSCIL test samples."""
    if episodes < 1 or query_per_class < 1 or novel_shots < 1:
        raise ValueError("Episodes, queries and novel shots must be positive")
    if not candidates or any(not math.isfinite(float(x)) or x < 0 for x in candidates):
        raise ValueError("Candidates must be finite and >=0")
    if not 0 <= beta <= 1 or not 0 <= shift_weight <= 1:
        raise ValueError("Invalid vision/text fusion weights")

    x = F.normalize(base_state["images_features"].detach().float(), dim=-1)
    y = base_state["images_targets"].detach().long().to(x.device)
    desc, mask, ids = pack_description_bank(
        base_state["vdt_features"].detach(),
        base_state["vdt_targets"].detach()
    )
    c = len(ids)
    if c < 4:
        raise ValueError("At least four base classes required for pseudo-incremental split")
    expected = torch.arange(c, device=ids.device)
    if not torch.equal(ids, expected):
        raise ValueError("Base class IDs must be contiguous starting at zero")
    per_class = [(y == i).nonzero(as_tuple=True)[0].cpu() for i in range(c)]
    required = novel_shots + query_per_class
    if any(len(indices) < required for indices in per_class):
        raise ValueError("Insufficient training images for disjoint support/query split")

    # CPU generator makes sampling reproducible and device-independent.
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) + 809071)
    episodes_data = []
    n_novel = max(1, min(c-1, round(0.25*c)))
    for episode_id in range(int(episodes)):
        order = torch.randperm(c, generator=gen).tolist()
        novel_ids = order[:n_novel]
        old_ids = order[n_novel:]
        ordering = old_ids + novel_ids
        supports = []
        q_features = []
        q_labels = []
        for new_id, actual_id in enumerate(ordering):
            indices = per_class[actual_id]
            shuffled = indices[torch.randperm(len(indices), generator=gen)]
            query_idx = shuffled[:query_per_class].to(x.device)
            other_idx = shuffled[query_per_class:].to(x.device)
            used_idx = other_idx[:novel_shots] if new_id >= len(old_ids) else other_idx
            supports.append(F.normalize(x[used_idx].mean(dim=0), dim=-1))
            q_features.append(x[query_idx])
            q_labels.append(torch.full((len(query_idx),), new_id,
                                       device=x.device, dtype=torch.long))
        visual = torch.stack(supports)
        if shift_weight > 0 and len(novel_ids):
            reference = visual[:len(old_ids)]
            novel = visual[len(old_ids):]
            similarity = float(tau) * (novel @ reference.T)
            correction = F.normalize(torch.softmax(similarity, dim=-1) @ reference, dim=-1)
            visual[len(old_ids):] = F.normalize(
                (1-shift_weight)*novel + shift_weight*correction, dim=-1
            )
        episodes_data.append({
            "visual_for_weights": torch.stack(supports),
            "visual_for_fusion": visual,
            "descriptions": desc[ordering],
            "mask": mask[ordering],
            "queries": torch.cat(q_features, dim=0),
            "labels": torch.cat(q_labels, dim=0),
            "num_pseudo_base": len(old_ids),
            "pseudo_novel_class_ids": novel_ids,
        })

    values=[]
    for strength in sorted(set(float(x) for x in candidates)):
        scores=[]
        for e in episodes_data:
            text,_,_ = contrastive_description_prototypes(
                e["descriptions"],e["mask"],e["visual_for_weights"],
                temperature=temperature,blend=blend,negative_strength=strength
            )
            fusion=F.normalize(float(beta)*text+
                               (1.0-float(beta))*e["visual_for_fusion"],dim=-1)
            pred=(e["queries"]@fusion.T).argmax(dim=-1)
            labels=e["labels"]
            base_mask=labels<e["num_pseudo_base"]
            novel_mask=~base_mask
            a=float((pred[base_mask]==labels[base_mask]).float().mean())
            b=float((pred[novel_mask]==labels[novel_mask]).float().mean())
            harmonic=2*a*b/(a+b) if (a+b)>0 else 0.0
            scores.append({
                "base_accuracy":round(100*a,4),
                "novel_accuracy":round(100*b,4),
                "harmonic_accuracy":round(100*harmonic,4),
            })
        values.append({
            "strength":strength,
            "episode_scores":scores,
            "harmonic_mean":round(sum(s["harmonic_accuracy"] for s in scores)/len(scores),6),
            "base_mean":round(sum(s["base_accuracy"] for s in scores)/len(scores),6),
            "novel_mean":round(sum(s["novel_accuracy"] for s in scores)/len(scores),6),
        })

    # Exact tie => select smallest strength; no test-based tie-breaking.
    selected=sorted(values,key=lambda r:(-r["harmonic_mean"],r["strength"]))[0]
    return selected["strength"],{
        "selection_source":"base_training_images_only",
        "seed":int(seed),
        "episodes":int(episodes),
        "pseudo_novel_classes_per_episode":int(n_novel),
        "query_images_per_class":int(query_per_class),
        "novel_support_images_per_class":int(novel_shots),
        "objective":"mean_pseudo_base_novel_harmonic_accuracy",
        "beta":float(beta),"vision_shift_weight":float(shift_weight),
        "vision_tau":float(tau),
        "candidate_results":values,
        "selected_strength":float(selected["strength"]),
    }
