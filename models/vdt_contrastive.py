"""Training-free, class-contrastive weighting of CLIP description features.

The only visual references are support/training class prototypes. No test
samples or labels enter description selection. Cross-class negatives use
only class prototypes available at the current FSCIL session.
"""
import math

import torch
from torch.nn import functional as F

from models.vdt_self_attention import mean_description_prototypes


@torch.no_grad()
def contrastive_description_prototypes(descriptions, valid_mask, support_visual,
                                       temperature=10.0, blend=0.5,
                                       negative_strength=1.0):
    """Select descriptions by own visual cosine minus hardest competing cosine.

    s(c,j) = <d(c,j),v(c)> - lambda * max_{k != c}<d(c,j),v(k)>
    w(c,:) = softmax_j(temperature*s(c,j))
    Output = normalize((1-blend)*mean + blend*weighted).
    All similarities are cosine in frozen CLIP space.
    """
    if descriptions.ndim != 3 or valid_mask.shape != descriptions.shape[:2]:
        raise ValueError("Expected descriptions [C,M,D] and mask [C,M]")
    c, m, dim = descriptions.shape
    if support_visual.shape != (c, dim):
        raise ValueError("Expected support_visual [C,D]")
    if c == 0 or not bool(valid_mask.any(dim=1).all()):
        raise ValueError("Every class must have a valid description")
    if not (math.isfinite(float(temperature)) and temperature >= 0 and
            math.isfinite(float(blend)) and 0 <= blend <= 1 and
            math.isfinite(float(negative_strength)) and negative_strength >= 0):
        raise ValueError("Invalid weighting hyperparameters")

    desc = F.normalize(descriptions.float(), dim=-1)
    visual = F.normalize(support_visual.float(), dim=-1)
    all_cos = torch.matmul(desc, visual.t())  # [C,M,C]
    own = all_cos.diagonal(dim1=0, dim2=2).transpose(0, 1)  # [C,M]
    if c > 1:
        not_self = torch.eye(c, dtype=torch.bool, device=desc.device).unsqueeze(1)
        hardest_other = all_cos.masked_fill(not_self, -float("inf")).max(dim=-1).values
    else:
        hardest_other = torch.zeros_like(own)

    margins = own - float(negative_strength) * hardest_other
    logits = (float(temperature) * margins).masked_fill(~valid_mask, -1e4)
    weights = torch.softmax(logits, dim=1)
    weighted = F.normalize((desc * weights.unsqueeze(-1)).sum(dim=1), dim=-1)
    mean = mean_description_prototypes(desc, valid_mask)
    fused = F.normalize((1.0 - blend) * mean + float(blend) * weighted, dim=-1)
    return fused, weights, {"own": own, "hardest_other": hardest_other,
                            "margin": margins}


@torch.no_grad()
def description_weight_diagnostics(descriptions, valid_mask, support_visual,
                                   weights, class_names, base_count,
                                   negative_strength=0.0):
    """Per-class entropy, description redundancy and visual margins.

    Normalized entropy is 0 for one valid description, and entropy/log(M)
    otherwise; close to 1 means near-uniform weights. All statistics use
    only CLIP text and support visuals, and are *not* test-set metrics.
    """
    if descriptions.ndim != 3 or valid_mask.shape != descriptions.shape[:2]:
        raise ValueError("Expected descriptions [C,M,D] and mask [C,M]")
    c,m,d = descriptions.shape
    if weights.shape != valid_mask.shape or support_visual.shape != (c,d):
        raise ValueError("Mismatch among descriptions, weights and visuals")
    if len(class_names) < c or not 0 <= base_count <= c:
        raise ValueError("Invalid names/base_count")

    desc = F.normalize(descriptions.float(), dim=-1)
    visual = F.normalize(support_visual.float(), dim=-1)
    own = (desc * visual.unsqueeze(1)).sum(dim=-1)
    # Need max over other available class support prototypes.
    all_cos = desc @ visual.t()
    if c > 1:
        neg = all_cos.masked_fill(
            torch.eye(c,dtype=torch.bool,device=desc.device).unsqueeze(1),
            -float("inf")).max(dim=-1).values
    else:
        neg = torch.zeros_like(own)
    rows = []
    for i in range(c):
        idx=valid_mask[i].bool()
        n=int(idx.sum().item())
        w=weights[i,idx].float()
        w=w/w.sum().clamp_min(1e-12)
        entropy=float((-w*w.clamp_min(1e-12).log()).sum())
        norm_entropy=entropy/math.log(n) if n>1 else 0.0
        valid_desc=desc[i,idx]
        if n>1:
            cosine=valid_desc@valid_desc.t()
            similarity=float((cosine.sum()-n)/(n*(n-1)))
        else:
            similarity=None
        candidate_scores=own[i,idx]-float(negative_strength)*neg[i,idx]
        rows.append({
            "class_id":i,"class_name":str(class_names[i]),
            "group":"base" if i<base_count else "novel",
            "num_descriptions":n,
            "normalized_entropy":round(norm_entropy,6),
            "top1_weight":round(float(w.max()),6),
            "intra_description_cos":round(similarity,6) if similarity is not None else None,
            "own_visual_cos_mean":round(float(own[i,idx].mean()),6),
            "hardest_negative_cos_mean":round(float(neg[i,idx].mean()),6),
            "weighted_margin_mean":round(float((w*candidate_scores).sum()),6),
            "unweighted_margin_mean":round(float(candidate_scores.mean()),6),
        })
    def aggregate(selected):
        def avg(key):
            x=[r[key] for r in selected if r[key] is not None]
            return round(sum(x)/len(x),6) if x else None
        return {
            "num_classes":len(selected),
            "normalized_entropy_mean":avg("normalized_entropy"),
            "top1_weight_mean":avg("top1_weight"),
            "intra_description_cos_mean":avg("intra_description_cos"),
            "weighted_margin_mean":avg("weighted_margin_mean"),
            "unweighted_margin_mean":avg("unweighted_margin_mean"),
        }
    return {
        "num_classes":c,"negative_strength":float(negative_strength),
        "base":aggregate(rows[:base_count]),
        "novel":aggregate(rows[base_count:]),
        "per_class":rows,
    }
