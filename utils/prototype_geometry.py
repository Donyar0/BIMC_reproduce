"""Read-only support-visual/text geometry diagnostics for FSCIL.

Prototype comparisons are computed from train/support visual prototypes and
frozen CLIP text encodings; test labels and test images are never used here.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def prototype_geometry(raw_text, candidate_text, support_visual, class_names,
                       base_count, top_k=30):
    """Pairwise text margins and matched visual alignment, in cosine units.

    A pair's separation is (1 - cosine). Positive delta means enlarged
    text distance. For each class report its nearest OTHER text class.
    """
    if raw_text.ndim != 2 or candidate_text.shape != raw_text.shape or support_visual.shape != raw_text.shape:
        raise ValueError("All prototype banks must be [C,D] of equal shape")
    c = raw_text.size(0)
    if len(class_names) < c or c < 2 or not (0 < base_count <= c):
        raise ValueError("Invalid class names/base count")
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    raw = F.normalize(raw_text.float(), dim=-1)
    adapted = F.normalize(candidate_text.float(), dim=-1)
    visual = F.normalize(support_visual.float(), dim=-1)
    raw_sim = raw @ raw.T
    new_sim = adapted @ adapted.T
    support_raw = (raw * visual).sum(-1)
    support_new = (adapted * visual).sum(-1)
    mask = torch.eye(c, dtype=torch.bool, device=raw.device)
    raw_nearest = raw_sim.masked_fill(mask, -float("inf")).argmax(-1)
    new_nearest = new_sim.masked_fill(mask, -float("inf")).argmax(-1)
    cls = []
    for i in range(c):
        cls.append({
            "class_id": i, "class_name": str(class_names[i]),
            "group": "base" if i < base_count else "novel",
            "raw_support_cos": round(float(support_raw[i]), 6),
            "candidate_support_cos": round(float(support_new[i]), 6),
            "delta_support_cos": round(float(support_new[i] - support_raw[i]), 6),
            "raw_closest_class": str(class_names[int(raw_nearest[i])]),
            "candidate_closest_class": str(class_names[int(new_nearest[i])]),
            "raw_closest_cos": round(float(raw_sim[i, raw_nearest[i]]), 6),
            "candidate_closest_cos": round(float(new_sim[i, new_nearest[i]]), 6),
        })
    triu = torch.triu_indices(c, c, offset=1, device=raw.device)
    pairs = []
    for i,j in zip(triu[0].tolist(),triu[1].tolist()):
        old = float(raw_sim[i,j])
        new = float(new_sim[i,j])
        pairs.append({
            "class_a_id": i, "class_a": str(class_names[i]),
            "class_b_id": j, "class_b": str(class_names[j]),
            "group": ("base-base" if j < base_count else
                       "novel-novel" if i >= base_count else "base-novel"),
            "raw_cos": round(old, 6), "candidate_cos": round(new, 6),
            "delta_separation": round(old - new, 6),
        })
    # Save the closest raw-text class pairs, including whether those pairs
    # became closer or farther after calibration. Also save largest changes.
    high_similarity = sorted(pairs, key=lambda p: -p["raw_cos"])[:top_k]
    collapsed = sorted(pairs, key=lambda p: p["delta_separation"])[:top_k]
    separated = sorted(pairs, key=lambda p: -p["delta_separation"])[:top_k]
    base = cls[:base_count]
    novel = cls[base_count:]
    def avg(rows, key):
        return round(sum(r[key] for r in rows)/len(rows), 6) if rows else None
    return {
        "num_seen_classes": c,
        "base_count": base_count,
        "support_alignment": {
            "raw_base_mean": avg(base, "raw_support_cos"),
            "candidate_base_mean": avg(base, "candidate_support_cos"),
            "raw_novel_mean": avg(novel, "raw_support_cos"),
            "candidate_novel_mean": avg(novel, "candidate_support_cos"),
            "delta_base_mean": avg(base, "delta_support_cos"),
            "delta_novel_mean": avg(novel, "delta_support_cos"),
        },
        "classes": cls,
        "most_similar_raw_pairs": high_similarity,
        "most_collapsed_pairs": collapsed,
        "most_separated_pairs": separated,
    }
