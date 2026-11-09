"""Training-free neighbor-aware text calibration for CLIP-based FSCIL.

Inspired by Flora, but uses pooled CLIP text embeddings instead of token features.
"""

import random

import torch
import torch.nn.functional as F


def sample_description_index(num_descriptions: int, seed: int, class_id: int) -> int:
    """Choose one description per class, independent of dataloader RNG/order."""
    if num_descriptions < 1:
        raise ValueError("Each class must have at least one nonempty description.")
    return random.Random(int(seed) * 1000003 + int(class_id)).randrange(num_descriptions)


@torch.no_grad()
def calibrate_neighbor_features(joint_features: torch.Tensor, topk: int, tau: float) -> torch.Tensor:
    """Calibrate concatenated [class-prompt, sampled-description] embeddings.

    All rows must represent *seen* categories in incremental class order.
    Cosine Top-K neighbors are chosen in the 2D concatenated feature space.
    The calibrated vector is converted back to CLIP's D-dimensional image
    space by equally combining its two D-dimensional components, without
    introducing a trainable projection.
    """
    if joint_features.ndim != 2 or joint_features.size(0) == 0 or joint_features.size(1) % 2:
        raise ValueError("joint_features must have shape [num_seen_classes, 2 * clip_dim].")
    if int(topk) < 0:
        raise ValueError("topk must be nonnegative.")

    original_dtype = joint_features.dtype
    normalized = F.normalize(joint_features.float(), dim=-1)
    k = min(int(topk), joint_features.size(0) - 1)

    if k:
        similarities = normalized @ normalized.t()
        similarities.fill_diagonal_(float('-inf'))
        neighbor_weights, neighbor_indices = similarities.topk(k, dim=1)
        neighbor_delta = (normalized[neighbor_indices] * neighbor_weights.unsqueeze(-1)).mean(dim=1)
        normalized = normalized + float(tau) * neighbor_delta

    class_part, description_part = normalized.chunk(2, dim=-1)
    combined = F.normalize(class_part, dim=-1) + F.normalize(description_part, dim=-1)
    return F.normalize(combined, dim=-1).to(original_dtype)
