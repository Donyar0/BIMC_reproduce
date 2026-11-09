"""Flora-inspired, training-free, multi-description calibration for BiMC.

CLIP returns global D-dimensional text embeddings rather than Flora's token
features. Keep every category-prompt/description pair in 2D space until
neighbor calibration, then aggregate its pairs to one D-dimensional classifier
prototype per class.
"""

import torch
import torch.nn.functional as F


@torch.no_grad()
def calibrate_neighbor_features(
    joint_features: torch.Tensor,
    pair_labels: torch.Tensor,
    topk: int,
    tau: float,
    chunk_size: int = 512,
) -> torch.Tensor:
    """Calibrate all paired descriptions using other *seen* class neighbors.

    joint_features: [number of all descriptions, 2 * clip_dim], each row is
        concat(normalized class-prompt feature, normalized description feature).
    pair_labels: [number of all descriptions], integer class IDs (0..C-1).
    topk: number of neighboring **classes**, excluding the pair's own class.
    tau: residual neighbor weight, as in Flora.
    Returns: [C, clip_dim] normalized per-class text prototypes.

    To stop classes with many descriptions from dominating the neighbors,
    each class supplies one centroid in the concatenated feature space.
    Each individual pair is calibrated using its top-k other-class centroids;
    pairs are then averaged *within their class*, with equal weight per pair.
    """
    if joint_features.ndim != 2 or joint_features.size(0) == 0 or joint_features.size(1) % 2:
        raise ValueError("joint_features must have shape [num_pairs > 0, 2 * clip_dim].")
    if pair_labels.ndim != 1 or pair_labels.numel() != joint_features.size(0):
        raise ValueError("pair_labels must have one class ID per category-description pair.")
    if int(topk) < 0:
        raise ValueError("topk must be nonnegative.")
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be positive.")

    labels = pair_labels.to(device=joint_features.device, dtype=torch.long)
    classes, inverse, counts = torch.unique(
        labels, sorted=True, return_inverse=True, return_counts=True
    )
    if not torch.equal(classes, torch.arange(len(classes), device=classes.device)):
        raise ValueError("Seen class IDs must be consecutive, starting at zero.")

    original_dtype = joint_features.dtype
    pairs = F.normalize(joint_features.float(), dim=-1)
    num_classes, dim = len(classes), pairs.size(1)

    # A single class-level neighbor anchor represents ALL its descriptions.
    class_centroids = pairs.new_zeros((num_classes, dim))
    class_centroids.index_add_(0, inverse, pairs)
    class_centroids = F.normalize(class_centroids / counts.unsqueeze(1), dim=-1)

    k = min(int(topk), num_classes - 1)
    result = pairs.new_zeros((num_classes, dim // 2))
    for start in range(0, pairs.size(0), int(chunk_size)):
        end = min(start + int(chunk_size), pairs.size(0))
        calibrated = pairs[start:end]
        if k:
            similarities = calibrated @ class_centroids.t()
            similarities.scatter_(1, inverse[start:end].unsqueeze(1), float('-inf'))
            values, indices = similarities.topk(k, dim=1)
            neighbors = (class_centroids[indices] * values.unsqueeze(-1)).mean(dim=1)
            calibrated = calibrated + float(tau) * neighbors

        prompt_part, desc_part = calibrated.chunk(2, dim=-1)
        pair_proto = F.normalize(
            F.normalize(prompt_part, dim=-1) + F.normalize(desc_part, dim=-1), dim=-1
        )
        result.index_add_(0, inverse[start:end], pair_proto)

    result = F.normalize(result / counts.unsqueeze(1), dim=-1)
    return result.to(original_dtype)
