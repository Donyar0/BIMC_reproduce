"""Frozen-CLIP orthogonal semantic residuals, independent of model weights.

All inputs are paired *selected training support* image/caption embeddings.
No gradients, training, test examples, or support resampling are involved.
"""
import torch
import torch.nn.functional as F


@torch.no_grad()
def paired_orthogonal_residual(anchor, caption_features, image_features,
                               min_residual_norm=1e-4):
    """Return per-caption residual evidence and class correction.

    anchor: [D] normalized class-level fine-grained BIMC text anchor.
    caption_features/image_features: [N,D] matched support embeddings
    from the same frozen CLIP model, in exactly the same order.

    The parallel part of caption e is (e @ a) * a; its orthogonal
    part is r_t = e - (e @ a) * a. The image residual is analogous.
    w_i = max(0, cos(r_t, r_v)) only if *both* norms exceed
    min_residual_norm. A zero or negative match contributes nothing.
    delta_c = sum_i w_i * r_t_i / N (not / sum(w)), so unreliable
    and missing evidence never magnifies a single surviving caption.
    """
    if anchor.ndim != 1 or caption_features.ndim != 2 or image_features.ndim != 2:
        raise ValueError("Expected anchor [D], captions [N,D], images [N,D].")
    if caption_features.shape != image_features.shape or caption_features.shape[1] != anchor.numel():
        raise ValueError("Paired support caption/image feature shapes do not match anchor.")
    if caption_features.shape[0] == 0:
        raise ValueError("Call paired_orthogonal_residual only on nonempty paired supports.")
    if min_residual_norm <= 0:
        raise ValueError("min_residual_norm must be positive.")

    a = F.normalize(anchor.float(), dim=0)
    e = F.normalize(caption_features.float(), dim=-1)
    v = F.normalize(image_features.float(), dim=-1)
    # Explicitly remove parallel components in each modality.
    text_parallel = (e @ a).unsqueeze(-1) * a.unsqueeze(0)
    visual_parallel = (v @ a).unsqueeze(-1) * a.unsqueeze(0)
    text_residual = e - text_parallel
    visual_residual = v - visual_parallel
    text_norm = text_residual.norm(dim=-1)
    visual_norm = visual_residual.norm(dim=-1)
    denom = (text_norm * visual_norm).clamp_min(1e-12)
    cosine = (text_residual * visual_residual).sum(dim=-1) / denom
    valid = (text_norm >= min_residual_norm) & (visual_norm >= min_residual_norm)
    cosine = torch.where(valid, cosine.clamp(-1.0, 1.0), torch.zeros_like(cosine))
    weights = cosine.clamp(0.0, 1.0)
    delta = (weights.unsqueeze(-1) * text_residual).sum(dim=0) / float(e.shape[0])
    return delta, weights, cosine, text_norm, visual_norm
