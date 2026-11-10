"""CLIP-A-self text adapter (Maniparambil et al., ICCV-W 2023) for BiMC.

Frozen CLIP encodes all category-conditioned descriptions. This module learns
self-attention across descriptions *within* each class on base-session images.
At all later FSCIL sessions the learned weights remain frozen.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def pack_description_bank(features, labels):
    """Pack variable-length per-class description embeddings.

    Returns padded [classes, max_descriptions, dim], valid mask [classes, max],
    and sorted class IDs. Padding never participates in attention or averages.
    """
    if features.ndim != 2 or features.size(0) == 0:
        raise ValueError("features must be a nonempty [N, D] tensor")
    if labels.ndim != 1 or labels.numel() != features.size(0):
        raise ValueError("labels must have one entry for each description")
    classes, counts = torch.unique(labels.long(), sorted=True, return_counts=True)
    max_count = int(counts.max().item())
    packed = features.new_zeros((classes.numel(), max_count, features.size(-1)))
    mask = torch.zeros((classes.numel(), max_count), dtype=torch.bool, device=features.device)
    for row, class_id in enumerate(classes):
        values = features[labels == class_id]
        packed[row, :values.size(0)] = values
        mask[row, :values.size(0)] = True
    return packed, mask, classes


def mean_description_prototypes(descriptions, valid_mask):
    """No-training mean of CLIP-encoded, class-conditioned descriptions.

    Unlike an untrained SelfAttentionTextAdapter (even with ratio=0), this
    applies neither learned projections nor LayerNorm.
    """
    if descriptions.ndim != 3 or valid_mask.shape != descriptions.shape[:2]:
        raise ValueError("Expected descriptions [C, M, D] and mask [C, M]")
    if not bool(valid_mask.any(dim=1).all()):
        raise ValueError("Every class must contain at least one description")
    valid = valid_mask.unsqueeze(-1)
    pooled = (descriptions.float() * valid).sum(dim=1) / valid.sum(dim=1)
    return F.normalize(pooled, dim=-1)


def visual_weighted_description_prototypes(descriptions, valid_mask, visual_prototypes,
                                            temperature=10.0, blend=0.5):
    """Training-free per-class description weighting from SUPPORT visual evidence.

    The visual_prototypes input must contain training images only (all base
    training images or five incremental shots), not any test-set features.
    Uses CLIP cosine similarity between each class's own visual prototype and
    its name-conditioned description embeddings; no cross-class leakage.
    At blend=0 or temperature=0, reduces to untrained description mean.
    """
    if descriptions.ndim != 3 or valid_mask.shape != descriptions.shape[:2]:
        raise ValueError("Expected descriptions [C, M, D] and mask [C, M]")
    if visual_prototypes.shape != (descriptions.size(0), descriptions.size(2)):
        raise ValueError("visual_prototypes must have shape [C, D]")
    if not bool(valid_mask.any(dim=1).all()):
        raise ValueError("Each class must have valid descriptions")
    if temperature < 0 or not 0.0 <= blend <= 1.0:
        raise ValueError("Temperature >= 0 and blend in [0,1] required")

    desc = F.normalize(descriptions.float(), dim=-1)
    visual = F.normalize(visual_prototypes.float(), dim=-1)
    similarities = (desc * visual[:, None, :]).sum(dim=-1)
    logits = (float(temperature) * similarities).masked_fill(~valid_mask, -1e4)
    weights = torch.softmax(logits, dim=1)
    weighted = F.normalize((desc * weights.unsqueeze(-1)).sum(dim=1), dim=-1)
    mean = mean_description_prototypes(desc, valid_mask)
    output = F.normalize((1.0 - blend) * mean + blend * weighted, dim=-1)
    return output, weights


class SelfAttentionTextAdapter(nn.Module):
    """The one-head, reduced-QKV, residual/LN CLIP-A-self design.

    Each row of the input is one class; attention only mixes *descriptions
    of that class*, never features of different classes.
    """

    def __init__(self, embed_dim=512, reduction=4, ratio=0.2, dropout=0.1):
        super().__init__()
        if embed_dim < reduction or reduction < 1:
            raise ValueError("Invalid embed_dim/reduction")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must lie in [0, 1]")
        hidden = embed_dim // reduction
        self.q_proj = nn.Linear(embed_dim, hidden, bias=False)
        self.k_proj = nn.Linear(embed_dim, hidden, bias=False)
        self.v_proj = nn.Linear(embed_dim, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.hidden = hidden
        self.ratio = ratio

    def forward(self, descriptions, valid_mask):
        """Compute one L2-normalized text prototype per class [C, D]."""
        if descriptions.ndim != 3:
            raise ValueError("descriptions must have shape [C, M, D]")
        if valid_mask.shape != descriptions.shape[:2]:
            raise ValueError("valid_mask must have shape [C, M]")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("Every class must have a description")

        # Cast cached fp16 CLIP embeddings to the adapter's fp32 precision.
        x = descriptions.to(dtype=self.q_proj.weight.dtype)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        logits = q @ k.transpose(-2, -1) / math.sqrt(self.hidden)
        logits = logits.masked_fill(~valid_mask.unsqueeze(1), -1e4)
        attn = self.attn_dropout(torch.softmax(logits, dim=-1))
        out = self.out_dropout(self.out_proj(attn @ v))

        # Matches original repository's residual mixing followed by LayerNorm.
        adapted = self.norm(2.0 * (self.ratio * out + (1.0 - self.ratio) * x))
        weights = valid_mask.unsqueeze(-1).to(adapted.dtype)
        per_class = (adapted * weights).sum(dim=1) / weights.sum(dim=1)
        return F.normalize(per_class, dim=-1)


class LinearTextAdapter(nn.Module):
    """Parameter-matched non-attention control: per-description D->D projection.

    For reduction=4 attention, this model has exactly the same number of
    trainable parameters: D^2 + 3D (Linear + bias + LayerNorm).
    It uses the same residual ratio, dropout, LN and mean pooling.
    """

    def __init__(self, embed_dim=512, ratio=0.2, dropout=0.1):
        super().__init__()
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must lie in [0, 1]")
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ratio = ratio

    def forward(self, descriptions, valid_mask):
        if descriptions.ndim != 3 or valid_mask.shape != descriptions.shape[:2]:
            raise ValueError("Expected descriptions [C, M, D] and mask [C, M]")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("Every class needs at least one description")
        x = descriptions.to(dtype=self.projection.weight.dtype)
        out = self.dropout(self.projection(x))
        y = self.norm(2.0 * (self.ratio * out + (1.0 - self.ratio) * x))
        mask = valid_mask.unsqueeze(-1).to(y.dtype)
        return F.normalize((y * mask).sum(1) / mask.sum(1), dim=-1)


class MLPTextAdapter(nn.Module):
    """Parameter-matched MLP control with per-description feature mixing.

    D->D/2 (no bias), GELU, D/2->D (with bias) + LN gives D^2+3D
    parameters, identical to Linear and attention with reduction=4.
    It does not mix different descriptions; only attention does that.
    """

    def __init__(self, embed_dim=512, ratio=0.2, dropout=0.1):
        super().__init__()
        if embed_dim % 2:
            raise ValueError("embed_dim must be even for parameter matching")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must lie in [0, 1]")
        self.fc1 = nn.Linear(embed_dim, embed_dim // 2, bias=False)
        self.fc2 = nn.Linear(embed_dim // 2, embed_dim, bias=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ratio = ratio

    def forward(self, descriptions, valid_mask):
        if descriptions.ndim != 3 or valid_mask.shape != descriptions.shape[:2]:
            raise ValueError("Expected descriptions [C, M, D] and mask [C, M]")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("Every class needs at least one description")
        x = descriptions.to(dtype=self.fc1.weight.dtype)
        out = self.dropout(self.fc2(F.gelu(self.fc1(x))))
        y = self.norm(2.0 * (self.ratio * out + (1.0 - self.ratio) * x))
        mask = valid_mask.unsqueeze(-1).to(y.dtype)
        return F.normalize((y * mask).sum(1) / mask.sum(1), dim=-1)
