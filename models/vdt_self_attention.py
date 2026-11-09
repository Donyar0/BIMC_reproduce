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
