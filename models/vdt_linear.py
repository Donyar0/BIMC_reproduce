"""Minimal linear text-prototype calibration for frozen-CLIP FSCIL.

Each class has variable-length name-conditioned description embeddings.
The base-trained LinearTextAdapter is frozen in every incremental session.
"""
import torch
from torch import nn
from torch.nn import functional as F


def pack_description_bank(features, labels):
    """Return padded descriptions [C,M,D], valid mask [C,M], sorted class IDs."""
    if features.ndim != 2 or features.size(0) == 0:
        raise ValueError("Features must be a nonempty [N,D] tensor")
    if labels.ndim != 1 or labels.numel() != features.size(0):
        raise ValueError("Each description must have one class label")
    classes, counts = torch.unique(labels.long(), sorted=True, return_counts=True)
    packed = features.new_zeros((len(classes), int(counts.max()), features.size(-1)))
    mask = torch.zeros(packed.shape[:2], dtype=torch.bool, device=features.device)
    for row, class_id in enumerate(classes):
        items = features[labels == class_id]
        packed[row, :len(items)] = items
        mask[row, :len(items)] = True
    return packed, mask, classes


class LinearTextAdapter(nn.Module):
    """Residual linear projection + LayerNorm + class-wise mean pooling.

    Same architecture and parameterization as the successful earlier
    VDT_MODE='linear' ablation (D^2+3D parameters at embedding dimension D).
    Does NOT use attention, visual-conditioned weighting, or test features.
    """
    def __init__(self, embed_dim=512, ratio=0.2, dropout=0.1):
        super().__init__()
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("ratio must be in [0,1]")
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ratio = float(ratio)

    def forward(self, descriptions, valid_mask):
        if descriptions.ndim != 3 or valid_mask.shape != descriptions.shape[:2]:
            raise ValueError("Expected [C,M,D] descriptions and [C,M] mask")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("Every class requires at least one valid description")
        x = descriptions.to(dtype=self.projection.weight.dtype)
        out = self.dropout(self.projection(x))
        adjusted = self.norm(2.0 * (self.ratio * out + (1.0 - self.ratio) * x))
        mask = valid_mask.unsqueeze(-1).to(dtype=adjusted.dtype)
        return F.normalize((adjusted * mask).sum(dim=1) / mask.sum(dim=1), dim=-1)
