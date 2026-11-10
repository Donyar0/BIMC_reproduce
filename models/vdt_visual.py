"""CLIP-A-self-style intra-class visual self-attention aggregation for FSCIL.

CLIP image encoder is frozen. Train the visual aggregation module *only* on
base-class training image features via disjoint support/query episodes. Each
episode uses a few support images to build a class prototype, then separate
base training images as CE queries. Incremental sessions reuse this same
frozen module on all available new-class support images (typically five).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


class VisionSelfAttentionAdapter(nn.Module):
    """One-head reduced-QKV residual self-attention followed by masked mean.

    Identical architectural pattern to the original CLIP-A-self-style
    text-description attention in this repository, but tokens are *images
    of the same class*, not descriptions. No inter-class attention.
    """

    def __init__(self, embed_dim=512, reduction=4, ratio=0.2, dropout=0.1):
        super().__init__()
        if embed_dim < reduction or reduction < 1:
            raise ValueError("Invalid embed_dim / attention reduction")
        if not 0 <= ratio <= 1:
            raise ValueError("Residual ratio must lie in [0,1]")
        hidden = embed_dim // reduction
        self.q_proj = nn.Linear(embed_dim, hidden, bias=False)
        self.k_proj = nn.Linear(embed_dim, hidden, bias=False)
        self.v_proj = nn.Linear(embed_dim, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)
        self.hidden = hidden
        self.ratio = float(ratio)

    def forward(self, images, valid_mask):
        if images.ndim != 3 or valid_mask.shape != images.shape[:2]:
            raise ValueError("Expected image tokens [C,K,D] and valid mask [C,K]")
        if not bool(valid_mask.any(dim=1).all()):
            raise ValueError("Each class needs at least one support image")
        x = images.to(dtype=self.q_proj.weight.dtype)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.hidden)
        logits = logits.masked_fill(~valid_mask.unsqueeze(1), -1e4)
        attention = self.attn_dropout(torch.softmax(logits, dim=-1))
        residual = self.out_dropout(self.out_proj(attention @ v))
        adapted = self.norm(2.0 * (self.ratio * residual +
                                    (1.0 - self.ratio) * x))
        weights = valid_mask.unsqueeze(-1).to(dtype=adapted.dtype)
        return F.normalize((adapted * weights).sum(dim=1) /
                           weights.sum(dim=1), dim=-1)


def sample_disjoint_visual_episode(features, targets, num_classes,
                                   support_per_class, query_per_class,
                                   generator=None):
    """Sample support and query from separate indices within BASE TRAIN.

    Returns [C,K,D] support, [C,K] mask, [C*Q,D] queries, global labels,
    and sampled support/query indices for audit/testing.
    """
    if support_per_class < 1 or query_per_class < 1:
        raise ValueError("Need positive support/query counts")
    if num_classes < 2:
        raise ValueError("Episode needs at least two classes")
    if targets.numel() != features.shape[0] or features.ndim != 2:
        raise ValueError("Expected base image features [N,D] and targets [N]")
    if generator is None:
        generator = torch.Generator(device="cpu").manual_seed(1)
    targets_cpu = targets.detach().cpu().long()
    support_rows, query_rows, support_idx, query_idx = [], [], [], []
    for class_id in range(num_classes):
        candidates = (targets_cpu == class_id).nonzero(as_tuple=True)[0]
        if len(candidates) < support_per_class + query_per_class:
            raise ValueError("Not enough base training images for class " +
                             str(class_id))
        order = torch.randperm(len(candidates), generator=generator)
        selected = candidates[order[:support_per_class + query_per_class]]
        si = selected[:support_per_class]
        qi = selected[support_per_class:]
        support_idx.append(si)
        query_idx.append(qi)
        support_rows.append(features[si.to(features.device)])
        query_rows.append(features[qi.to(features.device)])
    support_idx = torch.stack(support_idx)
    query_idx = torch.stack(query_idx)
    if bool(torch.isin(support_idx.reshape(-1),
                       query_idx.reshape(-1)).any()):
        raise AssertionError("Episode has support/query index overlap")
    support = torch.stack(support_rows, dim=0).detach()
    queries = torch.cat(query_rows, dim=0).detach()
    query_labels = torch.arange(num_classes,device=features.device).repeat_interleave(
        query_per_class
    )
    mask = torch.ones(support.shape[:2],device=features.device,dtype=torch.bool)
    return support, mask, queries, query_labels, support_idx, query_idx


def train_visual_aggregator(adapter, features, targets, num_classes,
                            epochs=20, steps_per_epoch=10,
                            support_per_class=5, query_per_class=5,
                            lr=1e-4, weight_decay=1e-4, seed=1):
    """Train visual self-attention with episodic, class-level prototype CE.

    Only adapter parameters get gradients; both query and support CLIP
    image features are detached. A different episode is sampled for
    each optimization step; no real test features/labels are accessed.
    """
    if epochs < 1 or steps_per_epoch < 1 or lr <= 0:
        raise ValueError("Invalid visual training epochs/steps/lr")
    if features.ndim != 2 or len(features) != len(targets):
        raise ValueError("Invalid base support features")
    class_ids = torch.unique(targets.long(),sorted=True)
    if not torch.equal(class_ids,torch.arange(num_classes,device=targets.device)):
        raise ValueError("Base training class IDs must be 0..C-1")
    features = F.normalize(features.detach().float(),dim=-1)
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(lr),
                                  weight_decay=float(weight_decay))
    for p in adapter.parameters():
        p.requires_grad_(True)
    adapter.train()
    history = []
    try:
        with torch.enable_grad():
            for epoch in range(int(epochs)):
                sum_loss = 0.0
                for _ in range(int(steps_per_epoch)):
                    supports, mask, queries, labels, _, _ = (
                        sample_disjoint_visual_episode(
                            features, targets, num_classes,
                            support_per_class,query_per_class,generator=gen
                        )
                    )
                    optimizer.zero_grad()
                    prototypes = adapter(supports,mask)
                    logits = 100.0 * (queries @ prototypes.t())
                    loss = F.cross_entropy(logits,labels)
                    loss.backward()
                    optimizer.step()
                    sum_loss += float(loss.detach())
                mean_loss = sum_loss / steps_per_epoch
                history.append(mean_loss)
                print("VISUAL_ADAPTER base epoch={}/{} episodic_CE={:.6f}".format(
                    epoch+1,epochs,mean_loss
                ))
    finally:
        adapter.eval()
        for p in adapter.parameters():
            p.requires_grad_(False)
    return history


@torch.no_grad()
def aggregate_visual_classes(adapter, features, targets, class_ids):
    """Aggregate every class's available training/support images, no sampling.

    One class at a time bounds attention memory when CIFAR base classes have
    ~500 training images. New classes typically have exactly five.
    """
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("Expected features [N,D] and labels [N]")
    if not isinstance(class_ids,torch.Tensor):
        class_ids = torch.as_tensor(class_ids,device=targets.device)
    adapter.eval()
    prototypes=[]
    for cid in class_ids.tolist():
        selected = features[targets == cid]
        if selected.shape[0] == 0:
            raise ValueError("No support image for class " + str(cid))
        tokens = F.normalize(selected.float(),dim=-1).unsqueeze(0)
        valid = torch.ones(tokens.shape[:2],dtype=torch.bool,device=tokens.device)
        prototypes.append(adapter(tokens,valid)[0])
    return F.normalize(torch.stack(prototypes,dim=0),dim=-1)
