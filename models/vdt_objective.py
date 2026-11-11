"""Classification objectives for the frozen-CLIP Linear text adapter.

text_ce: original image-to-text-prototype cross entropy.
fusion_ce: CE on the *actual* normalized BiMC text/visual fused prototype.

To avoid a supervised shortcut when the query image is part of its own
visual class prototype, fusion_ce uses a leave-one-out (LOO) positive
visual prototype. The negative-class prototypes remain the ordinary
training/support prototypes, matching FSCIL inference. No test samples
are used and CLIP remains frozen.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def support_feature_moments(features, targets, num_classes):
    """Per-class support-image sums and counts; images must be L2-normalized."""
    if features.ndim != 2 or targets.ndim != 1 or len(features) != len(targets):
        raise ValueError("Expected image features [N,D] and labels [N]")
    if num_classes < 1 or len(targets) == 0:
        raise ValueError("Need nonempty support classes")
    if int(targets.min()) < 0 or int(targets.max()) >= num_classes:
        raise ValueError("Support labels out of seen-class range")
    sums = features.detach().new_zeros((num_classes, features.shape[-1]))
    sums.index_add_(0, targets.long(), features.detach())
    counts = torch.bincount(targets.long(), minlength=num_classes)
    return sums, counts


def linear_adapter_ce(image_features, targets, text_prototypes,
                      visual_prototypes=None, beta=None, mode="text_ce",
                      support_sums=None, support_counts=None,
                      calibrate_loo=None, logit_scale=100.0):
    """CE for a batch, with optional fusion and LOO positive correction.

    visual_prototypes are computed from support/training images only.
    support_sums/counts should be computed from the *same session's*
    support images. The current query image is removed from its true
    class visual centroid for the CE's positive logit.
    """
    if image_features.ndim != 2 or text_prototypes.ndim != 2:
        raise ValueError("Image and class prototypes must be [N,D] and [C,D]")
    if targets.ndim != 1 or len(targets) != len(image_features):
        raise ValueError("Targets must match query batch")
    if mode not in ("text_ce", "fusion_ce"):
        raise ValueError("Unknown Linear Adapter training objective: " + str(mode))
    if targets.numel() == 0 or int(targets.min()) < 0 or int(targets.max()) >= len(text_prototypes):
        raise ValueError("Query labels out of seen-class range")

    images = F.normalize(image_features.detach().float(), dim=-1)
    text = F.normalize(text_prototypes.float(), dim=-1)
    if mode == "text_ce":
        return F.cross_entropy(float(logit_scale) * (images @ text.t()), targets.long())

    if visual_prototypes is None or beta is None:
        raise ValueError("fusion_ce requires support visual prototypes and beta")
    if not 0 < float(beta) <= 1:
        raise ValueError("Text fusion beta must be in (0,1]")
    if visual_prototypes.shape != text.shape:
        raise ValueError("Visual/text class prototypes must have identical shapes")
    if support_sums is None or support_counts is None:
        raise ValueError("fusion_ce requires support moments to remove self-match")

    visual = F.normalize(visual_prototypes.detach().float(), dim=-1)
    classifier = F.normalize(float(beta) * text + (1.0 - float(beta)) * visual, dim=-1)
    logits = float(logit_scale) * (images @ classifier.t())

    # The correct-class visual prototype for a training image must exclude
    # that same image. Otherwise fused CE can be artificially easy.
    idx = targets.long()
    if support_sums.shape != text.shape or support_counts.numel() != len(text):
        raise ValueError("Support moments must match the seen-class vocabulary")
    if not bool((support_counts[idx] >= 2).all()):
        raise ValueError("LOO fused CE needs at least two support images per class")
    with torch.no_grad():
        remaining = support_sums[idx].detach().float() - images
        loo_visual = F.normalize(remaining, dim=-1)
        if calibrate_loo is not None:
            loo_visual = F.normalize(calibrate_loo(loo_visual), dim=-1)
    own_classifier = F.normalize(
        float(beta) * text[idx] + (1.0 - float(beta)) * loo_visual, dim=-1
    )
    own_logit = float(logit_scale) * (images * own_classifier).sum(dim=-1)
    logits = logits.scatter(1, idx[:, None], own_logit[:, None])
    return F.cross_entropy(logits, idx)
