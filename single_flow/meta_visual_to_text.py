"""Episodic first-order meta-learning for visual-to-text Flow Matching.

All images here are *training images from real base classes*. Inside each
episode, class IDs are split into pseudo-base and pseudo-novel sets; support
and query of pseudo-novel classes are disjoint. The first-order meta gradient
is obtained from query FM supervision AFTER virtual support adaptation. This
is NOT merely conventional supervised Flow training on random class splits.
"""
import math

import torch

from .fm import DeepFlowMatchingNet
from .s1_flow import ids_checked, text_cache, unit

try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


def sample_episode(class_ids, labels, *, pseudo_new=10, shot=5, query=5,
                   base_batch=64, old_query_batch=64, generator=None):
    """Return source-index tensors. No original data is modified or copied."""
    ids = ids_checked(class_ids)
    if not (1 <= pseudo_new < len(ids)) or shot < 1 or query < 1:
        raise ValueError('Invalid pseudo-incremental class or support/query counts')
    if base_batch < 1 or old_query_batch < 1:
        raise ValueError('Base and old-query batches must be positive')
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError('Labels must be int64 [N]')
    if not set(labels.tolist()).issubset(set(ids)):
        raise ValueError('Episode labels must all be real base-class labels')
    order = torch.randperm(len(ids), generator=generator).tolist()
    novel_ids = [ids[i] for i in order[:pseudo_new]]
    base_ids = [ids[i] for i in order[pseudo_new:]]
    support, novel_query = [], []
    for cid in novel_ids:
        indices = torch.where(labels == cid)[0]
        if len(indices) < shot + query:
            raise ValueError('Every pseudo-new class needs disjoint support and query')
        perm = indices[torch.randperm(len(indices), generator=generator)]
        support.append(perm[:shot])
        novel_query.append(perm[shot:shot + query])
    base_pool = torch.where(torch.isin(labels, torch.tensor(base_ids)))[0]
    if len(base_pool) == 0:
        raise ValueError('No pseudo-base training examples available')
    perm = base_pool[torch.randperm(len(base_pool), generator=generator)]
    # Reusing base samples between FM and old-query objectives is allowed;
    # the strict disjointness requirement is pseudo-novel support vs query.
    base_idx = perm[:min(base_batch, len(perm))]
    old_idx = perm[-min(old_query_batch, len(perm)):]
    return dict(pseudo_base_ids=base_ids, pseudo_new_ids=novel_ids,
                support=torch.cat(support), novel_query=torch.cat(novel_query),
                base_fm=base_idx, old_query=old_idx)


def fit_meta(cache, *, epochs=20, episodes_per_epoch=30, pseudo_new=10,
             shot=5, query=5, base_batch=64, old_query_batch=64,
             inner_steps=1, inner_lr=1e-4,
             meta_lr=1e-4, weight_decay=1e-4, base_fm_weight=1.0,
             old_fm_weight=0.5, novel_fm_weight=1.0,
             blocks=24, hidden=None, seed=1, device='cpu'):
    """Train a meta-initialization from scratch, *without full-base pretraining*.

    In each episode:
      1. Reserve pseudo-novel class IDs; do base FM only on pseudo-base IDs.
      2. Take differentiable-in-parameters but first-order (no Hessian) inner
         SGD steps on pseudo-novel 5-shot support with only FM.
      3. Evaluate adapted Flow on disjoint pseudo-novel query and pseudo-base
         query using only FM; combine them with original-parameter base FM.
    No cross-entropy, classification scores, or margin penalties are trained.
    """
    if cache.get('stage') != 'base' or cache.get('split') != 'train':
        raise ValueError('Meta episodes require base train cache, never incremental/test')
    if (type(epochs) is not int or epochs < 1 or type(episodes_per_epoch) is not int or
            episodes_per_epoch < 1 or type(inner_steps) is not int or inner_steps < 1):
        raise ValueError('epochs, episodes_per_epoch and inner_steps must be positive integers')
    for value in (inner_lr, meta_lr):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Learning rates and temperature must be positive finite numbers')
    for value in (weight_decay, base_fm_weight, old_fm_weight, novel_fm_weight):
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Loss weights and decay must be nonnegative finite numbers')
    if novel_fm_weight == 0:
        raise ValueError('Novel query FM must be active for pseudo-incremental meta-learning')
    if type(blocks) is not int or blocks < 1:
        raise ValueError('blocks must be a positive integer')
    if not isinstance(hidden, (int, type(None))) or (hidden is not None and hidden < 1):
        raise ValueError('hidden must be positive or None')
    ids, text = text_cache(cache)
    images = unit(cache['image_features']).cpu()
    labels = cache['image_labels'].detach().cpu()
    if labels.ndim != 1 or labels.dtype != torch.long or len(labels) != len(images):
        raise ValueError('image_labels must be int64 [N]')
    if images.shape[1] != text.shape[1] or set(labels.tolist()) != set(ids):
        raise ValueError('Meta training requires images for exactly all base classes')
    # Validate support/query size BEFORE allocating GPU model/optimizer.
    counts = torch.stack([(labels == cid).sum() for cid in ids])
    if int(counts.min()) < shot + query:
        raise ValueError('Each base class must contain at least SHOT + QUERY training images')
    if not 1 <= pseudo_new < len(ids):
        raise ValueError('Pseudo-novel count must be smaller than base class count')
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    arch = dict(in_channels=text.shape[1], model_channels=hidden or text.shape[1],
                out_channels=text.shape[1], num_res_blocks=blocks)
    model = DeepFlowMatchingNet(**arch).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=meta_lr, weight_decay=weight_decay)
    text_device = text.to(device)
    row = {cid: i for i, cid in enumerate(ids)}
    y_all = torch.tensor([row[int(cid)] for cid in labels.tolist()], dtype=torch.long)
    records = []
    for epoch in range(epochs):
        totals = dict(loss=0., query_novel_fm=0., query_old_fm=0.,
                      base_fm=0., inner_support_fm=0.)
        for _ in range(episodes_per_epoch):
            ep = sample_episode(ids, labels, pseudo_new=pseudo_new, shot=shot,
                                query=query, base_batch=base_batch,
                                old_query_batch=old_query_batch, generator=gen)
            support = ep['support']
            novel_query = ep['novel_query']
            base = ep['base_fm']
            old_query = ep['old_query']
            sx = images[support].to(device)
            sy = y_all[support].to(device)
            nx = images[novel_query].to(device)
            ny = y_all[novel_query].to(device)
            bx = images[base].to(device)
            by = y_all[base].to(device)
            ox = images[old_query].to(device)
            oy = y_all[old_query].to(device)

            # Stateless virtual SGD adaptation: no mutation of the meta model.
            # Detached inner gradient = first-order MAML (not full second-order).
            params = dict(model.named_parameters())
            fast_params = params
            inner_loss = None
            for _step in range(inner_steps):
                t = torch.rand(len(sx), 1, device=device)
                xt = (1 - t) * sx + t * text_device[sy]
                predicted = functional_call(model, fast_params, (xt, t))
                fm_support = ((predicted - (text_device[sy] - sx)) ** 2).sum(-1).mean()
                inner_loss = fm_support
                gradients = torch.autograd.grad(inner_loss, tuple(fast_params.values()))
                fast_params = {name: param - inner_lr * grad.detach()
                               for (name, param), grad in zip(fast_params.items(), gradients)}
                # Creating a new graph at each step avoids second-order derivatives.

            def adapted_fm(features, targets):
                # Same random-time velocity supervision as training, not CE on t=0 logits.
                t = torch.rand(len(features), 1, device=device)
                target_text = text_device[targets]
                xt = (1 - t) * features + t * target_text
                predicted = functional_call(model, fast_params, (xt, t))
                return ((predicted - (target_text - features)) ** 2).sum(-1).mean()

            query_novel_fm = adapted_fm(nx, ny)
            query_old_fm = (adapted_fm(ox, oy) if old_fm_weight
                            else torch.zeros_like(query_novel_fm))

            # Outer base-only Flow loss: novel IDs are never in this term.
            t = torch.rand(len(bx), 1, device=device)
            xt = (1 - t) * bx + t * text_device[by]
            velocity = model(xt, t)
            base_fm = ((velocity - (text_device[by] - bx)) ** 2).sum(-1).mean()
            total = (novel_fm_weight * query_novel_fm +
                     old_fm_weight * query_old_fm + base_fm_weight * base_fm)
            if not torch.isfinite(total):
                raise RuntimeError('Nonfinite episodic meta loss')
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()
            totals['loss'] += total.detach().item()
            totals['query_novel_fm'] += query_novel_fm.detach().item()
            totals['query_old_fm'] += query_old_fm.detach().item()
            totals['base_fm'] += base_fm.detach().item()
            totals['inner_support_fm'] += inner_loss.detach().item()
        means = {k: value / episodes_per_epoch for k, value in totals.items()}
        records.append(means)
        print(f'meta_epoch={epoch+1} loss={means["loss"]:.6f} '
              f'query_novel_fm={means["query_novel_fm"]:.6f} '
              f'query_old_fm={means["query_old_fm"]:.6f} '
              f'base_fm={means["base_fm"]:.6f} '
              f'inner_support_fm={means["inner_support_fm"]:.6f}', flush=True)

    return dict(
        format_version=1, direction='visual_to_text',
        state_dict={key: val.detach().cpu().clone()
                    for key, val in model.state_dict().items()},
        architecture=arch, encoder_id=cache['encoder_id'], base_class_ids=ids,
        training=dict(mode='episodic_first_order_maml', seed=seed, epochs=epochs,
                      episodes_per_epoch=episodes_per_epoch,
                      pseudo_new=pseudo_new, shot=shot, query=query,
                      base_batch=base_batch, old_query_batch=old_query_batch,
                      inner_steps=inner_steps, inner_lr=inner_lr,
                      objective='fm_only', meta_lr=meta_lr,
                      weight_decay=weight_decay, base_fm_weight=base_fm_weight,
                      old_fm_weight=old_fm_weight,
                      novel_fm_weight=novel_fm_weight),
        meta_history=records)
