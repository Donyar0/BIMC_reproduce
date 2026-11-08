"""Shared text-to-visual flow and reverse query calibration."""
import math
import torch
import torch.nn.functional as F
from .fm import DeepFlowMatchingNet
from .s1_flow import unit, text_cycle_loss


def real_means(cache):
    features = unit(cache['image_features']).cpu()
    labels = cache['image_labels'].cpu()
    return unit(torch.stack([features[labels == c].mean(0) for c in cache['class_ids']]))


def load_model(checkpoint, device):
    model = DeepFlowMatchingNet(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def adapt(checkpoint, cache, memory, *, steps, batch_size, lr, weight_decay, device,
          cycle_weight=0.0):
    """Fit current support examples and replay one real mean per old class."""
    if steps < 0 or batch_size < 1 or lr <= 0:
        raise ValueError('Invalid adaptation parameters')
    if not math.isfinite(cycle_weight) or cycle_weight < 0:
        raise ValueError('cycle_weight must be finite and non-negative')
    if cache['encoder_id'] != checkpoint['encoder_id']:
        raise ValueError('Encoder mismatch')
    ids = cache['class_ids']
    text = unit(cache['text_prototypes']).cpu()
    labels = cache['image_labels'].cpu().tolist()
    index = {c: i for i, c in enumerate(ids)}
    sources = torch.cat([memory['text'], text[torch.tensor([index[c] for c in labels])]])
    targets = torch.cat([memory['real'], unit(cache['image_features']).cpu()])
    model = load_model(checkpoint, device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    fm_total, cycle_total = 0.0, 0.0
    for _ in range(steps):
        idx = torch.randint(len(sources), (batch_size,))
        x0, x1 = sources[idx].to(device), targets[idx].to(device)
        t = torch.rand(len(idx), 1, device=device)
        fm_loss = (model((1-t)*x0+t*x1, t) - (x1-x0)).square().sum(-1).mean()
        cycle_loss = text_cycle_loss(model, x0) if cycle_weight else None
        loss = fm_loss + cycle_weight * cycle_loss if cycle_loss is not None else fm_loss
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite flow adaptation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        fm_total += fm_loss.item()
        if cycle_loss is not None:
            cycle_total += cycle_loss.item()
    if steps:
        print(f'adapt steps={steps} fm_loss={fm_total/steps:.6f} '
              f'cycle_loss={cycle_total/steps:.6f}', flush=True)
    result = dict(checkpoint)
    result['state_dict'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return result


@torch.no_grad()
def build_bank(checkpoint, memory, *, steps, num_base, beta_base, beta_inc, device,
               previous_bank=None):
    """Generate prototypes for current classes and append a frozen prior bank."""
    if steps < 1 or not all(0 <= x <= 1 for x in (beta_base, beta_inc)):
        raise ValueError('Invalid generation parameters')
    if not 0 <= num_base <= len(memory['class_ids']):
        raise ValueError('Invalid number of base classes')
    if memory['encoder_id'] != checkpoint['encoder_id']:
        raise ValueError('Encoder mismatch')
    if previous_bank is not None:
        if previous_bank['encoder_id'] != checkpoint['encoder_id']:
            raise ValueError('Encoder mismatch')
        if set(previous_bank['class_ids']) & set(memory['class_ids']):
            raise ValueError('Current classes must not overlap previous bank')
    model = load_model(checkpoint, device).eval()
    x = memory['text'].to(device)
    for k in range(steps):
        x = x + model(x, torch.full((len(x), 1), k / steps, device=device)) / steps
    generated = F.normalize(x, dim=-1)
    beta = torch.full((len(x), 1), beta_inc, device=device)
    beta[:num_base] = beta_base
    visual = F.normalize((1-beta)*memory['real'].to(device) + beta*generated, dim=-1)
    current = dict(class_ids=list(memory['class_ids']), text=memory['text'].cpu().clone(),
                   real=memory['real'].cpu().clone(), generated=generated.cpu(),
                   prototypes=visual.cpu(), encoder_id=checkpoint['encoder_id'])
    if previous_bank is None:
        return current
    return dict(class_ids=list(previous_bank['class_ids']) + current['class_ids'],
                text=torch.cat([previous_bank['text'], current['text']]),
                real=torch.cat([previous_bank['real'], current['real']]),
                generated=torch.cat([previous_bank['generated'], current['generated']]),
                prototypes=torch.cat([previous_bank['prototypes'], current['prototypes']]),
                encoder_id=checkpoint['encoder_id'])


@torch.no_grad()
def calibrate_text_bank(model, bank, *, strength):
    """Apply visual-guided residual calibration to every seen text prototype.

    The source is each class's STORED real support-set visual mean. It is
    important not to use generated visual prototypes as visual evidence:
    those originate in the very text prototype we are calibrating.
    A single reverse Euler step is intentionally consistent with the
    existing reverse-query update. strength=0 preserves old behavior.
    """
    if not 0 <= strength <= 1:
        raise ValueError('Text calibration strength must be in [0, 1]')
    text = bank['text']
    real = bank['real']
    if text.ndim != 2 or real.shape != text.shape:
        raise ValueError('Text and real support prototype banks must align')
    text = unit(text)
    if strength == 0:
        return text
    visual = unit(real).to(next(model.parameters()).device)
    text = text.to(visual.device)
    time = torch.ones((len(visual), 1), device=visual.device)
    projected = F.normalize(visual - model(visual, time), dim=-1)
    return F.normalize((1 - strength) * text + strength * projected, dim=-1)


@torch.no_grad()
def classify_bidirectional(model, image_features, bank, *, alpha, gamma,
                           text_calibration=0.0, calibrated_text=None):
    if not 0 <= alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError('alpha and gamma must be in [0, 1]')
    z = unit(image_features)
    t = torch.ones((len(z), 1), device=z.device)
    calibrated = F.normalize(z - alpha * model(z, t), dim=-1)
    visual = z @ bank['prototypes'].to(z.device).T
    if calibrated_text is None:
        calibrated_text = calibrate_text_bank(
            model, bank, strength=text_calibration)
    semantic = calibrated @ calibrated_text.to(z.device).T
    scores = gamma * visual + (1-gamma) * semantic
    ids = torch.tensor(bank['class_ids'], device=z.device)
    return ids[scores.argmax(-1)], scores


@torch.no_grad()
def prepare_ensemble(bank, covariance, description_features, description_labels, device):
    """Cache session-constant covariance terms and description class indices."""
    prototypes = unit(bank['prototypes']).to(device)
    count, dim = prototypes.shape
    cov = covariance.to(device=device, dtype=torch.float32)
    if cov.shape != (dim, dim):
        raise ValueError('Covariance shape must match the prototype dimension')
    precision = torch.linalg.pinv(cov)
    projected = prototypes @ precision
    # In the class-wise softmax, the query-only Mahalanobis term cancels.
    weighted_prototypes = projected.T.contiguous()
    prototype_quadratic = (prototypes * projected).sum(dim=-1)

    descriptions = description_features.to(device=device, dtype=torch.float32)
    labels = description_labels.detach().cpu().long()
    if (descriptions.ndim != 2 or descriptions.shape[1] != dim or
            labels.ndim != 1 or len(labels) != len(descriptions) or
            set(labels.tolist()) != set(bank['class_ids'])):
        raise ValueError('Description features/labels must cover exactly the seen classes')
    description_indices = tuple(
        torch.where(labels == class_id)[0].to(device) for class_id in bank['class_ids']
    )
    return dict(weighted_prototypes=weighted_prototypes,
                prototype_quadratic=prototype_quadratic,
                descriptions_t=descriptions.T.contiguous(),
                description_indices=description_indices,
                num_classes=count)


def ensemble_scores(scores, image_features, prepared, num_base, ensemble_alpha, logit_scale=1.0):
    """BiMC-style probability fusion with cached, exact Mahalanobis softmax."""
    if not torch.isfinite(torch.tensor(logit_scale)) or logit_scale <= 0:
        raise ValueError('logit_scale must be finite and positive')
    if not 0 <= ensemble_alpha <= 1:
        raise ValueError('ensemble_alpha must be in [0, 1]')
    z = unit(image_features)
    n, classes = z.shape[0], prepared['num_classes']
    if scores.shape != (n, classes) or not 0 <= num_base <= classes:
        raise ValueError('Score dimensions or number of base classes are invalid')

    cov_logits = (
        2 * (z @ prepared['weighted_prototypes'])
        - prepared['prototype_quadratic'][None, :]
    ) / 512.0
    prob_cov = torch.softmax(cov_logits, dim=-1)

    similarities = z @ prepared['descriptions_t']
    knn = torch.stack([
        similarities[:, indices].max(dim=1).values
        for indices in prepared['description_indices']
    ], dim=1)
    prob_knn = torch.softmax(knn, dim=-1)
    prob_flow = torch.softmax(scores * logit_scale, dim=-1)
    base = ensemble_alpha * prob_flow[:, :num_base] + (1-ensemble_alpha) * prob_cov[:, :num_base]
    inc = ensemble_alpha * prob_flow[:, num_base:] + (1-ensemble_alpha) * prob_knn[:, num_base:]
    return torch.cat([base, inc], dim=1)
