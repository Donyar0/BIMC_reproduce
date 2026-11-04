"""Shared text-to-visual flow and reverse query calibration."""
import torch
import torch.nn.functional as F
from .fm import DeepFlowMatchingNet
from .s1_flow import unit


def real_means(cache):
    features = unit(cache['image_features']).cpu()
    labels = cache['image_labels'].cpu()
    return unit(torch.stack([features[labels == c].mean(0) for c in cache['class_ids']]))


def load_model(checkpoint, device):
    model = DeepFlowMatchingNet(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model


def adapt(checkpoint, cache, memory, *, steps, batch_size, lr, weight_decay, device):
    """Fit current support examples and replay one real mean per old class."""
    if steps < 0 or batch_size < 1 or lr <= 0:
        raise ValueError('Invalid adaptation parameters')
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
    for _ in range(steps):
        idx = torch.randint(len(sources), (batch_size,))
        x0, x1 = sources[idx].to(device), targets[idx].to(device)
        t = torch.rand(len(idx), 1, device=device)
        loss = (model((1-t)*x0+t*x1, t) - (x1-x0)).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite flow adaptation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    result = dict(checkpoint)
    result['state_dict'] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return result


@torch.no_grad()
def build_bank(checkpoint, memory, *, steps, num_base, beta_base, beta_inc, device):
    if steps < 1 or not all(0 <= x <= 1 for x in (beta_base, beta_inc)):
        raise ValueError('Invalid generation parameters')
    model = load_model(checkpoint, device).eval()
    x = memory['text'].to(device)
    for k in range(steps):
        x = x + model(x, torch.full((len(x), 1), k / steps, device=device)) / steps
    generated = F.normalize(x, dim=-1)
    beta = torch.full((len(x), 1), beta_inc, device=device)
    beta[:num_base] = beta_base
    visual = F.normalize((1-beta)*memory['real'].to(device) + beta*generated, dim=-1)
    return dict(class_ids=memory['class_ids'], text=memory['text'].clone(),
                real=memory['real'].clone(), generated=generated.cpu(), prototypes=visual.cpu(),
                encoder_id=checkpoint['encoder_id'])


@torch.no_grad()
def classify_bidirectional(model, image_features, bank, *, alpha, gamma):
    if not 0 <= alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError('alpha and gamma must be in [0, 1]')
    z = unit(image_features)
    t = torch.ones((len(z), 1), device=z.device)
    calibrated = F.normalize(z - alpha * model(z, t), dim=-1)
    visual = z @ bank['prototypes'].to(z.device).T
    semantic = calibrated @ bank['text'].to(z.device).T
    scores = gamma * visual + (1-gamma) * semantic
    ids = torch.tensor(bank['class_ids'], device=z.device)
    return ids[scores.argmax(-1)], scores
