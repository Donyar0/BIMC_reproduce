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


