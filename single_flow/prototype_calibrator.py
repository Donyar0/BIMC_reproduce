"""Optional exemplar-free correction of the frozen bidirectional Flow scores."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .flow_utils import transport, unit
from .text_distribution_flow import classify_bidirectional, load_model


class ResidualPrototypeCalibrator(nn.Module):
    """Learn small visual-prototype corrections and a Base/Novel logit bias."""

    def __init__(self, prototypes, num_base, whitening_matrix=None):
        super().__init__()
        prototypes = unit(prototypes.detach().float())
        dim = prototypes.shape[1]
        whitening = (torch.eye(dim, device=prototypes.device)
                     if whitening_matrix is None else whitening_matrix.detach().float())
        self.register_buffer('whitening', whitening)
        self.register_buffer('visual_basis', unit(prototypes @ whitening))
        self.register_buffer('novel_mask',
                             (torch.arange(len(prototypes), device=prototypes.device)
                              >= num_base).float())
        self.prototype_delta = nn.Parameter(torch.zeros_like(self.visual_basis))
        self.class_bias = nn.Parameter(torch.zeros(len(prototypes), device=prototypes.device))
        self.novel_bias = nn.Parameter(torch.zeros((), device=prototypes.device))

    def forward(self, features, flow_scores):
        query = unit(unit(features.float()) @ self.whitening)
        adjusted_basis = F.normalize(self.visual_basis + 0.1 * self.prototype_delta.tanh(),
                                     dim=-1)
        visual_correction = query @ (adjusted_basis - self.visual_basis).T
        bias = 0.1 * self.class_bias.tanh() + 0.1 * self.novel_bias.tanh() * self.novel_mask
        return flow_scores + visual_correction + bias

    def regularization(self):
        return (self.prototype_delta.square().mean() + self.class_bias.square().mean()
                + self.novel_bias.square())


@torch.no_grad()
def calibration_features(checkpoint, memory, bank, *, device, num_base,
                         pseudo_per_class, noise_scale, steps, solver):
    """Generate old-class features from Flow and use only current-session support images."""
    if pseudo_per_class < 1:
        raise ValueError('PSEUDO_PER_CLASS must be positive')
    old_count = sum(memory['session_lengths'][:-1])
    if old_count < num_base or old_count != len(bank['class_ids']) - memory['session_lengths'][-1]:
        raise ValueError('Calibration requires at least one historical session')
    rows = torch.arange(old_count, device=device).repeat_interleave(pseudo_per_class)
    text = memory['text'].to(device)[rows]
    sigma = memory['sigma'].to(device)[rows].unsqueeze(-1)
    start = unit(text + noise_scale * sigma * torch.randn_like(text))
    flow_model = load_model(checkpoint, device)
    old_features = torch.cat([transport(flow_model, part, steps, solver=solver)
                              for part in start.split(256)])

    support = memory['support_features'].to(device)
    support_labels = memory['support_labels'].to(device)
    current_rows = []
    current_features = []
    for row, class_id in enumerate(bank['class_ids'][old_count:], start=old_count):
        options = torch.nonzero(support_labels == class_id, as_tuple=False).flatten()
        if not len(options):
            raise ValueError(f'Missing current support images for class {class_id}')
        chosen = options[torch.randint(len(options), (pseudo_per_class,), device=device)]
        current_features.append(support[chosen])
        current_rows.append(torch.full((pseudo_per_class,), row, device=device, dtype=torch.long))
    features = torch.cat((old_features, *current_features))
    labels = torch.cat((rows, *current_rows))
    return flow_model, features, labels


def fit_prototype_calibrator(checkpoint, memory, bank, *, device, num_base,
                             pseudo_per_class, noise_scale, flow_steps, solver,
                             scoring_kwargs, steps, batch_size, lr, temperature,
                             reg_weight, whitening_matrix=None):
    if steps < 1 or batch_size < 1 or lr <= 0 or temperature <= 0 or reg_weight < 0:
        raise ValueError('Invalid prototype-calibration settings')
    flow_model, features, labels = calibration_features(
        checkpoint, memory, bank, device=device, num_base=num_base,
        pseudo_per_class=pseudo_per_class, noise_scale=noise_scale,
        steps=flow_steps, solver=solver)
    with torch.no_grad():
        raw_scores = torch.cat([
            classify_bidirectional(flow_model, part, bank, **scoring_kwargs)[1]
            for part in features.split(256)])
    calibrator = ResidualPrototypeCalibrator(
        bank['prototypes'].to(device), num_base,
        whitening_matrix.to(device) if whitening_matrix is not None else None).to(device)
    optimizer = torch.optim.AdamW(calibrator.parameters(), lr=lr, weight_decay=0)
    losses = []
    for _ in range(steps):
        selected = torch.randint(len(features), (batch_size,), device=device)
        scores = calibrator(features[selected], raw_scores[selected])
        loss = (F.cross_entropy(temperature * scores, labels[selected])
                + reg_weight * calibrator.regularization())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite prototype-calibration loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    calibrator.eval().requires_grad_(False)
    print(f'Prototype calibrator: support/replay={len(features)}, '
          f'loss={losses[-1]:.4f}, novel_bias={0.1 * calibrator.novel_bias.tanh().item():+.4f}',
          flush=True)
    return calibrator, losses
