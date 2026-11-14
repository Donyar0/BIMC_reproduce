"""Base-only text-distribution -> visual-prototype flow matching."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .fm import DeepFlowMatchingNet
from .image_text_flow import scheduled_lr, transport
from .s1_flow import unit

METHOD = 'tangent-text-to-visual-v1'
DIRECTION = 'text_to_visual'


def description_std(stats):
    """Diagonal tangent-space standard deviation, in class_index row order."""
    centers = unit(stats['text_proto'].detach().float())
    features = unit(stats['description_features'].detach().float())
    labels = stats['description_targets'].detach().long()
    rows = []
    for row, class_id in enumerate(stats['class_index']):
        group = features[labels == int(class_id)]
        if len(group) < 2:
            raise ValueError(f'Class {class_id} needs at least two GPT descriptions')
        center = centers[row]
        residual = group - center
        tangent = residual - (residual @ center).unsqueeze(-1) * center
        rows.append(tangent.std(dim=0, unbiased=True))
    return torch.stack(rows).cpu()


def sample_text(centers, std, row_ids, noise_scale, max_angle):
    center = centers[row_ids]
    delta = torch.randn_like(center) * std[row_ids] * noise_scale
    delta = delta - (delta * center).sum(-1, keepdim=True) * center
    angle = delta.norm(dim=-1, keepdim=True).clamp(max=max_angle)
    direction = F.normalize(delta, dim=-1)
    return F.normalize(center * angle.cos() + direction * angle.sin(), dim=-1)


def fit(cache, *, epochs, batch_size, lr, weight_decay, blocks, hidden, seed, device,
        noise_scale=1.0, max_angle=0.5, mix_probability=0.2,
        warmup_ratio=0.05, min_lr_ratio=0.01):
    if cache.get('stage') != 'base' or cache.get('split') != 'train':
        raise ValueError('Base training cache required')
    if not (epochs > 0 and batch_size > 0 and blocks > 0 and lr > 0 and
            weight_decay >= 0 and noise_scale >= 0 and max_angle > 0 and
            0 <= mix_probability <= 1):
        raise ValueError('Invalid Flow parameters')
    ids = list(cache['class_ids'])
    if ids != list(range(len(ids))):
        raise ValueError('Base class IDs must be contiguous')
    centers = unit(cache['text_prototypes']).to(device)
    visual = unit(cache['visual_prototypes']).to(device)
    std = cache['description_std'].float().to(device)
    if centers.ndim != 2 or centers.shape != visual.shape or std.shape != centers.shape or not torch.isfinite(std).all():
        raise ValueError('Prototype/std dimensions mismatch')
    torch.manual_seed(seed)
    dim = centers.shape[1]
    arch = dict(in_channels=dim, model_channels=hidden or dim,
                out_channels=dim, num_res_blocks=blocks)
    network = DeepFlowMatchingNet(**arch).to(device).train()
    optimizer = torch.optim.AdamW(network.parameters(), lr=lr, weight_decay=weight_decay)
    loader = DataLoader(TensorDataset(torch.arange(len(ids))), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    total_steps = epochs * len(loader)
    history = []
    step = 0
    for epoch in range(epochs):
        losses = []
        for (rows,) in loader:
            rows = rows.to(device)
            x0 = sample_text(centers, std, rows, noise_scale, max_angle)
            x1 = visual[rows]
            if len(ids) > 1 and mix_probability:
                partner = torch.randint(len(ids)-1, rows.shape, device=device)
                partner += (partner >= rows).long()
                lam = torch.rand(len(rows), 1, device=device)
                choose = (torch.rand(len(rows), 1, device=device) < mix_probability)
                other = sample_text(centers, std, partner, noise_scale, max_angle)
                x0 = torch.where(choose, F.normalize((1-lam)*x0+lam*other, dim=-1), x0)
                x1 = torch.where(choose, F.normalize((1-lam)*x1+lam*visual[partner], dim=-1), x1)
            t = torch.rand(len(rows), 1, device=device)
            xt = (1-t)*x0+t*x1
            loss = (network(xt,t)-(x1-x0)).square().sum(-1).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite Flow loss')
            scheduled_lr(optimizer, step, total_steps, lr, warmup_ratio, min_lr_ratio)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step += 1
        history.append(sum(losses)/len(losses))
        print(f'epoch={epoch+1} tangent_fm_loss={history[-1]:.6f}', flush=True)
    return dict(format_version=1, architecture=arch,
                state_dict={k:v.detach().cpu().clone() for k,v in network.state_dict().items()},
                encoder_id=cache['encoder_id'], flow_direction=DIRECTION, method=METHOD,
                base_class_ids=ids, loss_history=history,
                training=dict(base_only=True, epochs=epochs, noise_scale=noise_scale,
                              max_angle=max_angle, mix_probability=mix_probability))


def load_model(checkpoint, device):
    if checkpoint.get('flow_direction') != DIRECTION or checkpoint.get('method') != METHOD:
        raise ValueError('Incompatible Flow checkpoint')
    model = DeepFlowMatchingNet(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.eval()


@torch.no_grad()
def build_bank(checkpoint, current, *, steps, num_base, beta_base, beta_inc,
               device, previous_bank=None, solver='euler'):
    if current['encoder_id'] != checkpoint['encoder_id'] or not 0 <= beta_base <= 1 or not 0 <= beta_inc <= 1:
        raise ValueError('Bank metadata or beta mismatch')
    ids = list(current['class_ids'])
    if previous_bank is not None and (previous_bank['encoder_id'] != checkpoint['encoder_id'] or
                                      set(ids) & set(previous_bank['class_ids'])):
        raise ValueError('Bank overlap or encoder mismatch')
    text = unit(current['text']).cpu()
    real = unit(current['real']).cpu()
    generated = transport(load_model(checkpoint, device), text.to(device), steps, solver=solver).cpu()
    beta = torch.full((len(ids), 1), beta_inc)
    beta[:num_base] = beta_base
    new = dict(class_ids=ids, text=text, real=real, generated=generated,
               prototypes=unit((1-beta)*real+beta*generated), encoder_id=checkpoint['encoder_id'])
    if previous_bank is None:
        return new
    return dict(class_ids=previous_bank['class_ids']+ids,
                **{key:torch.cat((previous_bank[key],new[key]))
                   for key in ('text','real','generated','prototypes')},
                encoder_id=checkpoint['encoder_id'])


@torch.no_grad()
def classify_bidirectional(model, image_features, bank, *, alpha, gamma, steps, solver='euler'):
    if not 0 <= alpha <= 1 or not 0 <= gamma <= 1:
        raise ValueError('Invalid scoring weights')
    z = unit(image_features)
    reverse = transport(model, z, steps, reverse=True, solver=solver)
    text_query = unit((1-alpha)*z + alpha*reverse)
    scores = gamma*(z @ bank['prototypes'].to(z.device).T) + (1-gamma)*(text_query @ bank['text'].to(z.device).T)
    ids = torch.tensor(bank['class_ids'], device=z.device)
    return ids[scores.argmax(-1)], scores
