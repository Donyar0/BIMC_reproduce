"""One velocity field with original and updated-pair text-to-visual FM."""
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .fm import DeepFlowMatchingNet
from .s1_flow import text_cache, unit


METHOD = 'bidirectional_single_flow'
OBJECTIVE = 'updated_pair_velocity_v2'
SCORE_NAMES = ('query_to_generated', 'reverse_to_text', 'reverse_to_generated')


def real_visual_means(cache):
    """Use observed support-image means, not BiMC's optional novel calibration."""
    ids, text = text_cache(cache)
    images = unit(cache['image_features']).cpu()
    labels = cache['image_labels'].detach().long().cpu()
    if labels.ndim != 1 or len(images) != len(labels) or set(labels.tolist()) != set(ids):
        raise ValueError('Every current class needs its own training/support images')
    visual = unit(torch.stack([images[labels == cid].mean(0) for cid in ids]))
    if visual.shape != text.shape:
        raise ValueError('Visual and text prototype dimensions differ')
    return ids, text, visual


def integrate(model, features, steps, reverse=False):
    """Differentiable Euler integration; reverse traverses t=1 to t=0."""
    if steps < 1:
        raise ValueError('Integration steps must be positive')
    x = features.float()
    dt = (-1.0 if reverse else 1.0) / steps
    for k in range(steps):
        time = 1-k/steps if reverse else k/steps
        t = torch.full((len(x), 1), time, dtype=x.dtype, device=x.device)
        x = x + dt*model(x, t)
    if not torch.isfinite(x).all():
        raise RuntimeError('Nonfinite Flow integration')
    return x


def updated_pair_fm(model, text, visual, steps):
    """FM from reverse-updated visual (text side) to forward-updated text (visual side)."""
    if text.shape != visual.shape or text.ndim != 2:
        raise ValueError('Updated-pair FM requires matching class prototypes')
    updated_text = F.normalize(integrate(model, text, steps), dim=-1)
    updated_visual = F.normalize(integrate(model, visual, steps, reverse=True), dim=-1)
    t = torch.rand(len(text), 1, device=text.device)
    xt = (1-t)*updated_visual + t*updated_text
    target_velocity = updated_text - updated_visual
    return (model(xt, t)-target_velocity).square().sum(-1).mean()


def _train(cache, model, *, epochs, batch_size, lr, weight_decay,
           bidir_fm_weight, bidir_steps, seed, device):
    if (epochs < 1 or batch_size < 1 or lr <= 0 or weight_decay < 0 or
            bidir_fm_weight < 0 or bidir_steps < 1):
        raise ValueError('Invalid bidirectional Flow training settings')
    ids, text, visual = real_visual_means(cache)
    loader = DataLoader(TensorDataset(text, visual), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr*0.01)
    history = []
    for epoch in range(epochs):
        total_sum = fm_sum = bidir_sum = 0.0
        count = 0
        for source, target in loader:
            source, target = source.to(device), target.to(device)
            t = torch.rand(len(source), 1, device=device)
            xt = (1-t)*source + t*target
            fm = (model(xt, t) - (target-source)).square().sum(-1).mean()
            bidir_fm = updated_pair_fm(model, source, target, bidir_steps)
            loss = fm + bidir_fm_weight*bidir_fm
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite bidirectional Flow loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            n = len(source)
            total_sum += loss.item()*n
            fm_sum += fm.item()*n
            bidir_sum += bidir_fm.item()*n
            count += n
        used_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        history.append(dict(total=total_sum/count, fm=fm_sum/count,
                            bidir_fm=bidir_sum/count, lr=used_lr))
        print(f'epoch={epoch+1} fm={history[-1]["fm"]:.6f} '
              f'bidir_fm={history[-1]["bidir_fm"]:.6f} '
              f'total={history[-1]["total"]:.6f}', flush=True)
    return history


def fit_base(cache, *, epochs, batch_size, lr, weight_decay, blocks, hidden,
             bidir_fm_weight, bidir_steps, seed, device):
    if cache['stage'] != 'base' or cache['split'] != 'train':
        raise ValueError('Base Flow requires base training statistics')
    if blocks < 1:
        raise ValueError('Flow blocks must be positive')
    ids, text, _ = real_visual_means(cache)
    if ids != list(range(len(ids))):
        raise ValueError('Base class IDs must be zero-based and ordered')
    torch.manual_seed(seed)
    architecture = dict(in_channels=text.shape[1],
                        model_channels=hidden or text.shape[1],
                        out_channels=text.shape[1], num_res_blocks=blocks)
    model = DeepFlowMatchingNet(**architecture).to(device)
    history = _train(cache, model, epochs=epochs, batch_size=batch_size,
                     lr=lr, weight_decay=weight_decay,
                     bidir_fm_weight=bidir_fm_weight,
                     bidir_steps=bidir_steps, seed=seed, device=device)
    return dict(format_version=2, method=METHOD, objective=OBJECTIVE,
                architecture=architecture,
                encoder_id=cache['encoder_id'], base_class_ids=ids,
                state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                training=dict(bidir_fm_weight=bidir_fm_weight,
                              bidir_steps=bidir_steps),
                loss_history=history, adaptation_history=[])


def load_model(checkpoint, device, train=False):
    if checkpoint.get('method') != METHOD or checkpoint.get('objective') != OBJECTIVE:
        raise ValueError('Checkpoint is not an updated-pair velocity Flow; retrain it')
    model = DeepFlowMatchingNet(**checkpoint['architecture']).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    return model.train() if train else model.eval().requires_grad_(False)


def adapt(checkpoint, cache, *, epochs, batch_size, lr, weight_decay, seed, device):
    if cache['stage'] != 'incremental' or cache['split'] != 'train':
        raise ValueError('Flow adaptation requires incremental support statistics')
    if cache['encoder_id'] != checkpoint['encoder_id']:
        raise ValueError('Encoder mismatch')
    settings = checkpoint['training']
    model = load_model(checkpoint, device, train=True)
    history = _train(cache, model, epochs=epochs, batch_size=batch_size,
                     lr=lr, weight_decay=weight_decay,
                     bidir_fm_weight=settings['bidir_fm_weight'],
                     bidir_steps=settings['bidir_steps'], seed=seed, device=device)
    updated = dict(checkpoint)
    updated['state_dict'] = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    updated['adaptation_history'] = checkpoint['adaptation_history'] + [dict(
        class_ids=list(cache['class_ids']), loss_history=history)]
    return updated


@torch.no_grad()
def build_bank(checkpoint, cache, *, steps, device, existing=None,
               regenerate_existing=False):
    if cache['encoder_id'] != checkpoint['encoder_id']:
        raise ValueError('Encoder mismatch')
    ids, text = text_cache(cache)
    model = load_model(checkpoint, device)
    if existing is not None:
        old_ids = list(existing['class_ids'])
        if (existing['encoder_id'] != checkpoint['encoder_id'] or
                set(old_ids) & set(ids) or old_ids != list(range(len(old_ids))) or
                ids != list(range(len(old_ids), len(old_ids)+len(ids)))):
            raise ValueError('Current and previous class banks do not align')
        text = torch.cat((existing['text'], text))
        ids = old_ids + ids
    if existing is not None and not regenerate_existing:
        new_text = text[len(old_ids):].to(device)
        generated = torch.cat((existing['generated'],
            F.normalize(integrate(model, new_text, steps), dim=-1).cpu()))
    else:
        generated = F.normalize(integrate(model, text.to(device), steps), dim=-1).cpu()
    return dict(class_ids=ids, text=text.cpu(), generated=generated,
                encoder_id=checkpoint['encoder_id'])


@torch.no_grad()
def score_three(model, image_features, bank, *, steps, weights=(1, 1, 1)):
    if len(weights) != 3 or any(w < 0 for w in weights) or sum(weights) <= 0:
        raise ValueError('SCORE_WEIGHTS must be three nonnegative values with positive sum')
    query = unit(image_features)
    text = unit(bank['text']).to(query.device)
    generated = unit(bank['generated']).to(query.device)
    if text.shape != generated.shape or query.shape[1] != text.shape[1]:
        raise ValueError('Query and prototype bank dimensions differ')
    reverse_query = F.normalize(integrate(model, query, steps, reverse=True), dim=-1)
    scores = dict(query_to_generated=query @ generated.T,
                  reverse_to_text=reverse_query @ text.T)
    if weights[2] > 0:
        scores['reverse_to_generated'] = reverse_query @ generated.T
    scale = float(sum(weights))
    scores['fused'] = sum(float(w)*scores[name] for w,name in zip(weights,SCORE_NAMES)
                          if w > 0)/scale
    return scores
