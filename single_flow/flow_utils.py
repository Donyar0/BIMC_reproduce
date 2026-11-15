"""Shared feature normalization, learning-rate schedule, and ODE integration."""
import math
import torch
import torch.nn.functional as F

def unit(x):
    x = x.detach().float()
    if x.ndim != 2 or not torch.isfinite(x).all() or (x.norm(dim=-1) < 1e-8).any():
        raise ValueError('Features must be finite, nonzero [N,D] tensors')
    return F.normalize(x, dim=-1)


def scheduled_lr(optimizer, step, total_steps, peak_lr, warmup_ratio, min_lr_ratio):
    """Linear warmup followed by cosine decay, for both training stages."""
    if not (0 <= step < total_steps and peak_lr > 0 and
            0 <= warmup_ratio < 1 and 0 <= min_lr_ratio < 1):
        raise ValueError('Invalid learning-rate schedule')
    warmup = min(total_steps-1, max(1, round(total_steps*warmup_ratio)))
    if step < warmup:
        lr = peak_lr*(step+1)/warmup
    else:
        progress = (step-warmup+1)/(total_steps-warmup)
        lr = peak_lr*(min_lr_ratio + (1-min_lr_ratio)*(1+math.cos(math.pi*progress))/2)
    for group in optimizer.param_groups:
        group['lr'] = lr
    return lr


@torch.no_grad()
def transport(model, features, steps, reverse=False, solver='euler'):
    if steps < 1:
        raise ValueError('Integration steps must be positive')
    if solver not in ('euler', 'midpoint', 'heun', 'rk4'):
        raise ValueError('SOLVER must be euler, midpoint, heun or rk4')
    x = features.float().clone()
    step = (-1 if reverse else 1) / steps
    for k in range(steps):
        time = 1-k/steps if reverse else k/steps
        def field(state, at):
            t = torch.full((len(state), 1), at, device=state.device)
            return model(state, t)

        k1 = field(x, time)
        if solver == 'euler':
            update = k1
        elif solver == 'midpoint':
            update = field(x + step*k1/2, time + step/2)
        elif solver == 'heun':
            k2 = field(x + step*k1, time + step)
            update = (k1 + k2)/2
        else:
            k2 = field(x + step*k1/2, time + step/2)
            k3 = field(x + step*k2/2, time + step/2)
            k4 = field(x + step*k3, time + step)
            update = (k1 + 2*k2 + 2*k3 + k4)/6
        x = x + step*update
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite Flow integration')
    return F.normalize(x, dim=-1)
