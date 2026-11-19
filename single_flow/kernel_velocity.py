"""Nonparametric bidirectional transport from fixed text/visual class anchors."""

import torch
from torch import nn
import torch.nn.functional as F

from .flow_utils import unit


class KernelVelocityField(nn.Module):
    """Nadaraya-Watson velocity on linear paths between class prototypes."""

    def __init__(self, bandwidth=0.4):
        super().__init__()
        if bandwidth <= 0:
            raise ValueError('Kernel bandwidth must be positive')
        self.bandwidth = float(bandwidth)
        self.register_buffer('text', torch.empty(0, 0))
        self.register_buffer('visual', torch.empty(0, 0))

    @torch.no_grad()
    def append(self, text, visual):
        text, visual = unit(text), unit(visual)
        if text.shape != visual.shape or not len(text):
            raise ValueError('Kernel anchors must be nonempty matched [classes, dim] tensors')
        if self.text.numel() and text.shape[1] != self.text.shape[1]:
            raise ValueError('Kernel anchor dimension changed')
        self.text = torch.cat((self.text.to(text.device), text), dim=0) if self.text.numel() else text
        self.visual = (torch.cat((self.visual.to(visual.device), visual), dim=0)
                       if self.visual.numel() else visual)

    def forward(self, x, t, exclude=None):
        if self.text.numel() == 0:
            raise ValueError('Append anchors before querying the kernel field')
        if x.ndim != 2 or x.shape[1] != self.text.shape[1]:
            raise ValueError('Expected [batch, anchor dimension] queries')
        if t.ndim == 1:
            t = t[:, None]
        if t.shape != (len(x), 1):
            raise ValueError('Expected one time per query')
        center = self.text[None] + t[:, :, None] * (self.visual - self.text)[None]
        squared = (x[:, None] - center).square().sum(-1)
        logits = -squared / (2 * self.bandwidth ** 2)
        if exclude is not None:
            exclude = torch.as_tensor(exclude, device=x.device, dtype=torch.long)
            if exclude.shape != (len(x),) or len(self.text) < 2:
                raise ValueError('Leave-one-out transport needs one excluded class per query')
            logits[torch.arange(len(x), device=x.device), exclude] = -torch.inf
        weights = F.softmax(logits, dim=-1)
        return weights @ (self.visual - self.text)


@torch.no_grad()
def kernel_transport(field, features, steps=8, reverse=False, exclude=None):
    """Euler integration with a fixed leave-one-out mask for prototype generation."""
    if steps < 1:
        raise ValueError('Integration steps must be positive')
    x = unit(features)
    dt = (-1.0 if reverse else 1.0) / steps
    for step in range(steps):
        t = (1.0 - step / steps) if reverse else step / steps
        time = torch.full((len(x), 1), t, device=x.device)
        x = x + dt * field(x, time, exclude=exclude)
    return unit(x)
