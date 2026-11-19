"""Cumulative class-balanced Gram matrix and one target column per class."""

import math

import torch
import torch.nn.functional as F


class BalancedRandomFeatureRidge:
    def __init__(self, input_dim, feature_dim, regularization, seed, device,
                 target_mode='prototype'):
        if input_dim < 1 or feature_dim < 1 or regularization <= 0:
            raise ValueError('Invalid random-feature ridge dimensions or regularization')
        if target_mode not in ('prototype', 'class_mean'):
            raise ValueError('Unknown ridge target mode')
        self.device = torch.device(device)
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.projection = (torch.randn(input_dim, feature_dim, generator=generator)
                           / math.sqrt(input_dim)).to(self.device)
        self.gram = torch.zeros(feature_dim, feature_dim, device=self.device)
        self.cross = torch.empty(feature_dim, 0, device=self.device)
        self.class_ids = []
        self.regularization = float(regularization)
        self.target_mode = target_mode
        self.weights = None
        self.observation_count = 0

    @torch.no_grad()
    def project(self, features):
        if features.ndim != 2 or features.shape[1] != self.projection.shape[0]:
            raise ValueError('Expected [batch, CLIP feature dimension] features')
        features = F.normalize(features.to(self.device, torch.float32), dim=-1)
        return F.relu(features @ self.projection)

    @torch.no_grad()
    def add_session(self, class_ids, real_features, real_labels, prototypes, batch_size=256):
        """Each class contributes unit total mass, regardless of its shot count."""
        new_ids = [int(c) for c in class_ids]
        old_count = len(self.class_ids)
        if not new_ids or new_ids != list(range(old_count, old_count + len(new_ids))):
            raise ValueError('New class IDs must follow the existing contiguous order')
        if (real_features.ndim != 2 or real_labels.ndim != 1
                or len(real_features) != len(real_labels)
                or prototypes.shape != (len(new_ids), real_features.shape[1])
                or batch_size < 1):
            raise ValueError('Invalid session features, labels, or prototypes')
        labels = real_labels.to(self.device, torch.long)
        if ((labels < old_count) | (labels >= old_count + len(new_ids))).any():
            raise ValueError('Session features must belong to new classes only')
        counts = torch.bincount(labels - old_count, minlength=len(new_ids))
        if (counts == 0).any():
            raise ValueError('Every new class needs real support features')
        mean_phi = torch.zeros(len(new_ids), self.projection.shape[1], device=self.device)
        for start in range(0, len(labels), batch_size):
            end = min(start + batch_size, len(labels))
            phi = self.project(real_features[start:end])
            local = labels[start:end] - old_count
            sample_weight = counts[local].float().reciprocal()
            self.gram.addmm_(phi.T, phi * sample_weight[:, None])
            mean_phi.index_add_(0, local, phi)
        mean_phi /= counts.float()[:, None]
        new_columns = (self.project(prototypes).T if self.target_mode == 'prototype'
                       else mean_phi.T)
        self.cross = torch.cat((self.cross, new_columns), dim=1)
        self.class_ids.extend(new_ids)
        self.observation_count += len(labels)
        self.weights = None
        return counts.cpu().tolist()

    @torch.no_grad()
    def solve(self):
        if not self.class_ids:
            raise ValueError('Add session statistics before solving')
        system = self.gram.clone()
        system.diagonal().add_(self.regularization)
        self.weights = torch.cholesky_solve(self.cross, torch.linalg.cholesky(system))
        return self.weights

    @torch.no_grad()
    def scores(self, features):
        if self.weights is None:
            raise ValueError('Solve the ridge system before scoring')
        return self.project(features) @ self.weights


def shrink_visual_prototype(real, generated, counts, kappa):
    """Shrink 5-shot estimates more strongly than well-sampled Base centers."""
    if real.shape != generated.shape or len(counts) != len(real) or kappa < 0:
        raise ValueError('Invalid prototype shrinkage inputs')
    weight = (counts.to(real.device).float() / (counts.to(real.device).float() + kappa))[:, None]
    return F.normalize(weight * real + (1 - weight) * generated, dim=-1)
