"""Frozen random ReLU features and an exemplar-free cumulative ridge head."""
import math

import torch
import torch.nn.functional as F


class IncrementalRandomFeatureRidge:
    """Keep only Phi.T @ Phi and Phi.T @ one_hot(y) across sessions."""

    def __init__(self, input_dim, feature_dim, regularization, seed, device):
        if input_dim < 1 or feature_dim < 1 or regularization <= 0:
            raise ValueError('Invalid random-feature ridge dimensions or regularization')
        self.device = torch.device(device)
        generator = torch.Generator(device='cpu').manual_seed(int(seed))
        self.projection = (torch.randn(input_dim, feature_dim, generator=generator)
                           / math.sqrt(input_dim)).to(self.device)
        self.gram = torch.zeros(feature_dim, feature_dim, device=self.device)
        self.cross = torch.empty(feature_dim, 0, device=self.device)
        self.regularization = float(regularization)
        self.class_ids = []
        self.weights = None
        self.observation_count = 0

    @torch.no_grad()
    def project(self, features):
        if features.ndim != 2 or features.shape[1] != self.projection.shape[0]:
            raise ValueError('Expected [batch, CLIP feature dimension] features')
        z = F.normalize(features.to(device=self.device, dtype=torch.float32), dim=-1)
        return F.relu(z @ self.projection)

    @torch.no_grad()
    def add_session(self, class_ids, feature_groups, batch_size=256):
        """Append new classes and accumulate their real and synthetic features."""
        new_ids = [int(class_id) for class_id in class_ids]
        old_count = len(self.class_ids)
        if not new_ids or new_ids != list(range(old_count, old_count + len(new_ids))):
            raise ValueError('New class IDs must be consecutive and follow old classes')
        if batch_size < 1:
            raise ValueError('RIDGE.BATCH_SIZE must be positive')
        self.cross = torch.cat((self.cross,
                                self.cross.new_zeros(self.cross.shape[0], len(new_ids))), dim=1)
        counts = torch.zeros(len(new_ids), dtype=torch.long, device=self.device)
        for features, labels in feature_groups:
            if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels):
                raise ValueError('Ridge features and labels must have matching batch dimensions')
            for start in range(0, len(labels), batch_size):
                end = start + batch_size
                selected = labels[start:end].to(device=self.device, dtype=torch.long)
                if ((selected < old_count) | (selected >= old_count + len(new_ids))).any():
                    raise ValueError('Session update contains an old or unknown class label')
                phi = self.project(features[start:end])
                self.gram.addmm_(phi.T, phi)
                self.cross.index_add_(1, selected, phi.T)
                counts += torch.bincount(selected - old_count, minlength=len(new_ids))
                self.observation_count += len(selected)
        if (counts == 0).any():
            raise ValueError('Every new class needs at least one real or synthetic feature')
        self.class_ids.extend(new_ids)
        self.weights = None
        return counts.cpu().tolist()

    @torch.no_grad()
    def solve(self):
        """Solve (G + lambda I) W = C without forming a matrix inverse."""
        if not self.class_ids:
            raise ValueError('Add class features before solving ridge regression')
        system = self.gram.clone()
        system.diagonal().add_(self.regularization)
        factor = torch.linalg.cholesky(system)
        self.weights = torch.cholesky_solve(self.cross, factor)
        return self.weights

    @torch.no_grad()
    def scores(self, features):
        if self.weights is None:
            raise ValueError('Solve ridge regression before scoring')
        return self.project(features) @ self.weights


def _standardize_per_query(scores):
    centered = scores - scores.mean(dim=-1, keepdim=True)
    scale = centered.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)
    return centered / scale


def fuse_flow_ridge(flow_scores, ridge_scores, ridge_weight):
    """Combine raw-score branches after matching their per-query scales."""
    if flow_scores.shape != ridge_scores.shape or not 0 <= ridge_weight <= 1:
        raise ValueError('Invalid Flow/Ridge score fusion')
    return ((1.0 - ridge_weight) * _standardize_per_query(flow_scores)
            + ridge_weight * _standardize_per_query(ridge_scores))
