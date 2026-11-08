"""Swapped VSC dual classifier adapted to frozen-CLIP FSCIL.

Branch A: support visual prototype -> textual space -> calibrated text prototype;
          query visual feature -> textual space -> semantic classification.
Branch B: class text prototype -> visual space -> pseudo-visual prototype;
          original query visual feature -> visual classification.

This is a *direction-swapped adaptation* of Wang et al. (JSTARS 2025),
not a verbatim implementation of their SAR episodic meta-learning model.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualProjection(nn.Module):
    """A dimension-preserving trainable modality projection, initialized to identity."""

    def __init__(self, dim, hidden_dim, strength=0.2):
        super().__init__()
        self.strength = float(strength)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        x = F.normalize(x.float(), dim=-1)
        return F.normalize(x + self.strength * self.net(x), dim=-1)


class SwappedVSCHead(nn.Module):
    """Two directional mappings with a residual visual-guided text calibrator.

    All inputs are normalized CLIP-space vectors. Class axes of text_proto
    and visual_proto must be aligned in ascending global class-label order.
    """

    def __init__(self, dim, hidden_dim=512, projection_strength=0.2,
                 text_strength=0.1, attention_tau=8.0,
                 logit_scale=20.0, fusion_weight=0.5):
        super().__init__()
        if not 0.0 <= fusion_weight <= 1.0:
            raise ValueError("fusion_weight must be in [0, 1]")
        self.text_strength = float(text_strength)
        self.attention_tau = float(attention_tau)
        self.logit_scale = float(logit_scale)
        self.fusion_weight = float(fusion_weight)
        self.visual_to_text = ResidualProjection(dim, hidden_dim, projection_strength)
        self.text_to_visual = ResidualProjection(dim, hidden_dim, projection_strength)
        self.text_calibrator = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, dim),
        )
        nn.init.zeros_(self.text_calibrator[-1].weight)
        nn.init.zeros_(self.text_calibrator[-1].bias)

    def prototype_bank(self, text_proto, visual_proto):
        """Compute both classifier banks using only seen-class support statistics."""
        if text_proto.shape != visual_proto.shape or text_proto.ndim != 2:
            raise ValueError("Both prototype banks must have shape [num_classes, dim]")
        text_proto = F.normalize(text_proto.float(), dim=-1)
        visual_proto = F.normalize(visual_proto.float(), dim=-1)

        # Visual->text: attention over projected support visual prototypes.
        projected_visual = self.visual_to_text(visual_proto)
        attention = torch.softmax(
            self.attention_tau * (text_proto @ projected_visual.T), dim=-1
        )
        visual_context = attention @ projected_visual
        delta_text = self.text_calibrator(
            torch.cat([text_proto, visual_context], dim=-1)
        )
        calibrated_text = F.normalize(
            text_proto + self.text_strength * delta_text, dim=-1
        )

        # Text->visual: build one pseudo-visual classifier per seen class.
        pseudo_visual = self.text_to_visual(text_proto)
        return calibrated_text, pseudo_visual

    def forward(self, query_visual, text_proto, visual_proto):
        query_visual = F.normalize(query_visual.float(), dim=-1)
        calibrated_text, pseudo_visual = self.prototype_bank(
            text_proto, visual_proto
        )
        query_text = self.visual_to_text(query_visual)
        logits_text = self.logit_scale * (query_text @ calibrated_text.T)
        logits_visual = self.logit_scale * (query_visual @ pseudo_visual.T)
        probabilities = (
            self.fusion_weight * logits_text.softmax(dim=-1)
            + (1.0 - self.fusion_weight) * logits_visual.softmax(dim=-1)
        )
        return {
            "probabilities": probabilities,
            "logits_text": logits_text,
            "logits_visual": logits_visual,
            "calibrated_text": calibrated_text,
            "pseudo_visual": pseudo_visual,
        }

    def losses(self, query_visual, labels, text_proto, visual_proto,
               consistency_weight=0.1, indication_weight=0.1):
        """Train on BASE classes only, with four directional objectives.

        The two auxiliary terms are contrastive prototype surrogates adapted
        for CLIP/FSCIL; they are NOT identical to the SAR paper's equations.
        """
        outputs = self.forward(query_visual, text_proto, visual_proto)
        labels = labels.long()
        loss_text = F.cross_entropy(outputs["logits_text"], labels)
        loss_visual = F.cross_entropy(outputs["logits_visual"], labels)

        n = text_proto.size(0)
        identity_labels = torch.arange(n, device=text_proto.device)
        visual_in_text = self.visual_to_text(visual_proto)
        text_in_visual = outputs["pseudo_visual"]
        text_proto = F.normalize(text_proto.float(), dim=-1)
        visual_proto = F.normalize(visual_proto.float(), dim=-1)

        # Correct class semantics for projected visual support prototypes.
        loss_consistency = F.cross_entropy(
            self.logit_scale * visual_in_text @ text_proto.T, identity_labels
        )
        # Visual prototypes indicate the correct class for pseudo-visual vectors.
        loss_indication = F.cross_entropy(
            self.logit_scale * text_in_visual @ visual_proto.T, identity_labels
        )
        total = (
            loss_text + loss_visual
            + consistency_weight * loss_consistency
            + indication_weight * loss_indication
        )
        return total, {
            "text_cls": loss_text.detach(),
            "visual_cls": loss_visual.detach(),
            "visual_to_text_consistency": loss_consistency.detach(),
            "text_to_visual_indication": loss_indication.detach(),
        }
