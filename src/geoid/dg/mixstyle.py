"""Proposed event-aware MixStyle; model insertion requires model-team review.

Reference: Zhou et al., Domain Generalization with MixStyle (ICLR 2021).
Explicit event pairing adapts the reference two-domain batch ordering. Donors
may repeat. Statistics use sample variance, as in the reference implementation.
"""

import math

import torch
from torch import nn


class MixStyle(nn.Module):
    def __init__(self, p=0.5, alpha=0.1, eps=1e-6):
        super().__init__()
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p must be finite in [0,1]")
        if not math.isfinite(alpha) or alpha <= 0 or not math.isfinite(eps) or eps <= 0:
            raise ValueError("alpha and eps must be positive and finite")
        self.p, self.alpha, self.eps = float(p), float(alpha), float(eps)
        self.last_info = {}

    def forward(self, features, event_ids):
        self.last_info = {"applied": False, "reason": "disabled"}
        if not self.training or self.p == 0:
            return features
        if features.ndim != 4 or not features.is_floating_point():
            raise ValueError("Expected floating spatial features [B,C,H,W]")
        b, c, h, w = features.shape
        if min(b, c, h, w) < 1 or not torch.isfinite(features).all():
            raise ValueError("Require finite nonempty features")
        if len(event_ids) != b or any(not isinstance(e, str) or not e for e in event_ids):
            raise ValueError("Require nonempty event ID per sample")
        if h * w < 2 or len(set(event_ids)) < 2:
            self.last_info["reason"] = "small_grid_or_no_other_event"
            return features
        if self.p < 1 and torch.rand((), device=features.device).item() >= self.p:
            self.last_info["reason"] = "probability"
            return features
        donors = []
        for event in event_ids:
            candidates = [j for j, other in enumerate(event_ids) if event != other]
            donors.append(
                candidates[torch.randint(len(candidates), (), device=features.device).item()]
            )
        x = features.float() if features.dtype in (torch.float16, torch.bfloat16) else features
        mean = x.mean((2, 3), keepdim=True).detach()
        std = (x.var((2, 3), keepdim=True, correction=1) + self.eps).sqrt().detach()
        concentration = x.new_tensor(self.alpha)
        weight = torch.distributions.Beta(concentration, concentration).sample((b, 1, 1, 1))
        mean_mix = weight * mean + (1 - weight) * mean[donors]
        std_mix = weight * std + (1 - weight) * std[donors]
        output = ((x - mean) / std * std_mix + mean_mix).to(features.dtype)
        self.last_info = {"applied": True, "donor_indices": donors, "reason": "applied"}
        return output


def mixstyle(features, event_ids, *, training):
    """Compatibility entry point; prefer a persistent MixStyle module in encoders."""
    return MixStyle().train(training)(features, event_ids)
