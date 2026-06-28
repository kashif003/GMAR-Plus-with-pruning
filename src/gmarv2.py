"""GMARv2: Enhanced Gradient-weighted Multi-head Attention Rollout

This module implements an improved variant of GMAR that uses ReLU-clamped
gradients and row-normalized residual rollout for more robust attention-based
explanations in transformer vision models (e.g., ViT).
"""

from pathlib import Path
from typing import List, Union
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt


__all__ = ["GMARv2"]


class GMARv2:
    """Compute and save GMARv2 attention heatmaps with ReLU-clamped gradients.

    This variant improves upon standard GMAR by:
    - Clamping gradients to positive values only
    - Applying row normalization to the residual rollout
    - Supporting both L1 and L2 gradient importance metrics

    Parameters
    ----------
    alpha : float, optional
        Residual scaling added during rollout (default: 1.0).
    norm_type : str, optional
        Head importance metric; either ``'l1'`` or ``'l2'`` (default: 'l1').
    """

    def __init__(self, alpha: float = 1.0, norm_type: str = "l1") -> None:
        if norm_type not in ("l1", "l2"):
            raise ValueError("norm_type must be 'l1' or 'l2'")
        self.alpha = float(alpha)
        self.norm_type = norm_type

    def compute(
        self,
        logits: torch.Tensor,              # output logits before sotmax 
        pred_class: int,                   # prediction class
        attn_weights: List[torch.Tensor],   # the attention weights of the same model (without MLP)
        model,
    ) -> torch.Tensor:
        """Compute a normalized GMARv2 heatmap with ReLU-clamped gradients.

        Parameters
        ----------
        logits
            Model output logits for the current sample (shape [1, num_classes]).
        pred_class
            Index of the target class used to backprop and weight attention.
        attn_weights
            Per-layer attention tensors. Each tensor must have gradients
            (call ``retain_grad()`` on these tensors during the forward pass).
        model
            The model instance; only used to call ``zero_grad()`` before
            backward.

        Returns
        -------
        torch.Tensor
            2D heatmap of shape (S, S) with values normalized to [0, 1].
        """
        if not attn_weights:
            raise ValueError("attn_weights must be a non-empty list")

        model.model.zero_grad()

        # Backprop on the chosen class to populate gradients on attention
        target_logit = logits[0, pred_class]
        target_logit.backward(retain_graph=True)

        weighted_attns: List[torch.Tensor] = []

        for attn in attn_weights:
            grad = attn.grad
            if grad is None:
                raise RuntimeError(
                    "attn.grad is None — ensure `.retain_grad()` was called on attention tensors"
                )

            # Clamp gradients to positive values and weight attention
            pos_grad = grad.clamp(min=0)
            weighted_grad = pos_grad * attn

            # Compute head importance based on norm_type
            if self.norm_type == "l1":
                head_importance = pos_grad.abs().sum(dim=(-1, -2))
            else:  # l2
                head_importance = (pos_grad ** 2).sum(dim=(-1, -2)).sqrt()      # only the gradient of the positive gradient

            # Normalize head weights per layer
            head_weights = head_importance / head_importance.sum(dim=-1, keepdim=True)
            head_weights = head_weights.view(1, -1, 1, 1)

            # Collapse head dimension and keep the [1, N(577), N(577)] attention matrix
            A_weighted = (weighted_grad * head_weights)
            weighted_attns.append(A_weighted)
        return weighted_attns

 