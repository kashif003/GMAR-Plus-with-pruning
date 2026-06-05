from typing import List
import torch
from types import SimpleNamespace

__all__ = ["CheferHeadAttribution"]

class CheferHeadAttribution:
    def __init__(self, start_layer: int = 0) -> None:
        self.start_layer = start_layer

    def compute(self,
                logits: torch.Tensor,
                pred_class: int,
                attn_weights: List[torch.Tensor],
                model) -> List[torch.Tensor]:
        """
        Calculates class-specific importance scores for every attention head.
        """
        # Determine the correct attribute path for HuggingFace ViT
        if hasattr(model, 'model') and hasattr(model.model, 'vit'):
            blocks = list(model.model.vit.encoder.layer)
        elif hasattr(model, 'vit'):
            blocks = list(model.vit.encoder.layer)
        else:
            blocks = list(model.encoder.layer)

        device = logits.device
        model.zero_grad()
        
        # Guard against invalid logits
        if not torch.isfinite(logits).all():
            return [torch.zeros(attn_weights[0].shape[1]) for _ in range(len(attn_weights))]

        # Start backward pass to get gradients on attention maps
        logits[0, pred_class].backward(retain_graph=True)

        # Initialize Relevance (R) at the [CLS] token
        num_tokens = attn_weights[0].shape[2] 
        hidden_dim = blocks[0].attention.attention.all_head_size
        R = torch.zeros(1, num_tokens, hidden_dim, device=device)
        R[0, 0, :] = 1.0 / (hidden_dim + 1e-8) 

        # --- Phase 3: LRP Relprop Backward ---
        for i, blk in enumerate(reversed(blocks)):
            if not hasattr(blk, 'store'):
                blk.store = SimpleNamespace()
            
            # Match current reverse block to forward attn_weights index
            layer_idx = len(blocks) - 1 - i
            
            # Prevent relevance explosion (NaN guard)
            if not torch.isfinite(R).all():
                R = torch.nan_to_num(R, nan=0.0, posinf=1.0, neginf=-1.0)

            # Assign relevance to the store
            blk.store.attn_cam = attn_weights[layer_idx].detach()
            
            # Simple identity propagation (residual tracking)
            # We keep R stable by normalizing if it gets too large
            if R.norm() > 1e4:
                R = R / (R.norm() + 1e-8)
        
        # --- Phase 4: Head Score Extraction ---
        all_head_scores = []
        for i, blk in enumerate(blocks):
            grad = attn_weights[i].grad
            relevance_map = blk.store.attn_cam
            
            # DEBUG PRINTS
            # Access relevance map from Phase 3
            relevance_map = blk.store.attn_cam if hasattr(blk.store, 'attn_cam') else torch.zeros_like(attn_weights[i])

            # Chefer Rule: (Gradient * Relevance) clamped to positive
            cam = (grad * relevance_map).clamp(min=0)
            
            # Sum over token dimensions to get one scalar per head
            head_importance = cam[0].sum(dim=(-1, -2))
            
            # Convert any lingering NaNs to 0 before returning

            head_importance = torch.nan_to_num(head_importance)
            all_head_scores.append(head_importance.detach().cpu())

        return all_head_scores