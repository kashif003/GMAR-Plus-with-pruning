# ViT.py
"""CustomViT: ViT Model Wrapper with Custom Attention Extraction

This module provides a wrapper around the HuggingFace ViT model that:
- Loads fine-tuned checkpoints or hub model IDs
- Matches the image processor from the checkpoint
- Returns logits, predictions, class names, and per-layer attention maps
- Enables custom forward passes for explainability methods
"""

import json
from pathlib import Path
from typing import List, Tuple, Optional
from torch import nn
import torch
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    AutoConfig,
)


__all__ = ["CustomViT"]


class CustomViT(torch.nn.Module):
    """Wrapper around HuggingFace ViT for attention-based explanations.

    Loads a fine-tuned ViT checkpoint, manages preprocessing, and enables
    custom forward passes that capture per-layer attention matrices.

    Parameters
    ----------
    model_name : str, optional
        Path to fine-tuned checkpoint directory or HF hub ID
        (default: "checkpoints/vit_large_tinyimagenet/best/").
    device : str, optional
        Compute device ('cuda', 'cpu', etc.); auto-selected if None.
    ensure_size : Tuple[int, int], optional
        Target image size (H, W) for preprocessing (default: (224, 224)).
    """
    def __init__(
        self,
        model_name: str = "google/vit-large-patch16-384",
        device: Optional[str] = None,
        ensure_size: Tuple[int, int] = (224, 224),
    ) -> None:
        """Initialize CustomViT with model and processor."""
        super().__init__()
        self.device = device or ("cuda:7" if torch.cuda.is_available() else "cpu")
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        try:
            self.processor = ViTImageProcessor.from_pretrained(model_name, local_files_only=True)
        except Exception:
            self.processor = ViTImageProcessor.from_pretrained("google/vit-large-patch16-224")

        preproc_path = Path(model_name) / "preproc.json"
        if preproc_path.exists():
            try:
                pp = json.loads(preproc_path.read_text())
                if "image_mean" in pp: self.processor.image_mean = pp["image_mean"]
                if "image_std"  in pp: self.processor.image_std  = pp["image_std"]
            except Exception:
                pass

        
        if ensure_size is not None:
            H, W = ensure_size
            try:
                self.processor.size = {"height": H, "width": W}
            except Exception:
                pass

        cfg = self.model.config
        id2label = cfg.id2label or {}
        try:
            tmp = {int(k): v for k, v in id2label.items()}
        except Exception:
            tmp = id2label
        self.imagenet_classes = [tmp[i] for i in range(cfg.num_labels)]

    def forward(self, pixel_values=None, **kwargs):
        """Standard forward pass required by PyTorch and utility functions."""
        # This passes the 'pixel_values' (and any other args like 'output_attentions')
        # directly to the underlying HuggingFace model.
        return self.model(pixel_values=pixel_values, **kwargs)
    @property
    def config(self):
        return self.model.config

    def preprocess(self, img) -> torch.Tensor:
        """Preprocess a PIL image for ViT.

        Parameters
        ----------
        img
            PIL Image to preprocess.

        Returns
        -------
        torch.Tensor
            Normalized image tensor of shape [1, 3, H, W].
        """
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def forward_with_custom_attention(self, img_tensor: torch.Tensor):
        # 1. Start with embeddings
        hidden_states = self.model.vit.embeddings(img_tensor)
        attn_weights: List[torch.Tensor] = []
        self.model.train() # This ensures all internal parameters are ready to flow gradients
# OR
        for param in self.model.parameters():
            param.requires_grad = True
        
        for i, layer in enumerate(self.model.vit.encoder.layer):
            num_heads = layer.attention.attention.num_attention_heads
            head_dim = layer.attention.attention.attention_head_size
            qkv_layer = layer.attention.attention
            
            # Project Q, K, V
            q = qkv_layer.query(hidden_states)
            k = qkv_layer.key(hidden_states)
            v = qkv_layer.value(hidden_states)
            
            def transpose_for_scores(x):
                new_x_shape = x.size()[:-1] + (num_heads, head_dim)
                return x.view(*new_x_shape).permute(0, 2, 1, 3)

            q = transpose_for_scores(q)
            k = transpose_for_scores(k)
            v = transpose_for_scores(v)

            # 2. Calculate Attention Map
            attn_scores = torch.matmul(q, k.transpose(-1, -2)) / (head_dim ** 0.5)
            attn = torch.nn.functional.softmax(attn_scores, dim=-1)
            
            # CRITICAL: Keep this tensor in the graph
            attn.retain_grad() 
            attn_weights.append(attn)

            # 3. Reconstruct Hidden States
            context_layer = torch.matmul(attn, v)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_shape = context_layer.size()[:-2] + (layer.attention.attention.all_head_size,)
            context_layer = context_layer.view(*new_shape)
            
            # 4. Connect the output back to the main 'hidden_states' variable
            attention_output = layer.attention.output(context_layer, hidden_states)
            intermediate_output = layer.intermediate(attention_output)
            
            # Update the PLURAL variable so the NEXT layer receives this data
            hidden_states = layer.output(intermediate_output, attention_output)

        # 5. Final Output
        sequence_output = self.model.vit.layernorm(hidden_states)
        logits = self.model.classifier(sequence_output[:, 0, :])
        
        predicted_class = torch.argmax(logits, dim=-1).item()
        class_name = self.model.config.id2label[predicted_class]

        return logits, predicted_class, class_name, attn_weights