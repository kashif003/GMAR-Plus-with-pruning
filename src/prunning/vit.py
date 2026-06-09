# this is related to  the model all the fucntionalities with the model will be in this file.

import torch
from transformers import (
    ViTImageProcessor,
    ViTForImageClassification,
    AutoConfig,
)

import torch
from transformers import ViTForImageClassification, ViTImageProcessor

import torch
from transformers import ViTForImageClassification, ViTImageProcessor

class Custom_model(torch.nn.Module):
    def __init__(self, device, name="google/vit-large-patch16-384"):
        super().__init__()
        self.model_name = name
        self.device = device
        
        # Load model in eager mode to ensure the internal 4D matrix graph is built
        self.model = ViTForImageClassification.from_pretrained(
            self.model_name,
            attn_implementation="eager"
        )
        
        # Explicitly instruct the model to output the 4D attention weights
        # self.model.config.output_attentions = False
        # self.model.config.return_dict = False

        # Lists to store our clean 4D matrices
        self.attentions = []
        self.attention_gradients = []

        self.model.eval()

        try:
            self.processor = ViTImageProcessor.from_pretrained(self.model_name, local_files_only=True)
        except Exception as e:
            print("[INFO] Failed to load the processor:", e)
            print("[IMPORTANT] please load the processor separately.")
            self.processor = None
        
    def get_model(self):
        return self.model
    
    def config(self):
        return self.model.config
    
    def _create_tensor_hook(self):
        """
        Creates a custom lambda hook function that will save the gradient of the 
        specific tensor it is attached to.
        """
        def hook(grad):
            # This intercepts the gradient of the raw attention weight matrix
            # Shape will be exactly (1, 16, 577, 577)
            self.attention_gradients.append(grad.detach().cpu())
        return hook

    def clear(self):
        """Wipes tracking lists to clean memory between runs."""
        self.attentions = []
        self.attention_gradients = []
    
    def full_forward_pass(self, input_tensor, target_class=None):
        """
        Runs a forward pass and a batch-safe backward pass to capture 4D gradients.
        """
        self.clear() # Reset tracking arrays
        
        # 1. Run the forward pass
        input_tensor = input_tensor
        output = self.model(input_tensor, output_attentions=True)
        logits = output.logits
        
        # 2. Extract the native 4D attention matrices
        native_attentions = output.attentions
        
        # 3. Attach tensor hooks to the live graph
        for attn_tensor in native_attentions:
            self.attentions.append(attn_tensor.detach().cpu())
            attn_tensor.register_hook(self._create_tensor_hook())
            
        # 4. Determine the target classes for the entire batch
        if target_class is None:
            # Gets the highest scoring class index for EACH image in the batch
            target_class = logits.argmax(dim=-1) # Shape: (batch_size,)
            
        # 🎯 FIX FOR BATCHING: Gather the specific target class logit for each image
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_scores = logits[batch_indices, target_class] # Shape: (batch_size,)
        
        # 🎯 Turn the batch vector into a single scalar by taking the sum
        loss_scalar = target_scores.sum()
        
        # 5. Clear old network gradients and execute backprop on the scalar
        self.model.zero_grad()
        loss_scalar.backward()
        
        # Reverse the gradient list so it lines up index-for-index with the forward list
        self.attention_gradients.reverse()
        
        return output, self.attentions, self.attention_gradients

"""
custom_model = Custom_model(device="cpu")
model = custom_model.get_model()
dummy_input = torch.rand(2,3,384,384)

output, attention, gradient = custom_model.full_forward_pass(dummy_input) # output.logits.shape = (1,1000)
"""
import numpy as np
import torch

def make_final_score(attention, gradient, final_score_dict):
    """
    Computes head relevance scores for a batch and aggregates them 
    accumulatively into final_score_dict across the entire training/epoch loop.
    """
    if final_score_dict is None:
        final_score_dict = {}

    for layer_idx, (attn, grad) in enumerate(zip(attention, gradient)):
        key = int(layer_idx)
        
        if isinstance(attn, torch.Tensor):
            att_arry = attn.detach().cpu().numpy()
        else:
            att_arry = np.array(attn)
            
        if isinstance(grad, torch.Tensor):
            grad_arry = grad.detach().cpu().numpy()
        else:
            grad_arry = np.array(grad)
        
        weighted_attention = att_arry * grad_arry
        
        positive_relevance = np.maximum(weighted_attention, 0)
        
        if positive_relevance.ndim == 4:
            batch_head_scores = np.sum(positive_relevance, axis=(0, 2, 3))
        elif positive_relevance.ndim == 3:
            batch_head_scores = np.sum(positive_relevance, axis=(1, 2))
        
        if key in final_score_dict:
            final_score_dict[key] += batch_head_scores
        else:
            final_score_dict[key] = batch_head_scores

    return final_score_dict

"""final_dict = {}
for i in range(3):

    final = make_final_score(attention, gradient, final_dict)

print("done")"""