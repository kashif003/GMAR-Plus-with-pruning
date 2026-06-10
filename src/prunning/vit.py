import torch
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification

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

        # Lists/Dicts to store our structural data
        self.attentions = []
        self.attention_gradients = {}  # FIX: Initialized as a dictionary

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
        """Standard hook for full backward passes."""
        def hook(grad):
            self.attention_gradients_list.append(grad.detach().cpu())
        return hook

    def _create_tensor_hook_legrad(self, layer_idx):
        """Pure LeGrad hook: Extract 1D head importance score directly from gradients."""
        def hook(grad):
            # Apply the LeGrad ReLU
            positive_grad = torch.clamp(grad, min=0)
            
            # Collapse spatial token dimensions [Batch, Heads, Tokens, Tokens] -> [Batch, Heads]
            head_scores = positive_grad.sum(dim=[-2, -1])
            
            # Average across the batch -> [Heads]
            mean_head_scores = head_scores.mean(dim=0)
            
            # Store directly into the dictionary by layer index without order issues
            self.attention_gradients[layer_idx] = mean_head_scores.detach().cpu().numpy()
            
            return grad
        return hook

    def clear(self):
        """Wipes tracking lists to clean memory between runs."""
        self.attentions = []
        self.attention_gradients = {}  # FIX: Keep as a clean dictionary
        self.attention_gradients_list = []
    
    def full_forward_pass(self, input_tensor, target_class=None):
        """Runs a forward pass and a batch-safe backward pass."""
        self.clear() 
        self.attention_gradients_list = []
        
        output = self.model(input_tensor, output_attentions=True)
        logits = output.logits
        native_attentions = output.attentions
        
        for attn_tensor in native_attentions:
            self.attentions.append(attn_tensor.detach().cpu())
            attn_tensor.register_hook(self._create_tensor_hook())
            
        if target_class is None:
            target_class = logits.argmax(dim=-1) 
            
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_scores = logits[batch_indices, target_class] 
        loss_scalar = target_scores.sum()
        
        self.model.zero_grad()
        loss_scalar.backward()  # FIX: Added missing .backward()
        
        self.attention_gradients_list.reverse()
        return output, self.attentions, self.attention_gradients_list
    
    def legrad_forward_pass(self, inputs, target_class=None):
        """Pure LeGrad implementation tracking layer-wise gradients."""
        self.clear() 
    
        output = self.model(inputs, output_attentions=True)
        logits = output.logits
        native_attentions = output.attentions
        
        for layer_idx, attn_tensor in enumerate(native_attentions):
            self.attentions.append(attn_tensor.detach().cpu())
            
            # Keep intermediate tensor gradient graph alive
            attn_tensor.retain_grad()
            attn_tensor.register_hook(self._create_tensor_hook_legrad(layer_idx))
            
        if target_class is None:
            target_class = logits.argmax(dim=-1) 
            
        batch_indices = torch.arange(logits.size(0), device=logits.device)
        target_scores = logits[batch_indices, target_class] 
        loss_scalar = target_scores.sum()
        
        self.model.zero_grad()
        loss_scalar.backward()  # Triggers the custom hooks automatically
        
        return output, self.attentions, self.attention_gradients


# --- OUTSIDE THE CLASS FUNCTIONALITY ---

def accumulate_legrad_scores(legrad_gradients, final_score_dict=None):
    """
    Accumulates pre-calculated 1D layer-wise LeGrad head scores 
    across training iterations or evaluation datasets.
    """
    if final_score_dict is None:
        final_score_dict = {}

    for layer_idx, grad_scores in legrad_gradients.items():
        if layer_idx in final_score_dict:
            final_score_dict[layer_idx] += grad_scores
        else:
            final_score_dict[layer_idx] = np.copy(grad_scores)

    return final_score_dict


