# this file will be used to get the score from the GMAR.

from vit import Custom_model
from transformers import AutoImageProcessor, AutoModelForImageClassification    
import torch
import numpy as np

#-- 
dummy_input = torch.rand(1,3,384,384)
custom_model = Custom_model("cpu")

output, attention, gradient = custom_model.full_forward_pass(dummy_input)


from collections import defaultdict
import torch

def process_and_accumulate_gmar(attention_list, gradient_list, global_pruning_scores=None):
    """
    Calculates raw GMAR head scores for a single image's layers and accumulates 
    them into a running global dictionary across the entire dataset.
    
    Args:
        attention_list (list): List of attention tensors from full_forward_pass.
        gradient_list (list): List of gradient tuples/tensors from full_forward_pass.
        global_pruning_scores (dict): Running dictionary holding accumulated dataset scores.
        
    Returns:
        dict: Updated running dictionary with the current image's scores accumulated.
    """
    if global_pruning_scores is None:
        global_pruning_scores = {}

    for layer_idx, (att, grad) in enumerate(zip(attention_list, gradient_list)):
        
        if isinstance(att, (tuple, list)):
            att = att[0]
        if isinstance(grad, (tuple, list)):  
            grad = grad[0]
            
        if not isinstance(att, torch.Tensor):
            att = torch.tensor(att)
        if not isinstance(grad, torch.Tensor): 
            grad = torch.tensor(grad)
            
        grad = grad.to(att.device) 
        
        mul = att * grad
        head_scores = (mul ** 2).sum(dim=(-1, -2)).sqrt()
        
        img_layer_score = head_scores.squeeze(0).detach().cpu() # Shape: [num_heads]
        
        if layer_idx in global_pruning_scores:
            global_pruning_scores[layer_idx] += img_layer_score
        else:
            global_pruning_scores[layer_idx] = img_layer_score.clone()

    return global_pruning_scores

images = [dummy_input, dummy_input]

global_pruning_scores = {}
for idx, image in enumerate(images):

# FIX: Calling the pure legrad pass instead of the incomplete standard pass
    torch.cuda.empty_cache()
    img_tensor =   image # get_img_tensor(processor, image)
    output, attention, gradient = custom_model.full_forward_pass(img_tensor)
    global_pruning_scores =process_and_accumulate_gmar(attention, gradient, global_pruning_scores)

    print("done")