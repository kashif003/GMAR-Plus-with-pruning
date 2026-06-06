# this file will be used to get the attention head score for the chefar cam.

"""
1) forward pass
    + get attention matrix and also class prediction
2) calculate the gradients based on the prediction.
4) make the score for that layer
"""
from vit import Custom_model
from transformers import AutoImageProcessor, AutoModelForImageClassification    
from utils import get_jpeg_images,get_img_tensor
import torch
import numpy as np

custom_model = Custom_model(device="cuda:7")
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
images = get_jpeg_images("imagenet_val_1000")

final_score = {}

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

for idx, image in enumerate(images):
    torch.cuda.empty_cache()
    img_tensor = get_img_tensor(processor, image)

    output, attention, gradient = custom_model.full_forward_pass(img_tensor)

    final_score = make_final_score(attention, gradient, final_score)





