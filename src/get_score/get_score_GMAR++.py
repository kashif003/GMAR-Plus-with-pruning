# this file will be used to get the score from the GMAR++.

from vit import Custom_model
from transformers import AutoImageProcessor, AutoModelForImageClassification    
import torch
import numpy as np

#-- 

custom_model = Custom_model("cuda:6")

from collections import defaultdict
import torch

def accumulate_gmarpp_scores(gmarpp_gradients, final_score_dict=None):
    """
    Accumulates pre-calculated 1D layer-wise LeGrad head scores 
    across training iterations or evaluation datasets.
    """
    if final_score_dict is None:
        final_score_dict = {}

    for layer_idx, grad_scores in gmarpp_gradients.items():
        if layer_idx in final_score_dict:
            final_score_dict[layer_idx] += grad_scores
        else:
            final_score_dict[layer_idx] = np.copy(grad_scores)

    return final_score_dict


from utils import get_jpeg_images,get_img_tensor
from transformers import AutoImageProcessor, AutoModelForImageClassification    
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
images = get_jpeg_images("imagenet_val_1000")
from tqdm import tqdm
global_pruning_scores = {}
for idx, image in tqdm(enumerate(images)):

# FIX: Calling the pure legrad pass instead of the incomplete standard pass
    torch.cuda.empty_cache()
    img_tensor = get_img_tensor(processor, image)
    output, attention, gmarpp_grads = custom_model.gmarpp_forward_pass(img_tensor.pixel_values.to("cuda:6"))


    global_pruning_scores = accumulate_gmarpp_scores(gmarpp_grads, global_pruning_scores)


import json

print("[INFO] Converting tensors to native Python formats for JSON serialization...")

# Convert integer keys to strings, and PyTorch tensors to standard Python lists


# Now json.dump will work flawlessly!
json_ready_scores = {
    str(layer_idx): scores.tolist()
    for layer_idx, scores in global_pruning_scores.items()
}
with open("GMARPP_score.json", "w") as file:
    json.dump(json_ready_scores, file, indent=4)