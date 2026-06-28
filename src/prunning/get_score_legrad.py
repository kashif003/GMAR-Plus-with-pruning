# this file will be used to get the score from the LeGrad.
from vit import Custom_model
from transformers import AutoImageProcessor, AutoModelForImageClassification    
import torch
import numpy as np

#-- 

custom_model = Custom_model("cuda:7")

from collections import defaultdict
import torch

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


from utils import get_jpeg_images, get_img_tensor
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
images = get_jpeg_images("imagenet_val_1000")
from tqdm import tqdm
global_pruning_scores = {}
for idx, image in tqdm(enumerate(images)):

    torch.cuda.empty_cache()
    img_tensor = get_img_tensor(processor, image)
    output, attention, legrad_grads = custom_model.legrad_forward_pass(img_tensor.pixel_values.to("cuda:7"))

    global_pruning_scores = accumulate_legrad_scores(legrad_grads, global_pruning_scores)


import json

print("[INFO] Converting tensors to native Python formats for JSON serialization...")

json_ready_scores = {
    str(layer_idx): scores.tolist()
    for layer_idx, scores in global_pruning_scores.items()
}
with open("LeGrad_score.json", "w") as file:
    json.dump(json_ready_scores, file, indent=4)

print("[INFO] Successfully saved LeGrad scores to LeGrad_score.json!")