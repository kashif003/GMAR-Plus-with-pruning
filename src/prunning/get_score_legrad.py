# this file will be used to get the legrad score.

from vit import Custom_model
from transformers import AutoImageProcessor, AutoModelForImageClassification    
#from utils import get_jpeg_images,get_img_tensor
import torch
import numpy as np
"""
Pipeline:
    1. forward pass get the prediciton score
    2. backward pass of target prediciton wrt to the attention score in each layer.
    3. Apply Relu and get the scaler score.
    4. get the scaler score
    """

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


# --- RUNNING AN EXAMPLE LOOP ---
from vit import Custom_model
from utils import get_jpeg_images, get_img_tensor
from transformers import AutoImageProcessor

custom_model = Custom_model(device="cuda:7")
images = get_jpeg_images("imagenet_val_1000")
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")


from tqdm import tqdm
print("[INFO] Making Legrad score.")
global_pruning_scores = {}
for idx, image in tqdm(enumerate(images)):

# FIX: Calling the pure legrad pass instead of the incomplete standard pass
    torch.cuda.empty_cache()
    img_tensor = get_img_tensor(processor, image)
    output, attention, legrad_grads = custom_model.legrad_forward_pass(img_tensor.pixel_values)


    global_pruning_scores = accumulate_legrad_scores(legrad_grads, global_pruning_scores)

#-- saving a json file
import json
final_score = {}

for k, v in global_pruning_scores.items():
    final_score[k] = [score.item() for score in v ]

with open("legrad_score.json", "w") as file:
    json.dump(final_score, file)


