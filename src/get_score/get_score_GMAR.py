# this file will be used to get the score from the GMAR.

from collections import defaultdict
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification    

from ..utils import get_jpeg_images, get_img_tensor
from ..vit import Custom_model

#-- 

custom_model = Custom_model("cuda:7")
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
images = get_jpeg_images("imagenet_val_1000")


def accumulate_gmar_scores(gmar_gradients, final_score_dict=None):
    """
    Accumulates pre-calculated 1D layer-wise LeGrad head scores 
    across training iterations or evaluation datasets.
    """
    if final_score_dict is None:
        final_score_dict = {}

    for layer_idx, grad_scores in gmar_gradients.items():
        if layer_idx in final_score_dict:
            final_score_dict[layer_idx] += grad_scores
        else:
            final_score_dict[layer_idx] = np.copy(grad_scores)

    return final_score_dict


# --- Main Loop ---
global_pruning_scores = {}

for idx, image in tqdm(enumerate(images)):
    torch.cuda.empty_cache()
    img_tensor = get_img_tensor(processor, image)
    output, attention, gmar_grads = custom_model.gmar_forward_pass(img_tensor.pixel_values.to("cuda:7"))

    global_pruning_scores = accumulate_gmar_scores(gmar_grads, global_pruning_scores)


print("[INFO] Converting tensors to native Python formats for JSON serialization...")

# Now json.dump will work flawlessly!
json_ready_scores = {
    str(layer_idx): scores.tolist()
    for layer_idx, scores in global_pruning_scores.items()
}

with open("scores/GMAR_score.json", "w") as file:
    json.dump(json_ready_scores, file, indent=4)