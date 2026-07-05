# chefer_score.py
# Makes head importance scores using Chefer et al. (2021) method.

from ..vit import Custom_model
from transformers import AutoImageProcessor
from ..utils import get_jpeg_images, get_img_tensor
import torch
import numpy as np
import json
from tqdm import tqdm

custom_model = Custom_model(device="cuda:7")
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
images = get_jpeg_images("imagenet_val_1000")


def compute_lrp_relevance(attentions, gradients):
    """
    Computes LRP relevance per layer using Deep Taylor Decomposition.
    R^(nb) = (attention * gradient) / sum(attention * gradient)
    Returns list of relevance arrays, one per layer.
    """
    relevances = []
    for attn, grad in zip(attentions, gradients):
        if isinstance(attn, torch.Tensor):
            attn = attn.detach().cpu().numpy()
        else:
            attn = np.array(attn)

        if isinstance(grad, torch.Tensor):
            grad = grad.detach().cpu().numpy()
        else:
            grad = np.array(grad)

        # LRP: element-wise product of attention and gradient
        lrp = attn * grad  # [Batch, Heads, Tokens, Tokens]

        # Normalize to maintain conservation rule
        denom = np.sum(np.abs(lrp), axis=(-2, -1), keepdims=True) + 1e-8
        lrp = lrp / denom

        relevances.append(lrp)

    return relevances


def compute_chefer_head_scores(attentions, gradients):
    """
    Chefer et al. head importance score per layer:
    1. Compute LRP relevance R = normalize(attn * grad)
    2. Multiply gradient with relevance: grad x R
    3. ReLU -> keep positive values only
    4. Mean across heads -> scalar per head
    """
    relevances = compute_lrp_relevance(attentions, gradients)
    head_scores_per_layer = {}

    for layer_idx, (grad, rel) in enumerate(zip(gradients, relevances)):

        if isinstance(grad, torch.Tensor):
            grad = grad.detach().cpu().numpy()
        else:
            grad = np.array(grad)

        # Step 1 - gradient x LRP relevance: [Batch, Heads, Tokens, Tokens]
        combined = grad * rel

        # Step 2 - ReLU: keep only positive values
        positive = np.maximum(combined, 0)

        # Step 3 - sum across token dims -> [Batch, Heads]
        head_scores = np.sum(positive, axis=(-2, -1))

        # Step 4 - mean across batch -> [Heads]
        mean_scores = np.mean(head_scores, axis=0)

        head_scores_per_layer[layer_idx] = mean_scores

    return head_scores_per_layer


def accumulate_chefer_scores(layer_scores, final_score_dict=None):
    """Accumulates head scores across images."""
    if final_score_dict is None:
        final_score_dict = {}

    for layer_idx, scores in layer_scores.items():
        if layer_idx in final_score_dict:
            final_score_dict[layer_idx] += scores
        else:
            final_score_dict[layer_idx] = np.copy(scores)

    return final_score_dict


# --- Main Loop ---
global_pruning_scores = {}

for idx, image in tqdm(enumerate(images)):
    torch.cuda.empty_cache()

    img_tensor = get_img_tensor(processor, image)
    output, attentions, gradients = custom_model.chefer_forward_pass(
        img_tensor.pixel_values.to("cuda:7")
    )

    layer_scores = compute_chefer_head_scores(attentions, gradients)
    global_pruning_scores = accumulate_chefer_scores(layer_scores, global_pruning_scores)

print("[INFO] Total images processed:", idx + 1)
print("[INFO] Saving scores...")

json_ready_scores = {
    str(layer_idx): scores.tolist()
    for layer_idx, scores in global_pruning_scores.items()
}

with open("scores/chefer_score.json", "w") as f:
    json.dump(json_ready_scores, f, indent=4)

print("[INFO] Successfully saved Chefer scores to chefer_score.json!")