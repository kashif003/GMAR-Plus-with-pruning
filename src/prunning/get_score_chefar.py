from chefar import CheferHeadAttribution  # The class we created earlier
from transformers import AutoImageProcessor
from utils import get_img_tensor, get_jpeg_images
from vit import CustomViT
import torch
import json

# 1. Setup Model and Chefer Attribution
model = CustomViT()
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
# Note: CheferCAM requires the full model for LRP backward flow
chefer_attr = CheferHeadAttribution()

images = get_jpeg_images("imagenet_val_1000")
final_score = []

print(f"[INFO] Starting CheferCAM Head Attribution for {len(images)} images...")

for idx, image in enumerate(images):
    torch.cuda.empty_cache()# Clean slate for backward passes
    
    # Pre-processing
    img_tensor = get_img_tensor(processor, image)
    pixel_values = img_tensor["pixel_values"].to("cuda:6")

    # Forward pass through your custom ViT
    # CheferCAM logic needs the logits and the predicted class to start the backward flow
    logits, predicted_class, class_name, attn_weights = model.forward_with_custom_attention(pixel_values)

    # 2. Compute Head Scores using LRP + Gradients
    # This returns a list of 24 tensors (one per layer), each [num_heads]
    # Logic inside: (Grad * Relevance).clamp(min=0).sum()
    img_head_scores = chefer_attr.compute(
    logits=logits,
    pred_class=predicted_class,
    attn_weights=attn_weights,  # <--- Add this line
    model=model
)

    # 3. GLOBAL MIN-MAX NORMALIZATION
    # This ensures scores are comparable across layers for this specific image
    all_scores_flat = torch.cat([t.flatten() for t in img_head_scores])
    global_min = all_scores_flat.min()
    global_max = all_scores_flat.max()
    
    normalized_img_scores = [
        (t - global_min) / (global_max - global_min + 1e-8)
        for t in img_head_scores
    ]

    # 4. AGGREGATION (Running Sum)
    if len(final_score) == 0:
        final_score = normalized_img_scores
    else:
        final_score = [x + y for x, y in zip(final_score, normalized_img_scores)]

    # Memory Management
    del logits, predicted_class, img_head_scores, normalized_img_scores
    if (idx + 1) % 250 == 0:
        print(f"[INFO] Processed {idx + 1} images...")
        break

# 5. SAVE TO JSON
# Convert tensors to list for JSON serialization
json_ready = {
    str(layer_idx): layer_tensor.tolist()
    for layer_idx, layer_tensor in enumerate(final_score)
}

with open("chefer_head_scores.json", "w") as f:
    json.dump(json_ready, f, indent=4)

print("[SUCCESS] CheferCAM head scores saved to chefer_head_scores.json")