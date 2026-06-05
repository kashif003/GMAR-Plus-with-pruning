#  this file will be used to get the magnitude of each head of the  model.

#-- importing the model
from vit import CustomViT
import torch
model = CustomViT().model

from collections import defaultdict
# Use the list class as the factory
score = defaultdict(list)

for i, layer in enumerate(model.vit.encoder.layer):
    layer_attention = layer.attention.attention
    
    # Weights of query, key, and value
    q_weights = layer_attention.query.weight
    k_weights = layer_attention.key.weight
    v_weights = layer_attention.value.weight # Corrected: used .value instead of .key

    num_heads = model.config.num_attention_heads
    head_dim  = model.config.hidden_size // num_heads

    # Reshape to [num_heads, head_dim, hidden_size]
    q_heads = q_weights.view(num_heads, head_dim, -1)
    k_heads = k_weights.view(num_heads, head_dim, -1)
    v_heads = v_weights.view(num_heads, head_dim, -1)

    # Calculate magnitude for each head
    layer_magnitudes = []
    for h in range(num_heads):
        # We calculate the Frobenius norm (L2) of the head's weights
        # You can combine Q, K, and V or just track one. 
        # Here, we combine them to represent the "Total Head Magnitude"
        q_norm = torch.norm(q_heads[h])
        k_norm = torch.norm(k_heads[h])
        v_norm = torch.norm(v_heads[h])
        
        # Total magnitude for this specific head
        total_magnitude = (q_norm + k_norm + v_norm).item()
        layer_magnitudes.append(total_magnitude)
    
    # Store the list of 16 magnitudes for this layer
    score[f"{i}"] = layer_magnitudes

import json
with open("mag_score.json", "w") as f:
    json.dump(score, f)