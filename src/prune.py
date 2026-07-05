import torch
import torch_pruning as tp
from transformers import AutoImageProcessor, AutoModelForImageClassification 
from vit import Custom_model   
from thop import profile, clever_format

def prune_vit_heads(model, layer_indices, heads_to_prune_list, device):
    model.eval()
    target_model = model.model if hasattr(model, 'model') else model
    img_size = target_model.config.image_size 

    for i, layer_idx in enumerate(layer_indices):
        current_layer_heads = heads_to_prune_list[i]
        block = target_model.vit.layers[layer_idx]

        # Sort reverse is MANDATORY here.
        for head_to_prune in sorted(current_layer_heads, reverse=True):

            # We grab the LATEST metadata from the block
            num_heads_current = block.attention.num_attention_heads
            head_dim = block.attention.head_dim

            start = head_to_prune * head_dim
            end = (head_to_prune + 1) * head_dim
            idxs = list(range(start, end))

            # Build DG for the CURRENT state of the model
            example_inputs = torch.randn(1, 3, img_size, img_size).to(device)
            dg = tp.DependencyGraph().build_dependency(target_model, example_inputs=example_inputs)

            q_layer = block.attention.q_proj
            group = dg.get_pruning_group(q_layer, tp.prune_linear_out_channels, idxs=idxs)

            if dg.check_pruning_group(group):
                group.exec()

                # Update metadata so the forward pass reshapes the smaller matrix correctly
                new_num_heads = num_heads_current - 1
                block.attention.num_attention_heads = new_num_heads

                print(f"Layer {layer_idx}: Pruned Head {head_to_prune}. Remaining heads: {new_num_heads}")

    return model


def FLOPS_and_PARAMS(model, input_tensor):
    """Calculates and formats FLOPs and Parameters."""
    # We use verbose=False to keep the console clean
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params




