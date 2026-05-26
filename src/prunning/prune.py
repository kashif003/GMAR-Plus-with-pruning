import torch
import torch_pruning as tp

def prune_vit_heads(model, layer_indices, heads_to_prune_list, device):
    model.to(device)
    model.eval()

    target_model = model.model if hasattr(model, 'model') else model
    img_size = target_model.config.image_size 

    for i, layer_idx in enumerate(layer_indices):
        current_layer_heads = heads_to_prune_list[i]
        block = target_model.vit.encoder.layer[layer_idx]

        # Sort reverse is MANDATORY here. 
        # If you prune head 1 then head 5, the index of 5 doesn't change.
        # If you prune 5 then 1, the index of 1 stays exactly where it was.
        for head_to_prune in sorted(current_layer_heads, reverse=True):
            
            # We grab the LATEST metadata from the block
            num_heads_current = block.attention.attention.num_attention_heads
            
            # The head_dim (usually 64 or 71) NEVER changes in ViT, 
            # only the number of heads changes.
            total_features = block.attention.attention.query.out_features
            head_dim = total_features // num_heads_current
            
            start = head_to_prune * head_dim
            end = (head_to_prune + 1) * head_dim
            idxs = list(range(start, end))

            # Build DG for the CURRENT state of the model
            example_inputs = torch.randn(1, 3, img_size, img_size).to(device)
            dg = tp.DependencyGraph().build_dependency(target_model, example_inputs=example_inputs)

            q_layer = block.attention.attention.query
            group = dg.get_pruning_group(q_layer, tp.prune_linear_out_channels, idxs=idxs)
            
            if dg.check_pruning_group(group):
                group.exec()
                
                # Update metadata so the forward pass reshapes the smaller matrix correctly
                new_num_heads = num_heads_current - 1
                block.attention.attention.num_attention_heads = new_num_heads
                block.attention.attention.all_head_size = new_num_heads * head_dim
                
                print(f"Layer {layer_idx}: Pruned Head {head_to_prune}. Remaining heads: {new_num_heads}")

    return model


def FLOPS_and_PARAMS(model, input_tensor):
    """Calculates and formats FLOPs and Parameters."""
    # We use verbose=False to keep the console clean
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params


# --- Example Usage ---
from transformers import AutoImageProcessor, AutoModelForImageClassification 
from vit import CustomViT   
from utils import make_prediction
from thop import profile, clever_format


device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
dummy_input = torch.randn(1, 3, 384, 384).to(device)
model = CustomViT()
bflops, bparams = FLOPS_and_PARAMS(model, dummy_input)

model = prune_vit_heads(model, layer_indices=[1,2], heads_to_prune_list=[[1,5],[7]], device=device)
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
make_prediction(model,processor,device = device)

aflops, aparams = FLOPS_and_PARAMS(model, dummy_input)
print("[INFO] Beofre:", bflops, bparams)
print("[INFO] After:", aflops, aparams)