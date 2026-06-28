# this is the main file it will be sed to prune as well as get  the results

from utils import get_layers_and_heads
from transformers import AutoImageProcessor, AutoModelForImageClassification 
from vit import Custom_model   
from thop import profile, clever_format
import torch
from prune import FLOPS_and_PARAMS, prune_vit_heads

import argparse
import os
from dotenv import load_dotenv
import wandb
load_dotenv()
WANDB_API_KEY = os.getenv("WANDB_KEY")
wandb.login(key=WANDB_API_KEY)


parser = argparse.ArgumentParser()

parser.add_argument("--config", type=int, required=True)
parser.add_argument("--score_path", type=str, required=True)

args = parser.parse_args()
cfg = args.config
wandb.init(
    project="GMAR++ with pruning",
    name=f"legrad_score_{cfg}",
    config={
        "Pruning precentage": cfg
    }
)

layer_list, head_list = get_layers_and_heads(args.score_path, percentage=cfg)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("[INFO] Device:", device)


# before pruning
dummy_input = torch.randn(1, 3, 384, 384).to(device)
custom_model = Custom_model(device =device)
model = custom_model.get_model().to(device)
bflops, bparams = FLOPS_and_PARAMS(model, dummy_input)


model_2 = prune_vit_heads(model, layer_indices=layer_list, heads_to_prune_list=head_list, device=device)
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")
# after pruning
aflops, aparams = FLOPS_and_PARAMS(model_2, dummy_input)
wandb.log({"FLOPS before Pruning":bflops, "PARAMS before Pruning":bparams})
wandb.log({"FLOPS after Pruning":aflops, "PARAMS after Pruning":aparams})


from get_score.get_baseline_accuracy import validate

accuracy = validate(model_2)
wandb.log({"Final Accuracy": accuracy})



# need to work on the clustering on the rtlc.
    # finish it by today