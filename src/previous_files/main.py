import timm
from prune import prune_tiny_vit
import torch
import wandb
from configs import Config
from dataloaders import *
from fine_tune import *
from utils import load_model

def train_with_sweep():
    base_config = Config()
    model, params = load_model(return_params=True)

    with wandb.init(
        project="computer-vision",
        name= "basline_2",
        notes="replicating the model with the given yaml file with algoritm 1 from the paper(with pretrained weights)",
        config=base_config.as_dict(),   
        resume="allow",
    ) as run:

        # model as mentioned in the yaml file
        

        model.to(base_config.DEVICE)
        print(f"Model on {base_config.DEVICE} and has {params} parameters.")
        train_loader = get_train_loader(base_config)   # make these use cfg.BATCH_SIZE etc.
    
        val_loader = get_valid_loader(base_config)

        model=train(model, train_loader, val_loader, base_config)

        acc, loss = valid(model, val_loader, base_config)
        wandb.log({"final_val_acc": acc, "final_val_loss": loss})
        print("acc, loss", acc, loss)

if __name__ == "__main__":
    train_with_sweep()
