import torch
from torch.utils.data import DataLoader
# from vit import CustomViT
from datasets import load_dataset
from transformers import AutoImageProcessor
from tqdm import tqdm
import wandb

# 1. Setup Device
device = "cuda:1" if torch.cuda.is_available() else "cpu"
# print(f"\n[INFO] Device: {device}")

# 2. Load Model and Processor
# model = CustomViT()
# model = model.model.to(device)
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")

# 3. Load Streaming Dataset
# Note: ImageNet-1k validation has 50,000 images
dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="validation",
    streaming=True,
    trust_remote_code=True,
)

def transform(examples):
    rgb_images = [img.convert("RGB") for img in examples["image"]]
    inputs = processor(rgb_images, return_tensors="pt")
    inputs["labels"] = torch.tensor(examples["label"])
    return inputs

processed_dataset = dataset.shuffle(buffer_size=1000).map(
    transform, 
    batched=True, 
    remove_columns=["image", "label"]
)

batch_size = 32
val_loader = DataLoader(processed_dataset, batch_size=batch_size)

# 4. Evaluation Loop
def validate(model):
    model.eval()
    total_correct = 0
    total_samples = 0

    total_batches = 50000 // batch_size 

    print("Starting data stream...\n")
    pbar = tqdm(val_loader, desc="Evaluating")
    with torch.no_grad():
        for batch in pbar:
            images = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
        
            if images.ndim == 5:
                images = images.squeeze(1)
        
            outputs = model(images)
            preds = outputs.logits.argmax(dim=-1)
        
            batch_correct = (preds == labels).sum().item()
            total_correct += batch_correct
            total_samples += labels.size(0)
        
            current_acc = (total_correct / total_samples) * 100

            wandb.log({"Per batch Accuracy": current_acc})
            pbar.set_postfix({"accuracy": f"{current_acc:.2f}%"})

    final_accuracy = (total_correct / total_samples) * 100


    return final_accuracy

