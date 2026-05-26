import torch
from torch.utils.data import DataLoader
from vit import CustomViT
from datasets import load_dataset
from transformers import AutoImageProcessor
from tqdm import tqdm

device = "cuda:7"  if torch.cuda.is_available() else "cpu"
model = CustomViT()
model = model.model.to(device)
processor = AutoImageProcessor.from_pretrained("google/vit-large-patch16-384")

dataset = load_dataset(
    "ILSVRC/imagenet-1k",
    split="validation",
    streaming=True,
    trust_remote_code=True,
)

def transform(examples):
    # Ensure all images are RGB (converts grayscale 1-channel to 3-channel)
    rgb_images = [img.convert("RGB") for img in examples["image"]]
    
    # Now the processor will be happy because everything has 3 channels
    inputs = processor(rgb_images, return_tensors="pt")
    
    inputs["labels"] = torch.tensor(examples["label"])
    return inputs

processed_dataset = dataset.shuffle(buffer_size=1000).map(
    transform, 
    batched=True, 
    remove_columns=["image", "label"]
)

val_loader = DataLoader(processed_dataset, batch_size=32)

print("Starting data stream...")

for batch in val_loader:
    images = batch["pixel_values"].to(device)
    labels = batch["labels"].to(device)
    preds = model(images).logits.argmax()
    
    if images.ndim == 5:
        images = images.squeeze(1)

    print(f"Batch processed successfully!")
    print(f"Images shape: {images.shape}") # Expected: [32, 3, 384, 384]
    print(f"Labels shape: {labels.shape}") # Expected: [32]
    print(f"preds shape: {preds.shape}") # Expected: [32]
    
    break

print("Done!")