# this file is used to make data loaders
import torch
import random
import math
from datasets import Dataset, concatenate_datasets
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_from_disk, concatenate_datasets
from torchvision import transforms
import json
from utils import *
from pathlib import Path
from configs import Config # Assuming you have your config here

class StreamingTrainDataset(IterableDataset):
    def __init__(self, file_paths, chunk_size=10):
        """
        file_paths: List of string paths to .arrow files.
        chunk_size: How many files to load into RAM at once.
        """
        self.files = file_paths
        self.chunk_size = chunk_size
        
        # Standard ImageNet normalization for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.files)
        
    def __iter__(self):
        # ---------------------------------------------------------
        # 1. WORKER SPLIT LOGIC
        # ---------------------------------------------------------
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            # Single-process data loading
            my_files = self.files
        else:
            # Multi-process: Split workload evenly
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.files))
            
            my_files = self.files[iter_start:iter_end]
            
            # Print once per worker to confirm split (optional debugging)
            # print(f"[Worker {worker_id}] Processing files {iter_start} to {iter_end}")

        # ---------------------------------------------------------
        # 2. SHUFFLE & STREAM
        # ---------------------------------------------------------
        # Shuffle the order of files assigned to this worker
        random.shuffle(my_files)
        
        # Iterate through files in small chunks to save RAM
        for i in range(0, len(my_files), self.chunk_size):
            chunk_paths = my_files[i : i + self.chunk_size]
            
            # Load datasets from raw arrow files
            dataset_list = []
            for p in chunk_paths:
                try:
                    # FIX 1: Use Dataset.from_file for raw .arrow files
                    ds = Dataset.from_file(str(p))
                    dataset_list.append(ds)
                except Exception as e:
                    print(f"Warning: Skipping corrupt file {p}: {e}")
                    continue
            
            if len(dataset_list) > 0:
                # Concatenate the chunk
                combined_ds = concatenate_datasets(dataset_list)
                
                
                # Yield items one by one
                for item in combined_ds:
                    img = item['image'] # Ensure key matches your data ('image' or 'img')
                    label = item['label']
                    
                    # FIX 2: Handle Grayscale Images (1 channel) causing crashes
                    # If mode is 'L' (Grayscale) or 'CMYK', convert to 'RGB'
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # Apply Transforms
                    if not isinstance(img, torch.Tensor):
                        img = self.transform(img)
                        
                    yield img, label

def get_train_loader(config):
    # 1. Getting path properly (Same as your code)
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    BASE_DIR = PROJECT_DIR.parent

    # 2. Load JSON
    json_path = PROJECT_DIR/"files/datastet_names.json"
    with open(str(json_path), "r") as file:
        names_file = json.load(file)

    # 3. Construct Training Paths
    # Note: ensure "train" is the correct key in your json
    train_paths = [
        BASE_DIR / "_huggingface-shared/datasets/imagenet-1k/default/1.0.0/07900defe1ccf3404ea7e5e876a64ca41192f6c07406044771544ef1505831e8" / item 
        for item in names_file["training"]
    ]

    # 4. Create Streaming Dataset
    # We pass the LIST of paths, not the loaded datasets
    train_dataset = StreamingTrainDataset(
        file_paths=train_paths, 
        chunk_size=10 # Loads 20 arrow files at a time to save RAM
    )

    # 5. Create DataLoader
    # Note: Shuffle is False because the dataset handles shuffling internally
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE, # e.g., 500
        num_workers=16,
        pin_memory=True,
        collate_fn=None, # Add your Mixup Collator here if not doing it inside loop
        drop_last=True,
        shuffle=False
    )

    return train_loader



def get_valid_loader(config):
    # Getting path properly
    PROJECT_DIR = Path(__file__).resolve().parent.parent
    BASE_DIR= PROJECT_DIR.parent

    # loading validation dataloader
    json_path= PROJECT_DIR/"files"/"datastet_names.json"
    with open(str(json_path), "r") as file:
        names_file= json.load(file)
    validation_names= [BASE_DIR/"_huggingface-shared/datasets/imagenet-1k/default/1.0.0/07900defe1ccf3404ea7e5e876a64ca41192f6c07406044771544ef1505831e8/"/item for item in names_file["validation"]]
    dataset_list = [Dataset.from_file(str(name)) for name in validation_names]
    # concatinating dataset and making data loader
    valid_dataset=Mydataset(concatenate_datasets(dataset_list),batch_size=config.BATCH_SIZE,transforms= "vit" )
    valid_loader=valid_dataset.dataloader()
    # getting validation accuracy
    return valid_loader


if __name__== "__main__":
    from configs import Config
    config= Config()
    train_loader= get_train_loader(config)
    print(train_loader.dataset)