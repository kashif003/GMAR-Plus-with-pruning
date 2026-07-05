# utils.py - Contains utility functions for model pruning, dataset preparation, and evaluation.

import copy
from collections import defaultdict
from io import BytesIO
import json
from pathlib import Path

from datasets import Dataset as HFDataset
from PIL import Image
import requests
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torch_pruning as tp
import torchvision.transforms as T
from thop import clever_format, profile
from timm.data import create_transform
from tqdm.auto import tqdm
from transformers import AutoImageProcessor, AutoModelForImageClassification 

# --- Model & Image Loading Utilities ---

def load_model(name: str):
    """Loads a model and its corresponding image processor from Hugging Face."""
    processor = AutoImageProcessor.from_pretrained(name) 
    model = AutoModelForImageClassification.from_pretrained(name)
    return processor, model


def get_img_tensor(processor, image_path, return_logits=False):
    """Loads a local image file and converts it into a tensor format suitable for the model."""
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    input_tensor = processor(img, return_tensors="pt")
    return input_tensor


def get_jpeg_images(folder_path):
    """Retrieves all JPEG file paths within a directory."""
    folder = Path(folder_path)

    if not folder.is_dir():
        raise NotADirectoryError(f"Invalid folder path: {folder_path}")

    jpeg_images = [
        str(file)
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() == ".jpeg"
    ]

    return jpeg_images


def make_prediction(model, processor, device, url="https://t3.ftcdn.net/jpg/02/41/29/52/360_F_241295223_bIfEF64ZYw15rETnhigRBNQL0qFYbe92.jpg", return_logits=False):
    """Fetches an image from a URL and makes an evaluation prediction using the model."""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content)).convert("RGB")

    input_tensor = processor(img, return_tensors="pt").to(device)
    model = model.to(device)
    preds = model(**input_tensor, output_attentions=True)
    pred = preds.logits.argmax(-1)
    name = model.config.id2label[pred.item()]
    print(f'Model prediction: {name}')
    if return_logits:
        return preds.logits, pred.item(), preds.attentions


# --- Profiling & Evaluation Utilities ---

def FLOPS_and_PARAMS(model, input):
    """Counts and formats the model's parameters and FLOPs using thop."""
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params


def get_val_topk(model, dataloader, device="cpu"):
    """Evaluates the model on a validation dataset and computes Top-1 and Top-5 accuracy."""
    model.to(device)
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0

    for batch in tqdm(dataloader):
        input_tensor, label = batch
        input_tensor = input_tensor.to(device)
        label = label.to(device)
    
        outputs = model(input_tensor).logits
    
        # Top-1 and Top-5 predictions
        _, pred_top5 = outputs.topk(5, dim=1, largest=True, sorted=True)  # (batch_size, 5)
        top1_correct += (pred_top5[:, 0] == label).sum().item()
        top5_correct += sum([label[i] in pred_top5[i] for i in range(label.size(0))])
    
        total += label.size(0)

    top1_acc = top1_correct / total * 100
    top5_acc = top5_correct / total * 100

    print(f"Top-1 Accuracy: {top1_acc:.2f}%")
    print(f"Top-5 Accuracy: {top5_acc:.2f}%")

    return top1_acc, top5_acc


# --- Dataset & Transformation Pipeline ---

class Mydataset(Dataset):
    """Custom wrapper for processing ImageNet splits with model-specific transformations."""
    def __init__(self, dataset=None, transforms=None, batch_size=32, shuffle=False, num_workers=8, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index]["image"]
        if self.transforms:
            transforms = self.get_transforms()  # apply respective transformation
            input = transforms(image)
        else:
            transform = T.ToTensor()
            input = transform(image)

        label = self.dataset[index]["label"]
        return input, label
    
    def get_transforms(self):  # different transforms for different models
        if self.transforms == 'vit':
            transforms = T.Compose([
                T.Lambda(lambda x: x.convert("RGB")),  # ensure 3 channels
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            ])

        elif self.transforms == "deit":
            transforms = T.Compose([
                T.Lambda(lambda x: x.convert("RGB")),  # ensure 3 channels
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                T.RandomErasing(p=0.1, value='random')
            ])

        elif self.transforms == "swin":
            transforms = T.Compose([
                T.Lambda(lambda x: x.convert("RGB")),  # ensure 3 channels
                *create_transform(
                    input_size=224,
                    is_training=True,
                    auto_augment='rand-m9-mstd0.5-inc1',   
                    interpolation='bicubic',
                    re_prob=0.25,                                         
                    re_mode='pixel',
                    re_count=1,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)
                ).transforms
            ])
        elif self.transforms is None:
            raise ValueError('You have set transforms to None.')
        return transforms

    def dataloader(self):
        dataloader = DataLoader(
            dataset=self,   # only self because self is a dataset
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
        )
        return dataloader


def get_datasets_by_cv(cv_split, index):
    """Loads and merges specific train and validation subsets dictated by cross-validation splits."""
    split_number = f"split_{index+1}"

    training_datasets = [Dataset.from_file(path) for path in cv_split[split_number]["training"]]
    validation_datasets = [Dataset.from_file(path) for path in cv_split[split_number]["validation"]]

    training_dataset = ConcatDataset(training_datasets)
    valid_dataset = ConcatDataset(validation_datasets)
    return training_dataset, valid_dataset


# --- Pruning & XAI Mechanics ---

def prune_vit_out_channels(
        model, device,
        input_tensor=torch.randn(1, 3, 224, 224),
        prune_layer_idx=10,
        pruned_heads=[1],   # list of head indices to prune
    ):
    """Uses torch_pruning to structurally remove specific attention heads from a ViT Layer."""
    input_tensor = input_tensor.to(device)
    model.to(device)
    # Params before pruning
    _, b_params = FLOPS_and_PARAMS(model, input_tensor)
    print()
    print(f"params before pruning: {b_params}")
    # Prune specified layer head
    model = model.to("cpu")
    block = model.vit.layers[prune_layer_idx]
    # num of heads
    num_heads_before = block.attention.num_attention_heads
    head_dim = block.attention.head_dim
    dummy_input = torch.randn(1, 197, head_dim * num_heads_before).to("cpu")
    print(f'Shape of input: {dummy_input.shape}')
    print(f'Building dependency graph________________________________')
    DG = tp.DependencyGraph().build_dependency(block, example_inputs=dummy_input)

    # Build index list for selected heads
    idxs = []
    for h in pruned_heads:
        start = h * head_dim
        end = (h + 1) * head_dim
        idxs.extend(range(start, end))

    q = block.attention.q_proj
    group = DG.get_pruning_group(q, tp.prune_linear_out_channels, idxs=idxs)

    if DG.check_pruning_group(group):
        group.prune()

    # Set number of heads dynamically
    num_heads_after = num_heads_before - len(pruned_heads)
    block.attention.num_attention_heads = num_heads_after

    # Register hooks for every encoder block's layernorm_before
    hook_handles = []
    for idx, blk in enumerate(model.vit.layers):
        handle = blk.layernorm_before.register_forward_hook(
            lambda module, inp, out, idx=idx: print(
                f"Block {idx} layernorm_before input shape: {inp[0].shape} and output shape {out[0].shape}"
            )
        )
        hook_handles.append(handle)

    # Params after pruning
    _, b_params = FLOPS_and_PARAMS(model, torch.randn(1, 3, 224, 224))
    print()
    print(f"params After pruning: {b_params}")

    for handle in hook_handles:
        handle.remove()


def prune_with_XAI(original_model, score,
                    cv_names, hyperparams, scheduler, device, dataloader,
                    input_tensor=torch.randn(1, 3, 224, 224),
                    prune_layer_idx=10,
                    pruned_heads=[1], index_value=0):
    """Performs continuous targeted pruning using importance scores, falling back to finetuning upon severe accuracy drops."""
    # Deep copy the model
    model = copy.deepcopy(original_model)
    
    # Measure initial accuracy
    print(" calculating initial accuracy of the model.....................................")
    top1_acc_initial, top5_acc = get_val_topk(model, dataloader, device="cpu")
    print("accuracy before pruning:", top1_acc_initial)
    
    # Prepare scores
    score = list(score.values())
    all_values = [item for sublist in score for item in sublist]
    srtd_values = sorted(all_values.copy())  # Sorted ascending (lowest first)
    
    # Track layers that have already been pruned
    pruned_layers = set()
    
    while index_value < len(srtd_values):  # Add bounds to prevent IndexError
        value = srtd_values[index_value]
        print(f'Score to be pruned {value}')
        prune_layer_idx = None
        pruned_heads = []
        
        # Find the layer and head index for this value
        for idx, head in enumerate(score):
            if value in head:
                prune_layer_idx = idx
                head_index = head.index(value)  # Get the index as int
                pruned_heads = [head_index]  # Wrap in list for consistency
                break  # Stop after finding the first match
        
        if prune_layer_idx is None:
            print(f"Warning: Value {value} not found in any head; skipping.")
            index_value += 1
            continue

        # Avoid pruning the same layer multiple times
        if prune_layer_idx in pruned_layers:
            print(f"Layer {prune_layer_idx} already pruned once; skipping to avoid shape mismatch.")
            index_value += 1
            continue
        
        # Prune the model
        print(f'Pruning the head......................................')
        print(f'Pruning layer= {prune_layer_idx}')
        print(f'Pruning head index= {pruned_heads}')

        prune_vit_out_channels(
            model,
            input_tensor=input_tensor,
            prune_layer_idx=prune_layer_idx,
            pruned_heads=pruned_heads, device=device
        )

        # Mark this layer as pruned
        pruned_layers.add(prune_layer_idx)
        
        # Measure accuracy after pruning
        print("calculating model accuracy after pruning.....................................")
        top1_acc_current, top5_acc = get_val_topk(model, dataloader, device="cpu")
        print("accuracy After pruning:", top1_acc_current)
        
        # Check if accuracy drop is too large
        if top1_acc_initial - top1_acc_current >= 5:
            print("Accuracy drop too large; stopping pruning.")
            print(f'Fine tuning the model.')
            model = fine_tune(model, cv_names, hyperparams, scheduler, device, top1_acc_initial)
            break
        index_value += 1  # Move to the next lowest score
    return model


def fine_tune(model, cv_names, params: dict, scheduler, device, initial_accuracy):
    """Finetunes a pruned model using cyclical data loops until baseline performance recovery or early stopping hits."""
    num_epochs = params["epochs"]
    optimizer = params["optimizer"]
    criterion = params["loss_fn"]

    patience = 10  # Number of epochs to wait for improvement
    best_acc = 0.0
    counter = 0
    model.to(device)

    # loading the datasets
    for epoch in tqdm(range(num_epochs)):
        model.train()
        index = epoch % 7

        training_dataset, valid_dataset = get_datasets_by_cv(cv_names, index)
        dataset_train = Mydataset(training_dataset, batch_size=128, shuffle=True, transforms="vit")
        dataset_valid = Mydataset(valid_dataset, batch_size=128, shuffle=True, transforms="vit")
        train_loader = dataset_train.dataloader()
        print(f'length of train_loader: {len(train_loader)}')
        valid_loader = dataset_valid.dataloader()
        for imgs, labels in tqdm(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            print('calculating the validation loss..............................')
            top1_acc, top5_acc = get_val_topk(model, valid_loader, device=device)
            print(f'Validation accuracy: {top1_acc}')
        
        if abs(initial_accuracy + 5) <= top1_acc:
            print(f'accuracy has been recovered')
            return model
        
        if top1_acc > best_acc:
            best_acc = top1_acc
            counter = 0  # reset patience counter if there is improvement
        else:
            counter += 1  # increment if no improvement
            if counter >= patience:
                print("Early stopping triggered due to no improvement.")
                break  # stop training

    return model


def get_layers_and_heads(path, percentage=10, score_type="global"):
    """Retrieves and identifies the lowest-scoring heads for model pruning, breaking ties on late layers."""
    with open(path, "r") as file:
        score_file = json.load(file)
        
    score_list = []
    # 1. Keep track of (layer, head_index, score) right from the start
    for layer, heads in score_file.items():
        for head_idx, head_score in enumerate(heads):
            score_list.append((layer, head_idx, round(head_score, 4)))

    # 2. Tie-breaker Sort: 
    # Primary key: score (ascending -> lowest first)
    # Secondary key: layer number (descending -> higher layers first, using -int(x[0]))
    sorted_score = sorted(score_list, key=lambda x: (x[2], -int(x[0])))
    
    print("[INFO] Sorted score with tie-breaking:", sorted_score)
    
    # Calculate how many total heads to prune
    cutoff_index = int(len(sorted_score) * (percentage / 100))
    
    combined = defaultdict(list)
    
    if score_type == "global":
        # 3. Use head_idx to safely populate the dictionary
        for layer, head_idx, score in sorted_score[:cutoff_index]:
            combined[int(layer)].append(head_idx)

    return list(combined.keys()), list(combined.values())