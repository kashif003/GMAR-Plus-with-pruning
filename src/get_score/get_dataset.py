# run this file to get the imagenet validation set for 1k images
import random
from pathlib import Path
from datasets import load_dataset
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()

from huggingface_hub import login
hf_token = os.getenv("HUGGING_FACE_TOKEN")
login(token= hf_token )  
# ── config ────────────────────────────────────────────────────────────
NUM_IMAGES    = 1000
SEED          = 42        # fixed seed for reproducibility
OUTPUT_DIR    = Path("imagenet_val_1000")
SPLIT         = "validation"   # ImageNet val set — 50k images, 50 per class


def download_imagenet_1000():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ImageNet-1k validation set from Hugging Face...")
    print("(only metadata is loaded initially — images stream on demand)\n")

    # streaming=True means images are NOT all downloaded upfront
    # only the 1000 we select will be downloaded
    ds = load_dataset(
        "ILSVRC/imagenet-1k",
        split      = SPLIT,
        streaming  = True,
        trust_remote_code = True,
    )

    # shuffle with fixed seed and take 1000
    ds_shuffled = ds.shuffle(seed=SEED, buffer_size=5000).take(NUM_IMAGES)

    print(f"Downloading {NUM_IMAGES} random images to: {OUTPUT_DIR.resolve()}\n")

    label_counts = {}   # track how many per class
    saved        = 0

    for i, sample in enumerate(ds_shuffled):
        img: Image.Image = sample["image"].convert("RGB")
        label: int       = sample["label"]      # 0-999 ImageNet class index

        # filename: imagenet_{index:04d}_class{label:04d}.jpeg
        fname = OUTPUT_DIR / f"imagenet_{i:04d}_cls{label:04d}.jpeg"
        img.save(fname, format="JPEG", quality=95)

        label_counts[label] = label_counts.get(label, 0) + 1
        saved += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1:>4}/{NUM_IMAGES}] saved {fname.name}")

    # ── summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"  Done. {saved} images saved to {OUTPUT_DIR.resolve()}")
    print(f"  Unique classes covered : {len(label_counts)} / 1000")
    print(f"  Images per class range : "
          f"{min(label_counts.values())} – {max(label_counts.values())}")
    print(f"{'='*50}")

    # save a label mapping file for reference
    label_file = OUTPUT_DIR / "labels.txt"
    with open(label_file, "w") as f:
        f.write("filename,class_idx\n")
        for i in range(saved):
            # re-derive label from filename
            fname = f"imagenet_{i:04d}_cls"
            f.write(f"{fname}*\n")

    print(f"  Label index in filename: imagenet_NNNN_clsLLLL.jpeg")
    print(f"    NNNN = sequential index")
    print(f"    LLLL = ImageNet class index (0-999)")


if __name__ == "__main__":
    download_imagenet_1000()