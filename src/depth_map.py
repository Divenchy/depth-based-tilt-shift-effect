import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import argparse
from pathlib import Path
from transformers import pipeline
from PIL import Image


parser = argparse.ArgumentParser(description="Generate a depth map from an image.")
parser.add_argument("image", help="Path to input image")
args = parser.parse_args()

stem = Path(args.image).stem
out_dir = Path("../stage1")
out_dir.mkdir(exist_ok=True)

pipe = pipeline(
    task="depth-estimation",
    model="depth-anything/Depth-Anything-V2-Base-hf",
    use_fast=True,
)

image = Image.open(args.image)
depth = pipe(image)["depth"]
depth.save(out_dir / f"{stem}-depth-map.png")