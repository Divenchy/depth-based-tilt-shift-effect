import argparse
from transformers import pipeline
from PIL import Image
from pathlib import Path

parser = argparse.ArgumentParser(description="Generate depth map from an image.")
parser.add_argument("image", help="Input image filename, e.g. city_street.jpg")
args = parser.parse_args()
out_dir = "../stage1/"

stem = Path(args.image).stem

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Base-hf", useFast=True)
image = Image.open(args.image)

depth = pipe(image)["depth"]
depth.save(out_dir + f"{stem}-depth-map.png")
