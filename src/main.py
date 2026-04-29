import sys
import subprocess
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description="End-to-end depth-based tilt-shift pipeline.")
parser.add_argument("image", help="Input image filename, e.g. city_street.jpg")

parser.add_argument("--force-depth", action="store_true",
    help="Re-run depth estimation even if a cached map exists.")

parser.add_argument("--focus-x", type=int, default=None, help="Focus point x (default: image center).")
parser.add_argument("--focus-y", type=int, default=None, help="Focus point y (default: image center).")
parser.add_argument("--tilt-x", type=float, default=0.0, help="Focal-plane tilt along x (depth/pixel).")
parser.add_argument("--tilt-y", type=float, default=0.002, help="Focal-plane tilt along y (depth/pixel).")
parser.add_argument("--coc-scale", type=float, default=50.0, help="Depth deviation -> CoC scale.")
parser.add_argument("--max-coc", type=float, default=15.0, help="Max CoC radius in pixels.")

parser.add_argument("--num-levels", type=int, default=12, help="Number of pre-blurred Gaussian levels.")
parser.add_argument("--sigma-scale", type=float, default=1.0, help="CoC -> sigma multiplier.")
parser.add_argument("--coc-smooth", type=float, default=3.0, help="Sigma of pre-smoothing on CoC map.")

parser.add_argument("--saturation", type=float, default=1.4, help="HSV saturation multiplier.")
parser.add_argument("--contrast", type=float, default=0.3, help="S-curve strength in [0, 1].")
parser.add_argument("--warmth", type=float, default=0.15, help="Warm/cool shift; positive = warmer.")

args = parser.parse_args()

stem = Path(args.image).stem
image_rel_path = str(Path("../resources") / args.image)


# Stage 1: depth estimation

depth_map_path = Path(f"../stage1/{stem}-depth-map.png")

if depth_map_path.exists() and not args.force_depth:
    print(f"\n[Stage 1] Cached depth map found at {depth_map_path}, skipping.")
    print("          Use --force-depth to regenerate.")
else:
    print("\n[Stage 1] Estimating depth...")
    subprocess.run([sys.executable, "depth_map.py", image_rel_path], check=True)


# Stage 2: tilted focal plane and Circle of Confusion map

print("\n[Stage 2] Computing tilted focal plane and CoC map...")

out_dir_s2 = Path("../stage2_3")
out_dir_s2.mkdir(exist_ok=True)

depth = np.array(Image.open(depth_map_path).convert("L"), dtype=np.float32) / 255.0
h, w = depth.shape

u0 = args.focus_x if args.focus_x is not None else w // 2
v0 = args.focus_y if args.focus_y is not None else h // 2
u0 = int(np.clip(u0, 0, w - 1))
v0 = int(np.clip(v0, 0, h - 1))
focus_depth = depth[v0, u0]

uu, vv = np.meshgrid(np.arange(w, dtype=np.float32),
                     np.arange(h, dtype=np.float32))

focal_plane = focus_depth + args.tilt_x * (uu - u0) + args.tilt_y * (vv - v0)
focal_plane = np.clip(focal_plane, 0.0, 1.0)

signed_dist = depth - focal_plane
coc = np.clip(np.abs(signed_dist) * args.coc_scale, 0.0, args.max_coc)

np.save(out_dir_s2 / f"{stem}-coc.npy", coc)
Image.fromarray((coc / args.max_coc * 255).astype(np.uint8)).save(out_dir_s2 / f"{stem}-coc-vis.png")
Image.fromarray((focal_plane * 255).astype(np.uint8)).save(out_dir_s2 / f"{stem}-focal-plane-vis.png")

print(f"  Focus point:  ({u0}, {v0})")
print(f"  Focus depth:  {focus_depth:.3f}")
print(f"  CoC range:    {coc.min():.2f} — {coc.max():.2f} px")


# Stage 3: depth-dependent defocus rendering

print("\n[Stage 3] Rendering depth-dependent defocus...")
subprocess.run([
    sys.executable, "defocus_render.py", args.image,
    "--num-levels", str(args.num_levels),
    "--sigma-scale", str(args.sigma_scale),
    "--coc-smooth", str(args.coc_smooth),
], check=True)


# Stage 4: miniature color grading

print("\n[Stage 4] Applying miniature color grading...")
subprocess.run([
    sys.executable, "color_grade.py", args.image,
    "--saturation", str(args.saturation),
    "--contrast", str(args.contrast),
    "--warmth", str(args.warmth),
], check=True)


print(f"\nDone. Final output: ../stage4/{stem}-final.png")