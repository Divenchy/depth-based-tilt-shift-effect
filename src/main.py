import sys
import subprocess
import argparse
from pathlib import Path
from PIL import Image
import numpy as np

'''
Arg parsing
'''

parser = argparse.ArgumentParser()
parser.add_argument("image", help="Input image filename, e.g. city_street.jpg")
args = parser.parse_args()
out_dir = Path("../stage2_3")
out_dir.mkdir(exist_ok=True)

# Strip file extension
stem = Path(args.image).stem


'''
Depth Map Generation
Utilizing Depth Anything V2
'''

subprocess.run([sys.executable, "depth_map.py", args.image], check=True)


'''
Tilted Focal Plane
'''
depth_img = Image.open(f"../stage1/{stem}-depth-map.png").convert("L")
depth = np.array(depth_img, dtype=np.float32) / 255.0

H, W = depth.shape

u0, v0 = W // 2, H // 2
focus_depth = depth[v0, u0]

tilt_x = 0.0    # depth units per pixel, x-axis
tilt_y = 0.002  # depth units per pixel, y-axis

uu, vv = np.meshgrid(np.arange(W, dtype=np.float32),
                     np.arange(H, dtype=np.float32))

focal_plane = focus_depth + tilt_x * (uu - u0) + tilt_y * (vv - v0)
focal_plane = np.clip(focal_plane, 0.0, 1.0)

# Positive = farther than focal plane, negative = closer
signed_dist = depth - focal_plane


'''
Circle of Confusion
'''
# coc_scale: blur pixels per unit of depth deviation — main tuning knob
# max_coc: hard cap on blur radius (pixels), tells your teammate the range
coc_scale = 50.0
max_coc   = 15.0
coc = np.clip(np.abs(signed_dist) * coc_scale, 0.0, max_coc)

'''
Save Tilted Plane and CoC results
'''
np.save(out_dir / f"{stem}-coc.npy", coc)

coc_vis = (coc / max_coc * 255).astype(np.uint8)
Image.fromarray(coc_vis).save(out_dir / f"{stem}-coc-vis.png")

fp_vis = (focal_plane * 255).astype(np.uint8)
Image.fromarray(fp_vis).save(out_dir / f"{stem}-focal-plane-vis.png")

print(f"Depth at focus center ({u0}, {v0}): {focus_depth:.3f}")
print(f"CoC range: {coc.min():.2f} — {coc.max():.2f} px")
