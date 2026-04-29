import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def render_defocus(image, coc, num_levels=8, sigma_scale=1.0, coc_smooth=2.0):
    if coc_smooth > 0:
        coc = gaussian_filter(coc, sigma=coc_smooth)

    sigma_map = coc * sigma_scale
    sigma_max = float(sigma_map.max())

    if sigma_max < 0.5:
        return image.copy()

    sigmas = np.linspace(0.0, sigma_max, num_levels)

    blurred_stack = np.empty((num_levels,) + image.shape, dtype=np.float32)
    blurred_stack[0] = image
    for i in range(1, num_levels):
        blurred_stack[i] = gaussian_filter(image, sigma=(sigmas[i], sigmas[i], 0))

    level_idx = (sigma_map / sigma_max) * (num_levels - 1)
    lo = np.clip(np.floor(level_idx).astype(np.int32), 0, num_levels - 2)
    hi = lo + 1
    weight = (level_idx - lo).astype(np.float32)

    h, w = coc.shape
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    lower = blurred_stack[lo, ii, jj]
    upper = blurred_stack[hi, ii, jj]

    return (1.0 - weight[..., None]) * lower + weight[..., None] * upper


def main():
    parser = argparse.ArgumentParser(description="Stage 3: depth-dependent defocus rendering.")
    parser.add_argument("image", help="Image filename, looked up in ../resources/")
    parser.add_argument("--num-levels", type=int, default=8)
    parser.add_argument("--sigma-scale", type=float, default=1.0)
    parser.add_argument("--coc-smooth", type=float, default=2.0)
    args = parser.parse_args()

    stem = Path(args.image).stem

    img = np.array(Image.open(Path("../resources") / args.image).convert("RGB"),
                   dtype=np.float32) / 255.0
    coc = np.load(Path("../stage2_3") / f"{stem}-coc.npy").astype(np.float32)

    print(f"Image shape:    {img.shape}")
    print(f"CoC range:      {coc.min():.2f} — {coc.max():.2f} px")
    print(f"Building {args.num_levels} blur levels...")

    output = render_defocus(
        img, coc,
        num_levels=args.num_levels,
        sigma_scale=args.sigma_scale,
        coc_smooth=args.coc_smooth,
    )

    out_dir = Path("../stage3")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{stem}-defocused.png"
    Image.fromarray(np.clip(output * 255, 0, 255).astype(np.uint8)).save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()