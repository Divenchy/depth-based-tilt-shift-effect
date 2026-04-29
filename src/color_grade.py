import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


def boost_saturation(rgb, factor):
    if factor == 1.0:
        return rgb
    hsv = rgb_to_hsv(rgb)
    hsv[..., 1] = np.clip(hsv[..., 1] * factor, 0.0, 1.0)
    return hsv_to_rgb(hsv)


def apply_s_curve(rgb, strength):
    if strength == 0.0:
        return rgb
    hsv = rgb_to_hsv(rgb)
    v = hsv[..., 2]
    smoothstep = 3 * v ** 2 - 2 * v ** 3
    hsv[..., 2] = (1 - strength) * v + strength * smoothstep
    return hsv_to_rgb(hsv)


def apply_warmth(rgb, warmth):
    if warmth == 0.0:
        return rgb
    out = rgb.copy()
    out[..., 0] *= 1.0 + 0.15 * warmth
    out[..., 2] *= 1.0 - 0.15 * warmth
    return np.clip(out, 0.0, 1.0)


def grade_miniature(rgb, saturation=1.4, contrast=0.3, warmth=0.15):
    out = boost_saturation(rgb, saturation)
    out = apply_s_curve(out, contrast)
    out = apply_warmth(out, warmth)
    return out


def main():
    parser = argparse.ArgumentParser(description="Stage 4: miniature color grading.")
    parser.add_argument("image", help="Image filename; loads ../stage3/<stem>-defocused.png")
    parser.add_argument("--saturation", type=float, default=1.4)
    parser.add_argument("--contrast", type=float, default=0.3)
    parser.add_argument("--warmth", type=float, default=0.15)
    args = parser.parse_args()

    stem = Path(args.image).stem

    img = np.array(Image.open(Path("../stage3") / f"{stem}-defocused.png").convert("RGB"),
                   dtype=np.float32) / 255.0

    print(f"Image shape:  {img.shape}")
    print(f"Saturation:   {args.saturation}")
    print(f"Contrast:     {args.contrast}")
    print(f"Warmth:       {args.warmth}")

    output = grade_miniature(
        img,
        saturation=args.saturation,
        contrast=args.contrast,
        warmth=args.warmth,
    )

    out_dir = Path("../stage4")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{stem}-final.png"
    Image.fromarray(np.clip(output * 255, 0, 255).astype(np.uint8)).save(out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()