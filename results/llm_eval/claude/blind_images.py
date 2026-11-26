import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
SEED = 566


VIEWS = ("original", "square", "detector224")


def make_view(image: Image.Image, view: str) -> Image.Image:
    if view == "original":
        return image
    if view == "detector224":
        w, h = image.size
        scale = 224 / min(w, h)
        image = image.resize((max(224, round(w * scale)), max(224, round(h * scale))), Image.Resampling.BICUBIC)
    w, h = image.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    out = image.crop((left, top, left + side, top + side))
    if view == "detector224" and out.size != (224, 224):
        out = out.resize((224, 224), Image.Resampling.BICUBIC)
    return out


def pixel_hash(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Copy the shared images under neutral names, with the answer key kept separately.")
    parser.add_argument("--image-root", type=Path, required=True, help="Folder the manifest paths are relative to")
    parser.add_argument("--images", type=Path, default=HERE / "shared_image_list.csv")
    parser.add_argument("--blind-dir", type=Path, required=True, help="New, empty folder for Claude to read")
    parser.add_argument("--key", type=Path, default=None)
    parser.add_argument("--view", choices=VIEWS, default="original",
                        help="original, square (center crop), or detector224 (bicubic resize + 224 center crop like the CNN/ViT)")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if args.key is None:
        args.key = HERE / ("answer_key.csv" if args.view == "original" else f"answer_key_{args.view}.csv")
    images = pd.read_csv(args.images)
    if args.blind_dir.exists() and any(args.blind_dir.iterdir()):
        raise SystemExit(f"{args.blind_dir} is not empty; pick a new folder")
    if args.key.resolve().is_relative_to(args.blind_dir.resolve()):
        raise SystemExit("The answer key must not be inside the blind folder")
    missing = [p for p in images.path if not (args.image_root / p).is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} images not found under {args.image_root}, e.g. {missing[0]}")

    args.blind_dir.mkdir(parents=True, exist_ok=True)
    order = images.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    rows = []
    for i, row in order.iterrows():
        blind_id = f"img_{i + 1:04d}"
        with Image.open(args.image_root / row.path) as src:
            rgb = make_view(src.convert("RGB"), args.view)
            clean = Image.frombytes("RGB", rgb.size, rgb.tobytes())
        target = args.blind_dir / f"{blind_id}.png"
        clean.save(target, format="PNG")
        with Image.open(target) as check:
            if pixel_hash(check) != pixel_hash(clean):
                raise SystemExit(f"Pixel mismatch after saving {target}")
        rows.append({"blind_id": blind_id, **row.to_dict(), "view": args.view,
                     "width": clean.width, "height": clean.height})

    key = pd.DataFrame(rows)
    key.to_csv(args.key, index=False)
    print(f"Wrote {len(key)} images to {args.blind_dir} and the answer key to {args.key}")


if __name__ == "__main__":
    main()
