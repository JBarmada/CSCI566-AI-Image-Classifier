"""Create one controlled post-processing variant of a fixed image manifest."""
import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageOps


def main():
    source, dest, kind = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    if dest.exists() or kind not in {"jpeg95", "geometry"}:
        raise ValueError("Destination must be new and kind must be jpeg95 or geometry")
    with (source / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 150:
        raise ValueError("Expected 150 inputs")
    for row in rows:
        path = source / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Input hash mismatch: {path}")
        relative = Path(row["path"]).with_suffix(".png" if kind == "geometry" else ".jpg")
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            if kind == "geometry":
                ImageOps.fit(rgb, (512, 512), method=Image.Resampling.BICUBIC).save(target, format="PNG")
            else:
                rgb.save(target, format="JPEG", quality=95, subsampling=0)
        row["path"] = relative.as_posix()
        row["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    with (dest / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Created {len(rows)} {kind} images")


if __name__ == "__main__":
    main()
