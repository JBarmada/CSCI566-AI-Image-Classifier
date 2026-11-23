"""Make one uniform center-crop/resize/JPEG copy of every manifest image."""
import csv
import hashlib
import sys
from pathlib import Path

from PIL import Image, ImageOps


def main():
    source, dest = map(Path, sys.argv[1:])
    if dest.exists():
        raise FileExistsError(dest)
    with (source / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 150:
        raise ValueError("Expected 150 input images")
    for row in rows:
        path = source / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Input hash mismatch: {path}")
        target = dest / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            output = ImageOps.fit(image.convert("RGB"), (512, 512),
                                  method=Image.Resampling.BICUBIC, centering=(0.5, 0.5))
            output.save(target, format="JPEG", quality=95, subsampling=0)
        row["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    with (dest / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} center-cropped and resized JPEG images to {dest}")


if __name__ == "__main__":
    main()
