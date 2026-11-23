"""Count available camera EXIF fields without disclosing their contents."""
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


def main():
    root = Path(sys.argv[1])
    counts = {group: Counter() for group in ("real", "insight", "inpainting")}
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        path = root / row["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {path}")
        with Image.open(path) as image:
            exif = image.getexif()
            counter = counts[row["method"]]
            counter["images"] += 1
            counter["exif"] += bool(exif)
            counter["camera_make"] += bool(exif.get(271))
            counter["camera_model"] += bool(exif.get(272))
    for group, counter in counts.items():
        print(group, dict(counter))


if __name__ == "__main__":
    main()
