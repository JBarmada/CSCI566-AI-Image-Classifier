"""Compare selected image hashes with a GenImage inventory, without exporting images."""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image


def main():
    root, inventory_path = map(Path, sys.argv[1:])
    with inventory_path.open(newline="", encoding="utf-8") as f:
        inventory = list(csv.DictReader(f))
    fields = ("sha256", "source_sha256", "pixel_sha256")
    hashes = {field: {r[field] for r in inventory if r.get(field)} for field in fields}
    counts = {group: Counter() for group in ("real", "insight", "inpainting")}
    with (root / "manifest.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        path = root / row["path"]
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        if file_hash != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {path}")
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        c = counts[row["method"]]
        c["n"] += 1
        c["file_sha256_matches_inventory_file"] += file_hash in hashes["sha256"]
        c["file_sha256_matches_inventory_source"] += file_hash in hashes["source_sha256"]
        c["pixel_sha256_matches_inventory_pixels"] += pixel_hash in hashes["pixel_sha256"]
    report = {"inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
              "inventory_rows": len(inventory), "groups": counts}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
