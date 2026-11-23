"""Sample matching still-image paths from three pinned public ZIP archives."""
import concurrent.futures
import csv
import hashlib
import json
import random
import sys
import zipfile
from pathlib import Path

from inspect_remote_zip import RemoteZip

REV = "eb2a54fbf2567790bd3dc61d09fdaade6bb6c31c"
BASE = f"https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/{REV}"
GROUPS = ("wiki", "insight", "inpainting")
ROOT = Path(sys.argv[1])


def open_archive(group):
    remote = RemoteZip(f"{BASE}/{group}.zip")
    archive = zipfile.ZipFile(remote)
    names = {x.filename.split("/", 1)[1]: x.filename for x in archive.infolist()
             if not x.is_dir() and "/" in x.filename}
    return remote, archive, names


def extract_group(group, bundle, keys):
    remote, archive, names = bundle
    rows = []
    for index, key in enumerate(keys):
        data = archive.read(names[key])
        digest = hashlib.sha256(data).hexdigest()
        target = ROOT / group / f"{index:03d}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append({"path": target.relative_to(ROOT).as_posix(), "sha256": digest,
                     "method": "real" if group == "wiki" else group,
                     "label": 0 if group == "wiki" else 1,
                     "source_path": names[key], "pair_id": f"{index:03d}",
                     "smoke": int(index < 6)})
    return rows, remote.bytes_read


def main():
    if ROOT.exists():
        raise FileExistsError(ROOT)
    bundles = {group: open_archive(group) for group in GROUPS}
    common = set.intersection(*(set(bundle[2]) for bundle in bundles.values()))
    keys = random.Random(5662026).sample(sorted(common), 50)
    ROOT.mkdir(parents=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {g: pool.submit(extract_group, g, bundles[g], keys) for g in GROUPS}
        results = {g: futures[g].result() for g in GROUPS}
    rows = [row for g in GROUPS for row in results[g][0]]
    with (ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    info = {"revision": REV, "seed": 5662026, "common_paths": len(common),
            "sampled_paths": len(keys), "smoke_per_group": 6,
            "bytes_transferred": {g: results[g][1] for g in GROUPS},
            "selection": "uniform sample of sorted common relative paths"}
    (ROOT / "provenance.json").write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info, indent=2), flush=True)


if __name__ == "__main__":
    main()
