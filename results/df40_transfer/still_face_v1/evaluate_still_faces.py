"""Evaluate saved GenImage classifiers on a fixed, hash-verified image manifest."""
import csv
import json
import sys
from pathlib import Path

import torch
import evaluate_shared as ev


def read_rows(root):
    rows = ev.vit.read_csv(root / "manifest.csv")
    if len(rows) != 150 or len({r["sha256"] for r in rows}) != 150:
        raise ValueError("Expected 150 unique image hashes")
    if {r["method"] for r in rows} != {"real", "insight", "inpainting"}:
        raise ValueError("Unexpected groups")
    for row in rows:
        row["image_id"] = row["sha256"]
        row["path"] = str(root / row["path"])
    return rows


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    rows = read_rows(root)
    smoke = [i for i, row in enumerate(rows) if row["smoke"] == "1"]
    if len(smoke) != 18:
        raise ValueError("Expected 18 smoke images")
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    report = {"images": len(rows), "smoke_images": len(smoke), "training": False,
              "smoke_in_full_set": True, "threshold": 0.5,
              "manifest_sha256": ev.vit.digest(root / "manifest.csv"),
              "gpu": torch.cuda.get_device_name(0), "models": {}}
    output = []
    with torch.inference_mode():
        for mode in ("baseline", "augmented"):
            scores, checks = ev.cnn_run(mode, rows, smoke)
            report["models"]["cnn_" + mode] = checks
            output.extend({**r, "model": "cnn", "mode": mode, "prob_fake": p}
                          for r, p in zip(rows, scores))
        scores, checks = ev.vit_run(rows, smoke)
        report["models"]["vit"] = checks
        output.extend({**r, "model": "vit"} for r in scores)
    if len(output) != 4 * len(rows):
        raise ValueError("Incomplete prediction count")
    ev.vit.write_csv(out / "predictions.csv", output)
    report["status"] = "passed"
    (out / "ACCEPTANCE.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
