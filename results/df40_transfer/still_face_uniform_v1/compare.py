"""Compare pre-registered score rows before and after uniform processing."""
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent


def load(directory):
    manifest = directory / "manifest.csv"
    report = json.loads((directory / "ACCEPTANCE.json").read_text())
    if report["status"] != "passed" or report["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest():
        raise ValueError("Acceptance or manifest mismatch")
    with manifest.open(newline="", encoding="utf-8") as f:
        labels = {(r["pair_id"], r["method"]): r for r in csv.DictReader(f)}
    with (directory / "predictions.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result = {}
    for row in rows:
        key = (row["model"], row["mode"], row["method"], row["pair_id"])
        expected = labels[(row["pair_id"], row["method"])]
        if key in result or row["sha256"] != expected["sha256"] or row["label"] != expected["label"]:
            raise ValueError("Duplicate or mismatched prediction")
        result[key] = float(row["prob_fake"])
    if len(result) != 600:
        raise ValueError("Incomplete predictions")
    return result


def main():
    original = load(ROOT.parent / "still_face_v1")
    uniform = load(ROOT)
    if set(original) != set(uniform):
        raise ValueError("Samples differ between conditions")
    results = []
    for model in ("cnn", "vit"):
        for mode in ("baseline", "augmented"):
            for group in ("real", "insight", "inpainting"):
                keys = [k for k in original if k[:3] == (model, mode, group)]
                delta = [uniform[k] - original[k] for k in keys]
                results.append({"model": model, "mode": mode, "group": group,
                                "n": len(keys), "original_positive": sum(original[k] >= 0.5 for k in keys),
                                "uniform_positive": sum(uniform[k] >= 0.5 for k in keys),
                                "mean_score_change": sum(delta) / len(delta),
                                "changed_decisions": sum((original[k] >= 0.5) != (uniform[k] >= 0.5) for k in keys)})
    with (ROOT / "paired_changes.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
