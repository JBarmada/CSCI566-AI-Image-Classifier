"""Verify shared prediction coverage and summarize fixed-threshold scores."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def metrics(real, fake):
    rs = [float(r["prob_fake"]) for r in real]
    fs = [float(r["prob_fake"]) for r in fake]
    fp = sum(x >= 0.5 for x in rs)
    tp = sum(x >= 0.5 for x in fs)
    auc = sum((f > r) + 0.5 * (f == r) for f in fs for r in rs) / (len(rs) * len(fs))
    return {"real_n": len(rs), "fake_n": len(fs), "false_positive": fp,
            "true_positive": tp, "real_fpr": fp / len(rs),
            "fake_recall": tp / len(fs), "balanced_accuracy": ((1 - fp / len(rs)) + tp / len(fs)) / 2,
            "auroc": auc}


def main():
    manifest = read_csv(ROOT / "manifest.csv")
    rows = read_csv(ROOT / "predictions.csv")
    with (ROOT / "ACCEPTANCE.json").open(encoding="utf-8") as f:
        acceptance = json.load(f)
    if acceptance["status"] != "passed" or len(rows) != 600 or len(manifest) != 150:
        raise ValueError("Acceptance or row count failed")
    manifest_hash = hashlib.sha256((ROOT / "manifest.csv").read_bytes()).hexdigest()
    if manifest_hash != acceptance["manifest_sha256"]:
        raise ValueError("Manifest differs from evaluated bytes")
    expected = {(r["sha256"], r["method"], r["label"], r["pair_id"]) for r in manifest}
    if len(expected) != 150:
        raise ValueError("Duplicate manifest rows")
    results = []
    for model in ("cnn", "vit"):
        for mode in ("baseline", "augmented"):
            subset = [r for r in rows if r["model"] == model and r["mode"] == mode]
            actual = {(r["sha256"], r["method"], r["label"], r["pair_id"]) for r in subset}
            if len(subset) != 150 or actual != expected:
                raise ValueError("Model predictions do not match manifest")
            real = [r for r in subset if r["method"] == "real"]
            for method in ("insight", "inpainting"):
                fake = [r for r in subset if r["method"] == method]
                if len(real) != 50 or len(fake) != 50:
                    raise ValueError("Incorrect group count")
                results.append({"model": model, "mode": mode, "method": method,
                                **metrics(real, fake)})
    with (ROOT / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
