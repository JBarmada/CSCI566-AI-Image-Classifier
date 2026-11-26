"""Audit a complete Gemini face run and report per-method metrics."""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_eval_common import validate_predictions


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def auc(labels, scores):
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return None
    return sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives) / (
        len(positives) * len(negatives)
    )


def metrics(rows, group):
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    scores = [float(row["p_manipulated"]) / 100 for row in rows]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "group": group,
        "images": len(rows),
        "real_images": tn + fp,
        "fake_images": tp + fn,
        "accuracy": (tp + tn) / len(rows),
        "balanced_accuracy": (recall + specificity) / 2,
        "precision": precision,
        "fake_recall": recall,
        "real_false_positive_rate": 1 - specificity,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "auroc": auc(labels, scores),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("results/df40_transfer/gemini_face/protocol.json"),
    )
    args = parser.parse_args()

    manifest = read_csv(args.manifest)
    predictions = read_csv(args.predictions)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    successful = [row for row in predictions if row["status"] == "ok"]
    expected = {row["sha256"]: row for row in manifest}
    actual = {row["sha256"]: row for row in successful}
    failures = []
    try:
        validate_predictions(manifest, predictions, args.dataset, protocol)
    except (ValueError, KeyError, TypeError) as exc:
        failures.append(str(exc))
    if not manifest or len(expected) != len(manifest):
        failures.append("manifest is empty or contains duplicate hashes")
    if len(actual) != len(successful):
        failures.append("successful predictions contain duplicate hashes")
    if set(actual) != set(expected):
        failures.append("successful prediction hashes do not exactly match the manifest")
    for sha256 in set(actual) & set(expected):
        prediction = actual[sha256]
        source = expected[sha256]
        if prediction["dataset"] != args.dataset:
            failures.append(f"dataset mismatch for {sha256}")
        if prediction["label"] != source["label"] or prediction["method"] != source["method"]:
            failures.append(f"label or method mismatch for {sha256}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "dataset": args.dataset,
        "expected_images": len(manifest),
        "successful_predictions": len(successful),
        "manifest_sha256": digest(args.manifest),
        "predictions_sha256": digest(args.predictions),
        "protocol_sha256": digest(args.protocol),
        "audited_utc": datetime.now(timezone.utc).isoformat(),
    }
    if failures:
        (args.output_dir / "AUDIT.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(audit, indent=2))
        raise SystemExit(2)

    rows = [actual[row["sha256"]] for row in manifest]
    audit["status"] = "incomplete"
    (args.output_dir / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    reports = [metrics(rows, "all_images")]
    real = [row for row in rows if row["label"] == "0"]
    methods = sorted({row["method"] for row in rows if row["label"] == "1"})
    for method in methods:
        fake = [row for row in rows if row["label"] == "1" and row["method"] == method]
        reports.append(metrics(real + fake, method))

    with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=reports[0])
        writer.writeheader()
        writer.writerows(reports)
    events_path = args.predictions.parent / "responses.jsonl"
    if events_path.exists():
        attempts = [
            usage
            for line in events_path.read_text().splitlines() if line.strip()
            for usage in json.loads(json.loads(line).get("usage_json", "[]"))
        ]
        audit["api_usage"] = {
            "scope": "All recorded responses in responses.jsonl, including failed attempts",
            "responses": len(attempts),
            "input_tokens": sum((item.get("usage") or {}).get("input_tokens", 0) for item in attempts),
            "output_tokens": sum((item.get("usage") or {}).get("output_tokens", 0) for item in attempts),
            "missing_usage_responses": sum(item.get("usage") is None for item in attempts),
        }
    audit["status"] = "passed"
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
