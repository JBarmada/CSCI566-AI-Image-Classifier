"""Audit Gemini outputs and report unique-image and per-holdout metrics."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(rows, group):
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    scores = [float(row["p_fake"]) / 100 for row in rows]
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "group": group,
        "images": str(len(rows)),
        "accuracy": f"{(tp + tn) / len(rows):.10g}",
        "precision": f"{precision:.10g}",
        "fake_recall": f"{recall:.10g}",
        "specificity": f"{specificity:.10g}",
        "real_false_positive_rate": f"{1 - specificity:.10g}",
        "f1": f"{(2 * precision * recall / (precision + recall) if precision + recall else 0):.10g}",
        "balanced_accuracy": f"{(recall + specificity) / 2:.10g}",
        "auroc": f"{auc(labels, scores):.10g}",
        "tp": str(tp),
        "tn": str(tn),
        "fp": str(fp),
        "fn": str(fn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unique-manifest",
        type=Path,
        default=Path("results/llm_eval/pilot_unique_api_manifest.csv"),
    )
    parser.add_argument(
        "--combined-manifest",
        type=Path,
        default=Path("results/llm_eval/pilot_task_manifest.csv"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("results/llm_eval/gemini_predictions.csv"),
    )
    parser.add_argument("--protocol", type=Path, default=Path("results/llm_eval/protocol.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/llm_eval/gemini_pilot/summary"))
    args = parser.parse_args()

    unique = read_csv(args.unique_manifest)
    combined = read_csv(args.combined_manifest)
    predictions = read_csv(args.predictions)
    successful = [row for row in predictions if row["status"] in {"ok", "success"}]
    prediction_hashes = [row["sha256"] for row in successful]
    expected_hashes = {row["sha256"] for row in unique}
    failures = []
    if not unique or len(unique) != len(expected_hashes):
        failures.append("unique manifest is empty or contains duplicate hashes")
    if not combined:
        failures.append("task manifest is empty")
    if len(prediction_hashes) != len(unique) or set(prediction_hashes) != expected_hashes:
        failures.append("successful predictions do not exactly match the unique manifest")
    if len(prediction_hashes) != len(set(prediction_hashes)):
        failures.append("duplicate successful prediction hashes")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit = {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "unique_images": len(unique),
        "evaluation_rows": len(combined),
        "successful_predictions": len(successful),
        "prediction_file_sha256": digest(args.predictions),
        "unique_manifest_sha256": digest(args.unique_manifest),
        "combined_manifest_sha256": digest(args.combined_manifest),
        "protocol_sha256": digest(args.protocol),
        "audited_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    if failures:
        print(json.dumps(audit, indent=2))
        raise SystemExit(2)

    by_hash = {row["sha256"]: row for row in successful}

    def scored(manifest_row):
        prediction = by_hash[manifest_row["sha256"]]
        score = float(prediction["p_fake"])
        return {
            "label": manifest_row["label"],
            "prediction": str(int(score >= 50)),
            "p_fake": str(score),
        }

    unique_rows = [scored(row) for row in unique]
    reports = [metrics(unique_rows, "unique_images")]
    holdouts = defaultdict(list)
    for row in combined:
        holdout = row.get("heldout_generator") or row.get("holdout_generator")
        if not holdout:
            raise ValueError("Task manifest is missing a holdout column")
        holdouts[holdout].append(scored(row))
    for holdout in sorted(holdouts):
        reports.append(metrics(holdouts[holdout], holdout))

    fields = tuple(reports[0])
    write_csv(args.output_dir / "metrics.csv", reports, fields)
    print(json.dumps(audit, indent=2))
    print("Wrote unique-image and equally weighted per-holdout metrics")


if __name__ == "__main__":
    main()
