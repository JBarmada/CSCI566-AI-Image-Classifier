"""Compare frontier models, CNN, and ViT on the same 90-image pilot."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


MODES = ("baseline", "augmented")


def valid_score(value):
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Score must be finite and between zero and one")
    return score


def index_predictions(rows):
    indexed = {row["sha256"]: row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("Duplicate prediction hashes")
    return indexed


def read_csv(path):
    if path.suffix == ".gz":
        stream = gzip.open(path, "rt", newline="", encoding="utf-8")
    else:
        stream = path.open(newline="", encoding="utf-8")
    with stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def auc(labels, scores):
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires both classes")
    ordered = sorted(zip(scores, labels))
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        rank_sum += (((index + 1) + end) / 2) * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def metrics(rows, model, holdout="all"):
    labels = [int(row["label"]) for row in rows]
    scores = [valid_score(row[model]) for row in rows]
    predictions = [int(score >= 0.5) for score in scores]
    tp = sum(label == prediction == 1 for label, prediction in zip(labels, predictions))
    tn = sum(label == prediction == 0 for label, prediction in zip(labels, predictions))
    fp = sum(label == 0 and prediction == 1 for label, prediction in zip(labels, predictions))
    fn = sum(label == 1 and prediction == 0 for label, prediction in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "model": model,
        "holdout": holdout,
        "rows": str(len(rows)),
        "accuracy": f"{(tp + tn) / len(rows):.10g}",
        "precision": f"{precision:.10g}",
        "fake_recall": f"{recall:.10g}",
        "specificity": f"{specificity:.10g}",
        "f1": f"{(2 * precision * recall / (precision + recall) if precision + recall else 0):.10g}",
        "balanced_accuracy": f"{(recall + specificity) / 2:.10g}",
        "auroc": f"{auc(labels, scores):.10g}",
        "tn": str(tn),
        "fp": str(fp),
        "fn": str(fn),
        "tp": str(tp),
    }


def frontier_scores(path, expected, model=None):
    rows = read_csv(path)
    successful = [row for row in rows if row["status"] == "success"]
    scores = {row["sha256"]: valid_score(float(row["p_fake"]) / 100) for row in successful}
    if len(scores) != len(successful) or set(scores) != expected:
        raise ValueError(f"Frontier predictions do not match pilot: {path}")
    if len(rows) != len(successful) or any(row.get("error") for row in rows):
        raise ValueError(f"Unresolved frontier failures: {path}")
    if model and any(row.get("model") != model for row in rows):
        raise ValueError(f"Frontier model mismatch: {path}")
    return scores


def vit_scores(path, holdout, pilot_labels):
    rows = [
        row
        for row in read_csv(path)
        if row["split"] == "test"
        and row["k"] == "7"
        and row["rep"] == "0"
        and row["seed"] == "17"
        and row.get("condition") == "clean"
    ]
    output = {}
    for mode in MODES:
        selected = index_predictions([row for row in rows if row["mode"] == mode])
        missing = set(pilot_labels) - set(selected)
        if missing:
            raise ValueError(f"ViT {holdout} {mode} is missing {len(missing)} pilot images")
        if any(int(selected[sha]["label"]) != label for sha, label in pilot_labels.items()):
            raise ValueError(f"ViT label mismatch: {holdout} {mode}")
        output[mode] = {sha: valid_score(selected[sha]["prob_fake"]) for sha in pilot_labels}
    return output


def cnn_scores(root, holdout, pilot_labels):
    output = {}
    for mode in MODES:
        name = f"holdout_{holdout}_k7_rep0__{mode}__seed17"
        path = root / name / "test_predictions.csv"
        rows = index_predictions(read_csv(path))
        missing = set(pilot_labels) - set(rows)
        if missing:
            raise ValueError(f"CNN {holdout} {mode} is missing {len(missing)} pilot images")
        if any(int(rows[sha]["label"]) != label for sha, label in pilot_labels.items()):
            raise ValueError(f"CNN label mismatch: {holdout} {mode}")
        output[mode] = {sha: valid_score(rows[sha]["probability_fake"]) for sha in pilot_labels}
    return output


def build_rows(task, gemini, groq, vit_root, cnn_root):
    by_holdout = defaultdict(list)
    for row in task:
        by_holdout[row["heldout_generator"]].append(row)
    output = []
    for holdout in sorted(by_holdout):
        pilot = by_holdout[holdout]
        if len(pilot) != 20 or sum(row["label"] == "1" for row in pilot) != 10:
            raise ValueError(f"Unexpected pilot composition for {holdout}")
        labels = {row["sha256"]: int(row["label"]) for row in pilot}
        if len(labels) != 20 or set(labels.values()) != {0, 1}:
            raise ValueError(f"Duplicate identities or invalid labels for {holdout}")
        vit = vit_scores(vit_root / f"{holdout}.csv.gz", holdout, labels)
        cnn = cnn_scores(cnn_root, holdout, labels)
        for row in pilot:
            sha = row["sha256"]
            output.append(
                {
                    "heldout_generator": holdout,
                    "sha256": sha,
                    "path": row["path"],
                    "label": row["label"],
                    "gemini": f"{gemini[sha]:.10g}",
                    "groq_qwen": f"{groq[sha]:.10g}",
                    "cnn_baseline": f"{cnn['baseline'][sha]:.10g}",
                    "cnn_augmented": f"{cnn['augmented'][sha]:.10g}",
                    "vit_baseline": f"{vit['baseline'][sha]:.10g}",
                    "vit_augmented": f"{vit['augmented'][sha]:.10g}",
                }
            )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("results/llm_eval"))
    parser.add_argument("--vit-predictions", type=Path, default=Path("results/vit_benchmark/predictions"))
    parser.add_argument(
        "--cnn-runs", type=Path,
        default=Path("/project2/yzhao010_1531/csci566-genimage/results/cnn_benchmark/notebook/runs")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/llm_eval/matched_comparison")
    )
    args = parser.parse_args()

    task_path = args.root / "pilot_task_manifest.csv"
    unique_path = args.root / "pilot_unique_api_manifest.csv"
    gemini_path = args.root / "gemini_predictions.csv"
    groq_path = args.root / "groq_predictions.csv"
    protocol_path = args.root / "protocol.json"
    task = read_csv(task_path)
    unique = read_csv(unique_path)
    expected = {row["sha256"] for row in unique}
    if len(task) != 160 or len(unique) != 90 or len(expected) != 90:
        raise ValueError("Expected the frozen 160-row / 90-image pilot")
    labels = {row["sha256"]: row["label"] for row in unique}
    if ({row["sha256"] for row in task} != expected
            or len({row["heldout_generator"] for row in task}) != 8
            or any(row["label"] not in {"0", "1"} for row in unique)
            or any(row["label"] != labels[row["sha256"]] for row in task)):
        raise ValueError("Task and unique manifests disagree")
    protocol = json.loads(protocol_path.read_text())
    models_by_provider = {item["provider"]: item["model"] for item in protocol["providers"]}

    rows = build_rows(
        task,
        frontier_scores(gemini_path, expected, models_by_provider["Google Gemini API"]),
        frontier_scores(groq_path, expected, models_by_provider["Groq API"]),
        args.vit_predictions,
        args.cnn_runs,
    )
    models = (
        "gemini",
        "groq_qwen",
        "cnn_baseline",
        "cnn_augmented",
        "vit_baseline",
        "vit_augmented",
    )
    overall = [metrics(rows, model) for model in models]
    per_holdout = []
    holdouts = sorted({row["heldout_generator"] for row in rows})
    for holdout in holdouts:
        selected = [row for row in rows if row["heldout_generator"] == holdout]
        per_holdout.extend(metrics(selected, model, holdout) for model in models)

    write_csv(args.output_dir / "matched_predictions.csv", rows)
    write_csv(args.output_dir / "overall_metrics.csv", overall)
    write_csv(args.output_dir / "per_holdout_metrics.csv", per_holdout)
    audit = {
        "status": "passed",
        "evaluation_rows": len(rows),
        "unique_images": len(expected),
        "models": list(models),
        "task_manifest_sha256": digest(task_path),
        "unique_manifest_sha256": digest(unique_path),
        "gemini_predictions_sha256": digest(gemini_path),
        "groq_predictions_sha256": digest(groq_path),
        "protocol_sha256": digest(protocol_path),
        "vit_prediction_sha256": {
            holdout: digest(args.vit_predictions / f"{holdout}.csv.gz") for holdout in holdouts
        },
        "cnn_prediction_sha256": {
            f"holdout_{holdout}_k7_rep0__{mode}__seed17": digest(
                args.cnn_runs
                / f"holdout_{holdout}_k7_rep0__{mode}__seed17"
                / "test_predictions.csv"
            )
            for holdout in holdouts
            for mode in MODES
        },
        "audited_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print("Verified 160 matched rows across Gemini, Groq-Qwen, CNN, and ViT")


if __name__ == "__main__":
    main()
