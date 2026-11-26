"""Compare Gemini with saved models on the matched 189-image DF40 set."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from summarize import metrics, validate_predictions
from face_eval_common import probability


EXPECTED_IMAGES = 189
REFERENCE_MODELS = (
    ("cnn", "baseline"),
    ("cnn", "augmented"),
    ("vit", "baseline"),
    ("vit", "augmented"),
    ("vit", "face_adapted"),
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def normalized_rows(manifest, gemini, reference, api_label="gemini"):
    expected = {row["sha256"]: row for row in manifest}
    gemini_ok = [row for row in gemini if row["status"] == "ok"]
    if len(expected) != EXPECTED_IMAGES or len(gemini_ok) != EXPECTED_IMAGES:
        raise ValueError("Expected 189 unique manifest rows and successful Gemini rows")
    if {row["sha256"] for row in gemini_ok} != set(expected):
        raise ValueError("Gemini predictions do not exactly match the manifest")

    sources = [(api_label, "zero_shot", gemini_ok, "p_manipulated", 100.0)]
    for model, mode in REFERENCE_MODELS:
        rows = [row for row in reference if row["model"] == model and row["mode"] == mode]
        sources.append((model, mode, rows, "prob_fake", 1.0))

    outputs = []
    for model, mode, rows, score_field, divisor in sources:
        if len(rows) != EXPECTED_IMAGES or {row["sha256"] for row in rows} != set(expected):
            raise ValueError(f"{model}/{mode} does not exactly match the manifest")
        for row in rows:
            source = expected[row["sha256"]]
            if row["label"] != source["label"] or row["method"] != source["method"]:
                raise ValueError(f"Label or method mismatch for {model}/{mode}")
            score = float(row[score_field]) / divisor
            probability(score * 100)
            outputs.append({
                "sha256": row["sha256"],
                "path": source["path"],
                "pair": source["pair"],
                "video_group": source["video_group"],
                "method": source["method"],
                "label": source["label"],
                "model": model,
                "mode": mode,
                "score": f"{score:.10g}",
                "prediction": str(int(score >= 0.5)),
            })
    return outputs


def metric_rows(rows, methods, api_label="gemini"):
    reports = []
    model_modes = ((api_label, "zero_shot"),) + REFERENCE_MODELS
    for model, mode in model_modes:
        selected = [row for row in rows if row["model"] == model and row["mode"] == mode]
        scored = [
            {
                "label": row["label"],
                "prediction": row["prediction"],
                "p_manipulated": str(float(row["score"]) * 100),
            }
            for row in selected
        ]
        real = [row for row, source in zip(scored, selected) if source["label"] == "0"]
        groups = [("all_images", scored)]
        for method in methods:
            fake = [
                row for row, source in zip(scored, selected)
                if source["label"] == "1" and source["method"] == method
            ]
            groups.append((method, real + fake))
        for group, group_rows in groups:
            reports.append({"model": model, "mode": mode, **metrics(group_rows, group)})
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path,
        default=Path(
            "results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv"
        ),
    )
    parser.add_argument(
        "--api-predictions", "--gemini-predictions", dest="api_predictions", type=Path,
        default=Path(
            "results/df40_transfer/gemini_face/df40_five_method_v3/predictions.csv"
        ),
    )
    parser.add_argument(
        "--reference-predictions", type=Path,
        default=Path(
            "results/df40_transfer/new_data/fresh_candidate_results_v3/predictions.csv"
        ),
    )
    parser.add_argument(
        "--reference-acceptance", type=Path,
        default=Path(
            "results/df40_transfer/new_data/fresh_candidate_results_v3/ACCEPTANCE.json"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(
            "results/df40_transfer/gemini_face/df40_five_method_v3/comparison"
        ),
    )
    parser.add_argument("--api-label", choices=("gemini", "openai"), default="gemini")
    parser.add_argument("--dataset", default="df40_five_method_v3")
    parser.add_argument("--protocol", type=Path,
                        default=Path("results/df40_transfer/gemini_face/protocol.json"))
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    providers = {"openai": "OpenAI Responses API", "gemini": "Google Gemini API"}
    if protocol.get("provider") != providers[args.api_label]:
        parser.error("--api-label must match the provider in --protocol")
    if args.api_label == "openai" and args.output_dir == parser.get_default("output_dir"):
        parser.error("Choose an explicit --output-dir for OpenAI results")

    acceptance = json.loads(args.reference_acceptance.read_text(encoding="utf-8"))
    if acceptance.get("status") != "passed":
        raise ValueError("Existing saved-model acceptance record has not passed")
    if digest(args.manifest) != acceptance.get("manifest_sha256"):
        raise ValueError("Manifest differs from the accepted saved-model evaluation")

    manifest = read_csv(args.manifest)
    methods = sorted({row["method"] for row in manifest if row["label"] == "1"})
    api_predictions = read_csv(args.api_predictions)
    validate_predictions(manifest, api_predictions, args.dataset, protocol)
    matched = normalized_rows(
        manifest,
        api_predictions,
        read_csv(args.reference_predictions),
        args.api_label,
    )
    reports = metric_rows(matched, methods, args.api_label)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched_path = args.output_dir / "matched_predictions.csv"
    metrics_path = args.output_dir / "metrics.csv"
    write_csv(matched_path, matched)
    write_csv(metrics_path, reports)
    audit = {
        "status": "passed",
        "manifest_images": EXPECTED_IMAGES,
        "models": 6,
        "matched_prediction_rows": len(matched),
        "methods": methods,
        "manifest_sha256": digest(args.manifest),
        "api_label": args.api_label,
        "api_model": protocol["model"],
        "protocol_sha256": digest(args.protocol),
        "api_predictions_sha256": digest(args.api_predictions),
        "reference_predictions_sha256": digest(args.reference_predictions),
        "matched_predictions_sha256": digest(matched_path),
        "metrics_sha256": digest(metrics_path),
        "audited_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": (
            "The face-adapted ViT is a secondary reference because it was trained on "
            "three evaluated manipulation methods; it is not a zero-shot transfer model."
        ),
    }
    (args.output_dir / "AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
