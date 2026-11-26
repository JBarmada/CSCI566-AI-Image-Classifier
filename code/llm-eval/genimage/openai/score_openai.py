import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation_utils import (
    LLM_EVAL,
    OUTPUT_DIR,
    SHARED_IMAGES,
    VIEWS,
    request_key,
    score_metrics,
)


def report_for(log_path: Path, model_id: str, output_dir: Path,
               expected_digest: str | None = None,
               manifest_path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    with log_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {log_path}:{line_number}") from exc
    matching = [
        row for row in records if row.get("model_id_requested") == model_id
    ]
    digests = {row["configuration_sha256"] for row in matching}
    if expected_digest:
        digests &= {expected_digest}
    if len(digests) != 1:
        raise ValueError(
            "The log must contain exactly one configuration for this model; "
            "use --configuration-sha256 to select one."
        )
    digest = next(iter(digests))
    config_records = [row for row in matching if row["configuration_sha256"] == digest]
    latest: dict[tuple[str, ...], dict[str, Any]] = {}
    for row in config_records:
        key = request_key(row)
        previous = latest.get(key)
        if previous is None or int(row["attempt"]) >= int(previous["attempt"]):
            latest[key] = row

    manifest_path = manifest_path or OUTPUT_DIR / "manifests" / "input_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing prepared input manifest {manifest_path}; run prepare_views.py first."
        )
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    labels = pd.read_csv(SHARED_IMAGES, dtype={"sha256": str})
    pilot = pd.read_csv(LLM_EVAL / "pilot_unique_api_manifest.csv", dtype={"sha256": str})
    pilot_ids = set(pilot["sha256"])
    key_columns = ["sha256", "label", "source_generator"]
    label_map = labels[key_columns].rename(columns={
        "sha256": "source_image_id",
        "source_generator": "generator",
    })
    if manifest["source_image_id"].nunique() != 630:
        raise ValueError("Prepared manifest does not describe the 630 shared unique images")

    key_lookup = {
        request_key(row): row for row in latest.values()
    }
    joined = manifest.merge(label_map, on="source_image_id", how="left", validate="many_to_one")
    if joined["label"].isna().any():
        raise ValueError("Prepared manifest contains an image absent from Claude's shared list")

    output_rows = []
    for _, item in joined.iterrows():
        lookup = {
            "source_image_id": item["source_image_id"],
            "submitted_image_sha256": item["submitted_image_sha256"],
            "view": item["view"],
            "model_id_requested": model_id,
            "prompt_sha256": config_records[0]["prompt_sha256"],
            "configuration_sha256": digest,
        }
        record = key_lookup.get(request_key(lookup))
        if record and record["source_image_sha256"] != item["source_image_sha256"]:
            raise ValueError(f"Source hash mismatch for {item['source_image_id']}/{item['view']}")
        status = record.get("status", "missing") if record else "missing"
        output_rows.append({
            "source_image_id": item["source_image_id"],
            "source_image_sha256": item["source_image_sha256"],
            "submitted_image_sha256": item["submitted_image_sha256"],
            "view": item["view"],
            "blind_id": item["blind_id"],
            "model_id_requested": model_id,
            "model_id_returned": record.get("model_id_returned") if record else None,
            "configuration_sha256": digest,
            "p_fake": record.get("p_fake") if record and status == "success" else np.nan,
            "status": status,
            "error_type": record.get("error_type") if record else "MissingRequest",
            "error_message": record.get("error_message") if record else "No API response recorded",
            "raw_response_text": record.get("raw_response_text") if record else None,
            "api_request_id": record.get("api_request_id") if record else None,
            "attempts": record.get("attempt") if record else 0,
            "input_tokens": record.get("input_tokens") if record else None,
            "output_tokens": record.get("output_tokens") if record else None,
            "reported_cost_usd": record.get("reported_cost_usd") if record else None,
            "estimated_cost_usd": record.get("estimated_cost_usd") if record else None,
            "label": int(item["label"]),
            "generator": item["generator"],
            "in_pilot": item["source_image_id"] in pilot_ids,
        })
    predictions = pd.DataFrame(output_rows)
    expected = len(predictions)
    if expected != 1890 or predictions.duplicated(["source_image_id", "view"]).any():
        raise ValueError("Expected 1,890 unique image/view prediction rows")

    metrics_rows = []
    status_rows = []
    attempt_frame = pd.DataFrame(config_records)
    for view in VIEWS:
        group = predictions[predictions["view"] == view]
        view_attempts = (
            attempt_frame[attempt_frame["view"] == view]
            if not attempt_frame.empty else pd.DataFrame()
        )
        valid = group[group["status"] == "success"]
        row = score_metrics(valid)
        n_real = int((group["label"] == 0).sum())
        n_fake = int((group["label"] == 1).sum())
        counts = group["status"].value_counts().to_dict()
        row.update({
            "view": view,
            "n_expected": len(group),
            "n_unique_source_images": group["source_image_id"].nunique(),
            "expected_real": n_real,
            "expected_fake": n_fake,
            "class_balance_real": n_real / len(group) if len(group) else np.nan,
            "response_coverage": float(group["status"].isin(
                ["success", "refusal", "invalid_output"]
            ).mean()),
            "valid_score_coverage": float(group["status"].eq("success").mean()),
            "refusals": int(counts.get("refusal", 0)),
            "invalid_outputs": int(counts.get("invalid_output", 0)),
            "api_failures": int(counts.get("api_failure", 0)),
            "api_failed_attempts": int(view_attempts["status"].eq("api_failure").sum())
            if not view_attempts.empty else 0,
            "missing": int(counts.get("missing", 0)),
            "model_id_requested": model_id,
            "configuration_sha256": digest,
        })
        metrics_rows.append(row)
        status_rows.append({
            "view": view,
            "n_expected": len(group),
            "success": int(counts.get("success", 0)),
            "refusal": int(counts.get("refusal", 0)),
            "invalid_output": int(counts.get("invalid_output", 0)),
            "api_failure": int(counts.get("api_failure", 0)),
            "api_failed_attempts": int(view_attempts["status"].eq("api_failure").sum())
            if not view_attempts.empty else 0,
            "missing": int(counts.get("missing", 0)),
        })

    generator_rows = []
    for view in VIEWS:
        view_data = predictions[predictions["view"] == view]
        real = view_data[(view_data["label"] == 0) & (view_data["status"] == "success")]
        for generator in sorted(view_data.loc[view_data["label"] == 1, "generator"].unique()):
            fake = view_data[
                (view_data["generator"] == generator)
                & (view_data["label"] == 1)
                & (view_data["status"] == "success")
            ]
            subset = pd.concat([real, fake], ignore_index=True)
            row = score_metrics(subset)
            row.update({
                "view": view,
                "generator": generator,
                "n_expected": 140,
                "n_expected_real": 70,
                "n_expected_fake": 70,
                "response_coverage": float(
                    view_data[
                        (view_data["label"] == 0)
                        | ((view_data["label"] == 1) & (view_data["generator"] == generator))
                    ]["status"].isin(["success", "refusal", "invalid_output"]).mean()
                ),
                "valid_score_coverage": len(subset) / 140,
                "real_images_reused_across_generator_rows": True,
            })
            generator_rows.append(row)

    # Reproduce the Claude shape-only baseline from the original-view manifest dimensions.
    original = predictions[predictions["view"] == "original"].copy()
    original_manifest = manifest[manifest["view"] == "original"][
        ["source_image_id", "width", "height"]
    ]
    shape = original.merge(original_manifest, on="source_image_id", validate="one_to_one")
    shape["shape_prediction"] = (shape["width"].astype(int) == shape["height"].astype(int)).astype(int)
    shape["shape_score"] = shape["shape_prediction"].astype(float) * 100
    shape_metrics = score_metrics(
        shape,
        score_column="shape_score",
        prediction_column="shape_prediction",
    )
    shape_metrics.update({
        "baseline": "shape-only (square = fake)",
        "view": "original",
        "n_expected": len(shape),
        "n_unique_source_images": shape["source_image_id"].nunique(),
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "openai_predictions.csv", index=False)
    pd.DataFrame(metrics_rows).to_csv(output_dir / "openai_view_metrics.csv", index=False)
    pd.DataFrame(generator_rows).to_csv(
        output_dir / "openai_per_generator_metrics.csv", index=False
    )
    pd.DataFrame(status_rows).to_csv(output_dir / "openai_response_status.csv", index=False)
    pd.DataFrame([shape_metrics]).to_csv(
        output_dir / "openai_shape_baseline_metrics.csv", index=False
    )
    attempts = attempt_frame
    if not attempts.empty:
        for column in (
            "input_tokens", "output_tokens", "estimated_cost_usd", "reported_cost_usd"
        ):
            if column not in attempts:
                attempts[column] = np.nan
        cost_summary = attempts.groupby("view", dropna=False).agg(
            api_attempts=("attempt", "count"),
            input_tokens_across_attempts=("input_tokens", lambda values: values.sum(min_count=1)),
            output_tokens_across_attempts=("output_tokens", lambda values: values.sum(min_count=1)),
            estimated_cost_usd_across_attempts=(
                "estimated_cost_usd", lambda values: values.sum(min_count=1)
            ),
            reported_cost_usd_across_attempts=(
                "reported_cost_usd", lambda values: values.sum(min_count=1)
            ),
        ).reset_index()
        cost_summary["model_id_requested"] = model_id
        cost_summary["configuration_sha256"] = digest
        cost_summary.to_csv(output_dir / "openai_cost_summary.csv", index=False)
    print(pd.DataFrame(metrics_rows).round(4).to_string(index=False))
    print(f"\nWrote metrics and predictions to {output_dir}")
    return predictions, pd.DataFrame(metrics_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score OpenAI JSONL predictions without dropping failed image requests."
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Exact requested model ID.")
    parser.add_argument("--configuration-sha256")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "metrics")
    args = parser.parse_args()
    report_for(args.log, args.model, args.out_dir, args.configuration_sha256)


if __name__ == "__main__":
    main()
