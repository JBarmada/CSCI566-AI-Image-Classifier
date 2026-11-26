import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation_utils import (
    CLAUDE_DIR,
    LLM_EVAL,
    OUTPUT_DIR,
    SHARED_IMAGES,
    THRESHOLD,
    VIEWS,
    score_metrics,
)
from score_openai import report_for


CLAUDE_FILES = {
    "original": "claude_api_predictions.csv",
    "square": "claude_api_square_predictions.csv",
    "detector224": "claude_api_detector224_predictions.csv",
}


def load_claude_predictions() -> pd.DataFrame:
    frames = []
    for view, filename in CLAUDE_FILES.items():
        path = CLAUDE_DIR / filename
        if not path.is_file():
            continue
        data = pd.read_csv(path, dtype={"sha256": str})
        if "p_fake" not in data:
            raise ValueError(f"Claude prediction file is missing p_fake: {path}")
        if "status" in data:
            data = data[data["status"] == "success"].copy()
        data["view"] = view
        data["source_image_id"] = data["sha256"]
        data["claude_p_fake"] = pd.to_numeric(data["p_fake"], errors="coerce")
        frames.append(data[["source_image_id", "view", "claude_p_fake"]])
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["source_image_id", "view", "claude_p_fake"]
    )


def matched_model_comparison(gpt: pd.DataFrame, baseline: pd.DataFrame,
                             score_column: str, label: str) -> dict[str, Any]:
    left = gpt[gpt["status"] == "success"]
    right = baseline.dropna(subset=[score_column])
    joined = left.merge(
        right[["source_image_id", "view", score_column]],
        on=["source_image_id", "view"],
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        return {
            "comparison": label,
            "view": gpt["view"].iloc[0] if not gpt.empty else None,
            "n_matched": 0,
            "n_unique_images": 0,
            "score_correlation": np.nan,
            "threshold_agreement": np.nan,
        }
    if "label" not in joined:
        raise ValueError(f"{label} match is missing GPT ground-truth labels")
    gpt_binary = joined["p_fake"] >= THRESHOLD
    base_binary = joined[score_column] >= THRESHOLD
    gpt_metrics = score_metrics(joined)
    baseline_metrics = score_metrics(
        joined.rename(columns={score_column: "baseline_score"}),
        score_column="baseline_score",
    )
    result = {
        "comparison": label,
        "view": joined["view"].iloc[0],
        "n_matched": len(joined),
        "n_unique_images": joined["source_image_id"].nunique(),
        "score_correlation": joined[["p_fake", score_column]].corr().iloc[0, 1],
        "threshold_agreement": float((gpt_binary == base_binary).mean()),
    }
    result.update({f"gpt_{key}": value for key, value in gpt_metrics.items()})
    result.update({f"baseline_{key}": value for key, value in baseline_metrics.items()})
    return result


def compare_gpt_to_claude(gpt: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    claude = load_claude_predictions()
    rows = []
    for view in VIEWS:
        rows.append(matched_model_comparison(
            gpt[gpt["view"] == view],
            claude[claude["view"] == view],
            "claude_p_fake",
            "GPT vs Claude",
        ))
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "openai_vs_claude_matched.csv", index=False)
    return result


def compare_pilot(gpt: pd.DataFrame, output_dir: Path,
                  input_manifest: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pilot = pd.read_csv(LLM_EVAL / "pilot_unique_api_manifest.csv", dtype={"sha256": str})
    shared = pd.read_csv(SHARED_IMAGES, dtype={"sha256": str})
    intersection = set(pilot["sha256"]) & set(shared["sha256"])
    if len(intersection) != 90:
        raise ValueError(f"Expected exact 90-image GPT/Gemini/Qwen pilot intersection, got {len(intersection)}")

    old_predictions = {
        "Gemini": LLM_EVAL / "gemini_predictions.csv",
        "Qwen (Groq)": LLM_EVAL / "groq_predictions.csv",
    }
    original = gpt[gpt["view"] == "original"].copy()
    prepared_path = input_manifest or OUTPUT_DIR / "manifests" / "input_manifest.csv"
    prepared = pd.read_csv(prepared_path, dtype=str, keep_default_na=False)
    prepared_original = prepared[prepared["view"] == "original"][
        ["source_image_id", "submitted_pixel_sha256"]
    ]
    pilot_pixels = pilot[["sha256", "pixel_sha256"]].rename(
        columns={"sha256": "source_image_id", "pixel_sha256": "pilot_pixel_sha256"}
    )
    pixel_check = prepared_original.merge(
        pilot_pixels, on="source_image_id", how="inner", validate="one_to_one"
    )
    pixels_match = (
        len(pixel_check) == len(intersection)
        and (pixel_check["submitted_pixel_sha256"] == pixel_check["pilot_pixel_sha256"]).all()
    )

    rows = []
    for name, path in old_predictions.items():
        if not path.is_file():
            continue
        old = pd.read_csv(path, dtype={"sha256": str})
        old = old[old["status"] == "success"].copy()
        old["source_image_id"] = old["sha256"]
        old["comparison_score"] = pd.to_numeric(old["p_fake"], errors="coerce")
        gpt_valid = original[
            original["source_image_id"].isin(intersection)
            & (original["status"] == "success")
        ].copy()
        baseline_valid = old[old["source_image_id"].isin(intersection)].copy()
        matched = gpt_valid.merge(
            old[["source_image_id", "comparison_score"]],
            on="source_image_id",
            how="inner",
            validate="one_to_one",
        )
        gpt_eval = gpt_valid.drop(columns=["label"], errors="ignore").merge(
            shared[["sha256", "label"]].rename(columns={"sha256": "source_image_id"}),
            on="source_image_id",
            validate="one_to_one",
        )
        base_eval = baseline_valid[["source_image_id", "comparison_score"]].merge(
            shared[["sha256", "label"]].rename(columns={"sha256": "source_image_id"}),
            on="source_image_id",
            validate="one_to_one",
        ).rename(columns={"comparison_score": "p_fake"})
        row = {
            "comparison": f"GPT vs {name}",
            "view": "original",
            "pilot_intersection_images": len(intersection),
            "expected_real": int((shared["sha256"].isin(intersection) & (shared["label"] == 0)).sum()),
            "expected_fake": int((shared["sha256"].isin(intersection) & (shared["label"] == 1)).sum()),
            "n_gpt_valid": len(gpt_eval),
            "gpt_coverage": len(gpt_eval) / len(intersection),
            "n_baseline_valid": len(base_eval),
            "baseline_coverage": len(base_eval) / len(intersection),
            "n_matched_successful": len(matched),
            "n_unique_images": matched["source_image_id"].nunique(),
            "score_correlation": (
                matched[["p_fake", "comparison_score"]].corr().iloc[0, 1]
                if len(matched) >= 2 else np.nan
            ),
            "threshold_agreement": (
                float(((matched["p_fake"] >= THRESHOLD)
                       == (matched["comparison_score"] >= THRESHOLD)).mean())
                if len(matched) else np.nan
            ),
            "gpt_client_pixel_equivalence": "verified" if pixels_match else "not verified",
            "pilot_source_bytes_vs_gpt_png_encoding": (
                "different encodings; decoded RGB equality checked"
            ),
            "api_internal_preprocessing_equivalence": "not observable",
        }
        row.update({f"gpt_{key}": value for key, value in score_metrics(gpt_eval).items()})
        row.update({
            f"baseline_{key}": value for key, value in score_metrics(base_eval).items()
        })
        rows.append(row)
    pilot_comparison = pd.DataFrame(rows)
    pilot_comparison.to_csv(output_dir / "openai_vs_pilot_models.csv", index=False)

    # Legacy task rows reuse real images across generator tasks; preserve but label the row unit.
    task = pd.read_csv(LLM_EVAL / "pilot_task_manifest.csv", dtype={"sha256": str})
    if task["sha256"].nunique() != 90:
        raise ValueError("Pilot task manifest no longer contains 90 unique images")
    if not set(task["sha256"]).issubset(intersection):
        raise ValueError("Pilot task rows are not contained in the exact pilot intersection")
    evaluations = []
    sources: dict[str, pd.DataFrame] = {
        "GPT": original[original["status"] == "success"].rename(
            columns={"source_image_id": "sha256"}
        ),
    }
    claude_original = load_claude_predictions()
    claude_original = claude_original[claude_original["view"] == "original"].rename(
        columns={"source_image_id": "sha256", "claude_p_fake": "p_fake"}
    )
    sources["Claude"] = claude_original
    for name, path in old_predictions.items():
        if path.is_file():
            old = pd.read_csv(path, dtype={"sha256": str})
            sources[name] = old[old["status"] == "success"]
    for model_name, data in sources.items():
        mapped = task.merge(
            data[["sha256", "p_fake"]], on="sha256", how="left", validate="many_to_one"
        )
        valid = mapped.dropna(subset=["p_fake"]).copy()
        valid["source_image_id"] = valid["sha256"]
        metric = score_metrics(valid)
        evaluations.append({
            "model": model_name,
            "evaluation_unit": "legacy pilot task rows; repeated real images are not independent",
            "n_task_rows_expected": len(task),
            "n_unique_images_in_manifest": task["sha256"].nunique(),
            "n_task_rows_scored": len(valid),
            "n_unique_images_scored": valid["sha256"].nunique(),
            **metric,
        })
    pd.DataFrame(evaluations).to_csv(
        output_dir / "pilot_legacy_task_row_metrics.csv", index=False
    )
    return pilot_comparison, pd.DataFrame(evaluations)


def compare_vit(gpt: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    vit_root = LLM_EVAL.parent / "vit_master_v1"
    index_path = vit_root / "metadata" / "prediction_index.csv"
    if not index_path.is_file():
        return pd.DataFrame()
    index = pd.read_csv(index_path, dtype=str, keep_default_na=False)
    index = index[(index["split"] == "test") & (index["condition"] == "clean")]
    detector = gpt[(gpt["view"] == "detector224") & (gpt["status"] == "success")]
    rows = []
    for _, provenance in index.iterrows():
        prediction_path = vit_root / provenance["prediction_file"]
        if not prediction_path.is_file():
            continue
        data = pd.read_csv(prediction_path, dtype={"sha256": str})
        if not {"sha256", "prob_fake", "pred_label", "threshold"} <= set(data.columns):
            continue
        joined = detector.merge(
            data[["sha256", "prob_fake", "pred_label", "threshold", "label"]],
            left_on="source_image_id",
            right_on="sha256",
            how="inner",
            suffixes=("_gpt", "_vit"),
            validate="one_to_one",
        )
        if joined.empty:
            continue
        if not (joined["label_gpt"].astype(int) == joined["label_vit"].astype(int)).all():
            raise ValueError(f"GPT and ViT labels differ on matched hashes in {prediction_path}")
        # Preserve the ViT's saved prediction/threshold; do not refit or retune it.
        vit_eval = joined[[
            "source_image_id", "label_vit", "prob_fake", "pred_label"
        ]].rename(columns={
            "label_vit": "label",
            "prob_fake": "vit_score",
        })
        vit_metrics = score_metrics(
            vit_eval,
            score_column="vit_score",
            prediction_column="pred_label",
        )
        gpt_eval = joined[[
            "source_image_id", "label_gpt", "p_fake"
        ]].rename(columns={"label_gpt": "label"})
        gpt_metrics = score_metrics(gpt_eval)
        rows.append({
            "comparison": "GPT detector224 vs saved ViT clean test predictions",
            "experiment": provenance["experiment"],
            "holdout": provenance["holdout"],
            "k": provenance["k"],
            "rep": provenance["rep"],
            "mode": provenance["mode"],
            "condition": provenance["condition"],
            "split": provenance["split"],
            "seed": provenance["seed"],
            "config_file": provenance["config_file"],
            "protocol_hash": provenance["protocol_hash"],
            "evaluation_hash": provenance["evaluation_hash"],
            "vit_prediction_file": provenance["prediction_file"],
            "vit_threshold_preserved": ";".join(sorted(joined["threshold"].astype(str).unique())),
            "n_matched": len(joined),
            "n_unique_images": joined["source_image_id"].nunique(),
            "input_preparation_equivalence": (
                "not established; compare only exact source-hash intersection"
            ),
            "api_internal_preprocessing_equivalence": "not observable",
            "score_correlation": joined[["p_fake", "prob_fake"]].corr().iloc[0, 1],
            "binary_agreement": float(
                ((joined["p_fake"] >= THRESHOLD).astype(int)
                 == joined["pred_label"].astype(int)).mean()
            ),
            "gpt_auroc_on_matched_set": gpt_metrics["auroc"],
            "vit_auroc_on_matched_set": vit_metrics["auroc"],
        })
        rows[-1].update({f"gpt_{key}": value for key, value in gpt_metrics.items()})
        rows[-1].update({f"vit_{key}": value for key, value in vit_metrics.items()})
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "openai_vs_vit_matched.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare GPT to Claude, Gemini/Qwen pilot, and saved CNN/ViT predictions."
    )
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--configuration-sha256")
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "metrics")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    predictions, _ = report_for(
        args.log, args.model, args.out_dir, args.configuration_sha256
    )
    print("GPT vs Claude")
    print(compare_gpt_to_claude(predictions, args.out_dir).round(4).to_string(index=False))
    pilot, legacy = compare_pilot(predictions, args.out_dir)
    print("\nGPT vs Gemini/Qwen pilot")
    print(pilot.round(4).to_string(index=False))
    print("\nLegacy pilot task-row metrics (not independent-image metrics)")
    print(legacy.round(4).to_string(index=False))
    vit = compare_vit(predictions, args.out_dir)
    print(f"\nMatching saved ViT test comparisons: {len(vit)}")


if __name__ == "__main__":
    main()
