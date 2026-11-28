import argparse
import json
import math
from pathlib import Path

import pandas as pd

from evaluation_utils import CLAUDE_DIR, OUTPUT_DIR, VIEWS


GPT_55_INPUT_USD_PER_MILLION = 5.0
GPT_55_OUTPUT_USD_PER_MILLION = 30.0
GPT_55_IMAGE_MULTIPLIER = 1.2
GPT_55_PATCH_BUDGET = 2500
GPT_55_MAX_DIMENSION = 2048
PROMPT_TOKENS_ESTIMATE = 100
EXPECTED_OUTPUT_TOKENS_ESTIMATE = 16
MAX_OUTPUT_TOKENS = 128


def patch_count(width: int, height: int) -> int:
    return math.ceil(width / 32) * math.ceil(height / 32)


def gpt55_high_tokens(width: int, height: int) -> int:
    scale = min(1.0, GPT_55_MAX_DIMENSION / max(width, height))
    width = max(1, math.floor(width * scale))
    height = max(1, math.floor(height * scale))
    patches = patch_count(width, height)
    if patches > GPT_55_PATCH_BUDGET:
        shrink = math.sqrt((32**2 * GPT_55_PATCH_BUDGET) / (width * height))
        adjusted = shrink * min(
            math.floor(width * shrink / 32) / (width * shrink / 32),
            math.floor(height * shrink / 32) / (height * shrink / 32),
        )
        width = max(1, math.floor(width * adjusted))
        height = max(1, math.floor(height * adjusted))
        while patch_count(width, height) > GPT_55_PATCH_BUDGET:
            if width >= height:
                width -= 1
            else:
                height -= 1
    return math.ceil(patch_count(width, height) * GPT_55_IMAGE_MULTIPLIER)


def estimate(output_dir: Path = OUTPUT_DIR / "manifests") -> dict[str, object]:
    totals: dict[str, dict[str, int]] = {}
    for view in VIEWS:
        key_path = CLAUDE_DIR / (
            "answer_key.csv" if view == "original" else f"answer_key_{view}.csv"
        )
        dimensions = pd.read_csv(key_path)[["width", "height"]]
        tokens = [
            gpt55_high_tokens(int(row.width), int(row.height))
            for row in dimensions.itertuples(index=False)
        ]
        totals[view] = {
            "images": len(tokens),
            "image_input_tokens": sum(tokens),
            "min_image_tokens": min(tokens),
            "max_image_tokens": max(tokens),
        }
    image_tokens = sum(row["image_input_tokens"] for row in totals.values())
    requests = sum(row["images"] for row in totals.values())
    prompt_tokens = requests * PROMPT_TOKENS_ESTIMATE
    expected_output_tokens = requests * EXPECTED_OUTPUT_TOKENS_ESTIMATE
    max_output_tokens = requests * MAX_OUTPUT_TOKENS
    expected_cost = (
        (image_tokens + prompt_tokens) * GPT_55_INPUT_USD_PER_MILLION
        + expected_output_tokens * GPT_55_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    max_cost = (
        (image_tokens + prompt_tokens) * GPT_55_INPUT_USD_PER_MILLION
        + max_output_tokens * GPT_55_OUTPUT_USD_PER_MILLION
    ) / 1_000_000
    report = {
        "illustrative_only": True,
        "model_id": "gpt-5.5",
        "detail": "high",
        "source_dimensions": "Claude's saved answer-key manifests",
        "evaluations_excluding_retries": requests,
        "views": totals,
        "estimated_image_input_tokens": image_tokens,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_output_tokens_scenario": expected_output_tokens,
        "maximum_output_tokens_by_config": max_output_tokens,
        "input_price_usd_per_million": GPT_55_INPUT_USD_PER_MILLION,
        "output_price_usd_per_million": GPT_55_OUTPUT_USD_PER_MILLION,
        "estimated_cost_expected_output_scenario_usd": round(expected_cost, 4),
        "estimated_cost_max_output_bound_usd": round(max_cost, 4),
        "caveat": (
            "Image tokens use the published GPT-5.5 high-detail patch rules; "
            "prompt and 16-token typical-output assumptions are estimates. "
            "The maximum is a bound only for output tokens under max_output_tokens=128. "
            "Rates are public standard text-token rates, before discounts/taxes; "
            "actual billing must be checked after selecting the exact model and API tier."
        ),
    }
    (output_dir / "gpt55_cost_estimate_illustrative.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Illustrative GPT-5.5 high-detail estimate using published image-token rules."
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR / "manifests")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps(estimate(args.out_dir), indent=2))


if __name__ == "__main__":
    main()
