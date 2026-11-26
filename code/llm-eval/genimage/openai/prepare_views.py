import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from evaluation_utils import CLAUDE_DIR, OUTPUT_DIR, LLM_EVAL, SHARED_IMAGES, VIEWS


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def pixel_sha256(image: Image.Image) -> str:
    return hashlib.sha256(np.asarray(image.convert("RGB")).tobytes()).hexdigest()


def make_view(image: Image.Image, view: str) -> Image.Image:
    if view == "original":
        return image
    if view == "detector224":
        width, height = image.size
        scale = 224 / min(width, height)
        image = image.resize(
            (max(224, round(width * scale)), max(224, round(height * scale))),
            Image.Resampling.BICUBIC,
        )
    width, height = image.size
    side = min(width, height)
    left, top = (width - side) // 2, (height - side) // 2
    output = image.crop((left, top, left + side, top + side))
    if view == "detector224" and output.size != (224, 224):
        output = output.resize((224, 224), Image.Resampling.BICUBIC)
    return output


def validate_shared_set(images: pd.DataFrame) -> None:
    if len(images) != 630 or images["sha256"].nunique() != 630:
        raise ValueError("Claude shared_image_list.csv must contain 630 unique images")
    if images["label"].value_counts().to_dict() != {1: 560, 0: 70}:
        raise ValueError("Expected 560 fake and 70 real images")
    counts = images.assign(
        selection_group=images["source_generator"].where(images["label"] == 1, "real")
    )["selection_group"].value_counts()
    if len(counts) != 9 or not counts.eq(70).all():
        raise ValueError("Expected 70 images in each of 8 fake-generator groups and real")
    pilot = pd.read_csv(LLM_EVAL / "pilot_unique_api_manifest.csv", dtype={"sha256": str})
    shared_ids = set(images["sha256"])
    pilot_ids = set(pilot["sha256"])
    if len(pilot_ids) != 90 or not pilot_ids.issubset(shared_ids):
        raise ValueError("The shared list must include all 90 unique Gemini/Qwen pilot images")
    if "in_pilot" in images:
        recorded = images["in_pilot"].map(
            lambda value: str(value).strip().lower() in {"true", "1"}
        )
        if set(images.loc[recorded, "sha256"]) != pilot_ids:
            raise ValueError("shared_image_list.csv in_pilot flags disagree with the pilot manifest")


def prepare(image_root: Path, output_root: Path) -> dict[str, object]:
    images = pd.read_csv(SHARED_IMAGES, dtype={"sha256": str})
    validate_shared_set(images)
    keys: dict[str, pd.DataFrame] = {}
    for view in VIEWS:
        key_path = CLAUDE_DIR / (
            "answer_key.csv" if view == "original" else f"answer_key_{view}.csv"
        )
        key = pd.read_csv(key_path, dtype={"sha256": str})
        if len(key) != 630 or key["sha256"].nunique() != 630:
            raise ValueError(f"{key_path.name} does not contain 630 unique images")
        if set(key["sha256"]) != set(images["sha256"]):
            raise ValueError(f"{key_path.name} does not match the shared image list")
        base_ids = keys.get("original")
        if base_ids is not None:
            base_mapping = base_ids["blind_id"].to_dict()
            if any(
                base_mapping[sha256] != blind_id
                for sha256, blind_id in key.set_index("sha256")["blind_id"].items()
            ):
                raise ValueError(f"{key_path.name} uses different blind IDs from Claude's original view")
        keys[view] = key.set_index("sha256")

    manifests: list[dict[str, object]] = []
    dimensions_checked = 0
    for _, row in images.iterrows():
        source_path = image_root / str(row["path"])
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing one of the 630 shared images: {source_path}")
        source_bytes = source_path.read_bytes()
        source_sha256 = sha256_bytes(source_bytes)
        if source_sha256 != row["sha256"]:
            raise ValueError(f"Source SHA-256 mismatch for shared image ID {row['sha256']}")

        with Image.open(source_path) as source:
            source_rgb = source.convert("RGB")
            source_pixel_hash = pixel_sha256(source_rgb)
            for view in VIEWS:
                prepared = make_view(source_rgb, view)
                clean = Image.frombytes("RGB", prepared.size, prepared.tobytes())
                blind_id = str(keys[view].loc[str(row["sha256"]), "blind_id"])
                target = output_root / "inputs" / view / f"{blind_id}.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                clean.save(target, format="PNG")

                with Image.open(target) as decoded:
                    decoded_rgb = decoded.convert("RGB")
                    decoded_pixel_hash = pixel_sha256(decoded_rgb)
                expected_pixel_hash = pixel_sha256(clean)
                if decoded_pixel_hash != expected_pixel_hash:
                    raise ValueError(f"PNG pixel mismatch for {view}/{blind_id}.png")
                if view == "original" and decoded_pixel_hash != source_pixel_hash:
                    raise ValueError(
                        f"Metadata-only anonymization changed decoded RGB pixels for {blind_id}"
                    )

                key_row = keys[view].loc[str(row["sha256"])]
                if (clean.width, clean.height) != (
                    int(key_row["width"]), int(key_row["height"])
                ):
                    raise ValueError(
                        f"Prepared dimensions differ from Claude's {view} answer key "
                        f"for shared image ID {row['sha256']}"
                    )
                dimensions_checked += 1
                manifests.append({
                    "blind_id": blind_id,
                    "source_image_id": str(row["sha256"]),
                    "source_image_sha256": source_sha256,
                    "source_pixel_sha256": source_pixel_hash,
                    "submitted_image_sha256": sha256_bytes(target.read_bytes()),
                    "submitted_pixel_sha256": decoded_pixel_hash,
                    "view": view,
                    "width": clean.width,
                    "height": clean.height,
                    "image_file": str(target.relative_to(output_root)),
                })

    manifest_path = output_root / "manifests" / "input_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(manifests)
    if len(manifest) != 1890 or manifest.duplicated(
        ["source_image_id", "view"]
    ).any():
        raise ValueError("Prepared manifest must contain exactly 1,890 unique image/view pairs")
    manifest.to_csv(manifest_path, index=False)
    pilot = pd.read_csv(LLM_EVAL / "pilot_unique_api_manifest.csv", dtype={"sha256": str})
    pilot_pixels = pilot.set_index("sha256")["pixel_sha256"].to_dict()
    original_rows = manifest[manifest["view"] == "original"]
    pixel_match = {
        row["source_image_id"]: row["submitted_pixel_sha256"] == pilot_pixels[row["source_image_id"]]
        for _, row in original_rows.iterrows()
        if row["source_image_id"] in pilot_pixels
    }
    summary = {
        "shared_image_count": len(images),
        "real_image_count": int((images["label"] == 0).sum()),
        "fake_image_count": int((images["label"] == 1).sum()),
        "fake_generators": int(images.loc[images["label"] == 1, "source_generator"].nunique()),
        "pilot_intersection_images": len(set(images["sha256"]) & set(pilot["sha256"])),
        "pilot_original_view_decoded_pixel_matches": sum(pixel_match.values()),
        "pilot_original_view_decoded_pixel_equality_verified": (
            len(pixel_match) == 90 and all(pixel_match.values())
        ),
        "views": list(VIEWS),
        "prepared_image_view_count": len(manifest),
        "dimensions_checked_against_claude_answer_keys": dimensions_checked,
        "original_view_metadata_only_pixel_equality_checked": 630,
        "all_encoded_png_pixel_hashes_verified": len(manifest),
        "manifest": str(manifest_path),
    }
    summary_path = output_root / "manifests" / "preparation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate Claude's anonymized GenImage views from the exact shared list."
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(prepare(args.image_root, args.output_root), indent=2))


if __name__ == "__main__":
    main()
