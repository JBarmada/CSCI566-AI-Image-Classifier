import ast
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from compare_models import compare_pilot, compare_vit
from evaluation_utils import (
    CLAUDE_DIR,
    LLM_EVAL,
    PROMPT,
    PROMPT_SHA256,
    config_hash,
    parse_score,
    request_key,
    score_metrics,
)
from prepare_views import make_view, pixel_sha256, validate_shared_set
from run_openai import (
    COMPLETED_STATUSES,
    append_checkpoint,
    call_once,
    read_log,
    request_config,
    run,
)
from score_openai import report_for


class EvaluationTests(unittest.TestCase):
    def test_prompt_matches_existing_experiment_exactly(self) -> None:
        source = (LLM_EVAL / "run_pilot.py").read_text(encoding="utf-8")
        module = ast.parse(source)
        prompt = next(
            ast.literal_eval(node.value)
            for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "PROMPT"
                    for target in node.targets)
        )
        self.assertEqual(PROMPT, prompt)
        self.assertEqual(len(PROMPT_SHA256), 64)

    def test_shared_manifest_counts_and_pilot_intersection(self) -> None:
        images = pd.read_csv(CLAUDE_DIR / "shared_image_list.csv")
        validate_shared_set(images)
        pilot = pd.read_csv(LLM_EVAL / "pilot_unique_api_manifest.csv")
        self.assertEqual(len(set(images.sha256) & set(pilot.sha256)), 90)
        self.assertEqual(int(images.in_pilot.sum()), 90)

    def test_parser_accepts_only_finite_json_numbers_in_range(self) -> None:
        self.assertEqual(parse_score('{"p_fake": 49.5}'), 49.5)
        self.assertEqual(parse_score('{"p_fake": 50}'), 50.0)
        for raw in (
            "", "not json", '{"p_fake": "50"}', '{"p_fake": true}',
            '{"p_fake": NaN}', '{"p_fake": -1}', '{"p_fake": 101}',
            '```json\\n{"p_fake": 50}\\n```',
        ):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_score(raw)

    def test_metrics_and_denominators(self) -> None:
        data = pd.DataFrame({
            "source_image_id": ["r1", "r2", "f1", "f2"],
            "label": [0, 0, 1, 1],
            "p_fake": [10, 70, 60, 90],
        })
        metrics = score_metrics(data)
        self.assertEqual((metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]),
                         (1, 1, 0, 2))
        self.assertEqual(metrics["n_unique_images"], 4)
        self.assertAlmostEqual(metrics["auroc"], 0.75)

    def test_view_transform_matches_claude_pipeline(self) -> None:
        sys.path.insert(0, str(CLAUDE_DIR))
        try:
            from blind_images import make_view as claude_make_view
        finally:
            sys.path.pop(0)
        image = Image.new("RGB", (381, 509), (7, 23, 41))
        for view in ("original", "square", "detector224"):
            result = make_view(image.copy(), view)
            expected = claude_make_view(image.copy(), view)
            self.assertEqual(result.size, expected.size)
            self.assertEqual(pixel_sha256(result), pixel_sha256(expected))

    def test_checkpoint_and_resume_key_scoping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "input_manifest.csv"
            pd.DataFrame([
                {
                    "source_image_id": f"id-{i}",
                    "source_image_sha256": f"source-{i}",
                    "submitted_image_sha256": f"submitted-{i}",
                    "view": view,
                    "image_file": f"inputs/{view}/{i}.png",
                }
                for i, view in enumerate(("original", "original", "square"))
            ]).to_csv(manifest_path, index=False)
            config = request_config("test-model", "auto")
            digest = config_hash(config)
            success = {
                "source_image_id": "id-0",
                "submitted_image_sha256": "submitted-0",
                "view": "original",
                "model_id_requested": "test-model",
                "prompt_sha256": PROMPT_SHA256,
                "configuration_sha256": digest,
                "attempt": 1,
                "status": "success",
            }
            log_path = root / "requests.jsonl"
            append_checkpoint(log_path, success)
            self.assertEqual(len(read_log(log_path)), 1)
            self.assertEqual(request_key(success)[0], "id-0")
            self.assertEqual(COMPLETED_STATUSES, {"success", "refusal", "invalid_output"})
            self.assertNotEqual(
                request_key(success),
                request_key({**success, "view": "square"}),
            )
            self.assertNotEqual(
                config_hash(request_config("test-model", "auto")),
                config_hash(request_config("test-model", "high")),
            )

            # Matching model/config/view/hash is skipped; a different view is not.
            with open(log_path, "a", encoding="utf-8") as file:
                file.write(json.dumps({
                    **success,
                    "source_image_id": "id-1",
                    "submitted_image_sha256": "submitted-1",
                    "status": "api_failure",
                    "attempt": 1,
                }) + "\n")
            from io import StringIO
            from contextlib import redirect_stdout

            output = StringIO()
            with redirect_stdout(output):
                run(
                    manifest_path, "test-model", "auto", log_path,
                    view="original", limit=0, dry_run=True,
                )
            result = json.loads(output.getvalue())
            self.assertEqual(result["requests_planned"], 1)

            output = StringIO()
            with redirect_stdout(output):
                run(
                    manifest_path, "other-model", "auto", log_path,
                    view="original", limit=0, dry_run=True,
                )
            result = json.loads(output.getvalue())
            self.assertEqual(result["requests_planned"], 2)

    def test_api_payload_contains_only_one_image_and_exact_prompt(self) -> None:
        class FakeResponses:
            request: dict[str, object] | None = None

            def create(self, **kwargs: object) -> object:
                self.request = kwargs
                return object()

        class FakeClient:
            def __init__(self) -> None:
                self.responses = FakeResponses()

        client = FakeClient()
        call_once(client, b"prepared-png", "high", "test-model", {
            "max_output_tokens": 128,
            "temperature": "omitted",
        })
        request = client.responses.request
        self.assertIsNotNone(request)
        assert request is not None
        self.assertEqual(request["model"], "test-model")
        self.assertFalse(request["store"])
        self.assertEqual(request["tools"], [])
        self.assertNotIn("temperature", request)
        content = request["input"][0]["content"]
        self.assertEqual(len(content), 2)
        self.assertEqual(content[0]["type"], "input_image")
        self.assertEqual(content[0]["detail"], "high")
        self.assertTrue(content[0]["image_url"].startswith("data:image/png;base64,"))
        self.assertEqual(content[1], {"type": "input_text", "text": PROMPT})

    def test_comparison_joins_exact_pilot_and_saved_vit_intersections(self) -> None:
        shared = pd.read_csv(CLAUDE_DIR / "shared_image_list.csv", dtype={"sha256": str})
        pilot = pd.read_csv(LLM_EVAL / "pilot_unique_api_manifest.csv", dtype={"sha256": str})
        pixel_map = pilot.set_index("sha256")["pixel_sha256"].to_dict()
        gpt_rows = []
        for _, row in shared.iterrows():
            for view in ("original", "detector224"):
                gpt_rows.append({
                    "source_image_id": row["sha256"],
                    "view": view,
                    "status": "success",
                    "p_fake": 25.0 if int(row["label"]) == 0 else 75.0,
                    "label": int(row["label"]),
                })
        gpt = pd.DataFrame(gpt_rows)

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_rows = [{
                "source_image_id": row["sha256"],
                "submitted_pixel_sha256": pixel_map.get(
                    row["sha256"], f"pixel-{row['sha256']}"
                ),
                "view": "original",
            } for _, row in shared.iterrows()]
            input_manifest = root / "input_manifest.csv"
            pd.DataFrame(input_rows).to_csv(input_manifest, index=False)
            pilot_results, legacy = compare_pilot(gpt, root, input_manifest)
            self.assertEqual(set(pilot_results["n_matched_successful"]), {90})
            self.assertTrue(pilot_results["gpt_client_pixel_equivalence"].eq("verified").all())
            self.assertTrue((legacy["n_task_rows_expected"] == 160).all())
            self.assertTrue((legacy["n_unique_images_in_manifest"] == 90).all())

            vit = compare_vit(gpt, root)
            self.assertEqual(len(vit), 16)
            self.assertEqual(int(vit["n_matched"].sum()), 2240)
            self.assertTrue(vit["gpt_n_real"].eq(70).all())
            self.assertTrue(vit["gpt_n_fake"].eq(70).all())
            self.assertIn("holdout", vit.columns)
            self.assertIn("vit_threshold_preserved", vit.columns)

    def test_pilot_metrics_use_the_same_valid_image_intersection(self) -> None:
        saved = pd.read_csv(LLM_EVAL / "openai/metrics/openai_predictions.csv")
        with tempfile.TemporaryDirectory() as temp_dir:
            compared, _ = compare_pilot(saved, Path(temp_dir))
        self.assertTrue(compared["n_matched_successful"].eq(88).all())
        self.assertTrue(compared["gpt_n_valid"].eq(88).all())
        self.assertTrue(compared["baseline_n_valid"].eq(88).all())
        # Availability is retained separately from metrics on matched rows.
        self.assertTrue(compared["n_baseline_valid"].eq(90).all())

    def test_scoring_retains_refusal_failure_and_missing_denominators(self) -> None:
        shared = pd.read_csv(CLAUDE_DIR / "shared_image_list.csv", dtype={"sha256": str})
        rows = []
        for _, source in shared.iterrows():
            for view in ("original", "square", "detector224"):
                rows.append({
                    "blind_id": "img_0001",
                    "source_image_id": source["sha256"],
                    "source_image_sha256": source["sha256"],
                    "submitted_image_sha256": f"submitted-{source['sha256']}-{view}",
                    "source_pixel_sha256": f"source-pixel-{source['sha256']}",
                    "submitted_pixel_sha256": f"pixel-{source['sha256']}-{view}",
                    "view": view,
                    "width": 224,
                    "height": 224,
                    "image_file": f"inputs/{view}/img_0001.png",
                })

        model = "score-test-model"
        config = request_config(model, "auto")
        digest = config_hash(config)
        manifest_lookup = {
            (row["source_image_id"], row["view"]): row for row in rows
        }
        source_ids = shared["sha256"].tolist()
        records = []
        for source_id, status, response, score in (
            (source_ids[0], "success", '{"p_fake": 25}', 25.0),
            (source_ids[1], "refusal", "I cannot help with that request.", None),
            (source_ids[2], "api_failure", None, None),
        ):
            item = manifest_lookup[(source_id, "original")]
            records.append({
                "source_image_id": source_id,
                "source_image_sha256": item["source_image_sha256"],
                "submitted_image_sha256": item["submitted_image_sha256"],
                "view": "original",
                "model_id_requested": model,
                "model_id_returned": model,
                "prompt_sha256": PROMPT_SHA256,
                "configuration_sha256": digest,
                "attempt": 1,
                "status": status,
                "raw_response_text": response,
                "p_fake": score,
                "error_type": "APIError" if status == "api_failure" else None,
                "error_message": "test failure" if status == "api_failure" else None,
            })

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "input_manifest.csv"
            pd.DataFrame(rows).to_csv(manifest_path, index=False)
            log_path = root / "requests.jsonl"
            log_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            _, metrics = report_for(
                log_path, model, root / "metrics", manifest_path=manifest_path
            )
            original = metrics[metrics["view"] == "original"].iloc[0]
            self.assertEqual(original["n_expected"], 630)
            self.assertEqual(original["n_valid"], 1)
            self.assertEqual(original["refusals"], 1)
            self.assertEqual(original["api_failures"], 1)
            self.assertEqual(original["missing"], 627)
            self.assertAlmostEqual(original["response_coverage"], 2 / 630)
            self.assertTrue(pd.isna(original["balanced_accuracy"]))


if __name__ == "__main__":
    unittest.main()
