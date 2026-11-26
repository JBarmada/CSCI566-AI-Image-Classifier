import importlib.util
import hashlib
import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


summarize = load(
    "gemini_face_summarize", "results/df40_transfer/gemini_face/summarize.py"
)
sys.path.insert(0, str(ROOT / "results/df40_transfer/gemini_face"))
compare = load("face_comparison", "results/df40_transfer/gemini_face/compare_df40_five_method.py")
common = load("face_common_test", "results/df40_transfer/face_eval_common.py")


class GeminiFaceSummaryTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((ROOT / "results/df40_transfer/openai_face/protocol.json").read_text())
        self.source = {"sha256": "hash", "path": "image.jpg", "label": "1", "method": "test"}
        self.prediction = {**self.source, "status": "ok", "error": "", "dataset": "test",
                           "model": self.protocol["model"],
                           "prompt_sha256": hashlib.sha256(self.protocol["prompt"].encode()).hexdigest(),
                           "protocol_sha256": common.protocol_hash(self.protocol),
                           "prediction": "1", "p_manipulated": "75",
                           "raw_response": '{"p_manipulated":75}'}

    def test_audit_rejects_tampering_and_invalid_scores(self):
        summarize.validate_predictions([self.source], [self.prediction], "test", self.protocol)
        for key, value in [("model", "other"), ("prompt_sha256", "other"),
                           ("protocol_sha256", "other"), ("prediction", "0"),
                           ("p_manipulated", "nan"), ("p_manipulated", "101"),
                           ("raw_response", '{"p_manipulated":76}'),
                           ("status", "error"), ("label", "0")]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                summarize.validate_predictions([self.source], [{**self.prediction, key: value}],
                                               "test", self.protocol)
        with self.assertRaises(ValueError):
            summarize.validate_predictions([self.source], [self.prediction] * 2, "test", self.protocol)

    def test_openai_comparison_retains_provider_name(self):
        manifest = summarize.read_csv(
            ROOT / "results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv")
        reference = summarize.read_csv(
            ROOT / "results/df40_transfer/new_data/fresh_candidate_results_v3/predictions.csv")
        api = [{**row, "status": "ok", "p_manipulated": "75"} for row in manifest]
        rows = compare.normalized_rows(manifest, api, reference, "openai")
        self.assertEqual(len(rows), 189 * 6)
        self.assertIn("openai", {row["model"] for row in rows})
        self.assertNotIn("gemini", {row["model"] for row in rows})
        methods = sorted({row["method"] for row in manifest if row["label"] == "1"})
        reports = compare.metric_rows(rows, methods, "openai")
        self.assertEqual(len(reports), 36)

    def test_full_summary_and_comparison_without_api_calls(self):
        manifest_path = ROOT / "results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv"
        protocol_path = ROOT / "results/df40_transfer/openai_face/protocol.json"
        manifest = summarize.read_csv(manifest_path)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            predictions = output / "predictions.csv"
            fixtures = []
            for source in manifest:
                score = 80 if source["label"] == "1" else 20
                fixtures.append({**self.prediction, **source, "dataset": "df40_five_method_v3",
                                 "p_manipulated": str(score), "prediction": source["label"],
                                 "raw_response": json.dumps({"p_manipulated": score})})
            with predictions.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(fixtures[0]))
                writer.writeheader()
                writer.writerows(fixtures)
            summary_args = ["summary", "--dataset", "df40_five_method_v3",
                            "--manifest", str(manifest_path), "--protocol", str(protocol_path),
                            "--predictions", str(predictions), "--output-dir", str(output / "summary")]
            with patch.object(sys, "argv", summary_args), contextlib.redirect_stdout(io.StringIO()):
                summarize.main()
            self.assertEqual(json.loads((output / "summary/AUDIT.json").read_text())["status"], "passed")
            compare_args = ["compare", "--api-label", "openai", "--api-predictions", str(predictions),
                            "--manifest", str(manifest_path), "--protocol", str(protocol_path),
                            "--output-dir", str(output / "comparison")]
            with patch.object(sys, "argv", compare_args), contextlib.redirect_stdout(io.StringIO()):
                compare.main()
            audit = json.loads((output / "comparison/AUDIT.json").read_text())
            self.assertEqual(audit["api_label"], "openai")
            self.assertEqual(audit["matched_prediction_rows"], 1134)
            # A later failed audit must replace an earlier passed status.
            fixtures[0]["model"] = "wrong-model"
            with predictions.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(fixtures[0]))
                writer.writeheader()
                writer.writerows(fixtures)
            with patch.object(sys, "argv", summary_args), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit):
                    summarize.main()
            self.assertEqual(json.loads((output / "summary/AUDIT.json").read_text())["status"], "failed")
    def test_metrics(self):
        rows = [
            {"label": "0", "prediction": "0", "p_manipulated": "10"},
            {"label": "0", "prediction": "1", "p_manipulated": "70"},
            {"label": "1", "prediction": "1", "p_manipulated": "90"},
            {"label": "1", "prediction": "0", "p_manipulated": "20"},
        ]
        result = summarize.metrics(rows, "test")
        self.assertEqual((result["tp"], result["tn"], result["fp"], result["fn"]), (1, 1, 1, 1))
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["auroc"], 0.75)

    def test_face_manifest_matches_reference_predictions(self):
        manifest = summarize.read_csv(
            ROOT / "results/df40_transfer/still_face_v1/manifest.csv"
        )
        predictions = summarize.read_csv(
            ROOT / "results/df40_transfer/still_face_v1/predictions.csv"
        )
        expected = {row["sha256"] for row in manifest}
        self.assertEqual(len(expected), 150)
        for model in ("cnn", "vit"):
            for mode in ("baseline", "augmented"):
                actual = {
                    row["sha256"] for row in predictions
                    if row["model"] == model and row["mode"] == mode
                }
                self.assertEqual(actual, expected)

    def test_df40_manifest_matches_reference_predictions(self):
        manifest = summarize.read_csv(
            ROOT / "results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv"
        )
        predictions = summarize.read_csv(
            ROOT / "results/df40_transfer/new_data/fresh_candidate_results_v3/predictions.csv"
        )
        expected = {row["sha256"] for row in manifest}
        self.assertEqual(len(expected), 189)
        for model, mode in (
            ("cnn", "baseline"),
            ("cnn", "augmented"),
            ("vit", "baseline"),
            ("vit", "augmented"),
            ("vit", "face_adapted"),
        ):
            actual = {
                row["sha256"] for row in predictions
                if row["model"] == model and row["mode"] == mode
            }
            self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
