import hashlib
import importlib.util
import json
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "claude_face_evaluate", ROOT / "results/df40_transfer/claude_face/evaluate.py"
)
evaluate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluate)


class ClaudeFaceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((evaluate.HERE / "protocol.json").read_text())
        self.prompt_hash = hashlib.sha256(self.protocol["prompt"].encode()).hexdigest()
        self.row = {"sha256": hashlib.sha256(b"image").hexdigest(),
                    "path": "image.png", "label": "1", "method": "test"}

    def response(self, **changes):
        values = dict(id="msg_test", model=self.protocol["model"], stop_reason="end_turn",
                      content=[SimpleNamespace(type="text", text='{"p_manipulated":75}')],
                      usage=SimpleNamespace(model_dump=lambda: {
                          "input_tokens": 120, "output_tokens": 12}))
        return SimpleNamespace(**{**values, **changes})

    def run_image(self, client, data=b"image"):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(data)
            return evaluate.evaluate(self.row, "test", path, client,
                                     self.protocol, self.prompt_hash, 3)

    def test_frozen_prompt_threshold_and_manifest_match_accepted_study(self):
        original = json.loads((ROOT / "results/df40_transfer/openai_face/protocol.json").read_text())
        for key in ("prompt", "decision_threshold"):
            self.assertEqual(self.protocol[key], original[key])
        manifest = ROOT / "results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv"
        self.assertEqual(hashlib.sha256(manifest.read_bytes()).hexdigest(), evaluate.MANIFEST_SHA256)

    def test_request_has_one_image_and_frozen_prompt_without_source_metadata(self):
        request = evaluate.response_request(b"image", "image/png", self.protocol)
        self.assertEqual(request["model"], "claude-opus-5-5")
        self.assertEqual(len(request["messages"]), 1)
        content = request["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": self.protocol["prompt"]})
        self.assertEqual(content[1]["source"]["data"], "aW1hZ2U=")
        self.assertEqual(set(request), {"model", "max_tokens", "messages"})
        self.assertEqual(set(content[1]["source"]), {"type", "media_type", "data"})
        with self.assertRaises(ValueError):
            evaluate.response_request(b"image", "image/tiff", self.protocol)

    def test_success_is_accepted_by_shared_audit_and_saves_usage(self):
        from face_eval_common import validate_predictions
        client = Mock()
        client.messages.create.return_value = self.response()
        event = self.run_image(client)
        self.assertEqual(event["status"], "ok")
        validate_predictions([self.row], [event], "test", self.protocol)
        self.assertEqual(json.loads(event["usage_json"])[0]["usage"]["input_tokens"], 120)

    def test_truncation_refusal_wrong_model_and_invalid_scores_stop(self):
        responses = [self.response(stop_reason="max_tokens"), self.response(stop_reason="refusal"),
                     self.response(model="other-model")]
        for raw in ('{"p_manipulated": NaN}', '{"p_manipulated": true}',
                    '{"p_manipulated": 101}', 'not JSON'):
            responses.append(self.response(content=[SimpleNamespace(type="text", text=raw)]))
        for response in responses:
            with self.subTest(response=response):
                client = Mock()
                client.messages.create.return_value = response
                event = self.run_image(client)
                self.assertEqual(event["status"], "error")
                self.assertTrue(event["stop_run"])
                self.assertEqual(client.messages.create.call_count, 1)

    def test_hash_mismatch_prevents_api_call(self):
        client = Mock()
        self.assertEqual(self.run_image(client, b"changed")["status"], "error")
        client.messages.create.assert_not_called()

    def test_transient_error_retries_but_bad_key_stops(self):
        client = Mock()
        transient = RuntimeError("unavailable")
        transient.status_code = 503
        client.messages.create.side_effect = [transient, self.response()]
        with patch.object(evaluate.time, "sleep"):
            self.assertEqual(self.run_image(client)["status"], "ok")
        self.assertEqual(client.messages.create.call_count, 2)
        client = Mock()
        bad_key = RuntimeError("authentication failed")
        bad_key.status_code = 401
        client.messages.create.side_effect = bad_key
        self.assertTrue(self.run_image(client)["stop_run"])
        self.assertEqual(client.messages.create.call_count, 1)

    def test_shared_runner_uses_claude_and_resume_skips_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image.png").write_bytes(b"image")
            args = SimpleNamespace(output_dir=root, data_root=root, dataset="test", retries=3)
            factory = Mock()
            client = factory.return_value
            client.messages.create.return_value = self.response()
            evaluate.run_evaluation(args, [self.row], self.protocol, "test-key", factory,
                                    evaluate_image=evaluate.evaluate)
            evaluate.run_evaluation(args, [self.row], self.protocol, "test-key", factory,
                                    evaluate_image=evaluate.evaluate)
            self.assertEqual(client.messages.create.call_count, 1)

    def test_comparison_cli_accepts_claude_and_matches_all_reference_images(self):
        # Re-label a saved score fixture only inside a temporary directory. No API calls.
        from face_eval_common import protocol_hash
        source = ROOT / "results/df40_transfer/openai_face/df40_five_method_v3_v2/predictions.csv"
        with source.open() as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row.update(model=self.protocol["model"], protocol_sha256=protocol_hash(self.protocol))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            result = subprocess.run([
                sys.executable,
                str(ROOT / "results/df40_transfer/gemini_face/compare_df40_five_method.py"),
                "--api-label", "claude", "--api-predictions", str(path),
                "--protocol", str(evaluate.HERE / "protocol.json"),
                "--output-dir", str(Path(directory) / "comparison"),
            ], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            audit = json.loads((Path(directory) / "comparison/AUDIT.json").read_text())
            self.assertEqual(audit["status"], "passed")
            self.assertEqual(audit["api_label"], "claude")
            self.assertEqual(audit["matched_prediction_rows"], 1134)


if __name__ == "__main__":
    unittest.main()
