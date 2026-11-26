import importlib.util
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]


def load(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


evaluate = load(
    "openai_face_evaluate", "results/df40_transfer/openai_face/evaluate.py"
)


class OpenAIFaceEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads(
            (ROOT / "results/df40_transfer/openai_face/protocol.json").read_text()
        )
        self.prompt_hash = hashlib.sha256(self.protocol["prompt"].encode()).hexdigest()
        self.row = {"sha256": hashlib.sha256(b"image").hexdigest(),
                    "path": "image.jpg", "label": "1", "method": "test"}

    def test_parse_probability(self):
        self.assertEqual(evaluate.parse_probability('{"p_manipulated": 62.5}'), 62.5)

    def test_parse_probability_rejects_extra_fields(self):
        with self.assertRaises(ValueError):
            evaluate.parse_probability('{"p_manipulated": 50, "label": 1}')

    def test_response_request_uses_private_structured_image_input(self):
        protocol = {
            "model": "test-model",
            "prompt": "test prompt",
            "max_output_tokens": 32,
        }
        request = evaluate.response_request(b"image", "image/jpeg", protocol)
        self.assertFalse(request["store"])
        self.assertEqual(request["text"]["format"]["type"], "json_schema")
        content = request["input"][0]["content"]
        self.assertEqual(content[0], {"type": "input_text", "text": "test prompt"})
        self.assertTrue(content[1]["image_url"].startswith("data:image/jpeg;base64,"))

    def test_protocol_matches_gemini_prompt_and_threshold(self):
        openai_protocol = json.loads(
            (ROOT / "results/df40_transfer/openai_face/protocol.json").read_text()
        )
        gemini_protocol = json.loads(
            (ROOT / "results/df40_transfer/gemini_face/protocol.json").read_text()
        )
        self.assertEqual(openai_protocol["prompt"], gemini_protocol["prompt"])
        self.assertEqual(
            openai_protocol["decision_threshold"], gemini_protocol["decision_threshold"]
        )

    def test_freeze_prevents_mixed_settings_and_legacy_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.csv"
            manifest.write_text("frozen manifest")
            output = root / "output"
            with evaluate.run_lock(output):
                evaluate.freeze_run(output, manifest, [self.row], "test", self.protocol)
                evaluate.freeze_run(output, manifest, [self.row], "test", self.protocol)
                with self.assertRaises(ValueError):
                    evaluate.freeze_run(output, manifest, [self.row], "test",
                                        {**self.protocol, "image_detail": "low"})
                with self.assertRaises(RuntimeError):
                    with evaluate.run_lock(output):
                        pass
            legacy = root / "legacy"
            legacy.mkdir()
            (legacy / "responses.jsonl").write_text("{}\n")
            with self.assertRaises(ValueError):
                evaluate.freeze_run(legacy, manifest, [self.row], "test", self.protocol)

    def test_success_usage_and_resume_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            path.write_bytes(b"image")
            client = Mock()
            client.responses.create.return_value = SimpleNamespace(
                id="response-id", model=self.protocol["model"], status="completed",
                output_text='{"p_manipulated":75}',
                usage=SimpleNamespace(model_dump=lambda: {"input_tokens": 123, "output_tokens": 12}),
            )
            event = evaluate.evaluate(self.row, "test", path, client, self.protocol, self.prompt_hash, 3)
            self.assertEqual(event["status"], "ok")
            self.assertEqual(json.loads(event["usage_json"])[0]["usage"]["input_tokens"], 123)
            events_path = Path(directory) / "responses.jsonl"
            events_path.write_text(json.dumps(event) + "\n")
            args = (events_path, {self.row["sha256"]}, "test", self.protocol["model"], self.prompt_hash)
            _, completed = evaluate.load_events(*args, self.protocol, [self.row])
            self.assertEqual(completed, {self.row["sha256"]})
            event["prediction"] = "0"
            events_path.write_text(json.dumps(event) + "\n")
            with self.assertRaises(ValueError):
                evaluate.load_events(*args, self.protocol, [self.row])

    def test_quota_failure_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            path.write_bytes(b"image")
            client = Mock()
            error = RuntimeError("quota exhausted")
            error.code = "insufficient_quota"
            error.status_code = 429
            client.responses.create.side_effect = error
            event = evaluate.evaluate(self.row, "test", path, client, self.protocol, self.prompt_hash, 5)
            self.assertTrue(event["stop_run"])
            self.assertEqual(client.responses.create.call_count, 1)

    def test_transient_failure_retries_and_incomplete_response_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.jpg"
            path.write_bytes(b"image")
            client = Mock()
            transient = RuntimeError("temporarily unavailable")
            transient.status_code = 503
            client.responses.create.side_effect = [transient, SimpleNamespace(
                id="incomplete", model=self.protocol["model"], status="incomplete",
                output_text="", usage=None, incomplete_details="max_output_tokens",
            )]
            with patch.object(evaluate.time, "sleep"):
                event = evaluate.evaluate(self.row, "test", path, client, self.protocol, self.prompt_hash, 5)
            self.assertEqual(client.responses.create.call_count, 2)
            self.assertEqual(event["status"], "error")
            self.assertTrue(event["stop_run"])
            self.assertIn("incomplete", event["error"])


if __name__ == "__main__":
    unittest.main()
