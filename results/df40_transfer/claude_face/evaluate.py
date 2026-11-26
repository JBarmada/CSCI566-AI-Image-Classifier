"""Run Claude on the frozen face manifest using the shared checkpoint workflow."""

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_eval_common import freeze_run, retry_delay, retryable, run_lock
from openai_face.evaluate import (
    make_event, parse_probability, read_manifest, run_evaluation, sha256_bytes,
    verify_images,
)

HERE = Path(__file__).resolve().parent
MANIFEST_SHA256 = "1f18b71d5d4ebd59ff184d1db06cb33f82f99f034274f6aaaa51b72724afae51"


def response_request(data, mime_type, protocol):
    if mime_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
        raise ValueError(f"Unsupported Claude image type: {mime_type}")
    return {
        "model": protocol["model"],
        "max_tokens": protocol["max_output_tokens"],
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": protocol["prompt"]},
            {"type": "image", "source": {
                "type": "base64", "media_type": mime_type,
                "data": base64.b64encode(data).decode("ascii"),
            }},
        ]}],
    }


def evaluate(row, dataset, path, client, protocol, prompt_sha256, retries):
    raw, error, usage = "", "", []
    for attempt in range(1, retries + 1):
        try:
            data = path.read_bytes()
            if sha256_bytes(data) != row["sha256"]:
                raise ValueError(f"Image hash mismatch: {path}")
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            response = client.messages.create(**response_request(data, mime_type, protocol))
            usage.append({
                "response_id": response.id,
                "model": response.model,
                "stop_reason": response.stop_reason,
                "usage": response.usage.model_dump() if response.usage else None,
            })
            raw = "".join(block.text for block in response.content
                          if getattr(block, "type", "") == "text").strip()
            if response.model != protocol["model"]:
                raise ValueError(f"Unexpected returned model: {response.model}")
            if response.stop_reason != "end_turn":
                raise ValueError(f"Response stopped with {response.stop_reason}")
            score = parse_probability(raw)
            event = make_event(row, dataset, protocol, prompt_sha256, attempt, "ok", raw,
                               probability=score)
            event["usage_json"] = json.dumps(usage)
            return event
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            if not retryable(exc):
                break
            if attempt < retries:
                time.sleep(retry_delay(exc, attempt))
    event = make_event(row, dataset, protocol, prompt_sha256, attempt, "error", raw,
                       error=error)
    event["usage_json"] = json.dumps(usage)
    event["stop_run"] = True
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="df40_five_method_v3")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=HERE / "protocol.json")
    parser.add_argument("--limit", type=int, help="Smoke test; use a separate output folder")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.retries < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--retries and --limit must be positive")
    if sha256_bytes(args.manifest.read_bytes()) != MANIFEST_SHA256:
        raise ValueError("Manifest differs from the accepted 189-image face evaluation")
    rows = read_manifest(args.manifest)
    if len(rows) != 189:
        raise ValueError("This evaluation requires the frozen 189-image face manifest")
    # Verify every image before paying for even the first request, including smoke tests.
    verify_images(rows, args.data_root)
    if args.validate_only:
        print("Validated 189 images; no API calls made")
        return
    if args.limit is not None:
        rows = rows[:args.limit]
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("provider") != "Anthropic Messages API":
        raise ValueError("Expected an Anthropic protocol")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Set ANTHROPIC_API_KEY in the shell; never save it in the repository")
    from anthropic import Anthropic

    with run_lock(args.output_dir):
        freeze_run(args.output_dir, args.manifest, rows, args.dataset, protocol)
        run_evaluation(args, rows, protocol, api_key, Anthropic, evaluate_image=evaluate)


if __name__ == "__main__":
    main()
