"""Run resumable OpenAI vision inference on a frozen face manifest."""

import argparse
import base64
import csv
import hashlib
import json
import mimetypes
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from face_eval_common import (
    freeze_run, probability, protocol_hash, retry_delay, retryable,
    run_lock, validate_predictions,
)


FIELDS = (
    "dataset",
    "sha256",
    "path",
    "method",
    "label",
    "pair_id",
    "p_manipulated",
    "prediction",
    "status",
    "error",
    "attempts",
    "model",
    "prompt_sha256",
    "protocol_sha256",
    "usage_json",
    "stop_run",
    "timestamp_utc",
    "raw_response",
)


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def read_manifest(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"path", "sha256", "method", "label"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    hashes = [row["sha256"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError("Manifest SHA-256 values must be unique")
    if any(row["label"] not in {"0", "1"} for row in rows):
        raise ValueError("Manifest labels must be 0 or 1")
    return rows


def image_path(row, data_root):
    path = Path(row["path"])
    return path if path.is_absolute() else data_root / path


def verify_images(rows, data_root):
    for index, row in enumerate(rows, 1):
        path = image_path(row, data_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256_bytes(path.read_bytes()) != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {path}")
        print(f"verified {index}/{len(rows)} {row['sha256'][:12]}", flush=True)


def parse_probability(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"p_manipulated"}:
        raise ValueError("Response must contain only p_manipulated")
    return probability(value["p_manipulated"])


def load_events(path, allowed_hashes, dataset, model, prompt_sha256, protocol=None, rows=None):
    events = []
    completed = set()
    if not path.exists():
        return events, completed
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("sha256") not in allowed_hashes:
                raise ValueError(f"Unknown image hash on event line {line_number}")
            if (
                event.get("dataset") != dataset
                or event.get("model") != model
                or event.get("prompt_sha256") != prompt_sha256
                or (protocol is not None
                    and event.get("protocol_sha256") != protocol_hash(protocol))
            ):
                raise ValueError("Existing checkpoint uses a different dataset, model or prompt")
            events.append(event)
    latest = latest_events(events)
    for event in latest:
        if event.get("status") == "ok":
            if protocol is not None:
                source = next(row for row in rows if row["sha256"] == event["sha256"])
                validate_predictions([source], [event], dataset, protocol)
            completed.add(event["sha256"])
    return events, completed


def latest_events(events):
    latest = {}
    for event in events:
        latest[event["sha256"]] = event
    return [latest[digest] for digest in sorted(latest)]


def write_checkpoint(path, events):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(latest_events(events))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def response_request(data, mime_type, protocol):
    encoded = base64.b64encode(data).decode("ascii")
    return {
        "model": protocol["model"],
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": protocol["prompt"]},
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{encoded}",
                        "detail": protocol.get("image_detail", "high"),
                    },
                ],
            }
        ],
        "max_output_tokens": protocol["max_output_tokens"],
        "reasoning": {"effort": protocol.get("reasoning_effort", "low")},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "face_manipulation_probability",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "p_manipulated": {"type": "number", "minimum": 0, "maximum": 100}
                    },
                    "required": ["p_manipulated"],
                    "additionalProperties": False,
                },
            }
        },
        "store": False,
    }


def evaluate(row, dataset, path, client, protocol, prompt_sha256, retries):
    raw = ""
    error = ""
    attempts = 0
    usage = []
    for attempts in range(1, retries + 1):
        try:
            data = path.read_bytes()
            if sha256_bytes(data) != row["sha256"]:
                raise ValueError(f"Image hash mismatch: {path}")
            mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
            response = client.responses.create(**response_request(data, mime_type, protocol))
            usage.append({
                "response_id": response.id,
                "model": response.model,
                "status": response.status,
                "usage": response.usage.model_dump() if response.usage else None,
            })
            raw = response.output_text or ""
            if response.status != "completed":
                raise ValueError(f"Response {response.status}: {response.incomplete_details}")
            probability = parse_probability(raw)
            event = make_event(
                row, dataset, protocol, prompt_sha256, attempts, "ok", raw,
                probability=probability,
            )
            event["usage_json"] = json.dumps(usage)
            return event
        except Exception as exc:  # API exceptions vary by SDK release.
            error = f"{type(exc).__name__}: {exc}"
            if not retryable(exc):
                break
            if attempts < retries:
                time.sleep(retry_delay(exc, attempts))
    event = make_event(
        row, dataset, protocol, prompt_sha256, attempts, "error", raw, error=error
    )
    event["usage_json"] = json.dumps(usage)
    event["stop_run"] = True
    return event


def make_event(
    row, dataset, protocol, prompt_sha256, attempts, status, raw,
    probability=None, error="",
):
    return {
        "dataset": dataset,
        "sha256": row["sha256"],
        "path": row["path"],
        "method": row["method"],
        "label": row["label"],
        "pair_id": row.get("pair_id") or row.get("pair") or "",
        "p_manipulated": "" if probability is None else f"{probability:.10g}",
        "prediction": "" if probability is None else str(
            int(probability >= protocol["decision_threshold"])
        ),
        "status": status,
        "error": error,
        "attempts": str(attempts),
        "model": protocol["model"],
        "prompt_sha256": prompt_sha256,
        "protocol_sha256": protocol_hash(protocol),
        "usage_json": "[]",
        "stop_run": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_response": raw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--protocol", type=Path,
        default=Path("results/df40_transfer/openai_face/protocol.json"),
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Smoke-test only; do not summarize as full")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.retries < 1:
        parser.error("--retries must be positive")

    rows = read_manifest(args.manifest)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]
    if args.validate_only:
        verify_images(rows, args.data_root)
        print(f"Validated {len(rows)} images; no API calls made")
        return

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in the shell; never save it in the repository")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the OpenAI SDK from requirements-openai.txt") from exc

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    prompt_sha256 = sha256_bytes(protocol["prompt"].encode("utf-8"))
    with run_lock(args.output_dir):
        freeze_run(args.output_dir, args.manifest, rows, args.dataset, protocol)
        run_evaluation(args, rows, protocol, api_key, OpenAI)


def run_evaluation(args, rows, protocol, api_key, client_class, evaluate_image=evaluate):
    prompt_sha256 = sha256_bytes(protocol["prompt"].encode("utf-8"))
    events_path = args.output_dir / "responses.jsonl"
    predictions_path = args.output_dir / "predictions.csv"
    events, completed = load_events(
        events_path, {row["sha256"] for row in rows}, args.dataset,
        protocol["model"], prompt_sha256, protocol, rows,
    )
    client = client_class(
        api_key=api_key,
        timeout=protocol.get("timeout_seconds", 120),
        max_retries=0,
    )

    with events_path.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            if row["sha256"] in completed:
                continue
            event = evaluate_image(
                row, args.dataset, image_path(row, args.data_root), client,
                protocol, prompt_sha256, args.retries,
            )
            events.append(event)
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if event["status"] == "ok":
                completed.add(row["sha256"])
            write_checkpoint(predictions_path, events)
            print(f"{index}/{len(rows)} {event['status']} {row['sha256'][:12]}", flush=True)
            if event.get("stop_run"):
                print("Stopping after failure; inspect predictions.csv before resuming", flush=True)
                break

    write_checkpoint(predictions_path, events)
    successful = sum(event["status"] == "ok" for event in latest_events(events))
    print(f"Successful predictions: {successful}/{len(rows)}")
    if successful != len(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
