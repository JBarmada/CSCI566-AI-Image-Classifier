"""Run a resumable, checkpointed Gemini evaluation on unique GenImage files."""

import argparse
import csv
import hashlib
import json
import mimetypes
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types


FIELDS = (
    "sha256",
    "path",
    "label",
    "source_generator",
    "p_fake",
    "prediction",
    "status",
    "error",
    "attempts",
    "model",
    "prompt_sha256",
    "timestamp_utc",
    "raw_response",
)


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_file(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_events(path, allowed_hashes, model, prompt_sha256):
    events = []
    completed = set()
    if not path.exists():
        return events, completed
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            event = json.loads(line)
            digest = event.get("sha256")
            if digest not in allowed_hashes:
                raise ValueError(f"Unknown image hash on event line {line_number}: {digest}")
            if event.get("model") != model or event.get("prompt_sha256") != prompt_sha256:
                raise ValueError("Existing checkpoint uses a different model or prompt")
            events.append(event)
            if event.get("status") == "ok":
                completed.add(digest)
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


def parse_probability(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"p_fake"}:
        raise ValueError("Response must contain only p_fake")
    probability = float(value["p_fake"])
    if not 0 <= probability <= 100:
        raise ValueError("p_fake must be between 0 and 100")
    return probability


def evaluate(row, image_path, client, protocol, prompt_sha256, retries):
    raw = ""
    error = ""
    attempts = 0
    for attempts in range(1, retries + 1):
        try:
            data = image_path.read_bytes()
            if digest_bytes(data) != row["sha256"]:
                raise ValueError(f"Image hash mismatch: {image_path}")
            mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
            response = client.models.generate_content(
                model=protocol["model"],
                contents=[
                    types.Part.from_bytes(data=data, mime_type=mime_type),
                    protocol["prompt"],
                ],
                config=types.GenerateContentConfig(
                    temperature=protocol["temperature"],
                    max_output_tokens=protocol["max_output_tokens"],
                    response_mime_type="application/json",
                ),
            )
            raw = response.text or ""
            p_fake = parse_probability(raw)
            return {
                "sha256": row["sha256"],
                "path": row["path"],
                "label": row["label"],
                "source_generator": row["source_generator"],
                "p_fake": f"{p_fake:.10g}",
                "prediction": str(int(p_fake >= protocol["decision_threshold"])),
                "status": "ok",
                "error": "",
                "attempts": str(attempts),
                "model": protocol["model"],
                "prompt_sha256": prompt_sha256,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "raw_response": raw,
            }
        except Exception as exc:  # API errors vary by SDK release.
            error = f"{type(exc).__name__}: {exc}"
            if attempts < retries:
                time.sleep(min(60, (2 ** (attempts - 1)) + random.random()))
    return {
        "sha256": row["sha256"],
        "path": row["path"],
        "label": row["label"],
        "source_generator": row["source_generator"],
        "p_fake": "",
        "prediction": "",
        "status": "error",
        "error": error,
        "attempts": str(attempts),
        "model": protocol["model"],
        "prompt_sha256": prompt_sha256,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "raw_response": raw,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/llm_eval/pilot_unique_api_manifest.csv"),
    )
    parser.add_argument("--protocol", type=Path, default=Path("results/llm_eval/protocol.json"))
    parser.add_argument("--data-root", type=Path, default=Path("data/extracted/GenImage_Compact"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/llm_eval/gemini_pilot"))
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int, help="Optional smoke-test limit")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the current shell; do not save it in the repository")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    prompt_sha256 = digest_bytes(protocol["prompt"].encode("utf-8"))
    rows = load_csv(args.manifest)
    if not rows or len(rows) != len({row["sha256"] for row in rows}):
        raise ValueError("Manifest must contain at least one row and unique SHA-256 values")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    events_path = args.output_dir / "responses.jsonl"
    checkpoint_path = args.output_dir / "gemini_full_predictions.csv"
    allowed = {row["sha256"] for row in rows}
    events, completed = load_events(events_path, allowed, protocol["model"], prompt_sha256)
    client = genai.Client(api_key=api_key)

    with events_path.open("a", encoding="utf-8") as stream:
        for index, row in enumerate(rows, 1):
            if row["sha256"] in completed:
                continue
            image_path = args.data_root / row["path"]
            event = evaluate(row, image_path, client, protocol, prompt_sha256, args.retries)
            events.append(event)
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            if event["status"] == "ok":
                completed.add(row["sha256"])
            write_checkpoint(checkpoint_path, events)
            print(f"{index}/{len(rows)} {event['status']} {row['sha256'][:12]}", flush=True)

    write_checkpoint(checkpoint_path, events)
    successful = sum(event["status"] == "ok" for event in latest_events(events))
    print(f"Successful unique predictions: {successful}/{len(rows)}")
    if successful != len(rows):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
