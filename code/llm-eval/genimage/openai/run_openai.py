import argparse
import base64
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation_utils import (
    OUTPUT_DIR,
    PROMPT,
    PROMPT_SHA256,
    SHARED_IMAGES,
    VIEWS,
    config_hash,
    parse_score,
    request_key,
    safe_model_slug,
)
from prepare_views import validate_shared_set

MAX_OUTPUT_TOKENS = 128
MAX_ATTEMPTS_PER_RUN = 4
REQUEST_TIMEOUT_SECONDS = 120
RETRY_BASE_SECONDS = 5
RETRY_MAX_SECONDS = 60
COMPLETED_STATUSES = {"success", "refusal", "invalid_output"}


def request_config(model_id: str, image_detail: str,
                   temperature_zero: bool = False) -> dict[str, Any]:
    generation_parameters: dict[str, Any] = {
        "max_output_tokens": MAX_OUTPUT_TOKENS,
    }
    if temperature_zero:
        generation_parameters["temperature"] = 0
    else:
        generation_parameters["temperature"] = "omitted"
    return {
        "provider": "OpenAI",
        "endpoint": "Responses API",
        "model_id_requested": model_id,
        "image_detail": image_detail,
        "generation_parameters": generation_parameters,
        "tools": [],
        "conversation_history": False,
        "store": False,
        "prompt_sha256": PROMPT_SHA256,
    }


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def emit_dry_run(summary: dict[str, Any], summary_path: Path | None) -> None:
    rendered = json.dumps(summary, indent=2)
    if summary_path:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


def response_text(response: Any) -> tuple[str, bool]:
    texts: list[str] = []
    refused = False
    for item in response.output:
        for content in getattr(item, "content", []):
            if getattr(content, "type", None) == "refusal":
                refused = True
                text = getattr(content, "refusal", None)
                if text:
                    texts.append(text)
            elif getattr(content, "type", None) == "output_text":
                texts.append(content.text)
    return "\n".join(texts).strip(), refused


def usage_data(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "input_tokens_details": None,
            "output_tokens_details": None,
            "reported_cost_usd": None,
        }
    data = usage.model_dump(mode="json") if hasattr(usage, "model_dump") else {}
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
        "input_tokens_details": data.get("input_tokens_details"),
        "output_tokens_details": data.get("output_tokens_details"),
        "reported_cost_usd": data.get("cost"),
    }


def classify_api_error(exc: Exception) -> tuple[bool, str]:
    try:
        import openai
    except ImportError as import_error:
        raise RuntimeError("Install the official OpenAI Python SDK with `pip install openai`.") from import_error

    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True, "transient_network_error"
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        return status in {408, 409, 429} or status >= 500, f"http_{status}"
    if isinstance(exc, openai.APIError):
        return False, "nonretryable_api_error"
    return False, "client_or_local_error"


def call_once(client: Any, image_bytes: bytes, detail: str, model_id: str,
              generation_parameters: dict[str, Any]) -> Any:
    image_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    request: dict[str, Any] = {
        "model": model_id,
        "input": [{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": image_url, "detail": detail},
                {"type": "input_text", "text": PROMPT},
            ],
        }],
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
        "tools": [],
    }
    if generation_parameters.get("temperature") == 0:
        request["temperature"] = 0
    return client.responses.create(**request)


def make_record(item: dict[str, Any], config: dict[str, Any], digest: str,
                attempt: int, cost_rates: dict[str, float] | None) -> dict[str, Any]:
    return {
        "source_image_id": str(item["source_image_id"]),
        "source_image_sha256": str(item["source_image_sha256"]),
        "submitted_image_sha256": str(item["submitted_image_sha256"]),
        "view": str(item["view"]),
        "model_id_requested": config["model_id_requested"],
        "model_id_returned": None,
        "prompt_sha256": PROMPT_SHA256,
        "configuration_sha256": digest,
        "configuration": config,
        "cost_rates_usd_per_million_tokens": cost_rates,
        "image_detail_requested": config["image_detail"],
        "generation_parameters": config["generation_parameters"],
        "attempt": attempt,
        "request_started_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_completed_at_utc": None,
        "api_request_id": None,
        "response_id": None,
        "response_status": None,
        "status": None,
        "raw_response_text": None,
        "raw_response": None,
        "p_fake": None,
        "error_type": None,
        "error_message": None,
        "retryable": False,
        "retry_history": [],
        **usage_data(None),
        "estimated_cost_usd": None,
    }


def run(manifest_path: Path, model_id: str, detail: str, output: Path,
        view: str | None, limit: int, dry_run: bool,
        temperature_zero: bool = False,
        cost_rates: dict[str, float] | None = None,
        summary_path: Path | None = None) -> int:
    if not manifest_path.is_file():
        if not dry_run:
            raise FileNotFoundError(
                f"Prepared input manifest is missing: {manifest_path}; run prepare_views.py first."
            )
        shared = pd.read_csv(SHARED_IMAGES)
        validate_shared_set(shared)
        pilot = pd.read_csv(
            SHARED_IMAGES.parent.parent / "pilot_unique_api_manifest.csv",
            dtype={"sha256": str},
        )
        per_view = {name: 630 for name in VIEWS}
        if view:
            per_view = {view: per_view[view]}
        manifest_rows = sum(per_view.values())
        planned = manifest_rows
        if limit:
            planned = min(planned, limit)
            if view:
                per_view[view] = planned
            else:
                per_view = {
                    name: planned // len(VIEWS) + int(index < planned % len(VIEWS))
                    for index, name in enumerate(VIEWS)
                }
        config = request_config(model_id or "MODEL_NOT_SELECTED", detail, temperature_zero)
        emit_dry_run({
            "mode": "dry-run",
            "model_id_requested": config["model_id_requested"],
            "configuration_sha256": config_hash(config),
            "shared_images": len(shared),
            "real_images": int((shared["label"] == 0).sum()),
            "fake_images": int((shared["label"] == 1).sum()),
            "pilot_intersection_images": len(set(shared["sha256"]) & set(pilot["sha256"])),
            "planned_manifest_rows": manifest_rows,
            "requests_planned": planned,
            "views": per_view,
            "estimated_total_evaluations": 1890,
            "estimated_retries_excluded": True,
            "prepared_input_manifest_exists": False,
            "completed_keys_can_be_checked": False,
            "output_file": str(output),
            "api_key_required": False,
            "paid_requests_made": 0,
        }, summary_path)
        return 0

    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    required = {
        "source_image_id", "source_image_sha256", "submitted_image_sha256",
        "view", "image_file",
    }
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Input manifest is missing fields: {sorted(missing_columns)}")
    if manifest.duplicated(["source_image_id", "view"]).any():
        raise ValueError("Input manifest contains duplicate source-image/view rows")
    if view:
        if view not in VIEWS:
            raise ValueError(f"Unsupported view: {view}")
        manifest = manifest[manifest["view"] == view].copy()
    if not manifest["view"].isin(VIEWS).all():
        raise ValueError("Manifest contains an unsupported view")
    if manifest.empty:
        raise ValueError("No matching rows in input manifest")

    config = request_config(model_id or "MODEL_NOT_SELECTED", detail, temperature_zero)
    digest = config_hash(config)
    if dry_run:
        existing = read_log(output)
        completed_keys = {
            request_key(row)
            for row in existing
            if row.get("status") in COMPLETED_STATUSES
        }
        todo = [
            item for item in manifest.to_dict("records")
            if request_key({
                "source_image_id": item["source_image_id"],
                "submitted_image_sha256": item["submitted_image_sha256"],
                "view": item["view"],
                "model_id_requested": config["model_id_requested"],
                "prompt_sha256": PROMPT_SHA256,
                "configuration_sha256": digest,
            }) not in completed_keys
        ]
        if limit:
            todo = todo[:limit]
        emit_dry_run({
            "mode": "dry-run",
            "model_id_requested": config["model_id_requested"],
            "configuration_sha256": digest,
            "planned_manifest_rows": len(manifest),
            "already_completed_for_exact_configuration": len(manifest) - len(todo),
            "requests_planned": len(todo),
            "views": manifest["view"].value_counts().to_dict(),
            "estimated_total_evaluations": 1890,
            "estimated_retries_excluded": True,
            "output_file": str(output),
            "api_key_required": False,
            "paid_requests_made": 0,
        }, summary_path)
        return 0

    if model_id == "MODEL_NOT_SELECTED":
        raise ValueError("Provide the model ID selected by the user before execution.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("Install the official OpenAI SDK with `pip install openai`.") from exc

    existing = read_log(output)
    completed_keys = {
        request_key(row)
        for row in existing
        if row.get("status") in COMPLETED_STATUSES
    }
    attempt_counts: dict[tuple[str, ...], int] = {}
    for record in existing:
        key = request_key(record)
        attempt_counts[key] = max(attempt_counts.get(key, 0), int(record.get("attempt", 0)))

    pending = []
    for item in manifest.to_dict("records"):
        key_fields = {
            "source_image_id": item["source_image_id"],
            "submitted_image_sha256": item["submitted_image_sha256"],
            "view": item["view"],
            "model_id_requested": model_id,
            "prompt_sha256": PROMPT_SHA256,
            "configuration_sha256": digest,
        }
        if request_key(key_fields) not in completed_keys:
            pending.append((item, key_fields))
    if limit:
        pending = pending[:limit]

    print(
        f"Model {model_id}: {len(pending)} requests queued "
        f"({len(completed_keys)} exact completed request keys); output {output}",
        flush=True,
    )
    client = OpenAI(max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS)

    for index, (item, key_fields) in enumerate(pending, 1):
        image_path = manifest_path.parent.parent / item["image_file"]
        if not image_path.is_file():
            image_path = manifest_path.parent.parent.parent / item["image_file"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Prepared image is missing: {image_path}")
        image_bytes = image_path.read_bytes()
        image_hash = __import__("hashlib").sha256(image_bytes).hexdigest()
        if image_hash != item["submitted_image_sha256"]:
            raise ValueError(f"Submitted-image hash mismatch for {item['source_image_id']}")

        key = request_key(key_fields)
        history = [
            {
                "attempt": previous["attempt"],
                "error_type": previous.get("error_type"),
                "error_message": previous.get("error_message"),
                "retryable": previous.get("retryable", False),
                "timestamp_utc": previous.get("request_completed_at_utc"),
            }
            for previous in existing
            if request_key(previous) == key and previous.get("status") == "api_failure"
        ]
        first_attempt = attempt_counts.get(key, 0) + 1
        for local_attempt in range(MAX_ATTEMPTS_PER_RUN):
            attempt = first_attempt + local_attempt
            record = make_record(item, config, digest, attempt, cost_rates)
            record["retry_history"] = list(history)
            try:
                response = call_once(
                    client, image_bytes, detail, model_id, config["generation_parameters"]
                )
            except Exception as exc:
                retryable, error_type = classify_api_error(exc)
                message = f"{type(exc).__name__}: {str(exc)}"
                record.update({
                    "request_completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "api_failure",
                    "error_type": error_type,
                    "error_message": message[:2000],
                    "retryable": retryable,
                    "api_request_id": getattr(exc, "request_id", None),
                })
                append_checkpoint(output, record)
                history.append({
                    "attempt": attempt,
                    "error_type": error_type,
                    "error_message": message[:500],
                    "retryable": retryable,
                    "timestamp_utc": record["request_completed_at_utc"],
                })
                if not retryable or local_attempt == MAX_ATTEMPTS_PER_RUN - 1:
                    break
                delay_seconds = min(RETRY_BASE_SECONDS * (2 ** local_attempt), RETRY_MAX_SECONDS)
                print(
                    f"[{index}/{len(pending)}] transient API error; retry {local_attempt + 1} "
                    f"after {delay_seconds}s ({error_type})",
                    flush=True,
                )
                time.sleep(delay_seconds)
                continue

            raw, refused = response_text(response)
            response_error = getattr(response, "error", None)
            record.update({
                "model_id_returned": getattr(response, "model", None),
                "api_request_id": getattr(response, "_request_id", None),
                "response_id": getattr(response, "id", None),
                "response_status": getattr(response, "status", None),
                "raw_response_text": raw,
                "raw_response": response.model_dump(mode="json"),
                "request_completed_at_utc": datetime.now(timezone.utc).isoformat(),
                **usage_data(response),
            })
            if cost_rates and record["input_tokens"] is not None \
                    and record["output_tokens"] is not None:
                record["estimated_cost_usd"] = (
                    record["input_tokens"] * cost_rates["input"]
                    + record["output_tokens"] * cost_rates["output"]
                ) / 1_000_000
            if getattr(response, "status", None) == "failed" or response_error:
                record["status"] = "api_failure"
                record["error_type"] = "APIResponseFailure"
                record["error_message"] = str(response_error or "API returned failed status")[:2000]
            elif refused:
                record["status"] = "refusal"
            else:
                try:
                    record["p_fake"] = parse_score(raw)
                    record["status"] = "success"
                except ValueError as exc:
                    record["status"] = "invalid_output"
                    record["error_type"] = "InvalidModelOutput"
                    record["error_message"] = str(exc)
            append_checkpoint(output, record)
            break
        print(f"[{index}/{len(pending)}] {item['view']} -> checkpointed", flush=True)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run isolated GPT image evaluations through the official OpenAI Responses API."
    )
    parser.add_argument("--manifest", type=Path, default=OUTPUT_DIR / "manifests/input_manifest.csv")
    parser.add_argument("--model", default=None, help="Exact OpenAI model ID; required for execution.")
    parser.add_argument("--detail", choices=["low", "high", "original", "auto"], default="auto")
    parser.add_argument(
        "--temperature-zero",
        action="store_true",
        help="Send temperature=0 only after confirming support for the selected model.",
    )
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=OUTPUT_DIR / "manifests/dry_run_summary.json",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit new requests; 0 means no limit.")
    parser.add_argument("--dry-run", action="store_true", help="Summarize without making API calls.")
    parser.add_argument("--execute", action="store_true", help="Required to make paid API calls.")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit must be non-negative")
    if (args.input_price_per_million is None) != (args.output_price_per_million is None):
        parser.error("Provide both price-per-million values or neither")
    if any(value is not None and value < 0 for value in (
        args.input_price_per_million, args.output_price_per_million
    )):
        parser.error("Token prices must be non-negative")
    if not args.dry_run and not args.execute:
        parser.error("Use --dry-run or explicitly pass --execute")
    model_id = args.model or "MODEL_NOT_SELECTED"
    config = request_config(model_id, args.detail, args.temperature_zero)
    digest = config_hash(config)
    default_output = OUTPUT_DIR / "logs" / f"requests_{safe_model_slug(model_id)}_{digest[:12]}.jsonl"
    output = args.output or default_output
    status = run(
        args.manifest, model_id, args.detail, output, args.view, args.limit,
        args.dry_run, args.temperature_zero,
        (
            {"input": args.input_price_per_million, "output": args.output_price_per_million}
            if args.input_price_per_million is not None else None
        ),
        args.summary_output if args.dry_run else None,
    )
    sys.exit(status)


if __name__ == "__main__":
    main()
