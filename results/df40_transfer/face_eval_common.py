"""Protocol and result checks shared by the face API evaluations."""

import contextlib
import fcntl
import hashlib
import json
import math


def protocol_hash(protocol):
    return hashlib.sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def probability(value):
    if isinstance(value, bool):
        raise ValueError("Boolean is not a probability")
    score = float(value)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("Probability must be finite and between 0 and 100")
    return score


def validate_predictions(manifest, predictions, dataset, protocol):
    """Fail closed before publishing metrics or accepting resumed successes."""
    expected = {row["sha256"]: row for row in manifest}
    if not expected or len(expected) != len(manifest):
        raise ValueError("Manifest is empty or contains duplicate hashes")
    if len(predictions) != len(expected):
        raise ValueError("Expected exactly one successful prediction per manifest image")
    if {row["sha256"] for row in predictions} != set(expected):
        raise ValueError("Prediction hashes do not exactly match the manifest")
    prompt_hash = hashlib.sha256(protocol["prompt"].encode()).hexdigest()
    threshold = probability(protocol["decision_threshold"])
    for row in predictions:
        source = expected[row["sha256"]]
        if row.get("status") != "ok" or row.get("error"):
            raise ValueError("Prediction contains a failure")
        for key in ("label", "method", "path"):
            if row.get(key) != source.get(key):
                raise ValueError(f"Prediction {key} mismatch")
        if source["label"] not in {"0", "1"}:
            raise ValueError("Invalid manifest label")
        if (row.get("dataset") != dataset or row.get("model") != protocol["model"]
                or row.get("prompt_sha256") != prompt_hash
                or row.get("protocol_sha256") != protocol_hash(protocol)):
            raise ValueError("Prediction dataset, model, prompt or protocol mismatch")
        score = probability(row["p_manipulated"])
        if row.get("prediction") != str(int(score >= threshold)):
            raise ValueError("Prediction disagrees with the frozen threshold")
        raw = json.loads(row["raw_response"])
        if not isinstance(raw, dict) or set(raw) != {"p_manipulated"}:
            raise ValueError("Invalid raw response")
        if not math.isclose(probability(raw["p_manipulated"]), score, abs_tol=1e-7):
            raise ValueError("Saved score differs from raw response")


@contextlib.contextmanager
def run_lock(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "RUN.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another process is writing this output directory") from exc
        yield


def freeze_run(output_dir, manifest_path, rows, dataset, protocol):
    record = {
        "schema_version": 1,
        "dataset": dataset,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "selected_hashes": [row["sha256"] for row in rows],
        "protocol": protocol,
        "protocol_sha256": protocol_hash(protocol),
    }
    path = output_dir / "run.json"
    if path.exists():
        if json.loads(path.read_text()) != record:
            raise ValueError("Frozen run changed; use a new output directory")
    else:
        if any((output_dir / name).exists() for name in ("responses.jsonl", "predictions.csv")):
            raise ValueError("Legacy output has no frozen run record; use a new output directory")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n")
        temporary.replace(path)


def retryable(exc):
    """Retry transient service errors, never bad credentials, quota or bad input."""
    code = getattr(exc, "code", None)
    if code in {"insufficient_quota", "credit_balance_exhausted", "billing_hard_limit_reached"}:
        return False
    status = getattr(exc, "status_code", None)
    if status is None:
        status = code if isinstance(code, int) else None
    return (status in {408, 409, 429} or (isinstance(status, int) and status >= 500)
            or type(exc).__name__ in {"APIConnectionError", "APITimeoutError"})


def retry_delay(exc, attempt):
    headers = getattr(getattr(exc, "response", None), "headers", {})
    try:
        return max(0, float(headers["retry-after"]))
    except (KeyError, TypeError, ValueError):
        return min(60, 2 ** (attempt - 1))
