import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
THRESHOLD = 50.0


def parse(raw) -> float:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("empty response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    value = float(json.loads(text)["p_fake"])
    if not 0 <= value <= 100:
        raise ValueError(f"p_fake outside 0-100: {value}")
    return value


def read_responses(path: Path) -> pd.DataFrame:
    if path.suffix == ".csv":
        frame = pd.read_csv(path, dtype={"raw_response": str, "error": str})
        if "error" in frame:
            frame = frame[frame.error.isna() | (frame.error == "")]
        return frame.drop_duplicates("blind_id", keep="last")
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        if " | " not in line:
            raise SystemExit(f"{path.name} line {n} is not 'img_XXXX | response': {line[:60]}")
        blind_id, response = line.split(" | ", 1)
        rows.append({"blind_id": blind_id.strip(), "raw_response": response.strip()})
    return pd.DataFrame(rows, columns=["blind_id", "raw_response"])


def metrics(group: pd.DataFrame) -> dict:
    y, p = group.label.to_numpy(), group.p_fake.to_numpy()
    pred = (p >= THRESHOLD).astype(int)
    real, fake = y == 0, y == 1
    out = {"n": len(group), "n_real": int(real.sum()), "n_fake": int(fake.sum()),
           "accuracy": float((pred == y).mean()),
           "recall": float(pred[fake].mean()) if fake.any() else np.nan,
           "specificity": float(1 - pred[real].mean()) if real.any() else np.nan,
           "mean_p_fake_real": float(p[real].mean()) if real.any() else np.nan,
           "mean_p_fake_fake": float(p[fake].mean()) if fake.any() else np.nan}
    out["balanced_accuracy"] = np.nanmean([out["recall"], out["specificity"]])
    out["auroc"] = float(roc_auc_score(y, p)) if real.any() and fake.any() else np.nan
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description="Join Claude's blind scores with the answer key and compute metrics.")
    parser.add_argument("--raw", type=Path, default=HERE / "claude_responses.txt")
    parser.add_argument("--key", type=Path, default=HERE / "answer_key.csv")
    parser.add_argument("--model", required=True, help="Exact model shown in Claude, e.g. claude-opus-5-5")
    parser.add_argument("--out-dir", type=Path, default=HERE)
    parser.add_argument("--prefix", default="claude", help="Output file prefix, e.g. claude_api")
    args = parser.parse_args(argv)

    key = pd.read_csv(args.key)
    raw = read_responses(args.raw)
    if not {"blind_id", "raw_response"} <= set(raw.columns):
        raise SystemExit("responses need blind_id and raw_response")
    dupes = raw[raw.blind_id.duplicated(keep=False)]
    if not dupes.empty:
        raise SystemExit(f"Duplicate rows for: {sorted(dupes.blind_id.unique())[:10]}")
    unknown = set(raw.blind_id) - set(key.blind_id)
    if unknown:
        raise SystemExit(f"IDs not in the answer key: {sorted(unknown)[:10]}")

    rows = []
    merged = key.merge(raw, on="blind_id", how="left")
    for _, r in merged.iterrows():
        status, error, value = "success", "", np.nan
        if pd.isna(r.get("raw_response")):
            status, error = "missing", "no response recorded"
        else:
            try:
                value = parse(r.raw_response)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                status, error = "invalid", str(exc)
        rows.append({"sha256": r.sha256, "path": r.path, "model": args.model, "provider": "Anthropic",
                     "p_fake": value, "raw_response": r.get("raw_response", ""), "status": status,
                     "error": error, "timestamp_utc": r.get("timestamp_utc", ""),
                     "blind_id": r.blind_id, "label": r.label, "source_generator": r.source_generator,
                     "group": r.group})
    preds = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    preds.drop(columns=["label", "source_generator", "group"]).to_csv(
        args.out_dir / f"{args.prefix}_predictions.csv", index=False)

    ok = preds[preds.status == "success"]
    overall = pd.DataFrame([{"model": args.model, **metrics(ok)}])
    real = ok[ok.label == 0]
    per_gen = pd.DataFrame([{"model": args.model, "generator": g, **metrics(pd.concat([fakes, real]))}
                            for g, fakes in ok[ok.label == 1].groupby("source_generator")])
    status = preds.status.value_counts().rename_axis("status").reset_index(name="images")
    overall.to_csv(args.out_dir / f"{args.prefix}_overall_metrics.csv", index=False)
    per_gen.to_csv(args.out_dir / f"{args.prefix}_per_generator_metrics.csv", index=False)
    status.to_csv(args.out_dir / f"{args.prefix}_response_status.csv", index=False)

    print(status.to_string(index=False))
    print(overall.round(4).to_string(index=False))
    print(per_gen[["generator", "n_fake", "recall", "mean_p_fake_fake", "auroc"]].round(4).to_string(index=False))
    if (preds.status != "success").any():
        print("\nRe-run or record these before reporting:", ", ".join(preds[preds.status != "success"].blind_id[:20]))


if __name__ == "__main__":
    main()
