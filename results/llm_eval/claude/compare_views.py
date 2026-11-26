import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from score_claude import THRESHOLD, parse, read_responses

HERE = Path(__file__).resolve().parent
RUNS = {
    "original": ("claude_api_responses.csv", "answer_key.csv"),
    "square": ("claude_api_square_responses.csv", "answer_key_square.csv"),
    "detector224": ("claude_api_detector224_responses.csv", "answer_key_detector224.csv"),
}


def load(view: str, responses: Path, key: Path) -> pd.DataFrame:
    raw = read_responses(responses)
    raw["p_fake"] = [parse(x) for x in raw.raw_response]
    d = pd.read_csv(key).merge(raw[["blind_id", "p_fake"]], on="blind_id", how="inner")
    d["view"] = view
    return d


def summary(d: pd.DataFrame, score: str = "p_fake") -> dict:
    real, fake = d[d.label == 0], d[d.label == 1]
    return {"n": len(d), "auroc": roc_auc_score(d.label, d[score]),
            "recall": (fake[score] >= THRESHOLD).mean(),
            "specificity": (real[score] < THRESHOLD).mean(),
            "mean_p_fake_real": real[score].mean(), "mean_p_fake_fake": fake[score].mean()}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare Claude's API results across image views.")
    parser.add_argument("--dir", type=Path, default=HERE)
    args = parser.parse_args(argv)

    frames = [load(v, args.dir / r, args.dir / k) for v, (r, k) in RUNS.items()
              if (args.dir / r).exists() and (args.dir / k).exists()]
    if not frames:
        raise SystemExit("No API results found")
    data = pd.concat(frames, ignore_index=True)

    overall = [{"view": v, **summary(g)} for v, g in data.groupby("view", sort=False)]
    orig = data[data.view == "original"].copy()
    if not orig.empty:
        orig["square_only"] = (orig.width == orig.height).astype(float) * 100
        overall.append({"view": "shape-only baseline (square = fake)", **summary(orig, "square_only")})
    overall = pd.DataFrame(overall)

    rows = []
    for (view, gen), fakes in data[data.label == 1].groupby(["view", "source_generator"], sort=False):
        real = data[(data.view == view) & (data.label == 0)]
        rows.append({"view": view, "generator": gen, **summary(pd.concat([fakes, real]))})
    per_gen = pd.DataFrame(rows)
    wide = per_gen.pivot(index="generator", columns="view", values="recall")
    wide = wide[[v for v in RUNS if v in wide.columns]]

    overall.to_csv(args.dir / "claude_api_view_comparison.csv", index=False)
    per_gen.to_csv(args.dir / "claude_api_view_comparison_by_generator.csv", index=False)
    print(overall.round(4).to_string(index=False))
    print("\nFake recall (p_fake >= 50) by generator and view:")
    print(wide.round(3).to_string())


if __name__ == "__main__":
    main()
