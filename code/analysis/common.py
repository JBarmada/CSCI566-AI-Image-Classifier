from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
VIT_ROOT = REPO / "results" / "vit_benchmark"
OUT_ROOT = REPO / "results" / "analysis"

THRESHOLD = 0.5
BINS = 10
EPS = 1e-7
RUN_KEYS = ["experiment", "holdout", "k", "rep", "seed", "mode"]
JOIN_KEYS = ["experiment", "mode", "seed", "split", "image_id"]

HOLDOUT_ORDER = ["ADM", "BigGAN", "glide", "Midjourney", "stable_diffusion_v_1_4",
                 "stable_diffusion_v_1_5", "VQDM", "wukong"]
SHORT = {"ADM": "ADM", "BigGAN": "BigGAN", "glide": "GLIDE", "Midjourney": "Midjourney",
         "stable_diffusion_v_1_4": "SD1.4", "stable_diffusion_v_1_5": "SD1.5",
         "VQDM": "VQDM", "wukong": "Wukong"}
FAMILY = {"ADM": "diffusion", "BigGAN": "GAN", "glide": "diffusion",
          "Midjourney": "unassigned", "stable_diffusion_v_1_4": "diffusion",
          "stable_diffusion_v_1_5": "diffusion", "VQDM": "diffusion", "wukong": "diffusion"}


def load_vit_predictions(root: Path = VIT_ROOT) -> pd.DataFrame:
    files = sorted((root / "predictions").glob("*.csv.gz"))
    if not files:
        raise FileNotFoundError(f"No prediction files under {root / 'predictions'}")
    frame = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    return validate_predictions(frame, "ViT")


def validate_predictions(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    required = set(JOIN_KEYS) | {"holdout", "k", "rep", "label", "prob_fake"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} predictions are missing columns: {sorted(missing)}")
    if not frame.label.isin([0, 1]).all():
        raise ValueError(f"{name} labels must be 0/1")
    p = frame.prob_fake.to_numpy(float)
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError(f"{name} probabilities must be finite and in [0, 1]")
    if frame.duplicated(JOIN_KEYS).any():
        raise ValueError(f"{name} predictions contain duplicate run/split/image rows")
    frame = frame.copy()
    frame["family"] = frame.holdout.map(FAMILY).fillna("unassigned")
    return frame


def ece(labels, probs, bins: int = BINS) -> float:
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    index = np.minimum((probs * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        sel = index == b
        if sel.any():
            total += abs(probs[sel].mean() - labels[sel].mean()) * sel.sum()
    return float(total / len(labels))


def reliability_table(labels, probs, bins: int = BINS) -> pd.DataFrame:
    labels, probs = np.asarray(labels, float), np.asarray(probs, float)
    index = np.minimum((probs * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        sel = index == b
        rows.append({"bin": b, "lower": b / bins, "upper": (b + 1) / bins,
                     "count": int(sel.sum()),
                     "mean_prob_fake": probs[sel].mean() if sel.any() else np.nan,
                     "observed_fake_fraction": labels[sel].mean() if sel.any() else np.nan})
    return pd.DataFrame(rows)


def metrics(labels, probs, threshold: float = THRESHOLD) -> dict:
    labels, probs = np.asarray(labels, int), np.asarray(probs, float)
    pred = (probs >= threshold).astype(int)
    clipped = np.clip(probs, EPS, 1 - EPS)
    real, fake = labels == 0, labels == 1
    correct = pred == labels
    conf = np.maximum(probs, 1 - probs)
    return {
        "n": len(labels),
        "accuracy": float(correct.mean()),
        "recall_fake": float(pred[fake].mean()) if fake.any() else np.nan,
        "specificity_real": float(1 - pred[real].mean()) if real.any() else np.nan,
        "auroc": float(roc_auc_score(labels, probs)) if real.any() and fake.any() else np.nan,
        "ece": ece(labels, probs),
        "brier": float(np.mean((probs - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
        "mean_confidence": float(conf.mean()),
        "mean_prob_fake": float(probs.mean()),
        "fake_fraction": float(labels.mean()),
        "overconfidence": float(conf.mean() - correct.mean()),
    }


def per_run(frame: pd.DataFrame, prob_col: str = "prob_fake", threshold: float = THRESHOLD,
            keys=None) -> pd.DataFrame:
    keys = keys or RUN_KEYS + ["split"]
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        row.update(metrics(group.label, group[prob_col], threshold))
        rows.append(row)
    out = pd.DataFrame(rows)
    out["family"] = out.holdout.map(FAMILY).fillna("unassigned")
    return out


def summarize(runs: pd.DataFrame, value_cols, by=("holdout", "k", "mode")) -> tuple:
    by = list(by)
    grouped = runs.groupby(by)[value_cols]
    holdout = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_sd_repeats"))
    holdout["n_repeats"] = grouped.size()
    holdout = holdout.reset_index()
    outer = [c for c in by if c != "holdout"]
    means = holdout.groupby(outer)[[f"{c}_mean" for c in value_cols]]
    macro = means.mean().join(means.std().rename(columns=lambda c: c.replace("_mean", "_sd_holdouts")))
    macro["n_holdouts"] = means.size()
    return holdout, macro.reset_index()


def logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def out_dir(*parts) -> Path:
    path = OUT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
