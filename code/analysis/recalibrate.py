from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (BINS, EPS, RUN_KEYS, THRESHOLD, VIT_ROOT, load_vit_predictions,
                    logit, metrics, out_dir, reliability_table, sigmoid, summarize)
from style import MODE_COLOR, MODE_MARKER, TEXT_2, plt, save

METHODS = ["raw", "temperature", "platt"]
VALUES = ["accuracy", "recall_fake", "specificity_real", "ece", "brier", "log_loss",
          "overconfidence"]


def _nll(z, y):
    p = np.clip(sigmoid(z), EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def fit_temperature(val_probs, val_labels) -> float:
    z, y = logit(val_probs), np.asarray(val_labels, float)
    res = minimize_scalar(lambda log_t: _nll(z / np.exp(log_t), y), bounds=(-6, 6),
                          method="bounded")
    return float(np.exp(res.x))


def fit_platt(val_probs, val_labels) -> tuple[float, float]:
    z, y = logit(val_probs), np.asarray(val_labels, float)
    res = minimize(lambda w: _nll(w[0] * z + w[1], y), x0=[1.0, 0.0], method="L-BFGS-B")
    return float(res.x[0]), float(res.x[1])


def fit_threshold(val_probs, val_labels) -> float:
    p, y = np.asarray(val_probs, float), np.asarray(val_labels, int)
    candidates = np.unique(p)
    mids = np.concatenate([[0.0], (candidates[:-1] + candidates[1:]) / 2, [1.0]])
    best, best_t = -1.0, THRESHOLD
    for t in mids:
        pred = p >= t
        bal = 0.5 * (pred[y == 1].mean() + (~pred[y == 0]).mean())
        if bal > best + 1e-12:
            best, best_t = bal, t
    return float(best_t)


def apply(method: str, probs, params: dict):
    if method == "raw":
        return np.asarray(probs, float)
    z = logit(probs)
    if method == "temperature":
        return sigmoid(z / params["temperature"])
    if method == "platt":
        return sigmoid(params["platt_a"] * z + params["platt_b"])
    raise ValueError(method)


def recalibrate(preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    params_rows, metric_rows, test_frames = [], [], []
    for key, run in preds.groupby(RUN_KEYS, sort=True):
        val, test = run[run.split == "val"], run[run.split == "test"].copy()
        if val.empty or test.empty:
            raise ValueError(f"Run {key} needs both validation and test predictions")
        params = dict(zip(RUN_KEYS, key))
        params["temperature"] = fit_temperature(val.prob_fake, val.label)
        params["platt_a"], params["platt_b"] = fit_platt(val.prob_fake, val.label)
        params["val_threshold"] = fit_threshold(val.prob_fake, val.label)
        params_rows.append(params)
        for method in METHODS:
            p = apply(method, test.prob_fake, params)
            test[f"prob_{method}"] = p
            row = dict(zip(RUN_KEYS, key), method=method, threshold="0.5")
            row.update(metrics(test.label, p, THRESHOLD))
            metric_rows.append(row)
        row = dict(zip(RUN_KEYS, key), method="raw", threshold="val-selected")
        row.update(metrics(test.label, test.prob_fake, params["val_threshold"]))
        metric_rows.append(row)
        test_frames.append(test)
    return pd.DataFrame(params_rows), pd.DataFrame(metric_rows), pd.concat(test_frames)


def _ece_matrix(labels, probs):
    index = np.minimum((probs * BINS).astype(int), BINS - 1)
    total = np.zeros(probs.shape[0])
    for b in range(BINS):
        sel = index == b
        count = sel.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.abs(np.where(sel, probs, 0).sum(1) - np.where(sel, labels, 0).sum(1))
        total += np.where(count > 0, gap, 0)
    return total / probs.shape[1]


def paired_bootstrap(test: pd.DataFrame, k: int = 7, draws: int = 2000, seed: int = 566):
    rng = np.random.default_rng(seed)
    rows = []
    sub = test[test.k == k]
    for mode, mode_frame in sub.groupby("mode"):
        stats = {m: {"ece": [], "brier": []} for m in ["raw", "temperature"]}
        for holdout, h in mode_frame.groupby("holdout"):
            ids = {lab: np.sort(h[h.label == lab].image_id.unique()) for lab in (0, 1)}
            picks = np.concatenate([rng.choice(len(ids[lab]), (draws, len(ids[lab])))
                                    + (0 if lab == 0 else len(ids[0])) for lab in (0, 1)], 1)
            order = np.concatenate([ids[0], ids[1]])
            labels = np.r_[np.zeros(len(ids[0])), np.ones(len(ids[1]))][picks]
            for method in stats:
                per_rep_ece, per_rep_brier = [], []
                for _, r in h.groupby("rep"):
                    p = r.set_index("image_id").loc[order, f"prob_{method}"].to_numpy()[picks]
                    per_rep_ece.append(_ece_matrix(labels, p))
                    per_rep_brier.append(((p - labels) ** 2).mean(1))
                stats[method]["ece"].append(np.mean(per_rep_ece, 0))
                stats[method]["brier"].append(np.mean(per_rep_brier, 0))
        for metric in ["ece", "brier"]:
            raw = np.mean(stats["raw"][metric], 0)
            cal = np.mean(stats["temperature"][metric], 0)
            diff = raw - cal
            rows.append({"k": k, "mode": mode, "metric": metric,
                         "raw_mean": raw.mean(), "temperature_mean": cal.mean(),
                         "improvement_mean": diff.mean(),
                         "improvement_ci_low": np.quantile(diff, 0.025),
                         "improvement_ci_high": np.quantile(diff, 0.975),
                         "draws": draws, "resampling": "test images within holdout, by label"})
    return pd.DataFrame(rows)


def plot_before_after(macro: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    styles = {"raw": "-", "temperature": "--"}
    names = {"raw": "raw", "temperature": "temperature-scaled (fit on val)"}
    for ax, mode in zip(axes, ["baseline", "augmented"]):
        for method in ["raw", "temperature"]:
            t = macro[(macro["mode"] == mode) & (macro.method == method)
                      & (macro.threshold == "0.5")].sort_values("k")
            ax.plot(t.k, t.ece_mean, ls=styles[method], marker=MODE_MARKER[mode],
                    color=MODE_COLOR[mode] if method == "raw" else TEXT_2, label=names[method])
        ax.set(title=f"ViT {mode}", xlabel="Training generators (k)", xticks=[1, 2, 4, 7])
    axes[0].set_ylabel("Test ECE (macro over holdouts)")
    axes[0].legend()
    fig.suptitle("Test ECE before and after validation-only temperature scaling")
    fig.tight_layout()
    return save(fig, path)


def plot_reliability_after(test: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, mode in zip(axes, ["baseline", "augmented"]):
        t = test[(test["mode"] == mode) & (test.k == 7)]
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=TEXT_2, label="perfect calibration")
        for method, color in [("raw", MODE_COLOR["baseline"]),
                              ("temperature", MODE_COLOR["augmented"])]:
            r = reliability_table(t.label, t[f"prob_{method}"])
            r = r[r["count"] > 0]
            ax.plot(r.mean_prob_fake, r.observed_fake_fraction, marker="o", color=color,
                    label="raw" if method == "raw" else "temperature-scaled")
        ax.set(title=f"ViT {mode}, k=7 test", xlabel="Mean predicted P(fake) in bin",
               xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")
    axes[0].set_ylabel("Observed fraction fake")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return save(fig, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit-root", type=Path, default=VIT_ROOT)
    parser.add_argument("--draws", type=int, default=2000)
    args = parser.parse_args(argv)

    out, figs = out_dir("recalibration"), out_dir("figures")
    preds = load_vit_predictions(args.vit_root)
    params, test_metrics, test = recalibrate(preds)
    params.to_csv(out / "fitted_params.csv", index=False, float_format="%.6g")
    test_metrics.to_csv(out / "test_metrics_by_method.csv", index=False, float_format="%.6g")

    _, macro = summarize(test_metrics, VALUES, by=("holdout", "k", "mode", "method", "threshold"))
    macro.to_csv(out / "test_macro_by_method.csv", index=False, float_format="%.6g")
    _, pmacro = summarize(params, ["temperature", "platt_a", "platt_b", "val_threshold"],
                          by=("holdout", "k", "mode"))
    pmacro.to_csv(out / "fitted_params_macro.csv", index=False, float_format="%.6g")

    keep = ["experiment", "holdout", "k", "rep", "mode", "image_id", "label",
            "source_generator", "path", "prob_raw", "prob_temperature", "prob_platt"]
    test[keep].to_csv(out / "test_predictions_recalibrated.csv.gz", index=False,
                      float_format="%.6g")

    boot = paired_bootstrap(test, k=7, draws=args.draws)
    boot.to_csv(out / "bootstrap_k7_raw_vs_temperature.csv", index=False, float_format="%.6g")

    plot_before_after(macro, figs / "ece_before_after_temperature.png")
    plot_reliability_after(test, figs / "reliability_k7_temperature.png")
    cols = ["k", "mode", "method", "threshold", "accuracy_mean", "recall_fake_mean",
            "specificity_real_mean", "ece_mean", "brier_mean", "log_loss_mean"]
    print(macro[cols].round(4).to_string(index=False))
    print(pmacro.round(3).to_string(index=False))
    print(boot.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
