from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (HOLDOUT_ORDER, SHORT, VIT_ROOT, load_vit_predictions, out_dir,
                    per_run, reliability_table, summarize)
from style import K_COLOR, MODE_COLOR, MODE_MARKER, TEXT_2, plt, save

VALUES = ["accuracy", "recall_fake", "specificity_real", "auroc", "ece", "brier",
          "log_loss", "mean_confidence", "overconfidence", "mean_prob_fake"]


def verify_against_recorded(runs: pd.DataFrame, recorded: pd.DataFrame) -> dict:
    test = runs[runs.split == "test"].merge(
        recorded, on=["experiment", "mode", "seed"], suffixes=("", "_recorded"))
    checks = {}
    for col in ["accuracy", "recall_fake", "auroc", "ece", "brier", "log_loss"]:
        diff = (test[col] - test[f"{col}_recorded"]).abs().max()
        checks[col] = {"max_abs_diff": float(diff), "ok": bool(diff < 1e-6)}
    return {"runs_compared": int(len(test)), "expected": int(len(recorded)),
            "all_ok": bool(len(test) == len(recorded) and all(c["ok"] for c in checks.values())),
            "metrics": checks}


def pooled_reliability(preds: pd.DataFrame, by) -> pd.DataFrame:
    rows = []
    for key, group in preds.groupby(by):
        table = reliability_table(group.label, group.prob_fake)
        for name, value in zip(by, key if isinstance(key, tuple) else (key,)):
            table[name] = value
        rows.append(table)
    return pd.concat(rows, ignore_index=True)


def plot_reliability_by_k(bins: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
    for ax, mode in zip(axes, ["baseline", "augmented"]):
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=TEXT_2, label="perfect calibration")
        for k in [1, 2, 4, 7]:
            t = bins[(bins["mode"] == mode) & (bins.k == k) & (bins["count"] > 0)]
            ax.plot(t.mean_prob_fake, t.observed_fake_fraction, marker="o", color=K_COLOR[k],
                    label=f"k={k}")
        ax.set(title=f"ViT {mode}", xlabel="Mean predicted P(fake) in bin",
               xlim=(0, 1), ylim=(0, 1))
        ax.set_aspect("equal")
    axes[0].set_ylabel("Observed fraction fake")
    axes[1].legend(loc="lower right")
    fig.suptitle("Test reliability on the unseen generator (all holdouts and repeats pooled)")
    return save(fig, path)


def plot_reliability_k7_holdouts(bins: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.4), sharex=True, sharey=True)
    for ax, holdout in zip(axes.flat, HOLDOUT_ORDER):
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color=TEXT_2)
        for mode in ["baseline", "augmented"]:
            t = bins[(bins.holdout == holdout) & (bins["mode"] == mode) & (bins["count"] > 0)]
            ax.plot(t.mean_prob_fake, t.observed_fake_fraction, marker=MODE_MARKER[mode],
                    color=MODE_COLOR[mode], label=mode)
        ax.set_title(f"Held out: {SHORT[holdout]}")
        ax.set(xlim=(0, 1), ylim=(0, 1))
    for ax in axes[1]:
        ax.set_xlabel("Mean predicted P(fake)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Observed fraction fake")
    axes[0, 0].legend(loc="lower right")
    fig.suptitle("k=7 test reliability by held-out generator (3 repeats pooled)")
    fig.tight_layout()
    return save(fig, path)


def plot_metric_vs_k(macro: pd.DataFrame, path: Path):
    panels = [("ece", "ECE (lower is better)"), ("brier", "Brier score (lower is better)"),
              ("overconfidence", "Mean confidence − accuracy")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (metric, label) in zip(axes, panels):
        for mode in ["baseline", "augmented"]:
            t = macro[macro["mode"] == mode].sort_values("k")
            ax.errorbar(t.k, t[f"{metric}_mean"], yerr=t[f"{metric}_sd_holdouts"], capsize=3,
                        marker=MODE_MARKER[mode], color=MODE_COLOR[mode], label=mode)
        ax.set(title=label, xlabel="Training generators (k)", xticks=[1, 2, 4, 7])
    axes[0].legend()
    fig.suptitle("Test calibration vs k (equal-weight mean of 8 holdouts; bars = SD across holdouts)")
    fig.tight_layout()
    return save(fig, path)


def plot_probability_histograms(preds: pd.DataFrame, path: Path):
    edges = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, mode in zip(axes, ["baseline", "augmented"]):
        t = preds[(preds["mode"] == mode)]
        for label, name, color in [(0, "real images", MODE_COLOR["baseline"]),
                                   (1, "unseen-generator fakes", MODE_COLOR["augmented"])]:
            values = t[t.label == label].prob_fake
            ax.hist(values, bins=edges, weights=np.full(len(values), 1 / len(values)),
                    histtype="step", lw=2, color=color, label=name)
        ax.axvline(0.5, ls=":", color=TEXT_2, lw=1)
        ax.set(title=f"ViT {mode}, k=7 test", xlabel="Predicted P(fake)")
    axes[0].set_ylabel("Share of images")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    return save(fig, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit-root", type=Path, default=VIT_ROOT)
    args = parser.parse_args(argv)

    out, figs = out_dir("calibration"), out_dir("figures")
    preds = load_vit_predictions(args.vit_root)
    runs = per_run(preds)
    runs.to_csv(out / "per_run_metrics.csv", index=False, float_format="%.6g")

    recorded = pd.read_csv(args.vit_root / "summary" / "all_results.csv")
    check = verify_against_recorded(runs, recorded)
    (out / "verification.json").write_text(json.dumps(check, indent=2) + "\n")
    if not check["all_ok"]:
        raise SystemExit(f"Recomputed metrics disagree with recorded metrics: {check}")

    for split in ["test", "val"]:
        holdout, macro = summarize(runs[runs.split == split], VALUES)
        holdout.to_csv(out / f"{split}_by_holdout_k_mode.csv", index=False, float_format="%.6g")
        macro.to_csv(out / f"{split}_macro_by_k_mode.csv", index=False, float_format="%.6g")

    wide = runs.pivot_table(index=["experiment", "holdout", "k", "rep", "mode"], columns="split",
                            values=["ece", "brier", "accuracy", "recall_fake"]).reset_index()
    wide.columns = ["_".join(c).strip("_") for c in wide.columns]
    for m in ["ece", "brier", "accuracy", "recall_fake"]:
        wide[f"{m}_test_minus_val"] = wide[f"{m}_test"] - wide[f"{m}_val"]
    _, gap = summarize(wide.assign(split="test"), [f"{m}_test_minus_val" for m in
                                                    ["ece", "brier", "accuracy", "recall_fake"]])
    gap.to_csv(out / "val_to_test_shift_macro.csv", index=False, float_format="%.6g")

    test = preds[preds.split == "test"]
    by_k = pooled_reliability(test, ["mode", "k"])
    by_k.to_csv(out / "reliability_test_by_mode_k.csv", index=False, float_format="%.6g")
    k7 = pooled_reliability(test[test.k == 7], ["holdout", "mode"])
    k7.to_csv(out / "reliability_test_k7_by_holdout.csv", index=False, float_format="%.6g")

    _, macro = summarize(runs[runs.split == "test"], VALUES)
    plot_reliability_by_k(by_k, figs / "reliability_by_k.png")
    plot_reliability_k7_holdouts(k7, figs / "reliability_k7_by_holdout.png")
    plot_metric_vs_k(macro, figs / "calibration_vs_k.png")
    plot_probability_histograms(test[test.k == 7], figs / "probability_histograms_k7.png")
    print(macro[["k", "mode", "accuracy_mean", "ece_mean", "brier_mean",
                 "overconfidence_mean", "recall_fake_mean", "specificity_real_mean"]]
          .round(4).to_string(index=False))
    print("verification:", "passed" if check["all_ok"] else "FAILED")


if __name__ == "__main__":
    main()
