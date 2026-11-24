from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FAMILY, HOLDOUT_ORDER, OUT_ROOT, SHORT, out_dir
from style import MODE_COLOR, MODE_MARKER, TEXT_2, plt, save

CONFIDENT = 0.9


def add_outcomes(test: pd.DataFrame, prob_col: str = "prob_temperature") -> pd.DataFrame:
    t = test.copy()
    t["pred"] = (t.prob_raw >= 0.5).astype(int)
    t["wrong"] = t.pred != t.label
    t["error_type"] = np.select([t.wrong & (t.label == 1), t.wrong & (t.label == 0)],
                                ["missed_fake", "false_alarm"], "correct")
    conf = np.maximum(t[prob_col], 1 - t[prob_col])
    t["scaled_confidence"] = conf
    t["confident_error"] = t.wrong & (conf >= CONFIDENT)
    return t


def run_error_rates(t: pd.DataFrame) -> pd.DataFrame:
    keys = ["experiment", "holdout", "k", "rep", "mode"]
    fake, real = t[t.label == 1], t[t.label == 0]
    rates = pd.DataFrame({
        "miss_rate": fake.groupby(keys).wrong.mean(),
        "confident_miss_rate": fake.groupby(keys).confident_error.mean(),
        "false_alarm_rate": real.groupby(keys).wrong.mean(),
        "confident_false_alarm_rate": real.groupby(keys).confident_error.mean(),
    })
    errors = t[t.wrong].groupby(keys)
    rates["share_of_errors_that_are_misses"] = errors.apply(
        lambda g: (g.label == 1).mean(), include_groups=False)
    rates["share_of_errors_confident"] = errors.confident_error.mean()
    rates = rates.fillna(0).reset_index()
    rates["family"] = rates.holdout.map(FAMILY)
    return rates


def summarize_rates(rates: pd.DataFrame, by) -> pd.DataFrame:
    cols = [c for c in rates.columns if c.endswith("rate") or c.startswith("share")]
    keys = ["holdout"] + [b for b in by if b != "holdout"]
    per_holdout = rates.groupby(keys)[cols].mean()
    if "holdout" in by:
        return per_holdout.reset_index()
    return per_holdout.groupby(list(by))[cols].mean().reset_index()


def persistence(t: pd.DataFrame, k: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = t[t.k == k]
    fake = (sub[sub.label == 1]
            .groupby(["holdout", "mode", "image_id", "path"])
            .agg(runs=("wrong", "size"), times_wrong=("wrong", "sum"),
                 mean_prob_raw=("prob_raw", "mean"),
                 mean_prob_scaled=("prob_temperature", "mean"))
            .reset_index())
    real = (sub[sub.label == 0]
            .groupby(["mode", "image_id", "path", "source_generator"])
            .agg(runs=("wrong", "size"), times_wrong=("wrong", "sum"),
                 holdouts_wrong=("holdout", lambda h: h[sub.loc[h.index, "wrong"]].nunique()),
                 mean_prob_raw=("prob_raw", "mean"),
                 mean_prob_scaled=("prob_temperature", "mean"))
            .reset_index())
    for frame in (fake, real):
        frame["wrong_fraction"] = frame.times_wrong / frame.runs
    return fake, real


def concentration(fake_p: pd.DataFrame, real_p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, frame in [("fake (missed)", fake_p), ("real (false alarm)", real_p)]:
        for mode, f in frame.groupby("mode"):
            ever = f[f.times_wrong > 0]
            always = f[f.times_wrong == f.runs]
            rows.append({"images": kind, "mode": mode, "unique_images": len(f),
                         "runs_per_image": int(f.runs.median()),
                         "share_ever_wrong": len(ever) / len(f),
                         "share_always_wrong": len(always) / len(f),
                         "share_of_errors_from_always_wrong_images":
                             always.times_wrong.sum() / max(f.times_wrong.sum(), 1)})
    return pd.DataFrame(rows)


def plot_rates_by_holdout(summary: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    y = np.arange(len(HOLDOUT_ORDER))
    for ax, (col, title) in zip(axes, [("miss_rate", "Missed fakes (fake → real)"),
                                       ("false_alarm_rate", "False alarms (real → fake)")]):
        for offset, mode in [(-0.15, "baseline"), (0.15, "augmented")]:
            s = summary[summary["mode"] == mode].set_index("holdout").reindex(HOLDOUT_ORDER)
            ax.scatter(s[col], y + offset, color=MODE_COLOR[mode], marker=MODE_MARKER[mode],
                       s=50, label=mode, zorder=3)
            conf = s[col.replace("_rate", "").replace("miss", "confident_miss")
                     .replace("false_alarm", "confident_false_alarm") + "_rate"]
            ax.scatter(conf, y + offset, facecolors="none", edgecolors=MODE_COLOR[mode],
                       marker=MODE_MARKER[mode], s=50, zorder=3)
        ax.set(title=title, xlabel="Rate at k=7 (filled = all errors, hollow = confident)",
               xlim=(0, max(0.1, summary[col].max() * 1.15)))
    axes[0].set_yticks(y, [SHORT[h] for h in HOLDOUT_ORDER])
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Held-out generator")
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    return save(fig, path)


def plot_rates_vs_k(rates: pd.DataFrame, path: Path):
    s = summarize_rates(rates, ["k", "mode"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, (col, title) in zip(axes, [("miss_rate", "Missed fakes"),
                                       ("false_alarm_rate", "False alarms")]):
        for mode in ["baseline", "augmented"]:
            t = s[s["mode"] == mode].sort_values("k")
            ax.plot(t.k, t[col], marker=MODE_MARKER[mode], color=MODE_COLOR[mode], label=mode)
            conf = "confident_" + col
            ax.plot(t.k, t[conf], ls="--", marker=MODE_MARKER[mode], mfc="none",
                    color=MODE_COLOR[mode], label=f"{mode}, confident only")
        ax.set(title=title, xlabel="Training generators (k)", xticks=[1, 2, 4, 7])
    axes[0].set_ylabel("Test error rate (macro over holdouts)")
    axes[0].legend()
    fig.tight_layout()
    return save(fig, path)


def plot_persistence(fake_p: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    width = 0.38
    for offset, mode in [(-width / 2, "baseline"), (width / 2, "augmented")]:
        counts = fake_p[fake_p["mode"] == mode].times_wrong.value_counts(normalize=True)
        counts = counts.reindex(range(4), fill_value=0)
        ax.bar(counts.index + offset, counts.values, width, color=MODE_COLOR[mode], label=mode)
    ax.set(xticks=range(4), xlabel="Times missed across the 3 k=7 repeats",
           ylabel="Share of fake test images", title="Missed fakes are mostly the same images")
    ax.legend()
    return save(fig, path)


def example_grid(rows: pd.DataFrame, image_root: Path, path: Path, title: str):
    from PIL import Image

    n = len(rows)
    cols = min(6, n)
    fig, axes = plt.subplots(int(np.ceil(n / cols)), cols, figsize=(2.2 * cols, 2.5 * np.ceil(n / cols)))
    for ax in np.atleast_1d(axes).flat:
        ax.axis("off")
    for ax, (_, r) in zip(np.atleast_1d(axes).flat, rows.iterrows()):
        try:
            ax.imshow(Image.open(image_root / r.path).convert("RGB"))
        except OSError:
            ax.text(0.5, 0.5, "image not found", ha="center", color=TEXT_2)
        label = SHORT.get(r.get("holdout", r.get("source_generator")), "")
        ax.set_title(f"{label}  P(fake)={r.mean_prob_scaled:.2f}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return save(fig, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recalibrated", type=Path,
                        default=OUT_ROOT / "recalibration" / "test_predictions_recalibrated.csv.gz")
    parser.add_argument("--image-root", type=Path, default=None,
                        help="Folder the prediction paths are relative to (CARC GenImage_Compact)")
    args = parser.parse_args(argv)
    if not args.recalibrated.exists():
        raise SystemExit("Run code/analysis/recalibrate.py first.")

    out, figs = out_dir("errors"), out_dir("figures")
    t = add_outcomes(pd.read_csv(args.recalibrated))
    rates = run_error_rates(t)
    rates.to_csv(out / "per_run_error_rates.csv", index=False, float_format="%.6g")
    by_k = summarize_rates(rates, ["k", "mode"])
    by_k.to_csv(out / "error_rates_macro_by_k_mode.csv", index=False, float_format="%.6g")
    by_holdout = summarize_rates(rates, ["holdout", "k", "mode"])
    by_holdout.to_csv(out / "error_rates_by_holdout_k_mode.csv", index=False, float_format="%.6g")
    by_family = summarize_rates(rates, ["family", "k", "mode"])
    by_family.to_csv(out / "error_rates_by_family_k_mode.csv", index=False, float_format="%.6g")

    fake_p, real_p = persistence(t, k=7)
    fake_p.sort_values(["mode", "times_wrong", "mean_prob_scaled"], ascending=[True, False, True]) \
        .to_csv(out / "k7_fake_image_persistence.csv", index=False, float_format="%.6g")
    real_p.sort_values(["mode", "times_wrong", "mean_prob_scaled"], ascending=[True, False, False]) \
        .to_csv(out / "k7_real_image_persistence.csv", index=False, float_format="%.6g")
    conc = concentration(fake_p, real_p)
    conc.to_csv(out / "k7_error_concentration.csv", index=False, float_format="%.6g")
    real_src = (real_p.groupby(["mode", "source_generator"])
                .agg(images=("image_id", "size"), false_alarm_rate=("wrong_fraction", "mean"))
                .reset_index())
    real_src.to_csv(out / "k7_false_alarms_by_real_source_folder.csv", index=False,
                    float_format="%.6g")

    worst_miss = fake_p[fake_p["mode"] == "baseline"].nsmallest(12, "mean_prob_scaled")
    worst_alarm = real_p[real_p["mode"] == "baseline"].nlargest(12, "mean_prob_scaled")
    worst = pd.concat([worst_miss.assign(error="missed_fake"),
                       worst_alarm.assign(error="false_alarm")])
    worst.to_csv(out / "k7_baseline_most_confident_mistakes.csv", index=False, float_format="%.6g")

    plot_rates_by_holdout(by_holdout[by_holdout.k == 7], figs / "error_rates_k7_by_holdout.png")
    plot_rates_vs_k(rates, figs / "error_rates_vs_k.png")
    plot_persistence(fake_p, figs / "k7_missed_fake_persistence.png")
    if args.image_root:
        example_grid(worst_miss, args.image_root, figs / "examples_k7_missed_fakes.png",
                     "k=7 baseline: most confidently missed fakes")
        example_grid(worst_alarm, args.image_root, figs / "examples_k7_false_alarms.png",
                     "k=7 baseline: most confident false alarms on real images")

    print(by_k.round(4).to_string(index=False))
    print(by_family[by_family.k == 7].round(4).to_string(index=False))
    print(by_holdout[by_holdout.k == 7][["holdout", "mode", "miss_rate", "confident_miss_rate",
                                         "false_alarm_rate", "confident_false_alarm_rate"]]
          .round(4).to_string(index=False))
    print(conc.round(4).to_string(index=False))
    print(real_src.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
