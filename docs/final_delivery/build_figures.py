"""Regenerate the final poster and component figures from audited CSVs."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DF = ROOT / "results/df40_transfer"
FR = ROOT / "results/llm_eval"
BLUE, ORANGE, INK, MUTED = "#27658B", "#B85F28", "#142C3D", "#506675"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK,
                     "axes.edgecolor": "#9FB1BC", "xtick.color": INK, "ytick.color": INK,
                     "font.size": 18, "svg.fonttype": "none", "pdf.fonttype": 42})


def diversity(ax):
    data = pd.read_csv(DF / "clean_cnn_audit/matched_confidence_summary.csv")
    for model, color, marker in [("cnn", BLUE, "o"), ("vit", ORANGE, "s")]:
        for mode, style in [("baseline", "-"), ("augmented", "--")]:
            rows = data[data["mode"] == mode].sort_values("k")
            ax.plot(rows.k, rows[f"{model}_accuracy"]*100, color=color, marker=marker,
                    linestyle=style, linewidth=3, markersize=10,
                    label=f"{model.upper()} {mode}")
    ax.set(xlim=(.7, 7.3), ylim=(50, 100), xticks=[1, 2, 4, 7],
           ylabel="Clean accuracy (%)", xlabel="Training generators (k)")
    ax.legend(loc="upper left", ncol=2, fontsize=15, frameon=False)
    ax.grid(axis="y", alpha=.22)


def jpeg(ax):
    cnn = pd.read_csv(DF / "poster_assets/cnn_jpeg_class_macro.csv").pivot(
        index="class", columns="mode", values="rate")
    vit = pd.read_csv(DF / "poster_assets/vit_jpeg_class_macro.csv").set_index("class")
    rows = [(cnn.loc[c, "baseline"], cnn.loc[c, "augmented"], BLUE)
            for c in ["real", "fake"]]
    rows += [(vit.loc[c, "baseline_rate"], vit.loc[c, "augmented_rate"], ORANGE)
             for c in ["real", "fake"]]
    for i, (base, aug, color) in enumerate(rows):
        diff = 100*(aug-base)
        ax.barh(i, diff, height=.57, color=color)
        ax.text(diff/2, i, f"{diff:+.1f} pp", va="center", ha="center",
                color="white", fontweight="bold", fontsize=17)
        ax.text(1.0, i, f"{100*base:.1f} → {100*aug:.1f}%", va="center", fontsize=15)
    ax.set(yticks=range(4), yticklabels=["CNN real false alarms", "CNN fake recall",
                                      "ViT real false alarms", "ViT fake recall"],
           xlim=(-51, 24), xlabel="Augmented − baseline (percentage points)")
    ax.axvline(0, color=INK, lw=1.2)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.20)


def view_data():
    gpt = pd.read_csv(FR / "openai/metrics/openai_predictions.csv")
    scores = gpt[gpt.status == "success"].pivot(
        index="source_image_sha256", columns="view", values="p_fake")
    scores = scores.rename(columns=lambda x: "GPT "+x)
    labels = gpt.drop_duplicates("source_image_sha256").set_index("source_image_sha256").label
    for view, suffix in [("original", ""), ("square", "_square"), ("detector224", "_detector224")]:
        claude = pd.read_csv(FR / f"claude/claude_api{suffix}_predictions.csv")
        scores = scores.join(claude[claude.status == "success"].set_index("sha256").p_fake
                             .rename("Claude "+view))
    scores = scores.dropna().join(labels)
    assert scores.label.value_counts().to_dict() == {1: 520, 0: 70}
    assert (scores[scores.label == 0].drop(columns="label") < 50).all().all()
    return scores


def presentation(ax):
    data = view_data()
    fake = data[data.label == 1]
    views = ["original", "square", "detector224"]
    x = np.arange(3)
    for i, (name, color) in enumerate([("GPT", BLUE), ("Claude", ORANGE)]):
        vals = [(fake[f"{name} {v}"] >= 50).mean()*100 for v in views]
        bars = ax.bar(x + (i-.5)*.36, vals, .35, color=color, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=16)
    ax.set(xticks=x, xticklabels=["Original", "Square crop", "Detector224"],
           ylabel="Fake recall (%)", ylim=(0, 100))
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=.22)


def face_data():
    files = [DF / "openai_face/df40_five_method_v3_v2/comparison/matched_predictions.csv",
             DF / "claude_face/df40_five_method_v3_v1/comparison/matched_predictions.csv"]
    frames = [pd.read_csv(f) for f in files]
    data = pd.concat([frames[0], frames[1][frames[1].model == "claude"]])
    groups = data.groupby(["model", "mode"])
    for _, g in groups:
        assert g.sha256.nunique() == 189
        assert g.label.value_counts().to_dict() == {1: 135, 0: 54}
    return data


def face(ax):
    data = face_data()
    for model, mode, color, marker, label in [
        ("cnn", "baseline", BLUE, "o", "CNN"), ("vit", "baseline", ORANGE, "s", "ViT"),
        ("openai", "zero_shot", "#507043", "^", "GPT"), ("claude", "zero_shot", "#78568C", "D", "Claude"),
        ("vit", "face_adapted", "#222222", "*", "ViT adapted*")]:
        g = data[(data.model == model) & (data["mode"] == mode)]
        if g.empty and mode == "face_adapted":
            g = data[(data.model == model) & data["mode"].str.contains("adapt")]
        assert len(g) == 189, (model, mode, data["mode"].unique())
        real = g[g.label == 0].prediction.mean()*100
        fake = g[g.label == 1].prediction.mean()*100
        ax.scatter(real, fake, c=color, marker=marker, s=260, zorder=3, label=label)
    ax.plot([0,100], [0,100], color="#8699A5", linestyle="--", linewidth=2,
            label="Equal FPR & recall (50% balanced acc.)")
    ax.set(xlim=(-3,103), ylim=(-3,103), xlabel="Real false alarms (%)",
           ylabel="Fake recall (%)", xticks=[0,25,50,75,100], yticks=[0,25,50,75,100])
    ax.grid(alpha=.20)
    ax.legend(loc="lower right", fontsize=13, frameon=True)


def save_components():
    for fn in [diversity, jpeg, presentation, face]:
        fig, ax = plt.subplots(figsize=(12, 6.6))
        fn(ax)
        fig.tight_layout()
        for ext in ["svg", "png"]:
            fig.savefig(OUT / f"{fn.__name__}.{ext}", dpi=160, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    save_components()
