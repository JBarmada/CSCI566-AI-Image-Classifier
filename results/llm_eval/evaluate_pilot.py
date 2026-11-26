from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)

OUT = Path("results/llm_eval")
THRESHOLD = 50.0

# Load evaluation rows and model predictions.
task = pd.read_csv(OUT / "pilot_task_manifest.csv")

models = {
    "Gemini": pd.read_csv(OUT / "gemini_predictions.csv"),
    "Groq": pd.read_csv(OUT / "groq_predictions.csv"),
}

# Validate the task manifest.
assert len(task) == 160
assert task["sha256"].nunique() == 90
assert set(task["label"].unique()) == {0, 1}

evaluation = task.copy()

# Join predictions by SHA256, not filename or row order.
for name, predictions in models.items():
    predictions = predictions.copy()

    successful = predictions[predictions["status"] == "success"]

    assert successful["sha256"].nunique() == 90, (
        f"{name}: expected 90 unique successful predictions"
    )
    assert successful["p_fake"].notna().all()

    score_column = f"{name.lower()}_p_fake"

    prediction_map = successful.set_index("sha256")["p_fake"]

    evaluation[score_column] = (
        evaluation["sha256"].map(prediction_map)
    )

    assert evaluation[score_column].notna().all(), (
        f"{name}: missing predictions after joining"
    )

    evaluation[f"{name.lower()}_prediction"] = (
        evaluation[score_column] >= THRESHOLD
    ).astype(int)

# Metric calculation.
def calculate_metrics(group, score_column, prediction_column):
    y_true = group["label"].astype(int)
    scores = group[score_column].astype(float)
    y_pred = group[prediction_column].astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    return {
        "n_rows": len(group),
        "n_unique_images": group["sha256"].nunique(),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(
            y_true, y_pred
        ),
        "auroc": roc_auc_score(y_true, scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "mean_p_fake_real": scores[y_true == 0].mean(),
        "mean_p_fake_fake": scores[y_true == 1].mean(),
    }


# Overall metrics.
overall_rows = []

for name in ["gemini", "groq"]:
    result = calculate_metrics(
        evaluation,
        f"{name}_p_fake",
        f"{name}_prediction",
    )
    result["model"] = name.capitalize()
    overall_rows.append(result)

overall = pd.DataFrame(overall_rows)
overall = overall.set_index("model")

# Per-generator metrics.
generator_rows = []

for generator, group in evaluation.groupby("heldout_generator"):
    for name in ["gemini", "groq"]:
        result = calculate_metrics(
            group,
            f"{name}_p_fake",
            f"{name}_prediction",
        )
        result["heldout_generator"] = generator
        result["model"] = name.capitalize()
        generator_rows.append(result)

per_generator = pd.DataFrame(generator_rows)

# Model agreement on the same 160 evaluation rows.
agreement_rows = []

for name in ["gemini", "groq"]:
    agreement_rows.append({
        "comparison": "Gemini vs Groq",
        "evaluation_rows": len(evaluation),
        "prediction_agreement": (
            evaluation["gemini_prediction"]
            == evaluation["groq_prediction"]
        ).mean(),
        "probability_correlation": evaluation[
            ["gemini_p_fake", "groq_p_fake"]
        ].corr().iloc[0, 1],
    })

agreement = pd.DataFrame(agreement_rows)

# Save outputs.
evaluation.to_csv(
    OUT / "pilot_evaluation_rows.csv", index=False
)
overall.to_csv(
    OUT / "pilot_overall_metrics.csv"
)
per_generator.to_csv(
    OUT / "pilot_per_generator_metrics.csv", index=False
)
agreement.to_csv(
    OUT / "pilot_model_agreement.csv", index=False
)

# Display results.
metric_columns = [
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "f1",
    "balanced_accuracy",
    "auroc",
]

print("\n========== OVERALL RESULTS ==========")
print(overall[metric_columns].round(4).to_string())

print("\n========== CONFUSION MATRICES ==========")
for model in ["Gemini", "Groq"]:
    print(f"\n{model}")
    print(overall.loc[model, ["tn", "fp", "fn", "tp"]])

print("\n========== PER-GENERATOR RESULTS ==========")
print(
    per_generator[
        ["heldout_generator", "model"] + metric_columns
    ].round(4).to_string(index=False)
)

print("\n========== MODEL AGREEMENT ==========")
print(agreement.round(4).to_string(index=False))

print("\n========== SAVED FILES ==========")
for filename in [
    "pilot_evaluation_rows.csv",
    "pilot_overall_metrics.csv",
    "pilot_per_generator_metrics.csv",
    "pilot_model_agreement.csv",
]:
    print(OUT / filename)

print("\nEvaluation completed successfully.")
