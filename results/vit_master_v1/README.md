# Completed ViT ADM pilot: evaluation handoff

CARC job **13458685** finished on **October 26, 2026**. All **24 training runs and
24 clean test evaluations completed**, with zero missing results. This folder
contains copied outputs from that run, not new training or inference.

The study uses frozen CLIP ViT-L/14 plus a linear real/fake classifier, ADM held
out, k=1/2/4/7, dataset repeats 0/1/2, and classifier seed 17. Each experiment has a
baseline and one combined augmented treatment. Each validation/test export has
280 real and 280 fake rows. Labels: **real=0, fake=1**; threshold: **0.5**.

## Start here

| File | Use |
|---|---|
| [summary/all_results.csv](summary/all_results.csv) | One row per completed clean test evaluation, including metrics and source paths |
| [summary/summary_by_k.csv](summary/summary_by_k.csv) | Means and standard deviations across the three dataset repeats |
| [summary/augmentation_deltas.csv](summary/augmentation_deltas.csv) | Paired augmented-minus-baseline differences |
| [metadata/prediction_index.csv](metadata/prediction_index.csv) | Portable relative paths to all 48 validation/test prediction files, plus experiment and protocol metadata |
| [metadata/metrics_audit.csv](metadata/metrics_audit.csv) | Independent checks of saved metrics against the copied predictions |
| [metadata/batch_status.json](metadata/batch_status.json) | Original completed job status |
| [metadata/provenance.json](metadata/provenance.json) | Source locations, export time, and input-notebook hash |
| [metadata/artifact_inventory.csv](metadata/artifact_inventory.csv) | CARC source paths, sizes, and SHA-256 hashes of copied files |

Existing figures: [comparison](summary/ADM_clean_comparison.png),
[baseline AUROC heatmap](summary/baseline_clean_auroc_heatmap.png), and
[augmented AUROC heatmap](summary/augmented_clean_auroc_heatmap.png).

Under `runs/<run-name>/`, find the original configuration, training history,
completion marker, validation predictions/metrics/calibration, and the clean
test evaluation. The published source files total about 7.5 MB. Raw images,
features, model weights, and full logs remain on CARC.

`all_results.csv` retains absolute CARC source paths for provenance. Use the
relative paths in `prediction_index.csv` for local analysis:

```python
from pathlib import Path
import pandas as pd

base = Path("results/vit_master_v1")  # From the repository root
index = pd.read_csv(base / "metadata/prediction_index.csv")
test_files = index[index["split"] == "test"]
first = pd.read_csv(base / test_files.iloc[0]["prediction_file"])
```

## Work for the evaluation teammate

1. Use these exports for ViT analysis; no GPU or retraining is needed.
2. Collect CNN exports for the same dataset release and experiment IDs. Confirm
   matching image membership/labels using `image_id` or `sha256`. CNN exports
   currently provide paths; map them through their saved split CSV to obtain
   hashes. Do not join solely on row order or platform-specific absolute paths.
3. ViT's probability column is `prob_fake`; CNN's is `probability_fake`.
   Report accuracy, fake F1/recall, AUROC, and raw calibration consistently.
   Use validation predictions for any proposed fitted calibration/threshold;
   do not tune on test labels or overwrite this fixed-threshold baseline.
4. Produce performance-versus-k figures, matched CNN-ViT comparisons, and paired
   augmentation comparisons. Record missing CNN runs explicitly. Label CNN
   crop/flip separately from ViT blur/JPEG/resize/noise/filtering. ResNet is fully
   fine-tuned; CLIP's encoder is frozen, so compare the complete pipelines.

## Interpretation limits

- All test fakes are **ADM**. The `family_summary.csv` does not establish
  performance across unseen generator families.
- The pilot's test images change across k/repeats. Baseline/augmented share the
  same split within an experiment, but a trend across k is not a controlled
  comparison on one fixed test set. The corrected dataset is a future release.
- Repeats vary dataset draws/generator mixtures, not classifier seeds. Their
  standard deviations are not independent confidence intervals.
- JPEG/PNG compression differences remain a possible confound. These outputs do
  not establish that the model used the shortcut or that augmentation removed it.
- The existing exact-AUROC tie-break remains part of this pilot's selection rule.
  Correct it only in a separately versioned future protocol.

All 297 copied artifact hashes were checked against the CARC inventory. All 48
validation/test exports reproduce saved metrics within absolute tolerance 1e-6;
aggregate means/standard deviations also match. This checks exports, not original
image bytes or checkpoint weights. The index and audit are locally generated
handoff aids; they are not copied CARC artifacts listed in that inventory.

Keep this result snapshot intact. Publish future studies under separate names.
Full source: `/project2/yzhao010_1531/csci566-genimage/results/vit_master_v1/`;
source summary: `summaries/0e9eeb5f22d9/`.
