# Matched clean benchmark and confidence audit

The CNN completion marker reports 192 valid runs and zero missing runs. The
prediction export contains each run's validation and test predictions, config,
and completion marker. The attached aggregate files are copied from CARC.

`analyze_matched.py` matched the 192 CNN runs to the 192 existing ViT runs by
run name. Every paired test set contained the same 560 distinct SHA-256 image
IDs and labels, for 107,520 paired image predictions. Accuracy, AUROC, Brier
score and 10-bin expected calibration error were recomputed; all CNN values
agreed with saved results to numerical precision, and ViT values within 1e-6.
`matched_confidence_per_run.csv` includes per-run results and disagreement
counts. `matched_confidence_summary.csv` averages three dataset repeats in each
of eight holdouts with equal weight. Dataset repeats reuse test images; they are
not independent random seeds or 24 independent holdout datasets.

![Brier score and calibration error](matched_confidence.png)

At k=7 baseline, mean accuracy was 0.8330 for CNN and 0.8830 for ViT. Mean
10-bin ECE was 0.1138 for CNN and 0.2816 for ViT. ECE depends on the bins and
averaging scheme; it is not a universal measure of calibration. These results
compare the full existing pipelines, which use different pretraining and
fine-tuning, rather than isolating architecture. No recalibration was fitted.

`reliability_k7.csv` and `reliability_k7.png` pool predictions within each
score bin across the eight holdouts and three repeats. The plotted counts are
prediction rows, not independent images, and pooling can hide holdout-specific
differences. These figures describe the existing fixed-threshold models.

Reproduce on Windows after extracting the CNN prediction export under
`prediction_export/runs` and placing the ViT audit export at the adjacent
`audit_sources/vit_benchmark` workspace path:

```powershell
cd "C:\Users\creat\OneDrive - University of Southern California\Documents\ChatGPT\CSC566 Proj"
& "C:\Users\creat\miniconda3\envs\test\python.exe" deepfake_transfer\clean_cnn_audit\analyze_matched.py
& "C:\Users\creat\miniconda3\envs\test\python.exe" deepfake_transfer\clean_cnn_audit\plot_confidence.py
```

## Class-specific errors and disagreements

`error_breakdown.py` matches the 48 clean k=7 CNN–ViT runs by image hash
and label, and writes per-run and per-holdout results to
`error_breakdown_per_run.csv` and `error_breakdown_by_holdout.csv`. Each run
contains 280 real and 280 fake test images. At the fixed 0.5 threshold,
the baseline CNN and ViT incorrectly flag an average of 39.04 and 17.21
real images per run, respectively, and miss 54.50 and 48.33 fakes. The
augmented pipelines flag 49.04 and 12.08 real images and miss 106.42 and
84.38 fakes, respectively. These counts are averages over eight equally
weighted holdouts and three dataset repeats, not independent observations.

For baseline real images, the ViT alone is correct on 32.04 images per run
and the CNN alone on 10.21; for fake images, the respective counts are
36.92 and 30.75. On baseline fakes, both models err on 17.58 images per
run on average. These descriptive figures do not establish that combining
models would improve performance: a combination rule would require separate
validation. The script cross-checks image and label membership; its
disagreement and high-confidence-error counts were independently compared
with the existing 192-run audit. A high-confidence error means a wrong
prediction with fake probability at most 0.1 or at least 0.9. These numbers
refer only to this benchmark, threshold and pipeline.
