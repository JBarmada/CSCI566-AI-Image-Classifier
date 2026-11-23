# CNN–ViT robustness: V100 rerun

The four CNN holdouts whose original robustness replays failed the clean-score
gate were rerun on V100 GPUs. The rerun used the existing saved CNN checkpoints,
the original test images, batch size 64, mixed precision, and the same
preprocessing and decision threshold. For each checkpoint it evaluated clean,
JPEG quality 60, and resize-to-112-and-back conditions. No training occurred.
The original P100 results remain available in the earlier robustness extension.

All four V100 jobs completed successfully: 24 checkpoints and 72 evaluations.
An independent audit checked checkpoint and manifest hashes, prediction
membership and labels, completion markers, finite probabilities, and
recomputed accuracy, AUROC, ECE, Brier score and log loss. All 24 clean runs reproduced the original clean
scores exactly, passing the unchanged maximum-difference tolerance of 0.001.
The four unaffected holdouts retain their previously audited P100 results,
which also passed that gate. Thus all 48 CNN checkpoints in the updated paired
comparison meet the original clean replay check.

The V100 and P100 runs differed in six classifications over 40,320 prediction
rows: three clean, one JPEG, and two resize. These are repeated evaluations,
not 40,320 independent images. The largest per-run change in CNN accuracy was
0.001786; the largest per-run AUROC change was 0.000689. The findings are
numerically close to the preliminary comparison, but these counts show why we
preserved both versions. [INDEPENDENT_AUDIT.json](INDEPENDENT_AUDIT.json)
contains the per-run comparisons.

## Updated comparison

The following is the equal-weight average over the eight held-out generators,
after averaging the three repeats for each holdout. The corresponding ViT
results are the existing matching runs; their images, architecture-specific
preprocessing, and training differ from the CNN. Repeated images are not
independent samples.

| Training | Condition | CNN accuracy | ViT accuracy | CNN AUROC | ViT AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| Baseline | Clean | 0.8330 | 0.8830 | 0.9007 | 0.9564 |
| Baseline | JPEG 60 | 0.6443 | 0.6924 | 0.7888 | 0.8499 |
| Baseline | Resize 112 | 0.5020 | 0.5696 | 0.5807 | 0.8744 |
| Augmented | Clean | 0.7224 | 0.8278 | 0.8048 | 0.9361 |
| Augmented | JPEG 60 | 0.7065 | 0.7599 | 0.7825 | 0.8494 |
| Augmented | Resize 112 | 0.6885 | 0.8185 | 0.7506 | 0.8963 |

These are descriptive results on the tested generators, not proof of
performance on all synthetic-image or deepfake distributions. The clean-score
check supports reproducibility; it does not remove dataset dependence,
repeated-image dependence, or differences in model training.

## Evidence and reproducibility

- [paired_per_run.csv](paired_per_run.csv) has all 144 CNN–ViT matched rows,
  their GPU source and clean-check status. [paired_by_holdout.csv](paired_by_holdout.csv)
  and [paired_macro.csv](paired_macro.csv) contain the aggregations.
- `v100_results_v2.tar.gz` contains all four new holdout outputs, per-image scores,
  configurations and completion markers; it contains no source images or model
  weights. Extract it in this directory before rerunning `summarize_v100.py`.
  SHA-256: `a81516440c7c160b0d2eabb6142159cf4f1299cb025da398faae03e8587da2ff`.
- Jobs 14492182_3 through _6 ran on Tesla V100-PCIE-32GB with driver
  580.159.04, PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118, Pillow 12.1.1,
  and cuDNN 90701. `evaluate_v100.slurm` records the launch configuration.
- The first audit attempt on the login node failed when OpenBLAS tried to start
  too many threads, before processing results. Scheduled CPU audits with
  thread counts limited to one completed (14492761 and expanded audit 14494500). The inference jobs were unaffected.
- The prior GPU discrepancy investigation is in
  [cnn_replay_investigation](../cnn_replay_investigation/README.md). The
  [previous provisional comparison](../cnn_robustness_extension/README.md)
  and its [P100 prediction archive](../cnn_robustness_extension/robustness_results_v1.tar.gz)
  remain available for provenance.
