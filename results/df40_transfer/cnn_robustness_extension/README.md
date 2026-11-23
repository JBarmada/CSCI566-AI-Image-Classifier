# CNN and ViT k=7 robustness comparison

All eight CNN holdouts now have six checkpoints evaluated under clean,
JPEG 60 and resize-to-112-and-back conditions. The seven new holdouts add 84
degraded evaluations and 42 clean replays to the existing ADM results. The
existing ViT robustness study supplies the matching 144 condition/run rows.
`robustness_results_v1.tar.gz` preserves CNN predictions, configuration,
completion markers, summaries and notebook source. `paired_per_run.csv`,
`paired_by_holdout.csv`, and `paired_macro_provisional.csv` contain the matched
comparison. The plot shows the equal-weight mean over three repeats within
each of eight held-out generators. Dataset repeats share test images and are
not independent random seeds.

The independent audit checked checkpoint hashes, manifest membership, image
labels, finite scores, evaluation markers and recomputed accuracy and AUROC.
ADM, BigGAN, glide and Midjourney passed a predeclared 0.001 maximum clean
score difference against their original benchmark predictions. The other four
holdouts failed that strict replay tolerance for all six checkpoints each.
The largest difference across those runs was 0.004638671875, and three of
their 13,440 clean prediction rows changed classification at 0.5. Their largest
per-run AUROC difference was about 0.000077. Details are retained in
`AUDIT_INVENTORY.json`. The dataset, checkpoint, label and metric checks still
passed; the strict replay failures make the full eight-holdout summary
provisional.

For Stable Diffusion 1.4, a repeat of the new robustness code produced exactly
the same new scores, and setting deterministic cuDNN did not restore the old
scores. A replay using the original CNN notebook functions on one checkpoint
also differed from the saved source by 0.001953125. The original runtime
history is insufficient to establish the cause. The original benchmark's
saved clean scores remain authoritative for that benchmark; do not describe
the new CNN robustness results as fully passing the strict replay gate.

Both model pipelines use bicubic geometry before applying the same individual
JPEG or resize degradation, then their respective normalization. They differ
in architecture, pretraining and training strategy. The plots are descriptive,
and repeated images and related generators limit independence claims.
