# CNN and ViT k=7 robustness comparison

**Update:** The affected runs were rerun and audited on V100. The current
paired results are in [the V100 rerun](../cnn_robustness_v100/README.md).
This page preserves the earlier P100 comparison and its limitations.

All eight CNN holdouts now have six checkpoints evaluated under clean,
JPEG 60 and resize-to-112-and-back conditions. The seven new holdouts add 84
degraded evaluations and 42 clean replays to the existing ADM results. The
existing ViT robustness study supplies the matching 144 condition/run rows.
[robustness_results_v1.tar.gz](robustness_results_v1.tar.gz) preserves CNN predictions, configuration,
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

The discrepancy was investigated further using the original job logs and
controlled replays. The affected original jobs used V100 GPUs; the extension
used P100 GPUs. With the same replay software and checkpoint, one V100 replay
matched the original scores exactly while the P100 differed. All 24 affected
checkpoints matched exactly when replayed through both the original benchmark
notebook and the clean robustness notebook on a V100. The earlier claim that
the original SD 1.4 run used a P100 relied on a config field that a later audit could overwrite; original job logs show V100.
See [the replay investigation](../cnn_replay_investigation/README.md) for
controlled tests and limitations. The original P100 robustness outputs still
failed the unchanged gate, so the existing aggregate remains provisional.

Both model pipelines use bicubic geometry before applying the same individual
JPEG or resize degradation, then their respective normalization. They differ
in architecture, pretraining and training strategy. The plots are descriptive,
and repeated images and related generators limit independence claims.
