# CNN clean-score replay investigation

## Finding

The failed clean-score replays are explained by a GPU-dependent numerical
change in the tested setup. The original benchmark jobs for Stable Diffusion
1.4, Stable Diffusion 1.5, VQDM and wukong ran on Tesla V100 GPUs; the
robustness extension ran on Tesla P100 GPUs. Original jobs for the other four
holdouts ran on P100s and their clean replays passed. The evidence is in
[original_log_audit_v2.json](original_log_audit_v2.json); SHA-256 hashes identify
the original job logs used.

A controlled replay of one SD 1.4 baseline used the same checkpoint, 560
images, original notebook functions, batch size 64, mixed precision, and
software on both GPUs. The V100 reproduced all scores exactly; the P100's
maximum difference from the original was 0.001953125. The only deliberately
changed factor was the GPU model; device-specific numerical kernels and
hardware cannot be separated by this experiment. Both jobs used the same
NVIDIA driver (580.159.04), PyTorch 2.7.1+cu118, torchvision 0.22.1+cu118,
Pillow 12.1.1 and cuDNN 90701. See jobs 14491389 and 14491390 and their logs.

A V100 replay using the original notebook functions then matched the saved
scores exactly for all 24 affected checkpoints: 13,440 prediction rows, with
repeated images across checkpoints, maximum difference 0. It checked the
checkpoint SHA-256 against each completion marker and confirmed prediction
hash order. Job 14491538 completed successfully; per-checkpoint evidence is in
[matrix_results.json](matrix_results.json). The original 0.001 maximum-score
acceptance tolerance and 0.5 classification threshold were unchanged.

An independent replay of the **actual robustness notebook**, restricted to
the clean condition and run in a fresh output directory, also reproduced all
24 affected original score sets exactly on V100. Its checkpoint, image-hash,
label and 0.001 reference checks passed. Job 14491618 exited successfully;
[robust_clean_results.json](robust_clean_results.json) records all 24 runs.
This validates the clean pipeline under the original GPU model; it does not
change the existing P100 results.

## Provenance correction

The earlier report cited an SD 1.4 `config.json` as evidence the original GPU
was a P100. That field is not reliable original-GPU provenance. For the checked
run, the prediction file was last modified at 13:39 UTC on November 10,
whereas the config and upgraded completion marker were written at 22:03 UTC.
The benchmark notebook writes the current GPU into config before checking
whether a run is already complete, so a later audit can overwrite that field.
The original training job log reports V100. This corrects the previous
conclusion; it does not alter any original prediction or model file.

## Attempts and checks

1. Earlier audit: P100 robustness replay failed the 0.001 gate on 24 of 48
   checkpoints; largest difference 0.004638671875, three changed classifications
   across the 13,440 affected prediction rows. Repeating the P100 replay and
   setting deterministic cuDNN did not restore the old scores. Original
   notebook functions on P100 differed by up to 0.001953125 in one case.
   Original diagnostic job 14482900 failed on a missing plotting-only seaborn
   dependency; job 14482918 omitted that import and completed.
2. Original Slurm logs: four P100 and four V100 original holdout jobs
   align exactly with the pass/fail pattern. All original logs report PyTorch
   2.14.1+cu126; the original teammate environment is not accessible from the
   current account. Recorded `use_amp` is true in both runs, and the evaluated
   batch size is 64. Original and replay transformations use bicubic resize,
   center crop and ImageNet normalization. The existing audit verified hashes,
   labels and checkpoint identity; the V100 matrix additionally checked image
   order and checkpoint hashes for all 24 affected checkpoints.
3. Controlled P100/V100 replay, the 24-checkpoint original-notebook V100
   matrix, and the separate 24-checkpoint robustness-notebook V100 clean replay
   are described above. No training took place and no original files were
   modified. A first attempt to inspect a result file failed due to command
   quoting; a corrected read succeeded without affecting results.

## Interpretation and limits

The controlled comparison establishes a reproducible GPU-dependent difference
for the tested checkpoint and provides a consistent explanation across all
24 affected checkpoints. The original PyTorch/CUDA version differs from the
replay version; reproducing exactly on V100 does not prove that every software
combination is equivalent or identify the low-level operation responsible.
The old P100 clean results still failed the original 0.001 gate. Therefore the
existing full eight-holdout robustness aggregates remain provisional: an
independent successful V100 clean replay does not retroactively change those
P100 outputs, and the existing degraded-condition predictions were not
re-evaluated here. The original clean benchmark scores remain authoritative.

A conclusive full-gate remediation would rerun the affected clean and degraded
conditions on V100, validate all checkpoints, identities and metrics, and
rebuild the paired summaries from the accepted outputs while retaining the
original P100 results for provenance.
