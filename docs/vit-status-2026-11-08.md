# ViT remaining-work review: November 8, 2026

The core corrected ViT experiment is complete. All 192 runs, clean test
evaluations and the final independent CARC audit passed. Final Slurm audit job
13757886 completed with exit code 0. The earlier chat, **Plan ViT Collaboration
Split**, ended while the runs were still progressing; current files supersede
that historical status.

## Verified and published

- 8 holdouts, k=1/2/4/7, 3 dataset repeats, baseline/augmented, classifier seed 17.
- 192 best-head files, 192 validation exports and 192 clean test exports on CARC.
- Exact release fingerprint checked against all configs; all 288 input-manifest
  hashes, prediction identities/labels and metrics rechecked after copying locally.
- [GitHub analysis tables, figures, prediction exports and data dictionary](../results/vit_benchmark/README.md).
- [Drive manifest and full raw-analysis archives](https://drive.google.com/drive/folders/155zGH4KJr_cifkuqI29l0BbEocfMlb74).
  Drive listing readback confirms archive names and byte sizes. The folder is
  private to the connected account until it is shared/moved by an authorized editor.

Source checkout on CARC was `4fecd4ca6fd47ea835fa94375e9d3964a7ec5d49`;
recorded batch statuses provide per-job source provenance. Latest GitHub main at
review start was `ef63a58ab1cc77be03f594bf5b57ac16f07f2593`, including the November 7
CNN notebook alignment. The local Desktop checkout is older; its untracked
notebook, `test.py` and `train.csv` are user work and were preserved.

## Priorities for the ViT portion

| Priority | Work still needed | Concrete deliverable |
|---|---|---|
| 1 | Give teammates access to the exact corrected manifests | Shared/moved Drive handoff; verify fingerprint and relative image paths before using it |
| 2 | Finish and audit the matching CNN benchmark | Full result/prediction inventory; verify each input manifest and image identity against this release |
| 3 | Compare matched CNN and ViT pipelines | Performance-versus-k plots, per-holdout/family tables, paired error comparisons, explicitly named training treatments |
| 4 | Run degraded-image evaluation of existing checkpoints | Clean/JPEG/blur/resize/noise/filter tables for baseline and augmented models; fixed conditions and deterministic evaluation seed |
| 5 | Analyze probability quality and errors | Reliability/Brier/ECE analysis; optional calibrator fitted only on validation; inspect false negatives/positives and confounds |
| 6 | Quantify uncertainty honestly | Paired comparisons accounting for shared real/test images and related generator variants; state the single classifier seed and limited combination sampling |
| 7 | Finish report/poster and artifact preservation | Methods, literature, results, limits, figures, source versions, checkpoint backup and reproducibility instructions |

No additional ViT training is required to finish the existing 192-run clean
study. Degradation evaluation can reuse the saved heads and frozen backbone.
It still requires inference and a separate, recorded evaluation protocol; preserve
the accepted clean outputs. Do not alter training settings or overwrite checkpoints.

The runner already defines optional blur, JPEG, resize, noise and low-pass test
conditions, but recorded results contain only `clean`. Training augmentation
does not by itself demonstrate robustness. The current combined augmentation
recipe is not an ablation isolating each transformation's effect.

## CNN comparison readiness

The current `cnn.ipynb` targets the same corrected `benchmark/manifests`, eight
holdouts, k values, repeats and seed. A live read of
`/project2/yzhao010_1531/csci566-genimage/results/cnn_benchmark/notebook/`
found **47** training-complete markers and **47** test-prediction files at the
check. This is a snapshot and an inventory count, not an independent CNN audit.
The count can increase as the CNN team's jobs continue under their own account.

The example new CNN ADM/k1/rep0 config trains on `glide`, matching the corrected
plan. The incompatible ADM-only pilot tables, whose k1/rep0 used `Midjourney`,
have been removed from the repository. Collect the corrected-benchmark CNN
predictions and verify saved input CSV hashes and labels before joining them to
the ViT benchmark.

Compare complete pipelines with their pretraining/fine-tuning differences:
CNN is fine-tuned ResNet-50; ViT is a frozen CLIP encoder with a linear probe.
Review actual CNN config augmentation before pairing treatments. All comparison
probabilities must use the same held-out image bytes, label convention and test
condition. Use SHA-256 identity joins, not row order or experiment names alone.

## Proposal scope versus unfinished experiments

`CSCI566_submission.pdf` is a **project proposal**, not a final submission rubric.
It proposes CNN/transformer comparison, generator diversity, augmentation,
calibration, domain-adversarial training, contrastive learning, zero-shot
multimodal comparison, and deepfake transfer. This review treats those statements
as project context; they do not authorize launching new training, paid API jobs,
external submissions or messages to teammates.

The core k study, clean augmentation comparison and raw calibration exports are
complete. Domain-adversarial and task-specific contrastive training are not
implemented in the reviewed ViT pipeline. CLIP's pretrained contrastive backbone
does not mean this project trained the proposed contrastive objective.
Zero-shot multimodal and deepfake-transfer results were not found in the reviewed
ViT exports. The proposal describes deepfakes as a time-permitting extension in
its timeline, while zero-shot comparison appears in the planned main experiments.
Agree on retained scope and explicitly document deferred items in the report.

## What the completed results support

Baseline mean clean accuracy rises from 73.00% at k=1 to 88.30% at k=7;
mean AUROC rises from 0.8608 to 0.9564. Augmented accuracy rises from 70.74%
to 82.78%, and AUROC from 0.8420 to 0.9361. These are equally weighted means
of holdout means, with fixed image budgets and the sampled combinations.
Individual holdouts are not uniformly monotonic. This is evidence for average
improvement on this benchmark, not proof of universal unseen-generator detection.

Augmentation underperforms baseline on average on clean images at every k.
Degraded-image testing is the next direct way to test its intended benefit.
The raw-probability reliability plot shows substantial calibration mismatch;
k=7 mean ECE is 0.2816 baseline and 0.2513 augmented. Do not market a raw score
as a calibrated probability of authenticity. Validation/test gaps are descriptive
because the validation images also differ from the test images.

Eight generator variants do not provide eight independent families. Compression
history, image size, semantic content and undetected related images remain
limitations even after JPEG-table matching and duplicate checks.

## Suggested three-person ViT handoff

| Role | Immediate task |
|---|---|
| Jad | Resolve shared Drive access, preserve accepted outputs/checkpoints, coordinate exact-release CNN exports |
| ViT teammate A | Work from published clean tables/predictions on figures, paired comparisons, errors and report text |
| ViT teammate B | Define and run recorded degraded-image evaluations; help audit matching CNN exports |

These assignments are suggestions, not messages sent to the team or an agreed
division. A separate decision is needed on proposed extra training and extensions.
