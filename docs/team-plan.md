# Project status and proposed team plan

**Current status:** this page preserves an October planning snapshot. All 192
corrected ViT benchmark evaluations are now complete; use the
[November 8 remaining-work review](vit-status-2026-11-08.md) and
[completed benchmark handoff](../results/vit_benchmark/README.md) for current work.

Planning snapshot: **October 26, 2026**. The next progress report is approximately
two weeks away: progress, literature review, initial results, and an updated plan.
A final report and an in-person poster defense follow later.

**Completion update (October 26):** CARC job 13458685 reports all 24 ViT runs and
24 clean test evaluations complete. See the [results and evaluation handoff](../results/vit_master_v1/README.md).
Progress statements below describe the earlier planning snapshot.

Findings below were reviewed by Jad with ChatGPT. The work split is a proposal
for the seven-person team, not a record of agreed assignments.

## Main question

Does adding training generators improve detection of an unseen generator while
keeping total training images fixed? Compare CNN and ViT detector pipelines using
the same released CSVs and metrics. The [proposal](../CSCI566_submission.pdf) also
discusses calibration, augmentation, domain-adversarial/contrastive training,
zero-shot multimodal comparisons, and deepfake transfer.

There are eight generator variants, counting Stable Diffusion 1.4 and 1.5
separately, so k=7 can leave one unseen. With only seven total generators, k=7
would leave none unseen. Variants are not independent generator families.

## What exists now

- **ViT:** master notebook, synthetic tests, Jupyter fingerprint fix,
  preprocessing-address compatibility fix, and resumable Python/Slurm runner.
- **Model:** frozen CLIP ViT-L/14, normalized features, trainable linear real/fake
  head. This is supervised classification, not zero-shot CLIP.
- **Selected study:** ADM holdout; k=1,2,4,7; three dataset repeats; classifier
  seed 17; baseline plus augmented = 24 selected runs. Selected does not mean done.
- **Last reported progress:** k2/rep2 baseline. This page is not a live job check.
  Verify completion from outputs/logs under
  `/project2/yzhao010_1531/csci566-genimage/results/vit_master_v1/`.
- **CNN:** code uses shared CSV rows and provides no-augmentation and crop/flip
  modes. The reviewed committed notebook has no execution outputs confirming
  completed runs; collect actual exports/logs from the CNN team.

Keep the running pilot's protocol unchanged. Preserve its outputs when preparing
the next dataset/protocol version.

CNN is fully fine-tuned ResNet-50; CLIP's encoder is frozen. CNN augmented mode
uses crop/flip; ViT uses blur/JPEG/resize/noise/low-pass filtering. Label these as
different treatments and compare the complete detector pipelines, with their
pretraining/training differences explicit.

Dataset repeats change image draws and generator combinations; they are not
independent classifier seeds. CNN's footer mentioning seeds 17/18/19 is stale:
the reviewed code uses `base_seed=17` for each run.

## Proposed ownership

| People | Work now | Deliverable |
|---|---|---|
| Jad (ViT) | Finish, verify, and archive the current baseline/augmented ViT study. | Prediction exports, metrics, configurations, checkpoints, run inventory. |
| ViT teammate A | Compare CNN-ViT exports and summarize literature. | Comparison notebook, figures, calibration/error summaries, short literature review. |
| ViT teammate B | Temporarily help dataset work; own fixed splits and checks. | Fixed held-out tests across k/repeats, stable real splits, validation report. |
| Dataset members (2) | Prepare a separate compression-matched release with teammate B. | Measured image metadata, derivative JPEGs, new CSVs/hashes, original-image mapping. |
| CNN members (2) | Complete corresponding CNN experiments and export predictions. | Results for matching experiment IDs and clearly named training treatments. |

Teammates A/B do not duplicate Jad's training or modify the active study. CNN and
dataset code changes stay with their owners through focused reviews.

Teammate A can start from available ViT exports and CNN's `test_predictions.csv`
schema. Match release/experiment and image identities using SHA-256 (map CNN paths
through the saved CSV when needed); check labels/coverage before computing common
metrics. Plot accuracy, fake F1/recall, AUROC, and raw calibration. Record missing
CNN runs. Baseline/augmented are paired within an experiment; changing test sets
currently limit interpretation of trends across k. Do not fit thresholds on test.

## Next dataset release

Two findings need preparation work in a separate release:

1. **Fixed test images.** ADM k1/rep0 and k2/rep0 have 560 test images each but
   share only 223. Section 4b seeds a new random generator per experiment and
   resamples tests inside the k/repeat loops. Select tests once per holdout with
   a separate random generator and reuse across k/repeats. Reserve real test
   images before selecting real train/validation images; keep real split roles
   stable across experiments.
2. **Compression matching.** Reviewed manifests reference 2,000 unique real paths,
   all `.jpeg`, and 11,759 fake paths, all `.png`. This signals a possible pixel
   compression shortcut, not proof that our models exploit it. Measure actual
   formats, JPEG settings, and dimensions. Preserve real JPEGs and originals;
   create derivative fake JPEGs matching observed real settings/distributions
   reproducibly within each generator/split. Do not assume quality 96. Document
   remaining size/semantic biases; matching compression does not remove them all.

Check the new release for fixed test identities, stable real split roles,
class/generator counts, readable files, and no exact-content overlap across splits.
Keep original source IDs plus derivative hashes: conversion must not hide the same
source image appearing in multiple splits. Publish paths/release notes; both model
teams use the same release. Keep the current pilot files intact.

## Grader feedback and unfinished work

The grader warned about high-frequency artifact shortcuts and recommended spatial
and frequency augmentations (blur, JPEG recompression, filtering) alongside
domain-adversarial training. Extension comments encouraged zero-shot multimodal
and deepfake comparisons, plus breakdowns by generator family.

| Item | Current status / next action |
|---|---|
| Artifact-focused augmentation | Present as a combined ViT recipe; CNN crop/flip differs. Plan shared robustness conditions after the pilot. CLIP stays frozen, so its encoder is not retrained to remove shortcuts. |
| Calibration | ViT exports raw metrics/plots. Compare probability quality; fitted calibration or thresholds use validation data only. |
| Domain-adversarial / contrastive training | Not implemented. Plan a separate experiment if retained, or explain narrowed scope in the updated plan. |
| Family breakdown | Only ADM is held out currently. Add holdouts before broader conclusions. |
| Zero-shot multimodal comparison | Proposed, not implemented. Scope a fixed shared subset; record model/prompt/settings if pursued. |
| Deepfake transfer | Proposed extension; time permitting after the core comparison. |

The AUROC floating-point tie-break correction also remains future work. Use a
numerical tolerance and validate it in a **new** protocol/study; do not alter the
active pilot's checkpoint-selection rule or rewrite its checkpoints.

## Aim for the progress report

1. Finish/archive the ViT pilot and collect matching available CNN exports.
2. Produce initial comparison figures and a short literature review. Report pilot
   limitations and remaining runs honestly.
3. Prepare/verify the corrected dataset. Start a small matched CNN-ViT rerun after
   agreeing on the release and future protocol changes.
4. Update the plan for more holdouts and retained extensions. Every proposed
   method need not be finished before a progress report.

The final report/poster can grow from performance-versus-k figures, CNN-ViT
comparisons, augmentation effects, and robustness. Current conclusions stay
**ADM-specific**.

## Starting literature and evidence

- [GenImage project and paper](https://github.com/GenImage-Dataset/GenImage):
  dataset and cross-generator/degraded-image benchmarks.
- [Towards Universal Fake Image Detectors that Generalize Across Generative Models](https://arxiv.org/abs/2302.10174):
  pretrained vision-language features and linear probing for detection.
- [Fake or JPEG?](https://arxiv.org/abs/2403.17608): compression and image-size
  confounds in generated-image detection datasets.

Status sources: code review at `69ebb54`, supplied CARC logs, the
`vit_review_20261026T091349Z` bundle (a snapshot, not a finished study), the proposal,
and shared grader comments. Refresh job status from new logs/exports before
claiming completion or reporting additional metrics.
