# Team research handoff — November 23, 2026

## Start here

- [Team-review report draft](FINAL_REPORT_DRAFT.md): findings, methods, caveats
  and references. This is not a final submission; the team should review the
  course format, authorship, figures and citations.
- [Experiment log](FINAL_REPORT_EXPERIMENT_LOG.md): chronology, datasets,
  methods, decisions, validation and errors.
- [All research results](README.md): links to face transfer, clean comparison,
  robustness, zero-shot vision-language results and objective comparisons.

## What is complete

The corrected clean studies have 192 CNN and 192 ViT runs; 192 matched model
pairs were audited against shared test-image hashes and labels. The ViT k=7
JPEG60 and resize evaluation covers all eight holdouts and passed its audit.
The CNN extension completed the seven remaining holdouts, so outputs exist for
all eight; see the qualification below. Saved-model face transfer, fixed-prompt
zero-shot CLIP, a small Qwen vision-language test and exploratory face-adapted
and generator-objective experiments are documented in the results index.

The latest generator-objective comparison uses all eight holdouts, three
optimization seeds, the same repeat-0 data split and a frozen CLIP encoder.
Its 72 per-objective outputs and selected-head replays are available in the
[three-seed results](generator_objectives_seeds/README.md). The small average
differences do not establish a reliable winner.

The [DeepFakeFace still-photo pilot](still_face_v1/README.md) adds a shared
50-real, 100-fake inference-only comparison and a
[crop/resize and JPEG follow-up](still_face_geometry_v1/README.md). The baseline
ViT falsely flagged 4/50 real still photos; baseline CNN flagged 23/50. These
are small exploratory samples with unverified identity independence and
class-correlated image dimensions; they do not by themselves explain DF40
video-crop errors. No exact file or recorded decoded-pixel matches were found
against the compact GenImage inventory; altered images and pretraining overlap
remain possible.

## What needs care

1. **CNN robustness remains provisional.** Four holdouts exceeded the original
   0.001 clean-score replay tolerance. The maximum absolute score difference was
   0.0046387, and 3 of 13,440 affected predictions crossed the 0.5 threshold.
   Identity, labels, checkpoints and recomputed metrics passed other checks, but
   the reason for the discrepancy remains unknown. Do not describe the complete
   eight-holdout comparison as fully passing that audit. See the
   [robustness report](cnn_robustness_extension/README.md).
2. **DF40 transfer is exploratory.** Person-level independence and cross-data
   overlap are not established. Fake labels come from dataset provenance and do
   not prove every selected crop visibly exhibits manipulation. The adapted head
   is supervised and its test was inspected earlier.
3. **The Qwen evaluation is a small pilot:** one prompt, one model, one
   generator and 64 test images. It does not represent a broad VLM comparison.
4. **No pristine confirmation set was created for the objective extensions.**
   Additional seeds vary optimization, not dataset splits; all test data were
   previously examined. The model pipelines also differ in preprocessing and
   training, so they are not an architecture-only comparison.

## Access and demonstration

The prediction and checkpoint archives committed here contain derived results,
not the underlying face images or pretrained backbones. Large source artifacts
and training checkpoints remain on CARC; source images require access to the
team Drive or the CARC data location described in the results index. Check team
access rather than assuming each teammate has the same permissions.

The report and CSV files can be opened directly from GitHub. On Windows, after
cloning or pulling the team repository, a simple no-training demonstration is:

```powershell
cd "<path-to-cloned-CSCI566-AI-Image-Classifier>"
Import-Csv .\results\df40_transfer\generator_objectives_seeds\mean_by_holdout.csv |
  Format-Table holdout, objective, mean_accuracy, mean_auroc
```

The placeholder must be replaced by the teammate's actual clone directory.
For the original author's local working directory:

```powershell
cd "C:\Users\creat\Desktop\school\CSCI566\AI project"
Import-Csv .\results\df40_transfer\generator_objectives_seeds\mean_by_holdout.csv |
  Format-Table holdout, objective, mean_accuracy, mean_auroc
```

That local checkout may need a safe pull first; preserve any local edits. The
published report files are also available directly on GitHub without a pull.

## Most useful next step

Have the team review and accept the dataset protocol and report wording. For a
stronger research claim, establish source/person separation and evaluate on a
fresh untouched face test, and investigate or explicitly document the CNN
clean-score replay discrepancy before treating its robustness comparison as
fully verified. Prepare the poster and final presentation using reviewed figures.
