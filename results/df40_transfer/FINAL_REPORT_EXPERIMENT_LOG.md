# Final report evidence log: DF40 transfer

## 2026-11-23 — completed inference-only evaluation

**Question.** Do existing GenImage-trained CNN and ViT detectors transfer to
face-manipulation images? This is a preliminary test on a selected subset,
not an evaluation of all DF40 manipulations.

**Data.** The team-shared dataset contains 100 real Celeb-DF images and 400
DF40 fake images: 200 SimSwap, 100 FOMM and 100 Wav2Lip. Each method contains
one image per listed video group. The same 100 real images are reused for each
method comparison. The real images comprise 62 Celeb-real and 38 YouTube-real;
exact matching to each fake's source has not been established. All 500
downloaded images matched the SHA-256 values in the shared manifests and
decoded at the listed 256-by-256 dimensions. These checks establish agreement
with the team manifest, not independent authentication of every source label
or proof that every frame visibly exhibits a successful edit.

**Models and procedure.** One matched benchmark configuration was used:
holdout ADM, k=7, rep=0, seed=17, with baseline and augmented CNN ResNet-50
and frozen CLIP ViT-L/14 plus its previously trained linear head. Each model
used its original evaluation preprocessing and fake threshold 0.5. No model
was trained or calibrated on these images. This is not the complete 192-pair
comparison and the CLIP head evaluation is not text-prompt zero-shot CLIP.

**Checks.** CARC job 14481515 completed successfully on a Tesla P100. Both CNN
reference replays had maximum probability difference 0; ViT baseline and
augmented reference replays differed by at most 2.79e-8 and 2.49e-8,
respectively, within the 1e-5 gate. Repeated 24-image smoke inference matched
for both architectures. The smoke used the first six manifest entries in each
of real, SimSwap, FOMM, and Wav2Lip; those 24 images also appear in the full
descriptive evaluation. This differs from the earlier independent-smoke plan;
no score-based tuning was performed.

**Results.** AUROC compares fake versus real score rankings; false-positive
rate is the fraction of the 100 real images labeled fake at threshold 0.5.

| Model | SimSwap AUROC | FOMM AUROC | Wav2Lip AUROC | Real false-positive rate |
| --- | ---: | ---: | ---: | ---: |
| CNN baseline | 0.4383 | 0.37685 | 0.4520 | 99% |
| CNN augmented | 0.507625 | 0.4514 | 0.5292 | 57% |
| ViT baseline | 0.41845 | 0.7204 | 0.4331 | 90% |
| ViT augmented | 0.45485 | 0.6790 | 0.4494 | 83% |

**Interpretation.** False positives are high for all four configurations at
their existing threshold. ViT baseline ranks FOMM above real better than
chance on this subset, while most other method-level AUROCs are close to or
below 0.5. This suggests limited transfer in this test; source, identity,
processing, sampling and label uncertainties prevent assigning a cause.
The unequal SimSwap class sizes affect accuracy and F1; the reused real set
makes method comparisons dependent. Avoid claims about population performance
or all 40 DF40 methods from this small selected subset.

**Future work, not yet performed.** Audit source matching and processing;
consider calibration on separately collected validation data; compare a
new face-trained linear head on frozen CLIP features with the existing head
and an appropriate CNN baseline. Training, validation and final test should
be disjoint by source video and, where traceable and feasible, identity.
Since these results have been inspected, reserve a fresh test set for final
claims after choosing adaptations. Contrastive or domain-adversarial training
are later candidates, not completed results.

**Evidence.** `shared_results_v1/ACCEPTANCE.json` contains hashes and replay
checks; `shared_results_v1/predictions.csv` contains per-image scores and
smoke flags; `shared_results_v1/metrics.csv` and `metrics.json` contain
per-method metrics and confusion counts. `shared_dataset_v1/verification.json`
records image verification. `shared_results_v1/README.md` summarizes the run.
The original shared manifests and Drive file IDs are recorded in
`SHARED_DRIVE_INDEX.json`. These artifacts should be retained with the report.

## 2026-11-23 — exploratory error and validation-data audit

The error audit found false positives in both real sources: baseline CNN
62/62 Celeb-real and 37/38 YouTube-real, augmented CNN 34/62 and 23/38,
baseline ViT 58/62 and 32/38, augmented ViT 54/62 and 29/38. These
subgroups are small and do not establish the cause. A score-selected visual
review and its limitations are documented in `shared_results_v1/ERROR_AUDIT.md`.

Calibration remains pending: no separate verified validation set has been
assembled and no threshold has been selected. DF40 fake mirrors responded to
small range requests, but the official real archive returned a quota error.
The existing test manifests contain 56 distinct identifiable id tokens;
source/target roles and numeric YouTube identities require further mapping.
Different video names alone do not establish identity independence. The team
dataset-building code was not found in the local workspace or GitHub default
branch; the user has been asked to share it. Do not infer that it is absent
from every branch.

Correction: a subsequent check found two untracked notebooks in the separate
local Git checkout, `01_DF40_real_100.ipynb` and `02_DF40_fake_400.ipynb`.
Their descriptions and manifest fields match the team dataset construction.
They say released PNG bytes are copied, one crop per video group sampled with
seed 42, and exact smoke groups and hashes excluded; related clips require a
verified mapping. They have not yet been fully audited against the manifests.
The checkout reports 16 commits behind its last known origin/main and has
local changes. No git pull was performed during the earlier search.

## 2026-11-23 — notebook and independent-data follow-up

The local team notebooks were statically audited against the shared manifests;
see `notebook_audit.md`. They copy released PNGs without re-encoding and use
seed 42 with one crop per group, but their default exclusions cover only the
earlier smoke set, not the complete existing 500-image evaluation or all
related identities. Upstream real/fake crop consistency and exact original
source-video matching remain unverified.

The official DF40 Celeb-DF and FaceForensics++ preprocessed real-crop ZIPs
returned quota errors during bounded probes. Three fake-method mirrors allowed
bounded reads. Candidate metadata counts are in `new_data/CANDIDATE_COUNTS.json`:
excluding all recognizable identity tokens in the prior 500-image set leaves
no known-ID fake groups for the three CDF methods. Thirty-two unused numeric
real groups have unknown identities. These token counts are conservative,
not verified source-target mappings. A distinct identity-disjoint test cannot
be claimed from those CDF candidates. Neither new images nor thresholds were
produced at this point. The separate calibration tooling and its synthetic
checks are preparation, not empirical evaluation results.

## Team chat source clarification

The user supplied team chat indicating that Saisha downloaded the Celeb-DF
ZIP to run the real-crop notebook and uploaded the resulting 100-image real
folder. The chat links the official FaceForensics++ real ZIP and the DF40
metadata folder but does not establish that anyone retained or uploaded either
full ZIP. The shared folder still contains the 100 real crops and 400 fake
crops already evaluated, not an identified additional real-crop archive.
Both official ZIP download endpoints again returned quota-exceeded HTML on
bounded probes. A teammate's independently accessible copy or newly shared
link may unblock crop retrieval; identity/source separation still requires
separate verification. The team discussed extending beyond three methods,
but the recorded result covers only SimSwap, FOMM and Wav2Lip.

## Team-downloaded Celeb-DF archive verified

The user's completed `Celeb-DF-v2_real_data_for_DF40.zip` in the assignment
checkout is 2,168,805,771 bytes. Its index contains 28,174 PNG crops, including
all 5,620 real crop paths in the DF40 CDF test metadata for the three methods.
All 100 real crops used in the earlier evaluation match the ZIP bytes by
SHA-256. Extracted and decoded one crop per unused exact metadata video-group
name, yielding 78 additional real candidates in `new_data/real_candidates_v1`.
The manifest records original archive paths, image hashes and identity status.
Of the 78, 46 have a named identity token also present in the prior 500-image
set, and 32 have numeric video names with unresolved identities. This is a
candidate pool, with no assigned validation/fresh-test roles or fitted
thresholds. Related source clips and fake-side separation remain unverified.

## Additional DF40 fake crops collected

The user reviewed four displayed possible source pairs and reported that pairs
1, 2 and 4 looked like the same video, while pair 3 did not. This is a visual
observation, not confirmed source mapping. The user chose to proceed with the
FaceForensics++ portion of DF40 for the next evaluation.

Follow-up overlap audit: all 78 candidates in each method share at least one
`idN` token with the earlier evaluation. None has an exact image SHA-256 or
full group-name match. `new_data/fake_candidates_v1/overlap_review.csv` lists
the evidence for every crop. The suggested clip matches remain filename-based
inferences: DF40 metadata inspected here does not establish the target/source
roles or verify video links. Shared tokens support potential person overlap,
but cannot by themselves confirm exact source-video reuse. Do not describe
this candidate pool as identity-independent.

`new_data/fake_candidates_v1` contains 234 released CDF PNG crops, 78 each
for SimSwap, FOMM and Wav2Lip. The three pinned mirrors expose exactly the
official metadata crop path lists. The selection used one lower-middle frame
per group, whether the filename suggests overlap with earlier clips, and a
deterministic hash rank. Each crop passed the ZIP CRC and PNG decode checks.
The new crops have distinct SHA-256 hashes and exact group names, with no
match to the 500 earlier images or the 78 new real candidates. The manifest
records the archive URLs, paths, hashes and inferred source clip links.

Filename heuristics suggest prior source clip overlap for 53/78 SimSwap,
77/78 FOMM, and 12/78 Wav2Lip candidates. These are possible links, pending
verified source mapping. Treat the images as a candidate pool rather than an
independent split. No scores or calibration thresholds were computed on these
crops. See the folder's `audit.json` for selection and retrieval details.

## Exploratory threshold sweep on the shared DF40 set

Used the saved scores for all four trained configurations and zero-shot CLIP.
For each model, compared the original 0.5 rule with the lowest cutoff that
kept false positives at or below 10%, 20%, or 30% on the same 100 real images.
A score equal to the cutoff counts as fake. Tied scores can put the achieved
rate below the cap, as with the baseline CNN at the 10% and 20% caps.

At the 10% cap, zero-shot CLIP flagged 10/100 real images and detected
52/200 SimSwap, 44/100 FOMM, and 11/100 Wav2Lip images. The 20-row sweep is in
`shared_results_v1/exploratory_thresholds.csv`. The script checked all image
hashes against the shared manifest and reproduced the published 0.5 counts.
The cutoff choice and its outcome use the same exploratory data. Independent
calibration and test performance remain unmeasured.

## FaceForensics++ DF40 calibration and test

After the user selected FaceForensics++, the user downloaded the DF40 real
archive. Pinned DF40 FF metadata lists the same images under val and test, so
`new_data/prepare_ff_plan.py` partitions 70 linked video-ID pairs by a fixed
seed into 35 pairs per split. Each split contains 70 real crops and 70 fake
crops per SimSwap, FOMM and Wav2Lip. Exact named source IDs are separated
between the two partitions; person identity and cross-dataset overlap with
Celeb-DF are not established. The released crop bytes and metadata provenance
are recorded in `new_data/ff_complete_v1` and `ff_fake_v1` manifests. All 560
files passed hash checks. Fakes came from pinned third-party mirrors; official
byte identity is not independently verified.

CARC jobs 14482277 and 14482369 passed reference and repeat inference gates.
The four matched ADM k7 rep0 CNN and ViT configurations used frozen weights.
Thresholds were fixed using only the 70 validation real images, targeting at
most 10% empirical false positives, before the separate test was scored. At
the frozen thresholds, test results were:

| model | threshold | real FP / 70 | SimSwap detected / 70 | FOMM detected / 70 | Wav2Lip detected / 70 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CNN baseline | 0.9990234375000001 | 5 | 2 | 6 | 11 |
| CNN augmented | 0.7763671875000001 | 9 | 11 | 4 | 6 |
| ViT baseline | 0.615590274333954 | 5 | 9 | 17 | 3 |
| ViT augmented | 0.5530025362968446 | 4 | 13 | 13 | 8 |

The 10% target was validation-only; CNN augmented test FPR was 9/70 (12.9%).
Results show low fake recall at these operating points. Scores, verification,
frozen thresholds and original-0.5 comparisons are retained in
`new_data/ff_results_v1`. These estimates are preliminary for one set of four
checkpoints and 70 real/70 fake samples per method. Shared identities and
relationships beyond the parsed IDs remain unverified. No CLIP text-prompt
baseline was run in this experiment.

## Public alternative real images

The public `afatwapas/deepfake_face_classification` Hugging Face repository
describes itself as DF40-derived. A bounded ZIP-index inspection of its
`val.zip` found 1,606 files under `val/real/`; three individual images were
retrieved successfully and decoded at 512 x 512. The archive uses flattened
file names; the source-video identities, exact original datasets, release
provenance and overlap with our data were not established. It is not verified
as a mirror of the team's 256 x 256 original real-crop archive, nor as an
independent validation split. The dataset card has inconsistent licensing
statements. No images from this candidate were added to our evaluation.

## Paired errors and zero-shot CLIP — 2026-11-23

The existing CNN and ViT predictions were paired by image hash for matching
baseline and augmented configurations; see `paired_analysis/README.md` and
`summary.csv`. Both baseline models mislabeled 89/100 real images; both
augmented models mislabeled 51/100. Among augmented FOMM fakes, only ViT was
correct on 43/100 and only CNN on 2/100. Among augmented real images, only CNN
was correct on 32/100 and only ViT on 6/100. These counts are descriptive and
do not demonstrate that an ensemble will generalize.

CARC job 14481981 evaluated frozen CLIP ViT-L/14 image-text similarity with
the fixed prompts `a photo of a real face` and `a photo of a manipulated
face`, selected before this run. Both embeddings were normalized; softmax of
scaled cosine similarities produced the two-prompt score. No task-specific
classifier head, training, or threshold tuning was used. At the equal-prompt
0.5 decision threshold, false positives were 24/100 real images; fake recall
was 86/200 SimSwap, 60/100 FOMM, and 27/100 Wav2Lip. Per-method AUROC was
0.6316, 0.7459, and 0.5520 respectively. The scores are prompt-relative, not
calibrated probabilities. This one prompt pair and the already inspected shared
set make the findings exploratory; the test contains shared people and uses
the same real images across methods. See `zeroshot/prompts.json`,
`zeroshot/ACCEPTANCE.json`, `zeroshot/predictions.csv`, and `zeroshot/metrics.csv`.
