# Exploratory five-method comparison: frozen protocol

Prepared November 23, 2026 **before candidate scoring**. Owner: Jad. The
[data audit](FRESH_CANDIDATE_REVIEW.md) describes the duplicate cross-method
pair, its exclusion, Jad's visual check and remaining unknowns. This is a
source-ID-separated follow-up within FaceForensics++, not a person-independent
or external confirmation. It can be scored only as exploratory: the source
identities, mirror bytes and individual manipulation labels remain unverified.

## Sample and checkpoints

Use exactly the 203 rows in `fresh_candidate_review_v2/manifest.csv`, SHA-256
`1ccc573a2add842afafc23dbed69ebea337e94a2c4893725e7e1b6a0d7c4b7b4`.
The set has 58 real crops and 29 crops each for SimSwap, FOMM, Wav2Lip,
InSwapper and PIRender. It comprises 29 video-ID pairs; reusing the same real
images across methods does not make them separate observations. See the audit
for the fixed score-blind exclusion; do not change samples using model scores.

Evaluate the existing CNN and ViT baseline and augmented models used in the
shared FF evaluation (held-out ADM, k=7, repeat 0, seed 17). Also evaluate the
existing face-adapted ViT linear head saved as `head.pt` from job 14482534,
whose SHA-256 is
`8d0fdf3893fc7362801e8801c4fae9fe87e60f069a43803af3aade394e079af2`.
The shared frozen CLIP backbone SHA-256 is
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`.
Use each model's existing preprocessing and saved weights; check image and
checkpoint hashes, replay original reference predictions, and verify repeat
inference on a small fixed sample. No training, prompt optimization, model
selection or threshold fitting uses these images.

## Metrics and interpretation

For each model and method, report real false positives out of 58, fake
detections out of 29, false-positive rate, fake recall and AUROC comparing
that method with the same 58 real crops. Report both the original 0.5 threshold
and the previously selected FF-validation threshold. The latter is 0.9990234375000001
for CNN baseline, 0.7763671875000001 for CNN augmented, 0.615590274333954
for ViT baseline, 0.5530025362968446 for ViT augmented, and
0.5676835775375367 for the face-adapted head. These existing thresholds are
not guaranteed to meet a particular false-positive rate on this sample.

The face-adapted head trained on SimSwap, FOMM and Wav2Lip, so those three
are not unseen-method tests. InSwapper and PIRender are candidate unseen
methods under the recorded training manifest, subject to the archive-label
issue. Models differ in training and pretraining, so differences do not isolate
architecture or prove adaptation causality. This small sample and linked real
images do not support broad accuracy or identity-generalization claims. Preserve
all predictions, run metadata, errors and negative findings; do not select
which methods to report after inspecting results.
