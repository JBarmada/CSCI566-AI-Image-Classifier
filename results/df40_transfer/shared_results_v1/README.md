# DF40 shared-set inference, version 1

Run 14481515 on CARC completed with exit code 0. This is an inference-only
comparison of the existing ADM k7 rep0 seed17 baseline and augmented ViT and
CNN checkpoints. Both architectures use their original evaluation preprocessing
and a fake probability threshold of 0.5. No weights or threshold were fitted
on DF40. See ACCEPTANCE.json for checkpoint hashes and replay/repeat checks.

All 500 shared image bytes matched the supplied SHA-256 values and decoded at
the listed 256-by-256 dimensions. There are 100 real, 200 SimSwap, 100 FOMM,
and 100 Wav2Lip images. Each method comparison reuses the same 100 real images;
SimSwap therefore has twice as many fake images as either other comparison.
Each method has one listed image per video group, though groups may overlap
across methods. These are provenance labels rather than frame-level proof of a
successful manipulation. The real set contains 62 Celeb-real and 38
YouTube-real images; exact matching to each fake source is not established.

The operational smoke used the first six rows for each class/method from the
shared manifests. These 24 images are also included in the 500-image
descriptive evaluation, a departure from the earlier independent-smoke plan;
no tuning or score-based selection was performed. The smoke flag identifies
them in predictions.csv. This dataset is a limited subset and the four models
are one matched benchmark run, not the complete 192-pair comparison.

metrics.csv and metrics.json contain per-method confusion counts, accuracy,
fake recall, real false-positive rate, F1, and AUROC. The latter estimates how
often a fake receives a higher score than a real, with ties split evenly.
The same real samples are reused, so comparisons are dependent. Class
imbalance affects accuracy and F1 for SimSwap. These results are exploratory
and do not establish performance on all DF40 methods.
