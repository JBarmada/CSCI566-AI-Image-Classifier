# Exploratory error audit

The baseline CNN labeled 62/62 Celeb-real and 37/38 YouTube-real images fake.
The augmented CNN labeled 34/62 and 23/38 fake; the baseline ViT labeled
58/62 and 32/38 fake; the augmented ViT labeled 54/62 and 29/38 fake. These
small, nonrandom subgroups do not isolate a causal effect of source.

The contact sheet `vit_real_false_positive_review.jpg` displays the 12
highest-scoring real images for the baseline ViT. Faces are visible in each;
some images have dark triangular borders, and image quality and pose vary.
These observations cannot establish why a model predicted fake. The sheet was
selected by model score for diagnosis, not for curating evaluation data.

The manifests identify real and fake archive members and provide hashes, but
do not document all upstream face detection, crop alignment, compression, and
processing details, nor identify a verified matching original for each fake.
Comparing the team's construction code and source documentation is pending.
Do not remove examples because of model predictions.

Separate threshold calibration is pending genuinely separate data. The DF40
CDF metadata's validation and test aliases were previously found to contain
identical paths. Avoid treating them as independent sets. The shared test has
already been inspected and must not be used to fit and claim an unbiased test
threshold. Confirm source-video and known identity separation; record unknown
identity relationships and do not assert they are disjoint.
