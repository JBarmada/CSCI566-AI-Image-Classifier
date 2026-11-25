# Preliminary CPU face-candidate results

CARC job 14550254 completed inference on 189 fixed face crops with five model
variants (945 scores). No training or threshold fitting used these images.
[STATUS.json](STATUS.json) labels the run exploratory because the separate
[CPU reference replay](../cpu_reference_results_v1/README.md) exceeded the
original numerical tolerances. The P100 evaluation subsequently passed its
reference checks. CPU and P100 agreed on 944 of 945 decisions at threshold
0.5; nine baseline CNN decisions differed at its frozen validation threshold.
See the [paired comparison](../fresh_candidate_results_v3/README.md).

[predictions.csv](predictions.csv) contains the per-image scores;
[metrics_preliminary.csv](metrics_preliminary.csv) reports false positives,
fake detections and AUROC separately by method at both frozen thresholds.
[summarize_cpu_candidate.py](../summarize_cpu_candidate.py) verified 945 rows,
the complete set of 189 hashes per model, matching labels and methods, and
finite scores. It did not certify device equivalence, repeatability, mirror
provenance or individual labels. The original V100 attempt failed the
reference gate; the separate P100 retry passed it.

The dataset is small, the same real images recur across methods, identity and
cross-dataset separation are unverified, and third-party fake archive labels
have unresolved issues. The adapted head trained on three of these methods.
See the [frozen protocol](../FRESH_EVALUATION_PROTOCOL.md) and
[candidate audit](../FRESH_CANDIDATE_REVIEW.md). Treat every result here as
exploratory rather than independent confirmation.
