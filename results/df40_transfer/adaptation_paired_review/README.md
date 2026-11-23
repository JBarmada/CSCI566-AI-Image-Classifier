# Paired exploratory adaptation review

The three generator-objective feature adapters were compared on the same
holdout, optimization seed and test images. The paired table subtracts the
binary adapter's results from each alternative for all eight holdouts and
three seeds: 24 matched pairs per alternative. These are descriptive pairs,
not 24 independent experiments: the three seeds share the same data split,
test images recur, and all tests had previously been inspected.

| Alternative vs binary | Higher accuracy | Lower accuracy | Tied | Mean accuracy difference | Range |
| --- | ---: | ---: | ---: | ---: | ---: |
| Generator-adversarial | 13 | 8 | 3 | +0.22 percentage points | −1.43 to +2.68 points |
| Supervised contrastive | 8 | 13 | 3 | −0.13 percentage points | −2.50 to +1.07 points |

These figures show that outcomes vary across the matched runs; they do not
establish a reliable winner or justify a significance claim. The prior
holdout-level comparison averages the seeds first, producing 5/3 adversarial
and 4/4 contrastive better/worse holdouts. That is a different counting unit,
not a contradiction. Accuracy, AUROC, false-positive and true-positive deltas
for each pair are in [paired_deltas.csv](paired_deltas.csv). The source data
and prior audit are in [three-seed results](../generator_objectives_seeds/README.md).

This analysis concerns generator-objective adapters on GenImage, not the
separate supervised face-adapted head. A stronger comparison would freeze the
protocol and evaluate independent dataset splits and a fresh untouched test;
see [follow-up recommendations](../FACE_FOLLOWUP_DECISIONS.md). No training,
threshold changes or new test evaluations were performed for this review.
