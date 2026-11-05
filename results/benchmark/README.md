# Benchmark acceptance evidence

The corrected dataset is installed at
`/project2/yzhao010_1531/csci566-genimage/data/extracted/GenImage_Compact/benchmark`.
The image files remain in their original real-image folders and `_release_v2/jpeg_ai`.

- [Before/after numbers](before_after.json): ADM k1/rep0 versus k2/rep0 shared
  223/560 pilot test images; the corrected experiments share 560/560.
- [Installed-image acceptance](installed_acceptance.json): all 17,640 selected
  image bytes, source hashes, JPEG settings and split identities checked on CARC.
- [Release identity](RELEASE.json): immutable fingerprint and hashes of all manifests/audits.
- [Experiment plan](experiments.json): all 96 generator combinations and repeats.
- [Launch record](launch_record.json): CARC job IDs and deployed code revision;
  this is a submission snapshot, not live job status.
- [Local metadata acceptance](local_acceptance.json): all 96 experiments checked.
- [Similarity candidates](near_duplicate_candidates.csv): three candidate pairs;
  all six images excluded from the benchmark. Perceptual similarity is not proof
  of duplication; exclusion is conservative.

Each experiment has 1,400 real + 1,400 fake training images, 280 + 280 validation,
and 280 + 280 test. Fixed real splits and per-holdout fake tests are shared across
k/repeats. The builder uses seed 566; classifier training uses seed 17.

These files prove dataset acceptance, not completed model training. See
[the execution guide](../../docs/benchmark.md) for the GPU smoke test and 192-run
study. The original ADM pilot is preserved separately. JPEG history, dimensions
and semantic distribution remain limitations; do not call this a bias-free dataset.
