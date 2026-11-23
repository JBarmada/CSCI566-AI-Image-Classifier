# Eight-holdout exploratory objective comparison

The ADM pilot was extended to the other seven k=7 repeat-0 holdouts using the
same frozen CLIP features, adapter, seed, optimization settings, validation
selection and 0.5 threshold. The extension protocol was fixed before those
seven jobs ran, but after observing the ADM pilot. The GenImage test sets had
already been examined, so this remains exploratory. See `PROTOCOL.md` and the
original `../generator_objectives/PROTOCOL.md`.

| Objective | Mean accuracy | Mean AUROC | Accuracy versus baseline |
|---|---:|---:|---|
| Binary adapter | 0.9172 | 0.9741 | Reference |
| Generator-adversarial | 0.9217 | 0.9744 | Better in 6 holdouts, worse in 2 |
| Supervised contrastive | 0.9161 | 0.9742 | Better in 2 holdouts, worse in 6 |

Means weight the eight holdouts equally, one repeat and one optimization seed
per holdout. Each model has a 560-image test with 280 real and 280 fake images;
images can be reused across holdouts. These counts do not establish statistical
significance or future performance. The binary reference is the same feature
adapter trained without the extra objective, not the original linear head.
The CLIP encoder was frozen; the small adapter was trained. The adversarial
classifier sees only the seven training-generator labels for fake images.

Per-holdout results, including false positives and fake detections, are in
`per_holdout_objectives.csv`. On VQDM, contrastive accuracy is slightly higher
than baseline, but its false positives rise from 16 to 37 of 280 real images;
this illustrates why accuracy alone can hide important tradeoffs. One selected
checkpoint per objective reached the 20-epoch cap. These settings were not
optimized or tested over multiple seeds.

Jobs 14483084_0 through _6 all completed with exit code 0. The feature cache
identity, labels and original linear-head replay were checked before each job.
The independent checkpoint replay verified validation AUROC and test scores.
`summarize.py` rechecks all 24 prediction files, matching test hashes and labels
across objectives, recomputing metrics, and checking head hashes against replay
records. Outputs, weights, selection histories, and cluster logs for the seven
new holdouts are in `objective_outputs_14483084.tar.gz` (SHA-256
`d6f98001d8ee90e7df12d44df148a4ec4c904673b6ac802e3dbe5f40c356478c`);
ADM pilot artifacts are in `../generator_objectives/results_v1`.

Local collection initially hit an unrelated old Windows `tar` executable
failure; Python's archive reader successfully extracted the checksum-verified
archive. No research job was rerun because of this collection issue.

To reproduce the summary after extracting the archive into this directory and
retaining the adjacent ADM pilot results, from the project workspace:

```powershell
cd "C:\Users\creat\OneDrive - University of Southern California\Documents\ChatGPT\CSC566 Proj"
& "C:\Users\creat\miniconda3\envs\test\python.exe" deepfake_transfer\generator_objectives_extended\summarize.py
```
