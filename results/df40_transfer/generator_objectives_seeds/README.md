# Three-seed exploratory objective comparison

We repeated the pre-existing frozen-CLIP feature-adapter comparison with
optimization seeds 29 and 43, following the same fixed settings as seed 17.
These runs were planned after inspecting the seed-17 results. They use the
same k=7 repeat-0 training, validation and test manifests for all three seeds.
The encoder is frozen; the small adapter is trained. The seed changes adapter
initialization and training order, not the dataset split. All test sets were
previously examined, so the comparison is exploratory.

| Objective | Mean accuracy | Mean AUROC | Holdouts with better / worse accuracy than binary |
|---|---:|---:|---:|
| Binary adapter | 0.9162 | 0.9742 | Reference |
| Generator-adversarial | 0.9185 | 0.9741 | 5 / 3 |
| Supervised contrastive | 0.9150 | 0.9742 | 4 / 4 |

First average the three optimization seeds within each of the eight holdouts,
then weight the eight holdouts equally. The adversarial accuracy difference
versus baseline is about 0.22 percentage points; AUROC is slightly lower. This
is not evidence of a reliable improvement. There are only eight holdouts, and
seed runs and holdouts reuse test images. `mean_by_holdout.csv` exposes the
variation and false-positive/fake-detection tradeoffs. For example, the
adversarial mean accuracy is lower on VQDM and higher on SD 1.5; we have not
identified a causal reason. A fresh test and more dataset repeats are needed.

All 16 new CARC jobs in array 14483127 completed with exit code 0 and each
independently replayed its selected heads. Before training, every job verified
feature cache identity, labels, checkpoint hash and the original linear-head
validation/test scores. The summary audit checks per-seed manifest hashes,
matched test image hashes/labels across objectives, recomputes all 72 test
metrics from predictions, and verifies selected checkpoint hashes and replay
records. `per_holdout_seed.csv` contains all objective/holdout/seed rows.
Detailed new outputs, weights and job logs are in
`objective_seed_outputs_14483127.tar.gz`, SHA-256
`0b23a1802791cab01c8969cc3a21f1648b931c0968c773a95b1a7b76704d78b9`.
The seed-17 runs and their artifacts are in the neighboring
`generator_objectives` and `generator_objectives_extended` directories.

To reproduce the summary, extract both output archives into their respective
folders and retain the ADM pilot directory. Then from the original workspace:

```powershell
cd "C:\Users\creat\OneDrive - University of Southern California\Documents\ChatGPT\CSC566 Proj"
& "C:\Users\creat\miniconda3\envs\test\python.exe" deepfake_transfer\generator_objectives_seeds\summarize.py
```

Adapt the working path and Python environment for another computer. See
`PROTOCOL.md` and the earlier comparison protocols for the full settings.
