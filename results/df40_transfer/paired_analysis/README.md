# Paired CNN–ViT error analysis

The analysis pairs predictions for exactly the same image SHA-256 and the
same benchmark mode (baseline with baseline, augmented with augmented).
CNN and ViT each use their original checkpoint and preprocessing; predictions
use the fixed 0.5 fake-probability threshold. The comparison covers one
matched ADM holdout, k=7, rep=0, seed=17 configuration, not all 192 pairs.
No retraining, score-based sample removal, threshold fitting, or ensemble was
performed. `pairs.csv` records the per-image paired outcomes; `summary.csv`
records the counts below. Re-run with `python paired_analysis/analyze.py`
from the `deepfake_transfer` directory.

| Mode | Image class | N | Both correct | CNN only correct | ViT only correct | Both wrong |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Real | 100 | 0 | 1 | 10 | 89 |
| Baseline | SimSwap | 200 | 164 | 35 | 1 | 0 |
| Baseline | FOMM | 100 | 99 | 1 | 0 | 0 |
| Baseline | Wav2Lip | 100 | 87 | 13 | 0 | 0 |
| Augmented | Real | 100 | 11 | 32 | 6 | 51 |
| Augmented | SimSwap | 200 | 101 | 28 | 50 | 21 |
| Augmented | FOMM | 100 | 51 | 2 | 43 | 4 |
| Augmented | Wav2Lip | 100 | 53 | 15 | 25 | 7 |

Both baseline models misclassified 89 of 100 real images; both augmented
models misclassified 51 of the same 100. On fake images, the near-universal
positive predictions by the baseline CNN explain much of the apparent
baseline agreement on correct fake labels; this alone does not establish a
useful detector. In the augmented comparison, ViT alone correctly classified
43 FOMM images that CNN missed, while CNN alone correctly classified 2.
For augmented real images, CNN alone correctly classified 32 while ViT alone
correctly classified 6. The mixed direction shows method and class matter.

These are descriptive sample counts, not evidence that one architecture is
generally superior. The 100 real images are reused across methods; SimSwap
has 200 fake images, and identities or source videos can overlap across
methods. The images used in operational smoke checks are included. This
analysis does not establish independent video-level uncertainty or whether
the manipulations succeeded in each frame.
