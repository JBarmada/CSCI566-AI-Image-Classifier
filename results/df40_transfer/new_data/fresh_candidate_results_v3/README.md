# Five-method exploratory face comparison

CARC P100 job 14742998 completed inference on the frozen 189-image candidate.
It passed the original CNN and ViT reference checks, repeat checks and model
hash checks; see [ACCEPTANCE.json](ACCEPTANCE.json). The prior V100 job
14529608 stopped before scoring when the CNN baseline reference differed by
0.002441, above the unchanged 0.001 tolerance. The P100 retry used the same
GPU type as the earlier successful face evaluation; its CNN reference scores
matched exactly. This observes a device-dependent mismatch, not its cause.

The independent row audit checked all 945 CPU and GPU predictions against the
189 manifest hashes, labels and methods. At threshold 0.5, CPU and P100 made
the same decision on 944/945 scores; one augmented CNN decision differed. At
the previously selected validation thresholds, nine baseline CNN decisions
differed; the other four variants had no differences. The baseline CNN
validation threshold is 0.9990234375000001, close to the top of the score
range. See [per-model comparison](cpu_gpu_comparison.csv) and
[comparison script](../compare_cpu_gpu_candidate.py). Numerical tolerance
still failed for the CPU reference replay, so distinguish score reproducibility
from agreement at a specific threshold.

## P100 results at threshold 0.5

| Model | Real falsely flagged /54 | SimSwap /27 | FOMM /27 | Wav2Lip /27 | InSwapper /27 | PIRender /27 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CNN baseline | 53 | 26 | 27 | 27 | 26 | 27 |
| CNN augmented | 31 | 15 | 24 | 18 | 15 | 20 |
| ViT baseline | 42 | 17 | 25 | 19 | 19 | 26 |
| ViT augmented | 39 | 16 | 21 | 18 | 18 | 23 |
| Face-adapted ViT head | 20 | 20 | 26 | 14 | 17 | 21 |

Fake columns count images detected as fake. The 54 real crops are reused for
every method; the 27 fake crops per method come from the same 27 recorded
source-video-ID pairs. [metrics.csv](metrics.csv) also reports the frozen
validation-selected thresholds and AUROC; [predictions.csv](predictions.csv)
contains per-image scores. Recompute with
`python results/df40_transfer/new_data/summarize_fresh_candidate.py <new-output-directory>`
after copying the acceptance and prediction files to a fresh directory.

**Exploratory limits:** This is an upstream FaceForensics++ training-split
subset, not an external or verified person-independent test. Source identity,
Wav2Lip source roles, individual manipulation success and third-party mirror
method labels remain uncertain. Three pairs were removed score-blind; Jad's
review of two exclusions remains open. The adapted head trained on SimSwap,
FOMM and Wav2Lip, and its previously inspected test informed prior work.
Do not treat the results as causal evidence that adaptation reliably improves
unseen-method performance. See the [protocol](../FRESH_EVALUATION_PROTOCOL.md)
and [data review](../FRESH_CANDIDATE_REVIEW.md).
