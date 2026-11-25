# CPU reference replay

At Jad's request, CARC job 14549744 replayed existing saved checkpoints on
CPU without training. The V100 face evaluation job 14529608 later failed its
CNN reference gate; a P100 retry passed. The CPU reference run completed, but **all four model variants
exceeded their original maximum-score replay tolerance**:

| Model | Reference images | Largest score difference | Original tolerance | Decision changes at 0.5 |
| --- | ---: | ---: | ---: | ---: |
| CNN baseline | 64 | 0.005778 | 0.001 | 0 |
| CNN augmented | 64 | 0.003015 | 0.001 | 0 |
| ViT baseline | 16 | 0.0006933 | 0.00001 | 0 |
| ViT augmented | 16 | 0.0004745 | 0.00001 | 0 |

The CNN CPU run uses full-precision CPU operations; the original GPU replay
uses mixed precision. The CLIP encoder also runs on CPU. The comparison does
not isolate the source of each numerical difference. The zero decision changes
apply only to these small reference subsets and threshold 0.5, not the full
face evaluation or other thresholds. Do not claim CPU equivalence under the
original acceptance gate. See [REFERENCE.json](REFERENCE.json),
[per-image scores](reference_predictions.csv), and
[replay script](../cpu_reference_replay.py).

An additional inference-only CPU run on the frozen 189-image candidate, job
14550254, completed and produced 945 scores across five model variants. Its
[results](../cpu_candidate_results_v1/) are **preliminary**, because the
original replay gate failed. See the completed [CPU/P100 comparison](../fresh_candidate_results_v3/README.md).
Do not change the test manifest,
select thresholds on the candidate, or represent CPU-only scores as the
previously validated GPU results.
