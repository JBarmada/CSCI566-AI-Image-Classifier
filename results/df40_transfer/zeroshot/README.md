# CLIP text-prompt zero-shot comparison

CARC job 14481981 completed successfully on a P100 using the saved OpenAI CLIP
ViT-L/14 backbone. Before the run, `prompts.json` fixed the two prompts to
`a photo of a real face` and `a photo of a manipulated face`. The score is
the two-prompt softmax of learned-scale cosine similarities. Its 0.5 threshold
chooses the more similar prompt; the resulting value is not a calibrated
probability of manipulation. No head training, prompt tuning or threshold
fitting was performed. All 500 image hashes were checked before scoring and
the backbone hash was checked before and after.

| Method | Real | Fake | AUROC | Real false-positive rate | Fake recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| SimSwap | 100 | 200 | 0.6316 | 24/100 | 86/200 |
| FOMM | 100 | 100 | 0.7459 | 24/100 | 60/100 |
| Wav2Lip | 100 | 100 | 0.5520 | 24/100 | 27/100 |

These exploratory results use the already inspected 500-image sample and one
prompt pair. The same real images are reused across methods, and the source
and identity caveats in `../shared_results_v1/README.md` apply. Prompt results
may vary with wording and are not sufficient evidence for deployment.
`ACCEPTANCE.json` records hashes and environment; `predictions.csv` contains
individual scores, and `metrics.csv` contains confusion counts and metrics.
