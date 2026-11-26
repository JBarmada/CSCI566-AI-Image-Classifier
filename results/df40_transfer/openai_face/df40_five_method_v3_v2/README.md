# Matched OpenAI–CNN–ViT face evaluation

Completed November 26, 2026 on CARC. All 189 OpenAI predictions and the
six-model comparison passed their audits. The downloaded CSV files were
regenerated locally from the saved predictions and reproduced byte for byte.
This is an exploratory comparison, not a validated external deepfake benchmark.

## What was evaluated

- The frozen [v3 manifest](../../new_data/fresh_candidate_review_v3/manifest.csv):
  54 real face crops and 27 fake crops for each of SimSwap, FOMM, Wav2Lip,
  InSwapper (`inswap`) and PIRender, linked to 27 recorded video-ID pairs.
- OpenAI model `gpt-6.1-terra`, one image per request, one fixed zero-shot
  face-manipulation prompt, high image detail, low reasoning effort and a
  threshold of 50/100. The exact settings are preserved in [run.json](run.json).
  No model training, prompt tuning or threshold tuning was performed for this run.
- The same image hashes and labels as the accepted P100 CNN/ViT predictions.
  The face-adapted ViT is a separate supervised reference; it trained on three
  evaluated manipulation methods and must not be described as zero-shot.
- Only `OPENAI_API_KEY` was used for this run. Gemini/Claude were not called.
  The `gemini_face/` summary and comparison scripts perform offline analysis
  and accept an explicit OpenAI protocol/provider label.

## Overall results at the frozen threshold

Source: [comparison/metrics.csv](comparison/metrics.csv). All entries use the
same 189 images. Fake means the manifest's manipulation label.

| Model/configuration | Accuracy | Balanced accuracy | AUROC | Real falsely flagged /54 | Fakes detected /135 |
| --- | ---: | ---: | ---: | ---: | ---: |
| OpenAI, zero-shot | 33.86% | 50.37% | 0.5783 | 6 | 16 |
| CNN, baseline | 70.90% | 50.19% | 0.5690 | 53 | 133 |
| CNN, augmented | 60.85% | 55.37% | 0.5227 | 31 | 92 |
| ViT, baseline | 62.43% | 50.37% | 0.4930 | 42 | 106 |
| ViT, augmented | 58.73% | 49.44% | 0.5221 | 39 | 96 |
| ViT, face-adapted (supervised reference) | 69.84% | 67.78% | 0.7583 | 20 | 98 |

OpenAI correctly accepted 48/54 real crops but missed 119/135 fakes. Its real
false-positive rate was 11.11% and fake recall was 11.85%. The baseline CNN
showed the opposite tendency: it detected 133/135 fakes while falsely flagging
53/54 reals. The baseline ViT detected 106/135 fakes and falsely flagged 42/54
reals. These different error patterns explain why raw accuracy alone is a poor
summary here.

The set contains 135 fakes and 54 reals. Always predicting fake would achieve
71.43% accuracy and 50% balanced accuracy; always predicting real would achieve
28.57% accuracy and 50% balanced accuracy. OpenAI's 50.37% balanced accuracy is
near chance at this operating point, while AUROC 0.5783 describes its score
ordering across thresholds. Neither figure establishes useful general-purpose
detection or a statistically reliable difference between these models.

## Which manipulations OpenAI missed

Source: [summary/metrics.csv](summary/metrics.csv). Each method uses 27 fake
crops and the same shared 54 real crops; do not add the real counts across rows.

| Method | Detected /27 | Missed /27 | AUROC versus the shared real set |
| --- | ---: | ---: | ---: |
| SimSwap | 6 | 21 | 0.5844 |
| FOMM | 3 | 24 | 0.7003 |
| Wav2Lip | 3 | 24 | 0.5257 |
| InSwapper | 3 | 24 | 0.5340 |
| PIRender | 1 | 26 | 0.5470 |

FOMM illustrates the difference between ranking and the fixed decision rule:
its AUROC was 0.7003, but only 3/27 fakes crossed the frozen threshold. These
results do not justify selecting a new threshold using this test set. A future
threshold study needs separate validation images and a rule frozen before test.

## Evidence and reproduction

- [predictions.csv](predictions.csv): one final scored row per image, including
  model, prompt/protocol hashes, raw response and recorded usage.
- [responses.jsonl](responses.jsonl): append-only event log, response IDs and
  token usage; no image bytes or API key are included.
- [run.json](run.json): exact protocol, manifest hash and ordered image hashes.
- [summary/AUDIT.json](summary/AUDIT.json): 189/189 successful predictions,
  no failures, input hashes and token totals.
- [comparison/AUDIT.json](comparison/AUDIT.json): six model/configuration
  entries, 1,134 matched prediction rows and output/source checksums.
- [comparison/matched_predictions.csv](comparison/matched_predictions.csv):
  the matched image-level scores for all six entries.

The full run recorded 189 completed responses, each on its first attempt,
42,336 input tokens and 10,827 output tokens. The separate two-image smoke is
not included in those totals. Dollar billing is not inferred from token counts.
Use the [OpenAI guide](../README.md) to reproduce the offline summaries; no
further inference or retraining is necessary.

## Limits on interpretation

This is an upstream FaceForensics++ training-split subset with third-party
mirror provenance, not a new external dataset or verified identity-independent
test. Three pairs were excluded before scoring because of duplicate bytes or
questionable face correspondence. Source roles, individual manipulation success
and some method labels remain unverified; review of two exclusions is open.
Passing the result audit verifies saved membership, scores and settings, not
those unresolved dataset properties. See the
[selection review](../../new_data/FRESH_CANDIDATE_REVIEW.md).

Images share video-source groups, so 189 crops are not 189 independent people.
The models also differ in pretraining, training, preprocessing and decision-score
meaning. OpenAI's prompted manipulation score is not a calibrated probability.
This single model/prompt run cannot establish a general ranking of LLMs versus
specialized detectors. No significance or causal claims are made.
