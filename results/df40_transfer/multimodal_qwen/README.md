# Open vision-language comparison

This is a small, no-paid-API comparison using
[Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
at revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, evaluated with frozen
weights on CARC. It is one open model; GPT and Claude were not evaluated.

The fixed manifest contains 32 real and 32 ADM-generated images selected by
SHA-256 rank from the existing ADM k=7 rep0 test set. The fixed prompt asks
whether the image is a real photograph or AI-generated and requires `REAL`
or `FAKE`. The model receives images and prompt without the true labels.
Greedy decoding uses up to eight tokens. Raw outputs are preserved. All 64
responses were valid; no refusal or parse failures occurred. The two model
shards, input manifests and prompt were hash-verified.

| Model | Accuracy | Real false positives | Fake detections |
| --- | ---: | ---: | ---: |
| Qwen2.5-VL-3B zero-shot | 42/64 | 1/32 | 11/32 |
| CNN baseline | 48/64 | 7/32 | 23/32 |
| CNN augmented | 40/64 | 10/32 | 18/32 |
| ViT baseline | 61/64 | 2/32 | 31/32 |
| ViT augmented | 56/64 | 2/32 | 26/32 |

CNN and ViT scores use their existing 0.5 thresholds and paired saved
predictions for identical hashes; Qwen emits class labels, so it has no
comparable continuous score or AUROC here. The model uses its own image
preprocessing, so input pipelines differ. This sample covers one fake
generator, one fixed prompt and one checkpoint configuration, and its small
sample size limits generalization. The original 192-run test results were
already known; treat the comparison as exploratory.

The model, code, manifest, original prompt, all raw responses and acceptance
record are retained. `results_v1/comparison.json` holds the matched metrics.
Run `python deepfake_transfer/multimodal_qwen/summarize.py` from the workspace
with pandas/scikit-learn installed and the audited CNN/ViT exports in place.
