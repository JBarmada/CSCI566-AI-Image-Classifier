# Matched zero-shot GenImage pilot

This folder contains a completed, quota-safe frontier-model pilot built from the eight corrected `k=7`, repeat-0 GenImage holdouts. The committed manifests freeze 10 shared real images and 10 fake images per holdout: 90 unique API images and 160 matched evaluation rows.

## Completed API evaluation

The pilot successfully evaluated all 90 images with:

- Google `gemini-3.1-flash-lite`
- Groq-hosted `qwen/qwen3.8-27b` (this is Qwen served by Groq, not xAI Grok)

`pilot_overall_metrics.csv` and `pilot_per_generator_metrics.csv` contain the results. The attempted 2,520-image runs in `*_full_predictions.csv` are incomplete and must not be reported as full-benchmark results.

`protocol.json` records the prompt and generation settings used by `run_pilot.py`. API inference does not need to be repeated for the matched comparison.

## New team outputs awaiting integration

The [consolidation plan](../../docs/results-consolidation.md) records the
source commits and outstanding checks. These outputs are on other branches,
not yet present in this working tree:

- Claude: `e319d94` adds `results/llm_eval/claude/`, with the fixed 630-image
  shared list, neutral-ID preparation, isolated API runner, raw scored
  responses and metrics for original, square and detector224 views. Each
  view has 630 successful responses; the set includes all 90 pilot hashes.
- GPT: `7be92ec` on `zero-shot/gpt-genimage` adds
  `code/llm-eval/genimage/openai/` and `results/llm_eval/openai/`. The
  `gpt-6.2-sol` run has 1,844 valid scores and 46 invalid outputs across
  1,890 final image-view rows. This branch already includes the Claude work.

Use the 90-image intersection for comparisons with Gemini/Qwen, and report the
630-image Claude/GPT study separately. Match both sides on the same valid
source hashes and view. GPT's current pilot comparison mixes 88 valid GPT
images with 90-image baseline metrics; recomputation is required. Its saved
ViT table has only 1–4 all-real matches per old ADM export. Use corrected
ViT exports and obtain matching CNN predictions from CARC before a substantive
detector comparison. The GPT request-attempt log is not tracked at `7be92ec`
and must be obtained for its provenance audit. Do not rerun paid inference
to perform this consolidation.

The separate [OpenAI face evaluation](../df40_transfer/openai_face/df40_five_method_v3_v2/README.md)
uses 189 manipulated/real face images and model `gpt-6.1-terra`; its results
must not be pooled with this synthetic-image task.

## Audit the completed Gemini pilot

The independent audit checks exact manifest membership and duplicate successful
hashes, records input checksums, and recomputes metrics using the manifest labels:

```bash
python3 results/llm_eval/summarize_gemini.py
python3 results/llm_eval/summarize_gemini.py \
  --predictions results/llm_eval/groq_predictions.csv \
  --output-dir results/llm_eval/groq_pilot/summary
```

The original summary weights each holdout equally and therefore repeats the 10 shared real images eight times. The independent summaries also report the 90-unique-image view: Gemini accuracy is 71.11% and Groq-Qwen accuracy is 65.56%. Their holdout-weighted accuracies remain 83.75% and 76.25%, respectively.

The original API runners are retained for provenance and reproduction, not as
steps to repeat for the completed pilot. `run_full.py` now delegates to
`run_pilot.py`; `evaluate_gemini.py` is an alternative historical runner.
The partial full exports contain 95 Gemini rows and 91 Groq rows (one error).
They stay in place to preserve their original paths, but are not final results.
Use the [current workflow](../../docs/llm-evaluation.md) for the face evaluation.

## Compare Gemini, Groq-Qwen, CNN, and ViT

The matched comparison uses the committed API predictions, the audited ViT prediction exports, and the existing CNN run outputs. No retraining and no API calls are required.

From the CARC project root:

```bash
python3 results/llm_eval/compare_matched.py
```

The default CNN location is the absolute CARC path
`/project2/yzhao010_1531/csci566-genimage/results/cnn_benchmark/notebook/runs`.
Run from the repository root so the frozen pilot and published ViT exports
resolve. Elsewhere, pass `--root`, `--vit-predictions`, and `--cnn-runs` explicitly.
Only clean ViT test rows are selected; duplicate hashes and invalid scores fail.

Outputs are written to `results/llm_eval/matched_comparison/`:

- `matched_predictions.csv`: all six model/configuration scores on 160 matched rows
- `overall_metrics.csv`: matched overall metrics
- `per_holdout_metrics.csv`: generator-level metrics
- `AUDIT.json`: manifest and API-prediction hashes

Because the 10 real images occur in every holdout, the 160 rows are useful for an equally weighted holdout comparison but are not 160 independent images. The pilot contains 90 unique images.

## Tests

```bash
python3 -m unittest tests/test_matched_llm_comparison.py
```
