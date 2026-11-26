# Current LLM evaluation workflow

Updated November 26, 2026. This page distinguishes completed predictions from
code that is ready to run. It supersedes older pending-work statements.

## What is complete

- GenImage: Gemini and Groq-hosted Qwen each answered all 90 frozen pilot
  images. The 160 holdout evaluation rows reuse 10 real images eight times.
  The independent summary hashes match the saved inputs. See
  [the pilot guide](../results/llm_eval/README.md).
- CNN/ViT: clean and robustness benchmarks already have accepted results.
  Do not retrain them for the API comparisons.
- OpenAI faces: the cleaned runner passed a fresh two-image CARC smoke and
  completed all 189 frozen images on November 26. Both the summary audit and
  six-model matched comparison passed (1,134 prediction rows). OpenAI accuracy
  was 33.86%, balanced accuracy 50.37%, and AUROC 0.5783. Only the OpenAI key was
  used; no Gemini or Claude requests were made for this face run. The earlier
  smoke remains separate operational evidence. The
  [downloaded exports and analysis](../results/df40_transfer/openai_face/df40_five_method_v3_v2/README.md)
  are stored locally; offline regeneration reproduced the saved CSVs exactly.

## What to finish

1. Complete the existing GenImage pilot's matched CNN/ViT comparison. This
   needs saved predictions, not additional API calls.
2. Review the updated report draft and OpenAI face analysis as a team, then
   integrate the selected findings into the final poster and submitted report.
   Keep API keys out of Git.

The completed face exports remain on CARC under
`/project2/yzhao010_1531/csci566-genimage/results/df40_transfer/openai_face/df40_five_method_v3_v2/`.
The cleaned code was deployed separately at `openai_face_code_v2_20261126/`
without replacing existing project files. See the
[OpenAI guide](../results/df40_transfer/openai_face/README.md).

The 150-image DeepFakeFace evaluation remains optional, with image access to
be resolved separately. The selected 189-image set is an exploratory five-method
face test, not an identity-independent external test. Its 54 real images are
reused in the per-method comparisons. The face-adapted ViT is a separate
supervised reference, not a zero-shot model.

## Maintained entry points

- Face API inference: `openai_face/evaluate.py` and `gemini_face/evaluate.py`
  under `results/df40_transfer/`. Shared checks are in `face_eval_common.py`.
- Face summary and comparison: `gemini_face/summarize.py` and
  `gemini_face/compare_df40_five_method.py`. Both support OpenAI outputs;
  always pass the OpenAI protocol and provider label as documented.
- GenImage analysis: `results/llm_eval/compare_matched.py` and
  `results/llm_eval/summarize_gemini.py` reuse the completed pilot predictions.

`run_pilot.py`, `run_full.py`, and `evaluate_gemini.py` under `results/llm_eval/`
are historical/reproduction entry points, not steps in the face workflow.
`run_full.py` delegates to the pilot implementation instead of maintaining
a duplicate copy. The incomplete full prediction files remain in place for
provenance: Gemini has 95 rows, Groq has 91 rows including one failure.
Do not include them as completed benchmark results.

## Resume, audits and preservation

New face runs write `run.json` with the complete protocol, original manifest
hash and exact selected hashes. A nonblocking `RUN.lock` prevents concurrent
writers. Resuming with changed settings, membership or an old unversioned output
folder fails before API requests. Use a new output folder; do not delete old
results to bypass that check. Never expand a smoke folder into a full run.

Completed rows must match their labels, methods, paths, model, prompt hash,
protocol hash, raw response and frozen threshold. Audits reject duplicates,
missing/failed predictions and invalid scores. OpenAI logs usage and response
IDs in `responses.jsonl`; the summary totals recorded token usage, including
earlier failed response attempts. Billing remains the source for dollar costs.
Permanent API errors stop immediately; transient errors receive bounded retries,
then stop the run for inspection. Existing successful rows are retained.

Historical GPU discrepancy outputs and old plans remain useful evidence.
`DF40_RUNTIME_STATUS.json` is explicitly marked historical; current face data
limitations are in the [accepted candidate review](../results/df40_transfer/new_data/FRESH_CANDIDATE_REVIEW.md).

## Local verification without API access

```bash
python3 -B -m unittest discover -s tests -p 'test_*face_evaluation.py'
python3 -B -m unittest discover -s tests -p 'test_matched_llm_comparison.py'
```

These use saved manifests and mocked API responses. They need no API key or
GPU. The broader model tests additionally need the scientific Python environment.
