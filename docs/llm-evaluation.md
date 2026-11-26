# Current LLM evaluation workflow

Updated November 26, 2026. This page distinguishes completed predictions from
code that is ready to run. It supersedes older pending-work statements. The
[consolidation plan](results-consolidation.md) is the source inventory and
integration checklist; availability on a teammate branch is recorded separately
from integration and independent acceptance.

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
- Claude faces: the same 189 frozen sources completed on November 26 after
  full image validation and a separate two-image smoke. Both summary and
  six-model CNN/ViT comparison audits passed. Accuracy 70.37%, balanced accuracy
  55.93%, AUROC 0.6198; fake recall 89.63%, real specificity 22.22%.
  See the [Claude face guide](../results/df40_transfer/claude_face/README.md).
  Only the Anthropic key was used for this new run; OpenAI outputs are preserved.
  The original Claude face exports are also downloaded locally. Independent
  offline regeneration reproduced summary and matched CNN/ViT CSVs byte for
  byte; the response journal, source hashes and frozen membership were checked.
  See the [run notes](../results/df40_transfer/claude_face/df40_five_method_v3_v1/README.md).

## Integrated team experiments and original sources

- **Claude GenImage:** `e319d94`, included in `zero-shot-llm-evaluation`
  (`3d73b5b`) and the GPT GenImage branch. Three views of the same 630 sources
  have 630/630 successful scores each. All 90 pilot hashes are included.
  CSV-level membership/raw-score/overall-metric checks passed during branch
  review; the files are integrated, while complete image/protocol verification
  remains a separate acceptance question.
- **GPT GenImage:** `7be92ec` on `zero-shot/gpt-genimage` now includes code
  under `code/llm-eval/genimage/openai/` and saved results under
  `results/llm_eval/openai/`. The recorded model is `gpt-6.2-sol`, separate
  from our face model. There are 1,890 final rows across the same three
  630-image views: 1,844 valid scores, 46 invalid outputs, no missing final
  responses and no refusals in the saved tables. Code/results are available;
  the files are integrated, while attempt-level independent acceptance remains
  pending. This supersedes the
  earlier finding that the GPT-named branch had no implementation.
- **ViT calibration/errors:** `ebeba98` on `analysis/vit-confidence-errors`
  provides extended calibration, validation-fitted recalibration/thresholds,
  persistent-error analysis, figures and tests. It uses saved scores and
  needs no retraining. It is integrated and was independently regenerated;
  raw calibration/error summaries matched, with small fitted-parameter
  numerical differences documented in the project review.

The [plan](results-consolidation.md) links the pinned source commits and the
[project review](project-review-2026-11-26.md) records integration tests and
remaining scientific limitations. Original outputs are preserved.

## What to finish

1. Complete the GenImage matched comparison with the saved CNN export.
   Gemini/Qwen have 90 pilot images; Claude/GPT have a 630-image superset.
   Retain views and coverage separately. The integration corrected the earlier
   88-versus-90 pilot denominators and old ADM-only ViT discovery; use the
   [corrected comparisons](../results/llm_eval/openai/integration_review/README.md).
2. Obtain the GPT original request-attempt log from its owner for provenance
   verification. It is referenced by the results summary but not tracked in
   commit `7be92ec`; preserve invalid outputs rather than calling the API again.
3. Regenerate one consolidated analysis with separate GenImage and face sections.
   This needs saved predictions and offline checks, not additional API calls.
4. Review the updated report draft and OpenAI face analysis as a team, then
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

- Face API inference: `openai_face/evaluate.py`, `claude_face/evaluate.py` and `gemini_face/evaluate.py`
  under `results/df40_transfer/`. Shared checks are in `face_eval_common.py`.
- Face summary and comparison: `gemini_face/summarize.py` and
  `gemini_face/compare_df40_five_method.py`. Both support OpenAI and Claude outputs;
  always pass the matching provider protocol and label as documented.
- GenImage analysis: `results/llm_eval/compare_matched.py` and
  `results/llm_eval/summarize_gemini.py` reuse the completed pilot predictions.
  The integrated Claude/GPT entry points are inventoried in the consolidation
  plan; use saved scores rather than starting fresh inference.

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
