# OpenAI GPT zero-shot GenImage evaluation

The recorded `gpt-6.2-sol` run is available in the
[results summary](../../../../results/llm_eval/openai/RESULTS_SUMMARY.md).
During integration, the prompt test was aligned with the canonical
`run_pilot.py`, pilot performance metrics were restricted to common valid
images, and ViT discovery was changed to corrected k=7/repeat-0 clean exports.
See the [offline integration review](../../../../results/llm_eval/openai/integration_review/README.md).
Original exports are preserved; no API requests are needed for these corrections.

Scripts are kept in `code/llm-eval/genimage/openai/`; generated inputs, logs,
and reports are kept in `results/llm_eval/openai/`. This follows the selected
workspace location while keeping experiment outputs in the requested results
tree.

This experiment uses the exact 630 unique images in
`results/llm_eval/claude/shared_image_list.csv`: 70 real images and 70 fake images from each
of eight generators. It does not use or resample the separate 2,520-image
manifest. All 90 unique Gemini/Qwen pilot images are in the shared list.
The evaluation is 630 images × 3 views = **1,890 image evaluations**, before
retries.

No GPT model is selected in this repository. The runner requires an explicit
model ID for API execution; choose one and confirm its supported request
parameters before any paid call. It uses the official OpenAI Python SDK and
Responses API, not ChatGPT browser automation.

## Preparation and input privacy

The original Claude blind PNG folders are not saved in this repository, and the
GenImage source tree is not included in this local checkout. Stage the exact files listed in Claude's `carc_paths.txt` (the same download
described in [`results/llm_eval/claude/README.md`](../../../../results/llm_eval/claude/README.md)),
then run from the repository root:

```bash
mkdir -p ~/genimage_subset
rsync -av --files-from=results/llm_eval/claude/carc_paths.txt \
  NETID@discovery.usc.edu:/project2/yzhao010_1531/csci566-genimage/data/extracted/GenImage_Compact/ \
  ~/genimage_subset/

python code/llm-eval/genimage/openai/prepare_views.py \
  --image-root ~/genimage_subset
```

`prepare_views.py` reads Claude's existing shared list and answer keys, keeps
Claude's exact `blind_id` order, and reproduces `blind_images.py`:

- `original`: source decoded and converted to RGB; original dimensions.
- `square`: centered square crop at original resolution.
- `detector224`: BICUBIC resize of the short side to 224, then the same centered
  crop as Claude's script.
- Every submitted artifact is a metadata-free RGB PNG with a neutral `img_XXXX`
  filename. The answer keys, labels, generator names, original paths, and
  source filenames are never included in the API request.

Preparation verifies each source-file SHA-256 against the shared manifest,
checks the exact decoded output pixels and dimensions against Claude's
answer-key manifests, verifies same blind-ID/source correspondence across
views, checks client-decoded RGB equality against the pilot manifest for the
90-image intersection, and records the source byte hash, source RGB pixel hash,
submitted PNG byte hash, and submitted RGB pixel hash. The original-view
round-trip additionally requires pixel equality, verifying that removing
metadata did not change decoded RGB pixels. No model is called during
preparation.

The API request contains only the exact prompt and a base64 data URL for one
PNG. There is one independent request per image/view: no prior response,
conversation history, tools, web search, demonstrations, or extra metadata.
`store=False` is explicit. The runner does not read `.env`; set the key in the
process environment only.

## Model, request configuration, and prompt

Official documentation checked on 2026-11-26:

- [OpenAI image input and detail levels](https://developers.openai.com/api/docs/guides/images-vision)
- [OpenAI model catalog](https://developers.openai.com/api/docs/models)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)

The installed `openai` Python package in the execution environment was version
2.3.0; it exposes `OpenAI.responses.create`. The model catalog documents
image-input-capable models and the Responses API. Its available model IDs and
parameters can change; verify the exact selected model's page before running.

Defaults recorded in each request configuration:

- image detail: `auto` (override with `--detail`; supported choices depend on
  the model);
- `max_output_tokens=128`;
- temperature omitted until support is checked for the exact selected model.
  Add `--temperature-zero` only when that model supports `temperature=0`;
- no structured-output constraint is added, so the prompt itself remains
  unchanged;
- requested and returned model IDs, image detail, generation settings,
  configuration hash, prompt hash, timestamps, request/response IDs, raw
  response, parse status, errors, retry history, token usage, and available
  cost data are retained in JSONL.

The prompt is byte-for-byte identical to `PROMPT` in
[`results/llm_eval/run_full.py`](../../../../results/llm_eval/run_full.py):

```text
Analyze the provided image and estimate the probability that it was
generated by an AI system rather than captured by a real camera.

Return ONLY a JSON object in this format:
{"p_fake": 0}

p_fake must be a number from 0 to 100:
0 means certainly real; 100 means certainly AI-generated.

Do not assume that unusual visual characteristics necessarily
indicate AI generation.
```

`p_fake` must parse as a finite JSON number in `[0,100]`. Values `>= 50` are
classified fake. It is a **model-reported score**, not a calibrated
probability. Refusals, invalid outputs, API failures, and missing results are
kept distinct; valid-score metrics never silently discard their coverage
denominators.

## Dry run, smoke run, full run, and resume

Review the expected 1,890 request plan without a key or API call. The JSON
summary is saved at `results/llm_eval/openai/manifests/dry_run_summary.json`:

```bash
python code/llm-eval/genimage/openai/run_openai.py --dry-run
```

Choose and verify a model, then run a limited smoke test **only after approving
the estimated cost and paid request**:

```bash
read -s OPENAI_API_KEY && export OPENAI_API_KEY   # Hidden input; never put the value in a command or file.
python code/llm-eval/genimage/openai/run_openai.py \
  --model MODEL_ID --detail auto --limit 3 --execute
```

Check `results/llm_eval/openai/logs/requests_<model>_<config-hash>.jsonl`,
then score and compare:

```bash
python code/llm-eval/genimage/openai/score_openai.py \
  --log results/llm_eval/openai/logs/REQUEST_LOG.jsonl \
  --model MODEL_ID
python code/llm-eval/genimage/openai/compare_models.py \
  --log results/llm_eval/openai/logs/REQUEST_LOG.jsonl \
  --model MODEL_ID
```

After confirming the smoke run and approving the full run:

```bash
python code/llm-eval/genimage/openai/run_openai.py \
  --model MODEL_ID --detail auto --execute
```

Re-run the same command with exactly the same model, detail, and generation
configuration to resume. The default log name includes the model and
configuration hash. Completed successes, refusals, and invalid outputs are
not called again. A configuration change makes a different request key and
cannot reuse those predictions. Transient API errors have at most four
attempts per invocation (three retries with bounded exponential backoff);
every attempt is fsync-checkpointed to JSONL. Each API request has a 120-second
timeout so a stalled request is recorded as a retryable failure rather than
blocking the run indefinitely. Re-running resumes failed or missing keys. A
custom `--output` may be used, but retain one output file per configuration
for simpler auditing.

To run one view at a time, add `--view original`, `--view square`, or
`--view detector224`. Omitting `--view` plans/runs all three views.

The dry-run mode reports pending request counts and performs no API request.
`--execute` is mandatory for paid calls. Do not pass `--temperature-zero`
unless it is documented as supported by the selected model. Optionally supply
both `--input-price-per-million` and `--output-price-per-million` to record
per-response estimated costs from API token usage; the runner also preserves
provider-reported cost if present.

## Local checks and cost estimate

Run local validation before inference:

```bash
cd code/llm-eval/genimage/openai
python -m unittest -v test_openai_eval.py
python estimate_cost.py
python run_openai.py --dry-run
```

The illustrative cost report is written to
`results/llm_eval/openai/manifests/`. The estimator uses
the saved Claude dimensions and the currently documented GPT-5.5 high-detail
patch/multiplier rules and standard rates as one **example only**; it does not
select GPT-5.5. It assumes about 100 text-input tokens and a 16-token typical
output per request, and also gives a conservative upper output-token bound of
128 per request. Image token counts are estimated from the published resizing
rules. Actual cost depends on the selected model, detail support, model/API
pricing tier, output/reasoning tokens, and any pricing changes. Recompute and
present the exact model's estimate before approving any paid inference.

## Outputs and evaluation

Generated inputs, logs, and metrics stay under
`results/llm_eval/openai/`:

- `manifests/input_manifest.csv`: neutral identifiers, view, dimensions, and
  byte/pixel hashes only; no labels or source paths.
- `logs/*.jsonl`: immutable per-attempt request/response checkpoint records.
- `metrics/openai_predictions.csv`: scored rows joined locally to the answer
  key; label-bearing data is only added after inference.
- `metrics/openai_view_metrics.csv`: each view separately, with AUROC,
  accuracy, precision, recall, specificity, F1, balanced accuracy, confusion
  counts, real false-positive rate, fake miss rate, response coverage,
  valid-score coverage, refusals, invalid outputs, API failures, and missing
  calls. `api_failed_attempts` separately includes failures that were retried
  before a final response.
- `metrics/openai_per_generator_metrics.csv`: fake recall and other metrics
  per generator, paired with the 70 real images for that view. These
  generator-specific rows reuse those real images and are not additive.
- `metrics/openai_shape_baseline_metrics.csv`: Claude's shape-only control,
  recomputed from original-view manifest dimensions, labeling square as fake.
- `metrics/openai_cost_summary.csv`: per-view API-attempt token totals and any
  reported/estimated costs; retries are counted as additional attempts.
- `metrics/openai_vs_claude_matched.csv`: comparisons on identical
  source-image IDs and views.
- `metrics/openai_vs_pilot_models.csv`: GPT versus Gemini and Groq-hosted
  Qwen on the 90-image pilot intersection.
- `metrics/pilot_legacy_task_row_metrics.csv`: the historic 160 task rows;
  it explicitly reports 90 unique images because real images recur between
  generator tasks.
- `metrics/openai_vs_vit_matched.csv`: automatically discovered saved ViT
  clean test predictions matched by exact SHA-256, with experiment, holdout,
  repeat, mode, condition, test split, and saved threshold retained.

All per-view GPT metrics use up to 630 unique source images (70 real, 560 fake,
90% fake). AUROC and thresholded metrics use valid model scores only;
coverage/status counts use all 630 expected image requests. The combined
1,890 image-view evaluations are repeated measurements on the same 630 images,
not 1,890 independent images. Per-generator rows reuse the same 70 real
images. The legacy pilot task-row table is reported separately for precisely
this reason.

Claude's README documents a major shape confound: all fakes are square while
69 of 70 real images are non-square. Original-view performance close to the
shape-only baseline is not evidence of semantic AI-image detection. The square
and detector224 views address that specific shape/resolution shortcut, but
224×224 alone does not ensure identical model inputs. The API may apply its own
resize/patch preprocessing; CNN/ViT normalization, interpolation, and
preprocessing can also differ. Pilot pixel equality checks compare client-side
decoded RGB pixels where the required source files are available; internal
model preprocessing cannot be observed. Comparisons are paired only on the
actual shared hashes, with denominators reported.

The scripts currently live in `code/llm-eval/genimage/openai/`. The repository has saved ViT prediction exports and a provenance
index; the comparison script uses only matching saved clean **test** rows and
their saved `pred_label`/threshold. No matching CNN prediction CSV was found
in the checked local benchmark artifacts, so no CNN result is fabricated.

## Limitations

- GPT's image service can resize/patch images internally; the sent PNG hashes
  and client-side pixel checks do not prove identical model-internal tensors.
- Chat/API outputs may vary over time even with fixed parameters and model
  aliases. Record the exact returned model ID and timestamps.
- OpenAI usage data may omit dollar cost; in that case token-based estimates
  are retained and should be reconciled with the billing dashboard.
- OpenAI API data is not used to train/improve models by default unless an
  organization explicitly opts in. `store=False` disables Responses
  application-state storage; it does not disable default abuse-monitoring
  logging, which can retain content for up to 30 days. See [OpenAI data
  controls](https://developers.openai.com/api/docs/guides/your-data) and check
  the account's data-control settings before submitting images.
- The fixed 50 threshold and prompt are not tuned on these results.
- Failure coverage can affect valid-response metrics; always read the status
  table alongside performance tables.
- This is a fixed curated evaluation sample, not the separate full 2,520-image
  API manifest or a population-calibrated probability study.
