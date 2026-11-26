# OpenAI face-manipulation evaluation

This is the OpenAI Responses API equivalent of the Gemini evaluation. It uses
the same frozen manifests, prompt, 50% threshold, checksum verification,
per-image checkpoints and output columns. Gemini code and outputs are preserved.

On November 26, 2026, the cleaned runner passed a fresh two-image CARC smoke,
then completed all 189 images. The independent summary audit and matched
six-model comparison both passed (1,134 matched prediction rows). Outputs are
on CARC under `/project2/yzhao010_1531/csci566-genimage/results/df40_transfer/openai_face/df40_five_method_v3_v2/`;
the [exports and analysis](df40_five_method_v3_v2/README.md) are also stored in
this repository and were reproduced locally without API calls. OpenAI accuracy was
33.86%, balanced accuracy 50.37%, and AUROC 0.5783 on this exploratory subset.
Only `OPENAI_API_KEY` was used. The full run recorded 189 responses, 42,336 input
tokens and 10,827 output tokens, with no missing usage records.

The deployed code is isolated under the CARC project directory at
`openai_face_code_v2_20261126/`; run the commands below from that directory and
use the absolute output path above to inspect or resume this completed run.
The smoke outputs are in `openai_face/smoke_cleaned_v2_20261126/` under the
shared project's `results/df40_transfer/`. Legacy outputs are preserved and
deliberately cannot be resumed without a frozen `run.json`.
The [current project workflow](../../../docs/llm-evaluation.md) records scope.

The configured model is `gpt-6.1-terra`, selected for visual reasoning. Change
the model only before the first request; an existing run cannot be resumed with
a different model or prompt.

## Setup on CARC

```bash
python -m pip install -r results/df40_transfer/openai_face/requirements-openai.txt
read -s -p "OpenAI API key: " OPENAI_API_KEY; export OPENAI_API_KEY; printf '\n'
```

Never save the API key in the repository, notebook, Slurm file or log. Confirm
that dataset terms and team policy permit sending the face images to OpenAI.

## Validate without API calls

The commands below target the completed CARC run. From the isolated code root,
set its output path once. The full inference command will validate and skip its
189 existing successes; do not choose a fresh folder unless you intend to pay
for a new evaluation.

```bash
FACE_OUTPUT=/project2/yzhao010_1531/csci566-genimage/results/df40_transfer/openai_face/df40_five_method_v3_v2

python results/df40_transfer/openai_face/evaluate.py \
  --dataset df40_five_method_v3 \
  --manifest results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv \
  --data-root /project2/yzhao010_1531/csci566-genimage/data/df40_fresh_candidate_review_v3_20261123 \
  --output-dir "$FACE_OUTPUT" \
  --validate-only
```

## Smoke test

After full image validation, use a separate output folder for two API requests:

```bash
python results/df40_transfer/openai_face/evaluate.py \
  --dataset df40_five_method_v3_smoke \
  --manifest results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv \
  --data-root /project2/yzhao010_1531/csci566-genimage/data/df40_fresh_candidate_review_v3_20261123 \
  --output-dir /project2/yzhao010_1531/csci566-genimage/results/df40_transfer/openai_face/smoke_cleaned_v2_20261126 \
  --limit 2
```

## Full resumable run

```bash
python results/df40_transfer/openai_face/evaluate.py \
  --dataset df40_five_method_v3 \
  --manifest results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv \
  --data-root /project2/yzhao010_1531/csci566-genimage/data/df40_fresh_candidate_review_v3_20261123 \
  --output-dir "$FACE_OUTPUT"
```

Rerunning the same command skips successful images and retries failures. Results
are intentionally separate from Gemini results.

Run these commands from the repo root using the verified CARC Python interpreter
(for example, replace `python` with `"$HOME/cnn-python/bin/python"`). A GPU is not
needed for API inference. Check `Successful predictions: 189/189` before summaries.
The runner freezes the whole protocol and exact manifest selection, checks resumed
scores and prevents concurrent writers. It stops after a failure rather than
continuing to spend requests on a broken configuration. Transient service errors
are retried; bad credentials, exhausted credits and invalid responses stop for
inspection. Do not edit the protocol during a run or reuse a smoke folder for full
inference. Output-token exhaustion is recorded as an error, not a successful score.

## Audit and summarize

The existing summary script is provider-neutral and accepts the OpenAI protocol:

```bash
python results/df40_transfer/gemini_face/summarize.py \
  --dataset df40_five_method_v3 \
  --manifest results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv \
  --predictions "$FACE_OUTPUT/predictions.csv" \
  --protocol results/df40_transfer/openai_face/protocol.json \
  --output-dir "$FACE_OUTPUT/summary"
```

Only after `AUDIT.json` reports `passed`, create the matched six-model comparison:

```bash
python results/df40_transfer/gemini_face/compare_df40_five_method.py \
  --api-label openai \
  --api-predictions "$FACE_OUTPUT/predictions.csv" \
  --protocol results/df40_transfer/openai_face/protocol.json \
  --output-dir "$FACE_OUTPUT/comparison"
```

The comparison independently validates API provenance and matches the accepted
189-image CNN/ViT evaluation by SHA-256 and labels. It writes 1,134 prediction
rows across six model/configuration entries and per-method metrics. The adapted
ViT is a separate supervised reference. These are linked video-source groups,
not 189 independent people. The summary also totals recorded OpenAI token usage
from all response attempts in `responses.jsonl`; it does not estimate dollar cost.

When uploading this workflow to CARC, include the sibling `face_eval_common.py`
and the updated `gemini_face/summarize.py` and comparison scripts, not just this
directory. Upload explicit files without replacing the shared project directory.

Interpret `p_manipulated` as a prompted score, not a calibrated probability.
