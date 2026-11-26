# Claude face evaluation: the same 189 images

Completed November 26, 2026 on CARC: all 189 image checks, the separate two-image
smoke, full inference and both result audits passed. The full run recorded 189
successful responses, 54,999 input tokens and 24,801 output tokens. The
saved outputs on CARC contain the frozen run, response journal, predictions,
execution environment, summary and matched CNN/ViT comparison. They are at
`/project2/yzhao010_1531/csci566-genimage/results/df40_transfer/claude_face/df40_five_method_v3_v1/`.
The [saved outputs and run notes](df40_five_method_v3_v1/README.md) are now
downloaded locally. Summary and matched-comparison CSVs regenerated offline are
byte-identical to the CARC originals. The [status record](CARC_STATUS.json) and
[local reproduction record](df40_five_method_v3_v1/LOCAL_REPRODUCTION.json)
document checksum, source, response-log and frozen-membership checks.
This is separate from the completed OpenAI face study and Claude GenImage run.
The deployed code is isolated at
`/project2/yzhao010_1531/csci566-genimage/claude_face_code_v1_20261126/`.

Accuracy was 70.37%, balanced accuracy 55.93%, and AUROC 0.6198. Claude detected
121/135 fakes and falsely flagged 42/54 reals at the frozen threshold. This
does not establish a reliable face detector; see the class-specific metrics in
`df40_five_method_v3_v1/summary/metrics.csv`. No prompt or threshold tuning was
performed. The commands below reproduce or resume the completed evaluation;
keep the completed output path to skip successes rather than pay to score them
again in a new folder.

## Retrieve the completed outputs

From a local terminal, authenticate to USC when prompted:

```bash
scp alanjewa@discovery.usc.edu:/project2/yzhao010_1531/csci566-genimage/claude_face_results_v1_20261126.tgz /private/tmp/csci566-claude-face-results-20261126.tgz
```

Expected archive SHA-256:
`87dc30dd4db0560f1d7852b6129f8185e075dcad95a69f3608b770b70bd9b0b2`.
Verify it before extracting into this directory. The archive contains the full
run and separate smoke, with no images, weights or API credentials. After
download, regenerate the summary/comparison offline into a temporary directory
and compare their CSVs with the CARC originals. This verification passed locally
on November 26. The earlier manually transcribed archive checksum had a typo;
the value above is corrected, with the saved prediction and audit hashes also
independently verified.

Uses `claude-opus-5-5`, matching the saved Claude GenImage model, and the exact
OpenAI face prompt and 50% threshold. Source membership is pinned to manifest
SHA-256 `1f18b71d5d4ebd59ff184d1db06cb33f82f99f034274f6aaaa51b72724afae51`:
54 real and 135 manipulated images across five methods. One original source
image per isolated request; no method names, labels, paths or previous scores
are sent. Claude's 4,000-token budget matches its GenImage configuration;
provider image processing and JSON constraints differ from OpenAI, so this
does not claim identical API input tensors or inference settings.

The runner reuses checkpoint, locking and resume logic from
`../openai_face/evaluate.py` without importing the OpenAI SDK or using its key.
Every image hash is verified before the first request, including smoke tests.
Failures stop the run; only transient errors retry. Truncation, refusal, a
different returned model or invalid JSON/score is recorded as an error.
Existing successes are validated before skipping them. An unchanged protocol
and manifest are required to resume.

## CARC setup

Deploy into a fresh, isolated directory (for example
`claude_face_code_v1_20261126/`), preserving these relative paths. Do not replace
the shared `code/` or existing result directories. The package needs this
directory, `face_eval_common.py`, `openai_face/evaluate.py`,
`gemini_face/summarize.py`, `gemini_face/compare_df40_five_method.py`, the frozen
manifest, and accepted `fresh_candidate_results_v3/{predictions.csv,ACCEPTANCE.json}`.

From the isolated code root in a CARC Bash shell:

```bash
FACE_PYTHON="$HOME/cnn-python/bin/python"
"$FACE_PYTHON" -m pip install -r results/df40_transfer/claude_face/requirements-claude.txt
read -r -s -p 'Anthropic API key: ' ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY
printf '\n'

FACE_MANIFEST=results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv
FACE_DATA=/project2/yzhao010_1531/csci566-genimage/data/df40_fresh_candidate_review_v3_20261123
FACE_OUTPUT=/project2/yzhao010_1531/csci566-genimage/results/df40_transfer/claude_face/df40_five_method_v3_v1
FACE_PROTOCOL=results/df40_transfer/claude_face/protocol.json
```

Never paste the key into chat or save it in a file, notebook, log or command
history. No GPU is required. Run inference within a CARC compute allocation;
the API request runs remotely, with only lightweight local image IO.

## Validate, smoke, run

```bash
"$FACE_PYTHON" results/df40_transfer/claude_face/evaluate.py \
  --manifest "$FACE_MANIFEST" --data-root "$FACE_DATA" \
  --output-dir "$FACE_OUTPUT" --validate-only

"$FACE_PYTHON" results/df40_transfer/claude_face/evaluate.py \
  --dataset df40_five_method_v3_smoke \
  --manifest "$FACE_MANIFEST" --data-root "$FACE_DATA" \
  --output-dir "${FACE_OUTPUT}_smoke" --limit 2

"$FACE_PYTHON" results/df40_transfer/claude_face/evaluate.py \
  --manifest "$FACE_MANIFEST" --data-root "$FACE_DATA" \
  --output-dir "$FACE_OUTPUT"
```

The smoke uses the first two frozen manifest rows (both real); it checks API
compatibility rather than estimating performance. It is separate from the
189-image evaluation. Continue to the full run only after smoke success.
The full run must print `Successful predictions: 189/189`. Ctrl+C stops it;
rerun the identical command to resume saved successes. A request interrupted
before its response is saved may need to be paid for again.

## Audit and compare

```bash
"$FACE_PYTHON" results/df40_transfer/gemini_face/summarize.py \
  --dataset df40_five_method_v3 --manifest "$FACE_MANIFEST" \
  --predictions "$FACE_OUTPUT/predictions.csv" --protocol "$FACE_PROTOCOL" \
  --output-dir "$FACE_OUTPUT/summary"

"$FACE_PYTHON" results/df40_transfer/gemini_face/compare_df40_five_method.py \
  --api-label claude --api-predictions "$FACE_OUTPUT/predictions.csv" \
  --protocol "$FACE_PROTOCOL" --output-dir "$FACE_OUTPUT/comparison"
```

Require both `AUDIT.json` files to say `passed`. Summary includes accuracy,
balanced accuracy, fake recall, real false-positive rate, AUROC, confusion
counts and per-method metrics. Comparison contains Claude and five saved
CNN/ViT configurations (1,134 rows); the previous OpenAI run stays separate.
Responses, model IDs, stop reasons and token usage are retained in
`responses.jsonl`; `predictions.csv` has one latest result per source hash.

This remains an exploratory video-frame subset of 27 linked video pairs,
not a test of 189 independent identities. The adapted ViT is a separate
supervised reference. Treat model scores as prompted scores, not calibrated
probabilities. No prompt/threshold tuning on these evaluation images.

API format: [Anthropic vision documentation](https://platform.claude.com/docs/en/build-with-claude/vision).
