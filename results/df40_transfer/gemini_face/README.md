# Gemini face-manipulation evaluation

Current selected work uses OpenAI on the 189-image set; follow the
[OpenAI guide](../openai_face/README.md). This Gemini runner remains an optional
provider alternative and requires its own complete evaluation. The historical
150-image recommendation below is not a prerequisite for that OpenAI run.

This is a zero-shot Gemini evaluation on the exact frozen images already used
for the CNN and ViT face-transfer studies. It does not retrain a model or tune
the prompt on evaluation results. The runner verifies every image SHA-256,
sends only image bytes and the frozen prompt, checkpoints after every response,
and preserves raw responses and failures.

New runs freeze the complete protocol and selected manifest in `run.json`,
validate resumed predictions and lock each output folder to one writer. Use a
fresh folder for legacy outputs without this record; preserve existing results.
The summary now verifies model/prompt/protocol hashes, finite scores, raw responses
and threshold consistency as well as image membership. The shared implementation
in `../face_eval_common.py` must be uploaded alongside these files.

## Requirements

1. Confirm that the dataset terms and team policy permit sending these face
   images to Google. The repository does not establish that permission.
2. Make the image bytes accessible on CARC. They are not stored in GitHub.
3. Install `google-genai>=1.0,<2` in the active environment.
4. Set `GEMINI_API_KEY` in the shell. Never put the key in a script, notebook,
   Slurm file, Git commit, or output log.
5. Confirm that the Gemini project has quota for 339 image requests: 150 for
   DeepFakeFace and 189 for the optional five-method follow-up.

The recommended primary evaluation is the 150-image DeepFakeFace set. The
189-image FaceForensics++/DF40 set is exploratory and has unresolved identity,
source-role and provenance limitations documented in `FRONTIER_DATA_HANDOFF.md`.

## Validate locally on CARC without API calls

For DeepFakeFace, replace `STILL_FACE_ROOT` with the directory containing
`wiki/`, `insight/`, and `inpainting/`:

```bash
python results/df40_transfer/gemini_face/evaluate.py \
  --dataset deepfakeface_v1 \
  --manifest results/df40_transfer/still_face_v1/manifest.csv \
  --data-root STILL_FACE_ROOT \
  --output-dir results/df40_transfer/gemini_face/deepfakeface_v1 \
  --validate-only
```

For the five-method set:

```bash
python results/df40_transfer/gemini_face/evaluate.py \
  --dataset df40_five_method_v3 \
  --manifest results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv \
  --data-root /project2/yzhao010_1531/csci566-genimage/data/df40_fresh_candidate_review_v3_20261123 \
  --output-dir results/df40_transfer/gemini_face/df40_five_method_v3 \
  --validate-only
```

## Run Gemini

After validation, export the key and repeat the same command without
`--validate-only`. The run is resumable: rerunning the command skips successful
hashes and retries failed ones. Do not use `--limit` for the reported run.

```bash
export GEMINI_API_KEY='YOUR_KEY'
```

## Audit and summarize

DeepFakeFace:

```bash
python results/df40_transfer/gemini_face/summarize.py \
  --dataset deepfakeface_v1 \
  --manifest results/df40_transfer/still_face_v1/manifest.csv \
  --predictions results/df40_transfer/gemini_face/deepfakeface_v1/predictions.csv \
  --output-dir results/df40_transfer/gemini_face/deepfakeface_v1/summary
```

Five-method follow-up:

```bash
python results/df40_transfer/gemini_face/summarize.py \
  --dataset df40_five_method_v3 \
  --manifest results/df40_transfer/new_data/fresh_candidate_review_v3/manifest.csv \
  --predictions results/df40_transfer/gemini_face/df40_five_method_v3/predictions.csv \
  --output-dir results/df40_transfer/gemini_face/df40_five_method_v3/summary
```

Then create the matched six-model comparison:

```bash
python results/df40_transfer/gemini_face/compare_df40_five_method.py
```

This requires exactly 189 successful Gemini predictions and checks them
against the accepted 945 saved-model predictions. It compares Gemini, CNN and
ViT baseline/augmented, plus the face-adapted ViT as a separate secondary
reference. The adapted head is not zero-shot: it was trained on SimSwap, FOMM
and Wav2Lip, three methods present in this evaluation.

Each summary contains an independent membership/checksum audit and per-method
accuracy, balanced accuracy, false-positive rate, fake recall, F1 and AUROC.
Interpret Gemini confidence as a prompted score, not a calibrated probability.

## Matched CNN/ViT/Gemini comparison

After the 150-image Gemini audit passes, run:

```bash
python results/df40_transfer/gemini_face/compare_deepfakeface.py
```

This verifies that Gemini, CNN baseline/augmented and ViT baseline/augmented
all use the exact accepted 150-image manifest. It writes 750 matched prediction
rows, per-method metrics and `AUDIT.json` under
`deepfakeface_v1/comparison/`.
