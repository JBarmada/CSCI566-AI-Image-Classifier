# Corrected offline comparison review

Integration review, November 26, 2026. The source GPT run is `7be92ec`;
original predictions, per-view metrics and raw responses were preserved.
No API request, new image selection or retraining was performed.

## Corrections

1. The prompt regression test now reads the canonical `run_pilot.py`.
   `run_full.py` remains a thin wrapper. The prompt text and its hash are unchanged.
2. GPT/Gemini and GPT/Qwen performance metrics now use the same 88 valid
   pilot-image hashes on both sides. `n_baseline_valid=90` still records the
   baseline's available coverage; `baseline_n_valid=88` describes the actual
   matched metric denominator. On those 88 images, GPT AUROC is 0.994872,
   Gemini AUROC 0.978846 and Qwen AUROC 0.833974. These are source-image matches,
   not proof of identical API preprocessing or a broad performance ranking.
3. ViT discovery now reads the corrected `results/vit_benchmark` exports at
   k=7, repeat=0, seed=17, clean test only, retaining saved labels, thresholds,
   model protocol/evaluation hashes and prediction-file checksums. It produces
   16 comparisons: eight holdouts times baseline/augmented. Both classes are
   represented. There are 2,204 matched rows, representing 612 valid GPT source
   images reused across modes and holdouts, not 2,204 independent images.

The corrected ViT exports cover all 630 proposed sources before GPT's 18 invalid
detector224 outputs are excluded. Each holdout contributes the same 70 real
images and its corresponding valid subset of 70 fake images. Only the fake
generator's holdout is used, rather than testing it against a detector that
trained on that generator. Original/cropped/high-resolution views must remain
separate. Source matching does not establish identical internal model tensors.

## Verification and files

`AUDIT.json` records the source prediction checksum, output checksums, membership
counts and scope. The saved 1,890 GPT image/view keys were checked for uniqueness
and manifest membership; labels and submitted-image hashes matched their source
records. All 1,844 valid scores matched their saved raw JSON; 46 invalid outputs
remain explicit. The original attempt-level JSONL is unavailable in Git, so
retry/settings/usage provenance is not certified by these CSV checks.

- `openai_vs_pilot_models.csv`: corrected 88-image paired metrics.
- `openai_vs_claude_matched.csv`: valid shared-image/view comparisons.
- `openai_vs_vit_matched.csv`: corrected k=7/repeat-0 ViT comparisons.
- `pilot_legacy_task_row_metrics.csv`: historical task-row summaries; repeated
  real images remain clearly labeled and must not be counted as independent.

Original files under `../metrics/` are retained as historical exports. Prefer
this review for the corrected pilot and ViT comparisons. The original
per-view GPT and Claude results are unaffected. CNN matching is still pending
retrieval of its saved prediction export from CARC.

## Reproduce from saved scores

From the repository root with the scientific Python dependencies installed:

```bash
PYTHONPATH=code/llm-eval/genimage/openai python - <<'PY'
from pathlib import Path
import pandas as pd
from compare_models import compare_gpt_to_claude, compare_pilot, compare_vit

predictions = pd.read_csv('results/llm_eval/openai/metrics/openai_predictions.csv')
output = Path('tmp/openai-comparison-replay')
output.mkdir(parents=True, exist_ok=True)
compare_gpt_to_claude(predictions, output)
compare_pilot(predictions, output)
compare_vit(predictions, output)
PY
```

This reads saved CSVs and writes a separate replay directory. It uses no API key.
