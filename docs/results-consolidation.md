# Result consolidation plan

Updated November 26, 2026 after fetching the team branches. Consolidation is
feasible using saved predictions: no new training or paid inference is needed
to assemble and compare the available outputs. This document is the plan and
source inventory; the cross-branch consolidation has not been executed.

## Sources and integration state

- **Clean CNN/ViT and robustness:** accepted results are already indexed in
  [the research results directory](../results/df40_transfer/README.md).
  Keep the corrected 192-run clean studies and the accepted V100/P100 robustness
  combination distinct from older pilots and failed replay outputs.
- **OpenAI face evaluation:** committed on `feature/matched-frontier-comparison`
  at `1324ffc`. The [189-image exports and analysis](../results/df40_transfer/openai_face/df40_five_method_v3_v2/README.md)
  include the frozen run, raw scored responses, audits and matched six-model
  predictions. Local regeneration reproduced the original CSVs exactly.
  The resumable runner, protocol checks, concurrent-writer lock and offline
  comparison are implemented; 17 focused tests passed. These changes are pushed
  to the feature branch but are not yet in `main`.
- **Gemini/Groq-Qwen GenImage pilot:** the
  [90-image outputs](../results/llm_eval/README.md) are available on the current
  branch. The 160 legacy task rows reuse the same ten real images eight times.
  The separate 2,520-image attempts remain incomplete historical artifacts.
- **Claude GenImage:** source commit `e319d94`, included in
  `zero-shot-llm-evaluation` at `3d73b5b` and in `zero-shot/gpt-genimage`.
  [Source and outputs](https://github.com/JBarmada/CSCI566-AI-Image-Classifier/tree/e319d94/results/llm_eval/claude)
  contain 630 unique images: 70 real and 70 fake from each of eight generators.
  Original, square-crop and detector224 views each have 630 successful responses.
  Read-only checks confirmed all 90 pilot hashes are included, the same blind-ID
  mapping across views, raw-score consistency and the saved overall metrics.
  This is CSV-level verification, not a full image/protocol audit. The files
  are not integrated into the current branch or `main`.
- **GPT GenImage:** new source commit `7be92ec` on `zero-shot/gpt-genimage`
  adds [code](https://github.com/JBarmada/CSCI566-AI-Image-Classifier/tree/7be92ec/code/llm-eval/genimage/openai)
  and [outputs](https://github.com/JBarmada/CSCI566-AI-Image-Classifier/tree/7be92ec/results/llm_eval/openai).
  The saved model is `gpt-6.2-sol`, distinct from the `gpt-6.1-terra` face run.
  The same 630 GenImage sources have three views: 1,890 final rows, comprising
  1,844 valid scores and 46 invalid outputs. Valid counts are 617 original,
  615 square and 612 detector224. There are no missing final rows or refusals
  in the saved status tables. These outputs are available on that branch,
  not yet independently audited or integrated here. The earlier branch-review
  statement that GPT GenImage code was absent is superseded by this commit.
- **ViT confidence/error analysis:** source `ebeba98` on
  `analysis/vit-confidence-errors` contains
  [analysis, figures and tests](https://github.com/JBarmada/CSCI566-AI-Image-Classifier/tree/ebeba98/results/analysis).
  It reuses predictions, reports agreement with all 192 recorded clean-run
  metrics and fits calibrators/thresholds on validation data only. Integration
  and independent regeneration remain pending. The extended CNN counterpart
  has not been implemented; the existing raw CNN/ViT confidence audit remains
  available.

The older remote CNN Slurm/Python and ViT setup/documentation branches have no
unmatched non-merge patches relative to our current history. Do not treat old
branch commit IDs as new experimental results. Preserve them as history.

## Consolidation sequence

1. Integrate the reviewed branch contributions on a dedicated consolidation
   branch. Preserve the current OpenAI face workflow, audit checks, original
   exports and source attribution. The GPT branch already includes the Claude
   contribution; avoid adding it twice. Resolve documentation differences
   against this inventory instead of restoring old pending-work statements.
2. Build a source inventory recording dataset/task, manifest/source hashes,
   source branch/commit, model/configuration, preprocessing/view, threshold,
   checkpoint where relevant, expected/valid/invalid counts and audit status.
   Reference original files rather than moving or rewriting audited exports.
3. Recompute comparable metrics from saved rows. First complete the GenImage
   pilot comparison using its shared 90-image membership. Include Claude and
   valid GPT intersections with exact denominators. Report the 630-image
   Claude/GPT comparison separately; Gemini/Qwen have not scored the other
   540 images. Keep original, square and detector224 views separate.
4. Retrieve matching corrected CNN predictions from CARC and use the published
   corrected [ViT exports](../results/vit_benchmark/README.md). Check membership,
   labels, holdouts, repeats, conditions and preprocessing before comparing.
   If saved scores do not cover a proposed image set, report the missing
   coverage; restrict to supported intersections rather than fabricating rows.
5. Integrate/reproduce the ViT confidence analysis. Keep raw scores separate
   from validation-calibrated scores and preserve the fixed-threshold benchmark.
   Decide separately whether to implement equivalent CNN recalibration.
6. Publish one analysis entry point with separate sections for clean GenImage,
   robustness, frontier GenImage, and face/deepfake transfer. Include concise
   findings, reproducible tables/figures, coverage, limitations and evidence
   links. Suggested destination: `results/final_analysis/`; it is not created
   by this documentation update. Feed verified findings into the report/poster.

## Checks required before combining results

- **Common image membership:** GPT's saved Gemini/Qwen comparison reports
  88 valid GPT pilot images but baseline metrics with `baseline_n_valid=90`.
  Recompute both sides on the same 88-image intersection before calling those
  metrics matched. Preserve the full-90 baseline summary separately.
- **Correct detector exports:** the GPT branch's current ViT table has only
  72 repeated matches across 24 old ADM exports. Each intersection has 1–4
  images, all real and no fake. It cannot estimate AUROC or broad detector
  performance. Replace it with supported corrected-export intersections, and
  obtain the CNN exports from CARC rather than treating them as nonexistent.
- **Request provenance:** the GPT summary refers to 1,893 attempts including
  three retries, but its linked `logs/requests_gpt-6.2-sol.jsonl` is not tracked
  at `7be92ec`. Obtain the original attempt log from its owner to independently
  check retries, settings and usage. Do not label that attempt-level audit
  complete from final prediction CSVs alone.
- **Coverage:** retain GPT's 46 invalid outputs. Performance metrics use valid
  scores, while coverage uses all 630 expected sources per view. Use a common
  valid intersection for comparisons and disclose who was excluded. Do not
  silently call the API again or treat invalid outputs as correct predictions.
- **Preprocessing:** check submitted-image hashes/pixel mappings where available.
  Same source images or 224×224 dimensions do not prove identical CNN/ViT/API
  processing. Original shape is strongly correlated with class in the 630-image
  set; retain the square/resolution controls and avoid causal overstatements.
- **Different tasks:** GenImage synthetic-image detection and the 189-image
  face-manipulation test answer different questions. Do not pool their scores,
  samples or model rankings. Keep the optional 150-image DeepFakeFace API study
  distinct; it has not been run.
- **Aggregation:** show AUROC, balanced accuracy, real false positives, fake
  recall and counts alongside accuracy. Source-linked faces, repeated real
  images, views and dataset repeats are not independent observations. Fit any
  new calibration or thresholds on separate validation data, never this test.

## Completion criteria

The consolidated analysis is ready when every included result has a traceable
source and explicit audit status; comparison rows use verified common membership;
saved metrics and figures regenerate; partial/invalid/historical outputs are
labeled; and all report numbers link to their evidence. Preserve images and
weights on their existing storage and keep credentials out of the repository.
No new API key, API provider call, training job, branch merge or result rewrite
is part of this documentation update.
