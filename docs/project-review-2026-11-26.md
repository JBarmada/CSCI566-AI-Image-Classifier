# Project contributions, remaining work and integration review

Reviewed November 26, 2026 against all fetched team branches, saved predictions,
audit records and the existing research log. Attribution below is based on
observable contributions, not commit counts. Rewritten historical commits often
have duplicate hashes for the same patch, and work outside Git may be missing.
Authorship of an entire experiment should still be confirmed by its contributors.

## What each member completed, why, and what it achieved

### Jeslyn Yang

- Implemented the original fine-tuned ResNet-50 CNN notebook, corrected its
  early split/path handling, and committed initial CNN outputs. This provided
  the convolutional detector needed for comparison with the ViT pipeline.
  Evidence includes `68c10a6`, `69ebb54`, `547ff2c` and `e0dbc3e`, or their
  equivalent rewritten commits. The current corrected 192-run workflow was
  subsequently extended and audited by Aditya; distinguish the original model
  implementation from that later execution/verification work.
- Added GPT GenImage preparation, a resumable isolated-image Responses API
  runner, tests, scoring and model comparisons in `7be92ec`. Its purpose was
  to compare a frontier model against existing detectors and test image-shape
  and resolution effects. The saved study has 630 unique images in three
  views: 1,890 final rows, 1,844 valid scores and 46 invalid outputs. Original
  valid-score AUROC is 0.9910 and detector224 AUROC 0.9307; these do not mean
  the fixed threshold detects all fakes. Original fake recall is 55.2% and
  detector224 recall 26.8%. The original prompt/protocol was not retuned during
  integration. Missing attempt-level logs remain an audit limitation.

### Aditya Lanjewar

- Aligned the CNN to the corrected shared benchmark and added resumable CARC
  Slurm arrays, progress tracking, environment handling, migration/repair and
  independent final-audit fixes. This made the CNN study practical to run,
  restart and verify. All 192 corrected CNN runs passed their final audit.
  Evidence includes `01e1838`, `c766241`, `0c6bf3d`, the subsequent audit fixes,
  and the saved CNN completion/summary artifacts. Older branch hashes are
  patch-equivalent versions of the same work, not additional experiments.
- Prepared saved-checkpoint CNN k=7 robustness evaluation and its operational
  workflow. Jad later investigated the device replay discrepancy and audited
  the accepted reruns; the final robustness evidence is shared work.
- Implemented the resumable face API workflow and independent comparisons,
  completed/archived the 189-image OpenAI study and updated documentation in
  `1324ffc`. This tested whether generated-image detectors and a zero-shot
  model transfer to face manipulations without retraining. The face run has
  189/189 successful predictions and 1,134 matched model rows. OpenAI falsely
  flagged 6/54 reals but detected only 16/135 fakes; balanced accuracy was
  50.37%, near chance at the frozen threshold. This is a separate dataset/model
  from Jeslyn's GenImage evaluation.
- This integration review fixes the prompt-test incompatibility, common-image
  comparison denominators and corrected ViT export selection, preserving raw
  outputs and adding regression checks. These fixes require no new inference.

### Jad Barmada

- Built the shared corrected benchmark, dataset audits, frozen CLIP ViT-L/14
  feature/linear-head workflow, CARC runners and audited exports. This made
  generator-diversity experiments comparable at a fixed training budget and
  prevented changing test membership between settings. The accepted studies
  contain 192 runs for each detector. Evidence includes `417dd16`/`1c2ddb4`
  and `67c5261`/`d6937f2` plus their follow-up records.
- Completed/analyzed ViT robustness, investigated CNN replay discrepancies,
  verified V100/P100 reruns and exposed the JPEG operating-point tradeoff.
  Augmentation reduced some real-image false alarms while also reducing fake
  detections; accuracy gains alone did not capture that distinction.
- Added face-transfer, face-adapted-head, still-photo processing, zero-shot
  CLIP/Qwen and adversarial/contrastive feature-adapter studies; prepared report
  and poster evidence. These tested the proposal's broader generalization and
  artifact-reliance questions. Their results are exploratory, with source,
  identity and prior-test-inspection limitations already documented. The
  objective extension did not establish a reliable winning method.

### Shriya Rajesh

- Updated the early k=1 ViT starter and added the k=2 starter (`5af4363`,
  `49af90d`). This enabled early generator-combination experiments; those
  starters are now historical rather than the maintained master pipeline.
- Added the Gemini and Groq-hosted Qwen GenImage evaluation in `c000e5a`.
  This provided frontier-model baselines on a frozen shared sample. Both
  models completed the same 90 unique pilot images; 160 legacy task rows
  repeat ten real images across eight holdouts. Unique-image accuracies are
  71.11% and 65.56%, respectively. The larger 2,520-image attempts are partial
  and must not be counted as completed studies. A final matched CNN-inclusive
  frontier comparison is still outstanding.

### Rachel Lin

- Added the extended ViT calibration, validation-fitted temperature/Platt
  scaling, threshold selection, generator/family errors, persistent-image
  errors, tests and figures (`f3c3de6` through `ebeba98`). This explains what
  the ViT's confidence means and which errors persist beyond aggregate
  accuracy. At k=7, baseline ECE falls from 0.2816 to about 0.0676 after
  validation-only temperature scaling; classification at threshold 0.5 is
  unchanged. At low k, scaling can worsen log loss under generator shift.
- Added Claude evaluation and blinding/shape/resolution controls (`e319d94`).
  All 630 images received valid responses in each of three views. Original
  AUROC is 0.99977 with fake recall 75.5%; detector224 AUROC is 0.95353 with
  fake recall 56.25%. Real false positives are 0/70 in each view. These controls
  reveal sensitivity to image presentation, but do not prove the absence of
  all shortcuts or identify a single causal factor.

### Saisha Hongal

The research log records Saisha downloading Celeb-DF data and supplying the
100 real face crops used in the team's early face test. That supported the
real-versus-fake face comparison when official archive access was difficult.
Commit `dca65cd` assigns backlog owners. These support dataset preparation and
coordination credit; a separate model/code implementation is not established
by the available record. Literature, report, poster or other non-code work
should be confirmed with the team rather than inferred from assignments.

### Medini Patil

No identifiable code commit or completed implementation artifact was found
under this member's name in the reviewed history. The user also could not
confirm a code contribution. This is an evidence gap, not proof that no
non-code work occurred. Confirm any literature/data/report/presentation work
before assigning completed contribution credit.

## What remains most important

The main model training is complete. The largest implementation gap is a
single reproducible, coverage-aware analysis across the accepted outputs.
It should keep clean GenImage, robustness, frontier GenImage and face/deepfake
transfer as separate studies while exposing consistent metrics and evidence.

1. **CNN-inclusive frontier matching:** retrieve the saved CNN prediction
   export from CARC and join by hash, holdout, repeat and label. The corrected
   ViT export already covers the full 630-image shared source set; the earlier
   apparent lack of matches came from reading the old ADM pilot. The
   integration fixes now use corrected k=7/repeat-0 exports. Do not retrain.
2. **Finish provenance and coverage checks:** obtain the GPT request-attempt
   JSONL from its owner, retain all 46 invalid scores, and verify submitted
   image hashes/preprocessing wherever available. Its final CSVs alone cannot
   independently certify retries or billing. Do not silently rerun invalid
   images or change the threshold after inspecting this test.
3. **Publish the final analysis entry point:** use a source inventory and one
   repeatable command to regenerate accepted tables/figures, matched detector
   comparisons and evidence links. Corrected OpenAI comparison exports are
   now available in `results/llm_eval/openai/integration_review/`; they are one
   part of that broader remaining deliverable.
4. **Finish team deliverables:** select figures, review claims and contribution
   credit, integrate the final report/poster, and test an offline demo.
   These are completion tasks, not a need to invent more models.

Optional extensions are CNN temperature/Platt calibration and repeated-error
analysis, and a fresh person-independent face test. They would strengthen the
work but are not prerequisites for reporting the completed benchmark. The
adversarial/contrastive proposal already has exploratory feature-adapter results;
it has not been implemented as a new adversarial CNN backbone training study.

## Merge findings and fixes

The tested order was latest `main` (`4461244`), our face branch (`1324ffc`),
the GPT branch (`7be92ec`, already including Claude and the Gemini/Qwen merge),
the ViT confidence branch (`ebeba98`), and the pending documentation commit
(`15d5039`). Git reported no text conflicts for these distinct contributions.

The combined test run nevertheless initially had 78 passes and one failure:
the GPT prompt test still expected a constant inside `run_full.py`, which our
cleanup converted to a wrapper. Aligning that test to `run_pilot.py` preserves
the actual prompt and resolves the incompatibility.

Two analysis regressions were also demonstrated with failing checks and fixed:

- The pilot comparison measured GPT on 88 valid images and its baselines on
  all 90. Both performance denominators now use the same 88 hashes; original
  coverage is retained in separate columns.
- The GPT ViT discovery searched `vit_master_v1`, yielding only 72 repeated
  all-real matches across 24 old exports. Corrected `vit_benchmark` selection
  now yields 16 k=7/repeat-0 comparisons, both classes, and 2,204 matched rows
  over 612 valid GPT source images. Eighteen detector224 outputs remain invalid.

The corrected comparison files are separate from the original exports. Core
CNN/ViT training files and notebooks were not changed by the integration.
This is a tested source/analysis integration, not a rerun of GPU training or
any provider API.

## Repeated work: what is intentional and what should be reduced

- The old CNN/ViT setup branches and local pre-email-rewrite backup have zero
  unmatched non-merge patches against the integrated history. Their different
  hashes mostly reflect rewritten history. Do not merge their obsolete trees
  or rerun their experiments; their code changes are already represented.
- The GPT branch already includes Claude and the shared LLM branch. Merging
  all three separately adds ancestry/noise without three new implementations.
- The face Gemini/OpenAI runners duplicate manifest reading, image verification,
  event loading, atomic checkpoints and event construction. Protocol checks
  and locking are already shared in `face_eval_common.py`. A provider adapter
  could reduce the remaining duplication later, but changing frozen working
  inference code now adds risk and is not needed for the completed OpenAI run.
- `run_full.py` already delegates to `run_pilot.py`; the former full-copy
  duplication has been removed. `evaluate_gemini.py` remains historical and
  expects a different single-provider protocol shape than the current pilot
  `protocol.json`. It is not the maintained face or GPT runner; do not treat
  its default invocation as a validated new experiment.
- Multiple metric/comparison scripts overlap. Prefer one maintained entry
  point per task with common verified joins. Preserve historical raw exports
  and label superseded summaries rather than deleting evidence.
- Temperature scaling/error analysis is post-processing of saved scores,
  not another training run. The 630-image Claude/GPT study contains the
  90-image pilot, and three views reuse the same sources. Do not add these
  counts or present their averages as independent experiments.
- OpenAI face and OpenAI GenImage runs are different tasks, datasets, models
  and prompts. They are not redundant work and must not be pooled.

## Validation and limits

- The combined suite passes 80 tests and 28 subtests. It includes the original
  CNN/ViT synthetic CPU checks, face API mocks, ViT analysis checks and GPT
  GenImage regression tests. Provider APIs and model backbones are not called.
- Raw face exports retain their verified checksums. GPT raw predictions and
  original per-view outputs are preserved; only new corrected comparison files
  are generated. CSV-level membership, submitted hashes, labels, statuses and
  valid raw JSON scores were checked for all 1,890 GPT rows.
- ViT calibration, recalibration and error scripts were regenerated offline in
  an isolated directory. The raw calibration and aggregate error CSVs matched
  exactly. Fitted recalibration summaries had small numerical differences in
  the fresh dependency environment: mean ECE/Brier differences up to 1e-6 and
  log-loss summary differences up to 2.33e-4. Original results are preserved.
- Tests used an isolated Python 3.11 environment with NumPy 2.4.6, pandas 3.1.1,
  scikit-learn 1.9.1, SciPy 1.17.1, Pillow 13.0.0, PyTorch 2.15.0 and
  torchvision 0.29.2. This does not certify every historical API SDK/runtime or
  reproduce a CARC GPU training run. No automatic GitHub CI existed at review.
- Source/model differences, reused real images, repeated views and unresolved
  face identities/mirror labels limit generalization and uncertainty claims.
  No common valid-image score table by itself proves identical preprocessing
  or a universal ranking of architectures and frontier models.
- The 630-image GenImage set is 88.89% fake, so an all-fake rule attains 88.89%
  raw accuracy while balanced accuracy remains 50%. High AUROC can coexist
  with low fake recall at the frozen 50 threshold. Claude and GPT also use
  different output/reasoning budgets (the saved Claude runner permits 4,000
  output tokens versus GPT's 128); this is not a controlled compute-budget
  comparison. Do not retune the threshold on this test to improve the headline.

Reproduce tests from the repository root:

```bash
MPLBACKEND=Agg PYTHONPATH=tests:code/llm-eval/genimage/openai \
  python -m pytest -q tests code/llm-eval/genimage/openai/test_openai_eval.py
```

See the [consolidation plan](results-consolidation.md),
[corrected OpenAI comparisons](../results/llm_eval/openai/integration_review/README.md),
[face results](../results/df40_transfer/openai_face/df40_five_method_v3_v2/README.md)
and [ViT analysis](../results/analysis/README.md) for the supporting artifacts.
