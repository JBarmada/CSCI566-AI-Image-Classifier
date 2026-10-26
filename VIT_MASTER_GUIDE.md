# ViT master notebook

Open `ViT_Master_Experiments.ipynb` at the repository root and select the existing
**CSCI566 ViT** kernel in a GPU-backed CARC OnDemand session. The notebook is
self-contained: no companion Python module is required.

## Default experiment plan

- ADM held out; k = 1, 2, 4, 7; dataset repeats = 0, 1, 2; classifier seed = 17.
- Baseline and one combined augmentation recipe: **24 fresh training runs**.
- Frozen OpenAI CLIP ViT-L/14 plus a linear classifier in both modes.
- Same class counts, AdamW settings, threshold, and validation-based selection
  defaults as the existing ViT starter.
- Baseline uses normal deterministic CLIP preprocessing. Augmented training adds
  random blur, JPEG recompression, resizing, noise, and low-pass filtering to
  both labels, with new views each epoch. Exact settings are at the top.
- Validation and default final testing use clean, deterministic images.
- This plan does not include domain-adversarial training or zero-shot LLM tests.

## Run

1. Set all options in Section 1. Defaults use the shared CARC paths in the README.
2. Run all cells. Section 8 shows the verified 24-run plan; Section 9 runs it.
3. Section 10 generates tables, paired augmentation deltas, calibration diagnostics,
   comparison plots, and heatmaps. Existing source images/manifests are never edited.

`RUN_FINAL_TEST=True` is the default for a fixed benchmark. Change it to `False`
if test reporting is not yet authorized by the team. Test results never affect
checkpoint selection or the threshold. Do not tune this protocol on those results.

All configurable options are in the first code cell, including seeds, batch sizes,
epoch limits, augmentation strengths, evaluation conditions, and output paths.
`NUM_WORKERS=0` is deliberately enforced for reproducible augmented views and resumes.

## Storage and resume

Outputs: `/project2/yzhao010_1531/csci566-genimage/results/vit_master_v1/`.

Each experiment/mode/seed has its own stable directory under `runs/`. A configuration
fingerprint protects it from accidental reuse with changed scientific settings.
Use a new `STUDY_NAME` for intentional protocol changes. Completed matching runs are
verified and skipped. No existing starter-notebook runs are assumed completed.

The notebook saves a complete epoch checkpoint (`last.pt`) and a final selected head
(`best_head.pt`). An interrupted epoch is repeated deterministically from the previous
saved epoch. The baseline and augmented modes both keep the CLIP encoder frozen.
Only deterministic features are cached; augmented training views are never cached.

`MAX_RUNS_THIS_SESSION` limits how many pending runs get work in one invocation;
rerunning skips finished runs and continues through the remaining plan.

After a forcibly killed job, a `RUNNING.lock` may remain. Inspect its host/PID/job,
confirm the previous process has stopped, and remove only that stale lock. Never
remove a lock from an active run or launch two writers for the same run directory.

## Extra evaluations without retraining

Add `blur`, `jpeg`, `resize`, `noise`, or `lowpass` to `EVAL_CONDITIONS` and rerun
with unchanged training settings. Completed training is reused; new evaluations
get separate directories. This adds evaluation jobs, not training combinations.
Per-image noise is seeded from the image SHA-256 and evaluation condition so it is
reproducible. Share that recipe/seed with the CNN team.

The notebook reports raw probability calibration (ECE, Brier score, log loss and
reliability plots). It does not fit temperature scaling or adjust test thresholds.

## Additional holdouts and aggregation

After the dataset team creates corresponding manifests, add holdouts at the top.
All selected CSVs are validated before training begins. The current ADM-only plan
does not establish GAN-versus-diffusion holdout performance. Midjourney is left
unassigned in the editable family mapping because the exact source model is not
documented. Family summaries average holdout-level means.

Repeat statistics describe variation in generator subsets and sampled images, not
independent confidence intervals. Augmented-minus-baseline deltas are paired by exact
experiment and seed. With/without augmentation is one combined-recipe comparison;
separate augmentation ablations need additional explicitly named protocols.

## Verification

The code-version check fingerprints live Python implementations, including class
methods, without requiring source files. This supports Jupyter-defined classes
and stays stable when the same cells are executed again.

The included `tests/test_vit_master.py` uses tiny synthetic images and a toy frozen
encoder. It checks notebook compilation, metric/calibration boundaries, path and
hash validation, split leakage detection, deterministic degradation/caching,
interrupted training/resume equivalence, evaluation exports, exclusive locks,
session limits, completed-run skipping, configuration protection, and aggregation.

Run in an environment with the notebook dependencies:

```bash
python -m unittest discover -s tests -p 'test_vit_master.py' -v
```

These CPU checks do not validate actual CLIP weights, V100 execution, CARC permissions,
or the real dataset. The notebook performs those environment/data checks when run on
CARC. No real-data training results are shipped with this notebook.
