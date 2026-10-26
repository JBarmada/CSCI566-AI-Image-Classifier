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

### Run overnight with Slurm

From your repository directory on CARC (not inside a notebook cell):

```bash
sbatch run_vit_master.slurm
```

The job requests one GPU, four CPUs, 24 GB RAM and 24 hours using account
`yzhao010_1531`. It activates your existing `csci566-vit` environment. It does not
reinstall packages or pull code while running. You may close the browser once
`sbatch` returns a job ID. Queue time and completion within 24 hours are not guaranteed.

The runner executes the code cells of this same notebook, using its selected plan
(24 baseline/augmented runs by default), without depending on a Jupyter session.
It removes the interactive session work cap and stops on errors. Existing verified
training is skipped, partial training resumes at the last saved epoch, missing tests
are run, and summaries are generated. It never treats the presence of `last.pt` alone
as completed training. To validate without training:

```bash
sbatch run_vit_master.slurm --preflight-only
```

Watch `vit-master-JOBID.log` in the directory where you submitted the job.
The study's `batch_jobs/JOBID_TIMESTAMP/status.json` records `complete`,
`preflight_passed`, `failed`, or `interrupted_resumable`, with diagnostics.
A completed benchmark ends with `ALL SELECTED RUNS COMPLETE` and a summaries path.
Resubmit the same command after an interruption; do not change `STUDY_NAME` merely
because a session ended. A scheduler warning requests a clean stop 120 seconds before
the time limit; if a hard kill happens first, resume still uses the last full epoch.

Only one batch writer per study is allowed. The batch runner archives a stale
per-run lock only if the recorded PID is absent on the current host, or a successful
Slurm query shows no allocations at all on its recorded remote node. A failed query
or possibly active owner keeps the lock and stops with a diagnostic. Stop any old
notebook training before submission; never run both against the same study.

### Kernel-restart protocol fix

CLIP's transform `repr` contains a Python function memory address, for example
`<function _convert_image_to_rgb at 0x7f...>`. The address changes on restart and
previously caused a false `Protocol changed` error and unnecessary feature-cache
misses. Protocol/cache identities now strip only this transient address. Transform
names/parameters, manifests, settings, environment and numerical engine still match
strictly. When an old config differs only in this representation, the run retains
its original protocol hash so existing checkpoint/marker files remain valid. Original
configs/checkpoints are not rewritten. Each used run records `last_resume_check.json`.
Real differences list the changed fields and stop before training. Restore the original
settings or use a new study for actual protocol changes.

The traceback's `torch.cuda.empty_cache()` line is cleanup, not the source of the
protocol mismatch. Cleanup errors now cannot replace the original exception.

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
