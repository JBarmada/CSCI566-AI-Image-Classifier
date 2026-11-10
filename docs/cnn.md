# CNN batch workflow

The CNN benchmark uses `cnn.ipynb` as its single training implementation. The
batch runner selects one holdout per Slurm array task; each task runs k=1/2/4/7,
three dataset repeats and baseline/augmented training: 24 runs per holdout and
192 runs overall.

## Submit and monitor

Run from a complete, unchanged CARC checkout. The launcher uses the existing
`csci566-vit` environment by default because it contains the project's PyTorch
stack. Set `CNN_ENV_PREFIX` before submission if the CNN environment lives
elsewhere.

```bash
sbatch code/cnn/run_benchmark.slurm
python code/cnn/benchmark_status.py --watch
squeue --me
```

The array has eight tasks with at most two active at once. Index order is ADM,
BigGAN, glide, Midjourney, SD1.4, SD1.5, VQDM and wukong. Each task requests one
GPU for 24 hours. It reuses verified completed runs and resumes an interrupted
run from its last atomically written epoch.

Each holdout has an OS-managed batch lock, each run has a `RUNNING.lock`, and
job status is stored under:

```text
/project2/yzhao010_1531/csci566-genimage/results/cnn_benchmark/notebook/
  batch_jobs/<holdout>/<job-and-time>/status.json
  sessions/holdout_<holdout>/progress.json
  runs/<experiment>__<variant>__seed17/
```

The scheduler signals the runner two minutes before termination. Exit code 75
means the job stopped at a safe resume point. Resubmit one failed/interrupted
array index, for example:

```bash
sbatch --array=2 code/cnn/run_benchmark.slurm
```

Do not edit or pull the checkout during an active array. The runner snapshots the
notebook and CNN orchestration source in each batch-job directory.

## Final audit

Immediately after submitting the array, schedule the audit with the array job ID:

```bash
sbatch --dependency=afterok:ARRAY_JOB_ID code/cnn/finalize_benchmark.slurm
```

The dependency prevents finalization unless every array task succeeds. The CPU
audit independently checks all manifests, memberships, predictions, metrics and
checkpoint hashes. It exits nonzero unless all 192 runs pass and writes
`CNN_AUDIT.json` plus the aggregate CSV files under the shared output root.

For a lightweight validation that does not start training:

```bash
sbatch --array=0 code/cnn/run_benchmark.slurm --preflight-only
```
