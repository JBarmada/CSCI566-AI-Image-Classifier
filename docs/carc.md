# CARC setup and shared storage

See the [team plan](team-plan.md) for roles and priorities, and the
[ViT guide](vit.md) for study commands, resume behavior, and outputs.

## Personal code, shared data

Each teammate works in their own CARC clone:

```bash
cd ~
git clone https://github.com/JBarmada/CSCI566-AI-Image-Classifier.git
cd CSCI566-AI-Image-Classifier
```

If you already have a clone, use it. Save your work before updating; inspect
`git status` and use `git pull --ff-only`. Resolve local edits rather than
overwriting them. Keep an active study's code/environment fixed until it finishes.

Shared storage is at `/project2/yzhao010_1531/csci566-genimage`:

```text
csci566-genimage/
  data/
    genimagezips/                 original transferred archives
    extracted/GenImage_Compact/
      ADM/, BigGAN/, ...          original selected images
      _experiment_manifests/     experiments.json and per-experiment CSVs
  features/vit_master_v1/         deterministic CLIP feature cache
  results/vit_master_v1/
    runs/                        configs, inputs, checkpoints, evaluations
    batch_jobs/                  notebook snapshots and job status
    summaries/                   combined tables and figures
  checkpoints/                   shared checkpoint storage for other workflows
```

Authorized project members use the same image files; no copy into each home
directory is needed. Changes to shared files affect everyone.

The optional [shared-storage script](../setup_carc.sh) links `data/`, `features/`,
and `checkpoints/` in your clone to shared storage:

```bash
bash setup_carc.sh
ls -l data features checkpoints
```

It does not install Python packages. The master notebook's default absolute CARC
paths also work without these symlinks.

## ViT environment: once per teammate

Launch GPU-backed OnDemand JupyterLab with account `yzhao010_1531`, partition
`gpu`, one GPU, four CPUs, and sufficient memory (the batch script requests 24 GB).
Use `conda` in the software modules field. Choose a wall time for the session;
work is still subject to its allocation's time limit.

From a terminal in your repository folder:

```bash
bash setup_carc_vit.sh
```

The script creates/reuses `csci566-vit`, installs PyTorch and pinned OpenAI CLIP,
downloads the model, checks a real CUDA operation, and registers the **CSCI566 ViT**
kernel. Select that kernel in JupyterLab and restart the kernel before running.

The script defaults to CUDA 11.8 wheels for the project's P100-compatible setup.
A newer driver alone is not a reason to replace them. Each teammate creates a
personal environment/kernel; the dataset stays shared. Avoid package setup while
that environment is being used for training.

For a new terminal:

```bash
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csci566-vit
```

GPU-intensive work belongs in a GPU allocation. Use login nodes for submissions
and lightweight file operations.

## Working together

- Use your own clone/branch and publish focused pull requests.
- Run one writer per ViT study. Read existing exports for analysis work.
- Preserve current images/manifests/results. Corrected releases get separate paths,
  documented hashes, and a release note.
- Keep large artifacts on CARC. GitHub only backs up committed/pushed files;
  committing code does not back up checkpoints or raw data.
- End unused OnDemand sessions after saving. Closing a browser tab alone does not
  end the allocation.

```bash
squeue --me
myaccount
myquota
du -sh /project2/yzhao010_1531/csci566-genimage/data
```

## References

- [CARC: running jobs](https://www.carc.usc.edu/user-guides/hpc-systems/using-our-hpc-systems/running-jobs)
- [CARC: Conda](https://www.carc.usc.edu/user-guides/hpc-systems/software/conda)
- [CARC: Jupyter kernels](https://www.carc.usc.edu/user-guides/hpc-systems/software/jupyter-kernels)

The [setup script](../setup_carc_vit.sh) specifies the project's actual package
installation.
