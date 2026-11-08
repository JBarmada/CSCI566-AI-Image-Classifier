# CSCI566 - AI-Generated Image Detection

Does training on more image generators improve detection of an unseen generator?
We compare ResNet-50 and frozen CLIP ViT-L/14 using GenImage, shared CSVs, and
k = 1, 2, 4, 7 training generators at a fixed total training size.

## Start here

**November 8 update:** the corrected **192-run ViT benchmark is complete and
audited**. Use the [completed benchmark analysis handoff](results/vit_benchmark/README.md)
and [current remaining-work review](docs/vit-status-2026-11-08.md) for current status.

| Need | Read |
|---|---|
| Completed 192-run benchmark tables, all predictions, calibration and figures | [Benchmark results](results/vit_benchmark/README.md) |
| Corrected dataset, acceptance checks, and the 192-run ViT study | [Benchmark guide](docs/benchmark.md) |
| Progress, proposed team tasks, and next experiments | [Team plan](docs/team-plan.md) |
| Completed ViT pilot, predictions, and evaluation handoff | [ViT results](results/vit_master_v1/README.md) |
| CARC environment and shared storage | [CARC setup](docs/carc.md) |
| Run, resume, or understand the ViT study | [ViT guide](docs/vit.md) |
| CNN workflow | [CNN team's notebook](cnn.ipynb) |
| Dataset preparation | [Dataset team's notebook](GenImage_Final_Dataset_Workflow.ipynb) |

## ViT: one current workflow

| File | Purpose |
|---|---|
| [ViT_Master_Experiments.ipynb](ViT_Master_Experiments.ipynb) | Interactive entry point using `code/vit/engine/` |
| [code/vit/run_vit_master.py](code/vit/run_vit_master.py) | Runs that notebook without an interactive Jupyter session |
| [code/vit/run_benchmark.slurm](code/vit/run_benchmark.slurm) | Eight holdouts, at most two GPU jobs concurrently |
| [code/vit/setup_carc_vit.sh](code/vit/setup_carc_vit.sh) | One-time environment/kernel setup per teammate |
| [tests/](tests/) | Synthetic checks for the notebook and runner |

Run from your CARC repository folder when you are the assigned study operator.
Jad handles the current study; other teammates can work from its exports.
Only one process may write to each holdout's output folder at a time.
Submit from inside the checkout; the launcher finds the repository root.

```bash
# Once per teammate, in a GPU-backed CARC session:
bash code/vit/setup_carc_vit.sh

# Full preflight plus isolated GPU interruption/resume tests:
sbatch code/vit/smoke_vit.slurm
```

After that job reports `GPU SMOKE TEST PASSED`, submit training:

```bash
sbatch code/vit/run_benchmark.slurm
python code/vit/benchmark_status.py --watch
```

The master covers all four k values. The earlier k=1/k=2 notebooks are retained
under [archive/vit/](archive/vit/) as historical references.

## Shared data and results

Code lives in each person's `~/CSCI566-AI-Image-Classifier` clone. Large files live
under `/project2/yzhao010_1531/csci566-genimage`:

- Dataset: `data/extracted/GenImage_Compact/`
- Corrected CSVs: `data/extracted/GenImage_Compact/benchmark/manifests/`
- ViT outputs: `results/vit_benchmark/<holdout>/`
- ViT feature cache: `features/vit_benchmark/<holdout>/`
- Frozen pilot: `_experiment_manifests/` and `results/vit_master_v1/`

Keep images, features, checkpoints, and full logs on CARC. Commit code,
documentation, and small result tables to GitHub. Updated datasets and scientific
protocols get separate versions; preserve the current study while it runs.

Published pilot results concern **held-out ADM**. CNN crop/flip augmentation and ViT
blur/JPEG/resize/noise/filtering are separate treatments. The
[team plan](docs/team-plan.md) records limitations and proposed parallel work.
