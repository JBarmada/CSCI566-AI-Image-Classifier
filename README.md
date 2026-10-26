# CSCI566 - AI-Generated Image Detection

Does training on more image generators improve detection of an unseen generator?
We compare ResNet-50 and frozen CLIP ViT-L/14 using GenImage, shared CSVs, and
k = 1, 2, 4, 7 training generators at a fixed total training size.

## Start here

| Need | Read |
|---|---|
| Progress, proposed team tasks, and next experiments | [Team plan](docs/team-plan.md) |
| CARC environment and shared storage | [CARC setup](docs/carc.md) |
| Run, resume, or understand the ViT study | [ViT guide](docs/vit.md) |
| CNN workflow | [CNN team's notebook](cnn.ipynb) |
| Dataset preparation | [Dataset team's notebook](GenImage_Final_Dataset_Workflow.ipynb) |

## ViT: one current workflow

| File | Purpose |
|---|---|
| [ViT_Master_Experiments.ipynb](ViT_Master_Experiments.ipynb) | Model, settings, training, evaluation, and summaries |
| [run_vit_master.py](run_vit_master.py) | Runs that notebook without an interactive Jupyter session |
| [run_vit_master.slurm](run_vit_master.slurm) | Submits the runner to CARC |
| [setup_carc_vit.sh](setup_carc_vit.sh) | One-time environment/kernel setup per teammate |
| [tests/](tests/) | Synthetic checks for the notebook and runner |

Run from your CARC repository folder when you are the assigned study operator.
Jad handles the current study; other teammates can work from its exports.
Only one process may write to a study at a time.

```bash
# Once per teammate, in a GPU-backed CARC session:
bash setup_carc_vit.sh

# Submit preflight:
sbatch --time=00:30:00 run_vit_master.slurm --preflight-only
```

After that job finishes with `Preflight passed` and exit code 0, submit training:

```bash
sbatch run_vit_master.slurm
```

The master covers all four k values. The earlier k=1/k=2 notebooks are retained
under [archive/vit/](archive/vit/) as historical references.

## Shared data and results

Code lives in each person's `~/CSCI566-AI-Image-Classifier` clone. Large files live
under `/project2/yzhao010_1531/csci566-genimage`:

- Dataset: `data/extracted/GenImage_Compact/`
- Experiment CSVs: `data/extracted/GenImage_Compact/_experiment_manifests/`
- ViT outputs: `results/vit_master_v1/`
- ViT feature cache: `features/vit_master_v1/`

Keep images, features, checkpoints, and full logs on CARC. Commit code,
documentation, and small result tables to GitHub. Updated datasets and scientific
protocols get separate versions; preserve the current study while it runs.

Current results concern **held-out ADM**. CNN crop/flip augmentation and ViT
blur/JPEG/resize/noise/filtering are separate treatments. The
[team plan](docs/team-plan.md) records limitations and proposed parallel work.
