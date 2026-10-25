# CSCI566-AI-Image-Classifier
CSCI 566 — AI-Generated Image Detection

This repository contains the code, experiment configurations, and evaluation results for our USC CSCI 566 Deep Learning research project.

Our project investigates whether training AI-generated image detectors on images from multiple generators improves generalization to previously unseen generators.

We use the GenImage dataset and compare models trained on different numbers of generators ((k \in {1, 2, 4, 7})), while keeping the total number of training images constant.

1. Storage Architecture

We use two separate locations on USC CARC:

- "/home1" — GitHub repository and code: Each teammate maintains their own clone of the repository, allowing independent development and version control.
- "/project2" — Shared datasets and experiment outputs: All teammates access the same images, CLIP features, and model checkpoints without duplicating large files.

Personal GitHub Repository ("/home1")

Each teammate clones the repository into their own home directory:

/home1/<USC_NETID>/CSCI566-AI-Image-Classifier/
├── code/                  # Python scripts and notebooks
├── configs/               # Experiment configurations
├── splits/                # Dataset split CSVs (tracked when appropriate)
├── results/               # Small evaluation results and metrics
├── data/                  # Symlink to shared CARC data
├── features/              # Symlink to shared CLIP features
├── checkpoints/           # Symlink to shared model checkpoints
├── setup_carc.sh          # Creates shared storage symlinks
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md

The "data/", "features/", and "checkpoints/" entries are symbolic links, not actual copies of the files.

Changes to code should be committed and pushed to GitHub.

Shared CARC Storage ("/project2")

Our shared project directory is:

/project2/yzhao010_1531/csci566-genimage/
├── data/
│   ├── genimagezips/       # Original transferred ZIP archives
│   └── extracted/
│       └── GenImage_Compact/
│           ├── ADM/
│           ├── BigGAN/
│           ├── GLIDE/
│           ├── Midjourney/
│           ├── VQDM/
│           ├── wukong/
│           ├── stable_diffusion_v_1_4/
│           ├── stable_diffusion_v_1_5/
│           └── _experiment_manifests/
├── code/                   # Optional shared working files
├── splits/                 # Shared dataset manifests
├── configs/                # Shared experiment configurations
├── features/               # Extracted CLIP embeddings
├── checkpoints/            # Model weights and checkpoints
└── results/                # Full training and evaluation outputs

The extracted dataset layout shown above assumes all four transferred ZIP archives have been successfully extracted.

Files under "/project2" are shared directly among authorized CARC project members. They do not need to be copied into each teammate's home directory.

2. Getting Started

Step 1 — Connect to CARC

Connect using your USC account:

ssh <USC_NETID>@discovery.usc.edu

You must have permission to access the shared project directory.

Step 2 — Clone the GitHub Repository

From your CARC home directory:

cd ~
git clone git@github.com:JBarmada/CSCI566-AI-Image-Classifier.git
cd CSCI566-AI-Image-Classifier

If you already cloned the repository, run "git pull" to get the latest changes.

Step 3 — Connect Shared Storage

Run:

./setup_carc.sh

This script creates three symbolic links:

Local repository path| Shared CARC destination
"data/"| "/project2/yzhao010_1531/csci566-genimage/data/"
"features/"| "/project2/yzhao010_1531/csci566-genimage/features/"
"checkpoints/"| "/project2/yzhao010_1531/csci566-genimage/checkpoints/"

Verify them using:

ls -l data features checkpoints

You only need to run the setup script once per clone. It does not download or duplicate the dataset.

3. Working with the Dataset

Our selected GenImage dataset is stored in:

/project2/yzhao010_1531/csci566-genimage/data/extracted/GenImage_Compact/

When running scripts from the repository root, the equivalent path is:

data/extracted/GenImage_Compact/

For example:

from pathlib import Path

DATA_ROOT = Path("data/extracted/GenImage_Compact")

MANIFEST_DIR = (
    DATA_ROOT
    / "_experiment_manifests"
    / "holdout_ADM_k1_rep0"
)

train_csv = MANIFEST_DIR / "train.csv"

Relative paths assume the script or notebook is run with the repository root as its working directory.

The dataset is shared, so changes to files inside "data/" affect everyone. Avoid modifying or deleting original images or manifests without coordinating with the team.

4. Running Experiments

We use CARC's GPU resources to train and evaluate our detectors.

Experiments include:

- Frozen CLIP ViT feature extraction and lightweight classifier training.
- ResNet-50 training on selected generator combinations.
- Comparisons across different training-generator subsets.
- Evaluation on generators excluded from training.

Use GPU-backed CARC OnDemand JupyterLab sessions or Slurm jobs for GPU-intensive tasks. Do not run training on login nodes.

Large experiment outputs should be saved to shared storage:

from pathlib import Path

SHARED_ROOT = Path(
    "/project2/yzhao010_1531/csci566-genimage"
)

FEATURES_DIR = SHARED_ROOT / "features"
CHECKPOINT_DIR = SHARED_ROOT / "checkpoints"
RESULTS_DIR = SHARED_ROOT / "results"

Keep each experiment's outputs in a uniquely named directory to avoid overwriting teammates' results.

5. GitHub Workflow

Each teammate should work from their own repository clone.

Before starting work:

git pull

After making changes:

git status
git add code/ configs/ splits/ results/
git commit -m "Describe changes"
git push

For larger changes, use separate Git branches and pull requests to avoid conflicts.

What belongs on GitHub?

Commit to GitHub| Keep on CARC
Python scripts and notebooks| Raw GenImage images
Experiment configurations| ZIP archives
Small dataset split CSVs| Large CLIP embeddings
Requirements and documentation| Model checkpoints
Final metrics and small result tables| Temporary training outputs

Do not commit passwords, access tokens, SSH keys, or private configuration files.

Important: GitHub only backs up committed and pushed files. Our large dataset and checkpoints are not backed up by GitHub.

The original selected dataset is also retained on Google Drive as a separate backup.

6. Team Collaboration Guidelines

1. Edit code in your own Git clone, not directly in another teammate's home directory.
2. Use "/project2" for shared data and large outputs, so we avoid duplicating files.
3. Commit and push important code changes regularly.
4. Do not overwrite shared model checkpoints or experiment outputs. Use unique run names.
5. Keep experiment configurations and results reproducible, including seeds, generator subsets, and preprocessing settings.
6. Coordinate changes to shared datasets and manifests, since they affect every teammate.

7. Useful CARC Commands

Check storage quotas:

myquota

Check available computing allocations:

myaccount

Check running and queued jobs:

squeue -u "$USER"

Check the size of our shared dataset:

du -sh /project2/yzhao010_1531/csci566-genimage/data

Verify symbolic links:

ls -l data features checkpoints

---

Repository: https://github.com/JBarmada/CSCI566-AI-Image-Classifier

Shared CARC directory: "/project2/yzhao010_1531/csci566-genimage"

Main principle: GitHub manages our code and reproducibility; CARC "/project2" stores the shared data and large experiment artifacts.