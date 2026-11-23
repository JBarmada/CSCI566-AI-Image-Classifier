#!/bin/bash
#SBATCH --job-name=cnn-replay-control
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=control-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir "run_${SLURM_JOB_ID}"
cd "run_${SLURM_JOB_ID}"
cp ../cnn.ipynb ../replay_original.py .
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
/home1/barmada/.conda/envs/csci566-vit/bin/python -c 'import torch,torchvision,PIL; print("versions", torch.__version__, torchvision.__version__, PIL.__version__, "cudnn", torch.backends.cudnn.version())'
/home1/barmada/.conda/envs/csci566-vit/bin/python -u replay_original.py
