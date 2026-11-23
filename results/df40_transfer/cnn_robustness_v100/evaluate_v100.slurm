#!/bin/bash
#SBATCH --job-name=cnn-robust-v100
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=v100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --array=3-6%2
#SBATCH --output=robust-v100-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir "task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "task_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cp ../CNN_K7_Robustness_Evaluation.ipynb ../run_notebook.py .
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
/home1/barmada/.conda/envs/csci566-vit/bin/python -c 'import torch,torchvision,PIL; print("versions", torch.__version__, torchvision.__version__, PIL.__version__, "cudnn", torch.backends.cudnn.version())'
/home1/barmada/.conda/envs/csci566-vit/bin/python -u run_notebook.py "$SLURM_ARRAY_TASK_ID"
