#!/bin/bash
#SBATCH --job-name=cnn-robust-ext
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --array=0-6%2
#SBATCH --output=cnn-robust-%A_%a.log
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
/home1/barmada/.conda/envs/csci566-vit/bin/python -u run_notebook.py "$SLURM_ARRAY_TASK_ID"
