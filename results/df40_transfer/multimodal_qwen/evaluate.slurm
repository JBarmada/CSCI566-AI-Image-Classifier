#!/bin/bash
#SBATCH --job-name=vlm-genimage
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=eval-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="$PWD/dependencies" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
/home1/barmada/.conda/envs/csci566-vit/bin/python -u evaluate.py "results_${SLURM_JOB_ID}"
