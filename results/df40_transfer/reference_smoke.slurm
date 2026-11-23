#!/usr/bin/env bash
#SBATCH --job-name=df40-reference
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:10:00
#SBATCH --output=reference-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home1/barmada/.conda/envs/csci566-vit/bin/python -u smoke_inference.py \
  --reference-only --output "reference_${SLURM_JOB_ID}"
