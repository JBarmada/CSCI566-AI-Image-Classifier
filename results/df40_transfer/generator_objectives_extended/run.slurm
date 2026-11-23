#!/bin/bash
#SBATCH --job-name=gen-objectives-ext
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --array=0-6
#SBATCH --output=objectives-ext-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
holdouts=(BigGAN glide Midjourney stable_diffusion_v_1_4 stable_diffusion_v_1_5 VQDM wukong)
export GEN_OBJECTIVE_HOLDOUT="${holdouts[$SLURM_ARRAY_TASK_ID]}"
python_bin=/home1/barmada/.conda/envs/csci566-vit/bin/python
out="results_${SLURM_ARRAY_JOB_ID}_${GEN_OBJECTIVE_HOLDOUT}"
"$python_bin" -u run.py "$out"
"$python_bin" -u replay.py "$out" "$out/REPLAY.json"
