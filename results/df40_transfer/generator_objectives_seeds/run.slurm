#!/bin/bash
#SBATCH --job-name=gen-obj-seeds
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --array=0-15%8
#SBATCH --output=objectives-seed-%A_%a.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
holdouts=(ADM BigGAN glide Midjourney stable_diffusion_v_1_4 stable_diffusion_v_1_5 VQDM wukong)
seeds=(29 43)
export GEN_OBJECTIVE_HOLDOUT="${holdouts[$((SLURM_ARRAY_TASK_ID % 8))]}"
export GEN_OBJECTIVE_SEED="${seeds[$((SLURM_ARRAY_TASK_ID / 8))]}"
python_bin=/home1/barmada/.conda/envs/csci566-vit/bin/python
out="results_${SLURM_ARRAY_JOB_ID}_${GEN_OBJECTIVE_HOLDOUT}_seed${GEN_OBJECTIVE_SEED}"
"$python_bin" -u run.py "$out"
"$python_bin" -u replay.py "$out" "$out/REPLAY.json"
