#!/bin/bash
#SBATCH --job-name=df40-cpu-replay
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=cpu-reference-%j.log
set -euo pipefail
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
/home1/barmada/.conda/envs/csci566-vit/bin/python -u cpu_reference_replay.py "$1"
