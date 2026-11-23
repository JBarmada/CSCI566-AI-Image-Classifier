#!/bin/bash
#SBATCH --job-name=df40-ff-zero
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=p100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=ff-zero-%j.log
set -euo pipefail
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
/home1/barmada/.conda/envs/csci566-vit/bin/python -u ff_zeroshot.py "$1" "$2"
