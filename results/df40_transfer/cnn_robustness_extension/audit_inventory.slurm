#!/bin/bash
#SBATCH --job-name=cnn-robust-audit
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=audit-inventory-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
/home1/barmada/.conda/envs/csci566-vit/bin/python -u audit_inventory.py
