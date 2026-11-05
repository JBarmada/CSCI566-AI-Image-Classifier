#!/usr/bin/env bash
# Submit with --dependency=afterok:ARRAY_JOB_ID from the repository root.
#SBATCH --job-name=vit-final-audit
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=vit-final-audit-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
/home1/barmada/.conda/envs/csci566-vit/bin/python -u code/vit/finalize_benchmark.py
