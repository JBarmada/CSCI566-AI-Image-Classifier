#!/usr/bin/env bash
# Submit with --dependency=afterok:ARRAY_JOB_ID from inside the repository.
#SBATCH --job-name=cnn-final-audit
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=cnn-final-audit-%j.log
set -euo pipefail

repo="${SLURM_SUBMIT_DIR:?Submit from inside the repository}"
while [[ ! -f "$repo/code/cnn/finalize_benchmark.py" ]]; do
    [[ "$repo" != / ]] || { echo 'Cannot locate repository'; exit 1; }
    repo="$(dirname "$repo")"
done
cd "$repo"
cnn_env="${CNN_ENV_PREFIX:-$HOME/.conda/envs/csci566-vit}"
cnn_python="${CNN_PYTHON:-$cnn_env/bin/python}"
[[ -x "$cnn_python" ]] || { echo "Missing Python: $cnn_python" >&2; exit 1; }
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
export MPLBACKEND=Agg
"$cnn_python" -u code/cnn/finalize_benchmark.py
