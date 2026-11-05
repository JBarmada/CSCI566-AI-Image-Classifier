#!/usr/bin/env bash
#SBATCH --job-name=vit-smoke
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=vit-smoke-%j.log
set -euo pipefail
repo="${SLURM_SUBMIT_DIR:?Submit from inside the repository}"
while [[ ! -f "$repo/code/vit/run_vit_master.py" ]]; do
    [[ "$repo" != / ]] || { echo 'Cannot locate repository'; exit 1; }
    repo="$(dirname "$repo")"
done
cd "$repo"
module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csci566-vit
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="" python -m unittest discover -s tests -v
python -u code/vit/run_vit_master.py --preflight-only --study-name vit_benchmark_preflight
python -u code/vit/smoke_check.py
