#!/usr/bin/env bash
# Submit from anywhere inside this checkout: sbatch code/cnn/run_benchmark.slurm
#SBATCH --job-name=cnn-benchmark
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-7%2
#SBATCH --signal=B:USR1@120
#SBATCH --output=cnn-benchmark-%A_%a.log
#SBATCH --open-mode=append
set -euo pipefail

repo="${SLURM_SUBMIT_DIR:?Submit from inside the repository}"
while [[ ! -f "$repo/code/cnn/run_cnn_master.slurm" ]]; do
    [[ "$repo" != / ]] || { echo 'Cannot locate repository'; exit 1; }
    repo="$(dirname "$repo")"
done
holdouts=(ADM BigGAN glide Midjourney stable_diffusion_v_1_4 stable_diffusion_v_1_5 VQDM wukong)
source "$repo/code/cnn/run_cnn_master.slurm" --holdout "${holdouts[$SLURM_ARRAY_TASK_ID]}" "$@"
