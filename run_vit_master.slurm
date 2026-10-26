#!/usr/bin/env bash
# Submit from this repository: sbatch run_vit_master.slurm
# To change time: sbatch --time=12:00:00 run_vit_master.slurm
#SBATCH --job-name=vit-master
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=vit-master-%j.log
#SBATCH --open-mode=append

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch from the repository folder}"
test -f run_vit_master.py
test -f ViT_Master_Experiments.ipynb

module purge
module load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate csci566-vit
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

printf 'Job %s | host %s | start %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
# No setup/pip/Git pull in the job: run exactly the submitted code/environment.
# Settings come from the master notebook. Optional runner arguments follow sbatch's script name.
python -u run_vit_master.py --recover-stale-locks "$@" &
runner_pid=$!
forward_stop() {
    printf '\nScheduler stopping job; preserve the last complete epoch for resume.\n'
    kill -USR1 "$runner_pid" 2>/dev/null || true
}
trap forward_stop USR1 TERM
set +e
wait "$runner_pid"
status=$?
# wait can itself be interrupted by the scheduler signal before Python exits.
if kill -0 "$runner_pid" 2>/dev/null; then
    wait "$runner_pid"
    status=$?
fi
printf 'Runner exit code: %s | finish %s\n' "$status" "$(date -Is)"
exit "$status"
