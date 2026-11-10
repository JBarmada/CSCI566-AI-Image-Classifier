#!/usr/bin/env bash
# Submit from anywhere inside this repository.
# Optional runner arguments follow the script name.
#SBATCH --job-name=cnn-master
#SBATCH --account=yzhao010_1531
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --signal=B:USR1@120
#SBATCH --output=cnn-master-%j.log
#SBATCH --open-mode=append
set -eo pipefail

repo_dir="$(cd "${SLURM_SUBMIT_DIR:?Submit from inside the repository checkout}" && pwd -P)"
while [[ ! -f "$repo_dir/cnn.ipynb" || ! -f "$repo_dir/code/cnn/run_cnn_master.py" ]]; do
    if [[ "$repo_dir" == / ]]; then
        printf 'Cannot find the CNN repository above %s.\n' "$SLURM_SUBMIT_DIR" >&2
        exit 1
    fi
    repo_dir="$(dirname "$repo_dir")"
done
cd "$repo_dir"

cnn_env="${CNN_ENV_PREFIX:-$HOME/.conda/envs/csci566-vit}"
cnn_python="${CNN_PYTHON:-$cnn_env/bin/python}"
[[ -x "$cnn_python" ]] || { echo "Missing Python: $cnn_python" >&2; exit 1; }
export PATH="$(dirname "$cnn_python"):$PATH"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1

printf 'Job %s | host %s | start %s\n' "$SLURM_JOB_ID" "$(hostname)" "$(date -Is)"
"$cnn_python" -u code/cnn/run_cnn_master.py --recover-stale-locks "$@" &
runner_pid=$!
forward_stop() {
    printf '\nScheduler stopping job; preserving the last complete epoch.\n'
    kill -USR1 "$runner_pid" 2>/dev/null || true
}
trap forward_stop USR1 TERM
set +e
wait "$runner_pid"
status=$?
if kill -0 "$runner_pid" 2>/dev/null; then
    wait "$runner_pid"
    status=$?
fi
printf 'Runner exit code: %s | finish %s\n' "$status" "$(date -Is)"
exit "$status"
