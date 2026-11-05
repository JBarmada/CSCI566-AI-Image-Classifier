#!/usr/bin/env bash
#SBATCH --job-name=benchmark-audit
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=benchmark-audit-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
python_bin=/home1/barmada/.conda/envs/csci566-vit/bin/python
data_root=/project2/yzhao010_1531/csci566-genimage/data/extracted/GenImage_Compact
tools_dir=/scratch1/barmada/vit-benchmark-tools/code/benchmark
"$python_bin" "$tools_dir/install_benchmark.py" --archive /scratch1/barmada/vit-benchmark-release.zip --data-root "$data_root" --prefix _release_v2
"$python_bin" "$tools_dir/install_benchmark.py" --archive /scratch1/barmada/vit-benchmark-manifests.zip --data-root "$data_root" --prefix benchmark
"$python_bin" "$tools_dir/audit_benchmark.py" --benchmark "$data_root/benchmark" --data-root "$data_root" --report /project2/yzhao010_1531/csci566-genimage/results/benchmark_acceptance.json
