#!/bin/bash
#SBATCH --job-name=vlm-setup
#SBATCH --account=yzhao010_1531
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --output=setup-%j.log
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=/home1/barmada/.conda/envs/csci566-vit/bin/python
"$PY" -m pip install --no-deps --target "$PWD/dependencies" \
  transformers==4.51.3 accelerate==1.6.0 tokenizers==0.21.1 \
  safetensors==0.5.3 huggingface-hub==0.30.2 qwen-vl-utils==0.0.10 \
  requests==2.32.3 urllib3==2.4.0 idna==3.10 certifi==2025.1.31 \
  charset-normalizer==3.4.1 pyyaml==6.0.2 regex==2024.11.6 \
  packaging==24.2 filelock==3.18.0 fsspec==2025.3.2 psutil==7.0.0
export PYTHONPATH="$PWD/dependencies"
export HF_HUB_DISABLE_XET=1
"$PY" -c 'from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="Qwen/Qwen2.5-VL-3B-Instruct", revision="66285546d2b821cf421d4f5eb2576359d3770cd3", local_dir="model", max_workers=3), flush=True)'
