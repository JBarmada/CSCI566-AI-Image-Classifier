#!/usr/bin/env bash
# Run once per teammate from a CARC OnDemand JupyterLab terminal:
#   bash setup_carc_vit.sh
# Safe to rerun: reuses your environment; never touches the dataset.
# Sources:
# https://www.carc.usc.edu/user-guides/hpc-systems/software/jupyter-kernels
# https://pytorch.org/get-started/previous-versions/#v271
# https://github.com/openai/CLIP

set -eo pipefail
trap 'printf "\nSetup stopped at line %s. Fix the error above, then rerun this script.\n" "$LINENO" >&2' ERR

ENV_NAME="csci566-vit"
CLIP_COMMIT="a1d071733d7111c9c014f024669f959182114e33"

printf '\n=== 1/6: Connect to Conda ===\n'
if type module >/dev/null 2>&1; then
    module load conda
fi
if ! command -v conda >/dev/null 2>&1; then
    printf 'Conda not found. Add conda to Software modules to load in the OnDemand form.\n' >&2
    exit 1
fi
source "$(conda info --base)/etc/profile.d/conda.sh"

printf '\n=== 2/6: Create or reuse your Python environment ===\n'
if conda activate "$ENV_NAME" 2>/dev/null; then
    printf 'Reusing %s\n' "$ENV_NAME"
else
    conda create --name "$ENV_NAME" python=3.11 pip git ipykernel -y
    conda activate "$ENV_NAME"
fi
python - <<'PY'
import sys
print("Python:", sys.executable)
if sys.version_info[:2] != (3, 11):
    raise SystemExit("This setup expects Python 3.11 in csci566-vit. Please use a Python 3.11 environment.")
PY

# Older drivers use CUDA 11.8 wheels; newer drivers can use 12.6/12.8.
# This is a conservative selection, followed by an actual GPU operation below.
# Override if needed: VIT_CUDA_WHEEL=cu128 bash setup_carc_vit.sh
printf '\n=== 3/6: Install PyTorch and notebook packages ===\n'
DRIVER_MAJOR=0
if command -v nvidia-smi >/dev/null 2>&1; then
    DRIVER_OUTPUT="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true)"
    FIRST_DRIVER="${DRIVER_OUTPUT%%$'\n'*}"
    DRIVER_MAJOR="${FIRST_DRIVER%%.*}"
    if [[ ! "$DRIVER_MAJOR" =~ ^[0-9]+$ ]]; then DRIVER_MAJOR=0; fi
fi

if [[ -n "${VIT_CUDA_WHEEL:-}" ]]; then
    CUDA_WHEEL="$VIT_CUDA_WHEEL"
elif (( DRIVER_MAJOR >= 570 )); then
    CUDA_WHEEL=cu128
elif (( DRIVER_MAJOR >= 560 )); then
    CUDA_WHEEL=cu126
else
    CUDA_WHEEL=cu118
fi
case "$CUDA_WHEEL" in
    cu118|cu126|cu128) ;;
    *) printf 'VIT_CUDA_WHEEL must be cu118, cu126 or cu128.\n' >&2; exit 1 ;;
esac
printf 'PyTorch CUDA wheel: %s\n' "$CUDA_WHEEL"
python -m pip install --upgrade pip
# Use an explicit matching pair so all teammates get the same baseline versions.
python -m pip install "torch==2.7.1+${CUDA_WHEEL}" "torchvision==0.22.1+${CUDA_WHEEL}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"
python -m pip install pandas numpy scikit-learn matplotlib pillow tqdm ipykernel \
    ftfy regex "setuptools<81"

printf '\n=== 4/6: Install the original CLIP implementation ===\n'
if ! command -v git >/dev/null 2>&1; then
    conda install --name "$ENV_NAME" git -y
fi
python -m pip install "git+https://github.com/openai/CLIP.git@${CLIP_COMMIT}"
python -m pip check

printf '\n=== 5/6: Check packages, cache CLIP weights and check GPU ===\n'
python - <<'PY'
import importlib
import os
import torch
import clip

for package in ["torchvision", "pandas", "numpy", "sklearn", "matplotlib", "PIL", "tqdm", "ipykernel"]:
    importlib.import_module(package)
print("All notebook imports work.")
print("PyTorch:", torch.__version__)

# Download to your own cache so the notebook can load the model later.
# Does not train or access the project dataset. Skip with VIT_DOWNLOAD_WEIGHTS=0.
if os.environ.get("VIT_DOWNLOAD_WEIGHTS", "1") != "0":
    print("Preparing ViT-L/14 weights in your CLIP cache (first download may take a while)...", flush=True)
    model, _ = clip.load("ViT-L/14", device="cpu", jit=False)
    del model
    print("CLIP weights ready.")

if torch.cuda.is_available():
    # Exercise the GPU, not just the device-discovery call.
    x = torch.ones((16, 16), device="cuda")
    assert (x @ x).sum().item() == 4096
    torch.cuda.synchronize()
    print("GPU check passed:", torch.cuda.get_device_name(0))
else:
    print("Packages installed, but no usable GPU is visible in this session.")
    print("Before training: launch OnDemand JupyterLab with partition=gpu and GPUs=1.")
    print("If this is already a GPU session, check its allocation/driver before training.")
PY

printf '\n=== 6/6: Register your notebook kernel ===\n'
python -m ipykernel install --user --name "$ENV_NAME" --display-name "CSCI566 ViT"
printf '\nSetup complete.\n'
printf 'In your notebook: Kernel -> Change Kernel -> CSCI566 ViT.\n'
printf 'If it is missing, refresh JupyterLab; relaunch the OnDemand app if needed.\n'
printf 'Run the notebook from the top. Its GPU check must pass before training.\n'
printf 'For a new terminal: conda activate %s\n' "$ENV_NAME"
printf 'Each teammate runs this script once under their own CARC account.\n'
