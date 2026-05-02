#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="python3.11"
VENV_PATH="$HOME/rabitq_env"
CUDA_HOME_PATH="/usr/local/cuda-12.1"
TORCH_CUDA_ARCH_LIST_VALUE="8.0"
CUDA_PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu121"
NIAH_REPO_URL="https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git"
NIAH_DIR="${REPO_DIR}/LLMTest_NeedleInAHaystack"

log() {
  printf '\n[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}

append_bashrc_line() {
  local line="$1"
  local bashrc="$HOME/.bashrc"
  if [ ! -f "$bashrc" ]; then
    touch "$bashrc"
  fi
  if ! grep -Fqx "$line" "$bashrc"; then
    printf '%s\n' "$line" >> "$bashrc"
  fi
}

log "Repo dir: ${REPO_DIR}"
log "Virtualenv: ${VENV_PATH}"
log "Python: ${PYTHON_BIN}"
log "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST_VALUE}"

require_command sudo
require_command apt
require_command wget
require_command git

log "Installing system packages"
sudo apt update
sudo apt install -y \
  build-essential \
  g++ \
  g++-12 \
  git \
  wget \
  python3.11 \
  python3.11-dev \
  python3.11-venv

log "Installing CUDA 12.1 toolkit"
tmp_deb="/tmp/cuda-keyring_1.0-1_all.deb"
wget -O "${tmp_deb}" \
  https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i "${tmp_deb}"
sudo apt update
sudo apt install -y cuda-toolkit-12-1

export PATH="${CUDA_HOME_PATH}/bin:${PATH}"
export CUDA_HOME="${CUDA_HOME_PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST_VALUE}"
log "Writing CUDA environment variables to ~/.bashrc"
append_bashrc_line "export PATH=${CUDA_HOME_PATH}/bin:\$PATH"
append_bashrc_line "export CUDA_HOME=${CUDA_HOME_PATH}"
append_bashrc_line "export LD_LIBRARY_PATH=${CUDA_HOME_PATH}/lib64:\$LD_LIBRARY_PATH"

require_command "${PYTHON_BIN}"

if [ ! -d "${VENV_PATH}" ]; then
  log "Creating virtualenv"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
fi

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

log "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip "setuptools<71" wheel

log "Installing PyTorch CUDA 12.1"
python -m pip install torch==2.4.1 --index-url "${CUDA_PYTORCH_INDEX_URL}"

log "Installing core dependencies"
python -m pip install ninja numpy==1.26.4
python -m pip install tqdm rouge fuzzywuzzy python-Levenshtein jieba pytest
python -m pip install sentencepiece safetensors huggingface_hub tokenizers

log "Installing flash-attn"
python -m pip install flash-attn==2.6.3 --no-build-isolation

if [ -d "${NIAH_DIR}/.git" ]; then
  log "LLMTest_NeedleInAHaystack already exists, skipping clone"
elif [ -d "${NIAH_DIR}" ]; then
  log "LLMTest_NeedleInAHaystack directory exists without .git, skipping clone"
else
  log "Cloning LLMTest_NeedleInAHaystack"
  git clone "${NIAH_REPO_URL}" "${NIAH_DIR}"
fi

log "Applying NIAH sorted glob patch"
git -C "${NIAH_DIR}" apply "${REPO_DIR}/eval/niah_sorted_glob.patch" 2>/dev/null || \
  log "Patch already applied or not needed, skipping"

log "Building llm_rabitq extension"
python -m pip install -e "${REPO_DIR}/llm_rabitq" --no-build-isolation

log "Running llm_rabitq tests"
(
  cd "${REPO_DIR}/llm_rabitq"
  python -m pytest test/ -v
)

log "Verifying environment"
python - <<'PY'
import importlib
import os
import subprocess
import sys

def print_version(name):
    module = importlib.import_module(name)
    print(f"{name}: {getattr(module, '__version__', 'unknown')}")

print(f"python: {sys.version.split()[0]}")
print(f"venv:   {sys.prefix}")

try:
    import torch
    print(f"torch:  {torch.__version__}")
    print(f"cuda:   {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu:    {torch.cuda.get_device_name(0)}")
    else:
        print("gpu:    not visible to torch")
except Exception as exc:
    print(f"torch import failed: {exc}")

for mod in ["transformers", "accelerate", "datasets", "flash_attn", "rabitq"]:
    try:
        print_version(mod)
    except Exception as exc:
        print(f"{mod}: import failed ({exc})")

for cmd in [["nvcc", "--version"], ["nvidia-smi"]]:
    try:
        print(f"\n$ {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print(f"{cmd[0]} not found")
PY

log "Installing Python requirements"
python -m pip install -r "${REPO_DIR}/requirements.txt"

cat <<EOF

Setup complete.

Activate the environment with:
  source "${VENV_PATH}/bin/activate"

Optional next steps:
  huggingface-cli login
  export NIAH_EVALUATOR_API_KEY=...

EOF
