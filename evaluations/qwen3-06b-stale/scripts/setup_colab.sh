#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

python -m pip install --upgrade pip
python -m pip install -e "${PROJECT_DIR}"

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("No CUDA GPU detected. In Colab, select a GPU runtime before continuing.")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "Colab environment is ready."

