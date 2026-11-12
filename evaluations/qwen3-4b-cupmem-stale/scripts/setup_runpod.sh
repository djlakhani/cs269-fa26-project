#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_DIR="${CUPMEM_WORKSPACE_DIR:-/workspace}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable" >&2
  exit 1
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "RTX A6000"; then
  echo "Expected an RTX A6000" >&2
  nvidia-smi --query-gpu=name --format=csv,noheader >&2
  exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
  echo "AWS CLI is required" >&2
  exit 1
fi

AVAILABLE_GIB="$(df -Pk "${WORKSPACE_DIR}" | awk 'NR==2 {printf "%.0f", $4/1024/1024}')"
if (( AVAILABLE_GIB < 25 )); then
  echo "At least 25 GiB free is required under ${WORKSPACE_DIR}" >&2
  df -h "${WORKSPACE_DIR}" >&2
  exit 1
fi

UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ ! -x "${UV_BIN}" ]]; then
  echo "uv installation failed" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
"${UV_BIN}" python install 3.12
"${UV_BIN}" sync --locked --python 3.12
"${UV_BIN}" run --locked python - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"vram_gib={torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}")
if torch.cuda.get_device_capability(0)[0] < 8:
    raise SystemExit("An Ampere-or-newer GPU is required")
PY

echo "RunPod environment ready"
