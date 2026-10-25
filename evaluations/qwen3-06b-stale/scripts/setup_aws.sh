#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Use the AWS NVIDIA-driver Deep Learning AMI." >&2
  exit 1
fi
if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep -q "NVIDIA L4"; then
  echo "Expected an NVIDIA L4 GPU on g6.xlarge." >&2
  nvidia-smi --query-gpu=name --format=csv,noheader >&2
  exit 1
fi
if ! command -v aws >/dev/null 2>&1; then
  echo "AWS CLI is required. Install awscli before continuing." >&2
  exit 1
fi

UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  if [[ -x "${HOME}/.local/bin/uv" ]]; then
    UV_BIN="${HOME}/.local/bin/uv"
  else
    UV_BIN="$(command -v uv || true)"
  fi
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "uv installation completed, but the uv executable could not be located." >&2
  exit 1
fi

cd "${PROJECT_DIR}"
"${UV_BIN}" sync --locked

"${UV_BIN}" run --locked python - <<'PY'
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

print(f"torch={torch.__version__}")
print(f"cuda={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable to the locked PyTorch environment")
print(f"gpu={torch.cuda.get_device_name(0)}")
print(f"capability={torch.cuda.get_device_capability(0)}")
print(f"vram_gib={torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}")
if torch.cuda.get_device_capability(0)[0] < 8:
    raise SystemExit("Flash SDPA requires an Ampere-or-newer GPU")

q = torch.zeros((1, 16, 16, 128), dtype=torch.float16, device="cuda")
k = torch.zeros((1, 8, 16, 128), dtype=torch.float16, device="cuda")
v = torch.zeros_like(k)
with sdpa_kernel(SDPBackend.FLASH_ATTENTION), torch.inference_mode():
    torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=True, enable_gqa=True
    )
print("flash_sdpa_gqa_preflight=passed")
PY

echo "AWS environment is ready. Run scripts/run_aws.sh next."
