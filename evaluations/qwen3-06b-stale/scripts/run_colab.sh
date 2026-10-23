#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 /content/drive/MyDrive/PATH [smoke|full]" >&2
  exit 2
fi

ARTIFACT_ROOT="$1"
MODE="${2:-full}"
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Mode must be 'smoke' or 'full'." >&2
  exit 2
fi
if [[ ! -d /content/drive/MyDrive ]]; then
  echo "Google Drive is not mounted at /content/drive/MyDrive." >&2
  exit 2
fi

python - "${ARTIFACT_ROOT}" <<'PY'
from pathlib import Path
import sys

drive = Path("/content/drive/MyDrive").resolve()
target = Path(sys.argv[1]).expanduser().resolve()
if target != drive and drive not in target.parents:
    raise SystemExit(f"Artifact root must be inside mounted Google Drive: {drive}")
PY

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CACHE_DIR="${STALE_CACHE_DIR:-${ARTIFACT_ROOT}/hf-cache}"
export PYTHONPATH="${PROJECT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${CACHE_DIR}"

mkdir -p "${ARTIFACT_ROOT}" "${CACHE_DIR}"

python -m stale_qwen_eval prepare \
  --artifact-root "${ARTIFACT_ROOT}" \
  --cache-dir "${CACHE_DIR}"

if [[ "${MODE}" == "smoke" ]]; then
  python -m stale_qwen_eval run \
    --artifact-root "${ARTIFACT_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --resume \
    --max-queries 1
else
  python -m stale_qwen_eval run \
    --artifact-root "${ARTIFACT_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --resume
fi

python -m stale_qwen_eval report --artifact-root "${ARTIFACT_ROOT}"
echo "Artifacts are saved under ${ARTIFACT_ROOT}"

