#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 RUN_ID [smoke|full] [DESCRIPTION]" >&2
  exit 2
fi
RUN_ID="$1"
MODE="${2:-smoke}"
DESCRIPTION="${3:-Qwen3-4B MEME Cascade and Absence evaluation}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Run ID contains an invalid character." >&2
  exit 2
fi
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Mode must be smoke or full." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACT_BASE="${QWEN4B_ARTIFACT_BASE:-/home/ubuntu/qwen3-4b-artifacts}"
CACHE_DIR="${QWEN4B_CACHE_DIR:-/home/ubuntu/huggingface-cache}"
S3_BASE_URI="${QWEN4B_S3_BASE_URI:-s3://stale-qwen-evals-589498925064-us-east-1-an/stale-qwen-evals}"
SYNC_INTERVAL="${QWEN4B_SYNC_INTERVAL_SECONDS:-60}"
ARTIFACT_ROOT="${ARTIFACT_BASE}/${RUN_ID}"
S3_RUN_URI="${S3_BASE_URI%/}/runs/${RUN_ID}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION}}"
BUCKET_PATH="${S3_BASE_URI#s3://}"
BUCKET="${BUCKET_PATH%%/*}"

mkdir -p "${ARTIFACT_ROOT}" "${CACHE_DIR}"
cd "${PROJECT_DIR}"
UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "uv is unavailable; run bash scripts/setup_aws.sh first." >&2
  exit 1
fi
aws s3api head-bucket --bucket "${BUCKET}"
aws s3 sync "${S3_RUN_URI}" "${ARTIFACT_ROOT}" --exclude ".*" --exclude "*/.*" --only-show-errors

sync_artifacts() {
  aws s3 sync "${ARTIFACT_ROOT}" "${S3_RUN_URI}" --exclude ".*" --exclude "*/.*" --only-show-errors
}
SYNC_PID=""
cleanup() {
  if [[ -n "${SYNC_PID}" ]]; then
    pkill -TERM -P "${SYNC_PID}" 2>/dev/null || true
    kill "${SYNC_PID}" 2>/dev/null || true
    wait "${SYNC_PID}" 2>/dev/null || true
    SYNC_PID=""
  fi
  sync_artifacts
}
trap cleanup EXIT INT TERM
(
  while true; do
    sleep "${SYNC_INTERVAL}"
    sync_artifacts
  done
) &
SYNC_PID=$!

PREPARE_ARGS=(
  meme-prepare
  --artifact-root "${ARTIFACT_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --run-id "${RUN_ID}"
  --description "${DESCRIPTION}"
  --filler-input-target 29000
)
"${UV_BIN}" run --locked qwen3-memory-eval "${PREPARE_ARGS[@]}"
sync_artifacts

RUN_ARGS=(
  --artifact-root "${ARTIFACT_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --view all
  --attention-backend flash_sdpa
  --resume
)
if [[ "${MODE}" == "smoke" ]]; then RUN_ARGS+=(--max-episodes 1); fi
"${UV_BIN}" run --locked qwen3-memory-eval meme-run "${RUN_ARGS[@]}"
"${UV_BIN}" run --locked qwen3-memory-eval meme-report --artifact-root "${ARTIFACT_ROOT}" --view all
sync_artifacts

echo "MEME artifacts: ${S3_RUN_URI}"
