#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 BASE_RUN_ID KEY_PAIR_RUN_ID [smoke|full] [DESCRIPTION]" >&2
  exit 2
fi
BASE_RUN_ID="$1"
KEY_RUN_ID="$2"
MODE="${3:-smoke}"
DESCRIPTION="${4:-Qwen3-4B STALE four-view evaluation}"
PATTERN='^[A-Za-z0-9][A-Za-z0-9._-]*$'
if [[ ! "${BASE_RUN_ID}" =~ ${PATTERN} || ! "${KEY_RUN_ID}" =~ ${PATTERN} ]]; then
  echo "Run IDs contain an invalid character." >&2
  exit 2
fi
if [[ "${BASE_RUN_ID}" == "${KEY_RUN_ID}" ]]; then
  echo "The baseline and key-pair run IDs must differ." >&2
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
BASE_ROOT="${ARTIFACT_BASE}/${BASE_RUN_ID}"
KEY_ROOT="${ARTIFACT_BASE}/${KEY_RUN_ID}"
BASE_S3="${S3_BASE_URI%/}/runs/${BASE_RUN_ID}"
KEY_S3="${S3_BASE_URI%/}/runs/${KEY_RUN_ID}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION}}"
BUCKET_PATH="${S3_BASE_URI#s3://}"
BUCKET="${BUCKET_PATH%%/*}"

mkdir -p "${BASE_ROOT}" "${KEY_ROOT}" "${CACHE_DIR}"
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
aws s3 sync "${BASE_S3}" "${BASE_ROOT}" --exclude ".*" --exclude "*/.*" --only-show-errors
aws s3 sync "${KEY_S3}" "${KEY_ROOT}" --exclude ".*" --exclude "*/.*" --only-show-errors

sync_all() {
  aws s3 sync "${BASE_ROOT}" "${BASE_S3}" --exclude ".*" --exclude "*/.*" --only-show-errors
  aws s3 sync "${KEY_ROOT}" "${KEY_S3}" --exclude ".*" --exclude "*/.*" --only-show-errors
}
SYNC_PID=""
cleanup() {
  if [[ -n "${SYNC_PID}" ]]; then
    pkill -TERM -P "${SYNC_PID}" 2>/dev/null || true
    kill "${SYNC_PID}" 2>/dev/null || true
    wait "${SYNC_PID}" 2>/dev/null || true
    SYNC_PID=""
  fi
  sync_all
}
trap cleanup EXIT INT TERM
(
  while true; do
    sleep "${SYNC_INTERVAL}"
    sync_all
  done
) &
SYNC_PID=$!

PREPARE_ARGS=(
  prepare
  --artifact-root "${BASE_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --samples-per-type 25
  --run-id "${BASE_RUN_ID}"
  --description "${DESCRIPTION}: truncated and evidence-only"
)
"${UV_BIN}" run --locked qwen3-memory-eval "${PREPARE_ARGS[@]}"

BASE_ARGS=(
  --artifact-root "${BASE_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --view all
  --attention-backend flash_sdpa
  --resume
)
if [[ "${MODE}" == "smoke" ]]; then BASE_ARGS+=(--max-scenarios 1); fi
"${UV_BIN}" run --locked qwen3-memory-eval run "${BASE_ARGS[@]}"
"${UV_BIN}" run --locked qwen3-memory-eval report --artifact-root "${BASE_ROOT}" --view all
sync_all

KEY_PREPARE_ARGS=(
  prepare-key-pairs
  --source-scenarios "${BASE_ROOT}/scenarios.json"
  --artifact-root "${KEY_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --run-id "${KEY_RUN_ID}"
  --description "${DESCRIPTION}: chronological and reversed key pairs"
  --source-reference "${BASE_S3}/scenarios.json"
)
"${UV_BIN}" run --locked qwen3-memory-eval "${KEY_PREPARE_ARGS[@]}"

KEY_ARGS=(
  --artifact-root "${KEY_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --view all
  --attention-backend flash_sdpa
  --resume
)
if [[ "${MODE}" == "smoke" ]]; then KEY_ARGS+=(--max-scenarios 1); fi
"${UV_BIN}" run --locked qwen3-memory-eval run "${KEY_ARGS[@]}"
"${UV_BIN}" run --locked qwen3-memory-eval report --artifact-root "${KEY_ROOT}" --view all
sync_all

echo "STALE baseline: ${BASE_S3}"
echo "STALE key pairs: ${KEY_S3}"
