#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 RUN_ID [smoke|full] [DESCRIPTION]" >&2
  exit 2
fi

RUN_ID="$1"
MODE="${2:-smoke}"
DESCRIPTION="${3:-}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "RUN_ID may contain only letters, numbers, period, underscore, and hyphen." >&2
  exit 2
fi
if [[ "${MODE}" != "smoke" && "${MODE}" != "full" ]]; then
  echo "Mode must be 'smoke' or 'full'." >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACT_BASE="${STALE_ARTIFACT_BASE:-/home/ubuntu/stale-artifacts}"
CACHE_DIR="${STALE_CACHE_DIR:-/home/ubuntu/huggingface-cache}"
S3_BASE_URI="${STALE_S3_BASE_URI:-s3://stale-qwen-evals-589498925064-us-east-1-an/stale-qwen-evals}"
ARTIFACT_ROOT="${ARTIFACT_BASE}/${RUN_ID}"
S3_RUN_URI="${S3_BASE_URI%/}/runs/${RUN_ID}"
SYNC_INTERVAL_SECONDS="${STALE_SYNC_INTERVAL_SECONDS:-60}"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION}}"
S3_BUCKET_AND_PREFIX="${S3_BASE_URI#s3://}"
S3_BUCKET="${S3_BUCKET_AND_PREFIX%%/*}"

if [[ "${S3_BASE_URI}" != s3://* || -z "${S3_BUCKET}" ]]; then
  echo "STALE_S3_BASE_URI must be an s3:// URI." >&2
  exit 2
fi

mkdir -p "${ARTIFACT_ROOT}" "${CACHE_DIR}"
cd "${PROJECT_DIR}"

aws s3api head-bucket --bucket "${S3_BUCKET}"
# Resume cleanly on a replacement instance. An empty prefix is harmless.
aws s3 sync "${S3_RUN_URI}" "${ARTIFACT_ROOT}" \
  --exclude ".*" --exclude "*/.*" --only-show-errors

sync_artifacts() {
  aws s3 sync "${ARTIFACT_ROOT}" "${S3_RUN_URI}" \
    --exclude ".*" --exclude "*/.*" --only-show-errors
}

SYNC_PID=""
cleanup() {
  if [[ -n "${SYNC_PID}" ]]; then
    kill "${SYNC_PID}" 2>/dev/null || true
    wait "${SYNC_PID}" 2>/dev/null || true
  fi
  sync_artifacts
}
trap cleanup EXIT INT TERM

(
  while true; do
    sleep "${SYNC_INTERVAL_SECONDS}"
    sync_artifacts
  done
) &
SYNC_PID=$!

PREPARE_ARGS=(
  --artifact-root "${ARTIFACT_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --samples-per-type 25
  --run-id "${RUN_ID}"
)
if [[ -n "${DESCRIPTION}" ]]; then
  PREPARE_ARGS+=(--description "${DESCRIPTION}")
fi
uv run --locked stale-qwen-eval prepare "${PREPARE_ARGS[@]}"

sync_artifacts

RUN_ARGS=(
  --artifact-root "${ARTIFACT_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --view all
  --attention-backend flash_sdpa
  --resume
)
if [[ "${MODE}" == "smoke" ]]; then
  RUN_ARGS+=(--max-scenarios 1)
fi

uv run --locked stale-qwen-eval run "${RUN_ARGS[@]}"
uv run --locked stale-qwen-eval report --artifact-root "${ARTIFACT_ROOT}" --view all
sync_artifacts

echo "Local artifacts: ${ARTIFACT_ROOT}"
echo "S3 artifacts: ${S3_RUN_URI}"
