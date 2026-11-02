#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 SOURCE_RUN_ID RUN_ID [smoke|full] [DESCRIPTION]" >&2
  exit 2
fi

SOURCE_RUN_ID="$1"
RUN_ID="$2"
MODE="${3:-smoke}"
DESCRIPTION="${4:-}"
RUN_ID_PATTERN='^[A-Za-z0-9][A-Za-z0-9._-]*$'
if [[ ! "${SOURCE_RUN_ID}" =~ ${RUN_ID_PATTERN} || ! "${RUN_ID}" =~ ${RUN_ID_PATTERN} ]]; then
  echo "Run IDs may contain only letters, numbers, period, underscore, and hyphen." >&2
  exit 2
fi
if [[ "${SOURCE_RUN_ID}" == "${RUN_ID}" ]]; then
  echo "SOURCE_RUN_ID and RUN_ID must differ so the original evaluation is not modified." >&2
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
SOURCE_SCENARIOS_DIR="${CACHE_DIR}/source-scenarios/${SOURCE_RUN_ID}"
SOURCE_SCENARIOS="${SOURCE_SCENARIOS_DIR}/scenarios.json"
S3_SOURCE_URI="${S3_BASE_URI%/}/runs/${SOURCE_RUN_ID}/scenarios.json"
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

mkdir -p "${ARTIFACT_ROOT}" "${CACHE_DIR}" "${SOURCE_SCENARIOS_DIR}"
cd "${PROJECT_DIR}"

UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "uv is unavailable. Run scripts/setup_aws.sh before starting an evaluation." >&2
  exit 1
fi

aws s3api head-bucket --bucket "${S3_BUCKET}"
aws s3 cp "${S3_SOURCE_URI}" "${SOURCE_SCENARIOS}" --only-show-errors
# Restore any checkpoints if this run is resuming on a replacement instance.
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
  --source-scenarios "${SOURCE_SCENARIOS}"
  --artifact-root "${ARTIFACT_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --run-id "${RUN_ID}"
  --source-reference "${S3_SOURCE_URI}"
)
if [[ -n "${DESCRIPTION}" ]]; then
  PREPARE_ARGS+=(--description "${DESCRIPTION}")
fi
"${UV_BIN}" run --locked stale-qwen-eval prepare-key-pairs "${PREPARE_ARGS[@]}"
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

"${UV_BIN}" run --locked stale-qwen-eval run "${RUN_ARGS[@]}"
"${UV_BIN}" run --locked stale-qwen-eval report \
  --artifact-root "${ARTIFACT_ROOT}" --view all
sync_artifacts

echo "Source scenarios: ${S3_SOURCE_URI}"
echo "Local artifacts: ${ARTIFACT_ROOT}"
echo "S3 artifacts: ${S3_RUN_URI}"
