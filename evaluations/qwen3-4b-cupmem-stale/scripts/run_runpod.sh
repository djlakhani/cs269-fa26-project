#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 RUN_ID [smoke|subset|full]" >&2
  exit 2
fi
RUN_ID="$1"
MODE="${2:-smoke}"
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo "Run ID contains an invalid character" >&2
  exit 2
fi
if [[ "${MODE}" != "smoke" && "${MODE}" != "subset" && "${MODE}" != "full" ]]; then
  echo "Mode must be smoke, subset, or full" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARTIFACT_BASE="${CUPMEM_ARTIFACT_BASE:-/workspace/cupmem-artifacts}"
CACHE_DIR="${CUPMEM_CACHE_DIR:-/workspace/huggingface-cache}"
S3_BASE_URI="${CUPMEM_S3_BASE_URI:-s3://stale-qwen-evals-589498925064-us-east-1-an/cup-mem}"
SYNC_INTERVAL="${CUPMEM_SYNC_INTERVAL_SECONDS:-60}"
ARTIFACT_ROOT="${ARTIFACT_BASE}/${RUN_ID}"
S3_RUN_URI="${S3_BASE_URI%/}/runs/${RUN_ID}"
VLLM_BASE_URL="${CUPMEM_VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
VLLM_RAW_LOG="${CACHE_DIR}/vllm-${RUN_ID}.log"
export AWS_REGION="${AWS_REGION:-us-east-1}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION}}"

if [[ ! "${SYNC_INTERVAL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "CUPMEM_SYNC_INTERVAL_SECONDS must be a positive integer" >&2
  exit 2
fi
if [[ ! "${S3_BASE_URI}" =~ ^s3://[^/]+/.+ ]]; then
  echo "CUPMEM_S3_BASE_URI must be an s3://bucket/prefix URI" >&2
  exit 2
fi

if [[ -z "${JUDGE_OPENAI_API_KEY:-}" ]]; then
  echo "JUDGE_OPENAI_API_KEY is required" >&2
  exit 1
fi
mkdir -p "${ARTIFACT_ROOT}/logs" "${CACHE_DIR}"
cd "${PROJECT_DIR}"
UV_BIN="$(command -v uv || true)"
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "uv is unavailable; run scripts/setup_runpod.sh first" >&2
  exit 1
fi

BUCKET_PATH="${S3_BASE_URI#s3://}"
BUCKET="${BUCKET_PATH%%/*}"
aws s3api head-bucket --bucket "${BUCKET}"
aws s3 sync "${S3_RUN_URI}" "${ARTIFACT_ROOT}" --exclude ".*" --exclude "*/.*" --only-show-errors
exec > >(tee -a "${ARTIFACT_ROOT}/logs/runner.log") 2>&1

PREFLIGHT_FILE="$(mktemp)"
printf 'cupmem-s3-preflight\n' >"${PREFLIGHT_FILE}"
PREFLIGHT_KEY="${S3_RUN_URI%/}/preflight-${RUN_ID}.txt"
aws s3 cp "${PREFLIGHT_FILE}" "${PREFLIGHT_KEY}" --only-show-errors
aws s3api head-object --bucket "${BUCKET}" --key "${PREFLIGHT_KEY#s3://${BUCKET}/}" >/dev/null
rm -f "${PREFLIGHT_FILE}"

sync_artifacts() {
  aws s3 sync "${ARTIFACT_ROOT}" "${S3_RUN_URI}" --exclude ".*" --exclude "*/.*" --only-show-errors
}

SYNC_PID=""
VLLM_PID=""
cleanup() {
  set +e
  if [[ -n "${SYNC_PID}" ]]; then
    kill "${SYNC_PID}" 2>/dev/null
    wait "${SYNC_PID}" 2>/dev/null
  fi
  if [[ -n "${VLLM_PID}" ]]; then
    kill "${VLLM_PID}" 2>/dev/null
    wait "${VLLM_PID}" 2>/dev/null
  fi
  "${UV_BIN}" run --locked cupmem-stale-eval archive-log \
    --source "${VLLM_RAW_LOG}" \
    --destination "${ARTIFACT_ROOT}/logs/vllm.log.zst"
  sync_artifacts
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

(
  while true; do
    sleep "${SYNC_INTERVAL}"
    sync_artifacts
  done
) &
SYNC_PID="$!"

"${UV_BIN}" run --locked cupmem-stale-eval prepare \
  --artifact-root "${ARTIFACT_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --seed 42 \
  --samples-per-type 50 \
  --distractors-per-scenario 10
sync_artifacts

"${UV_BIN}" run --locked vllm serve Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --served-model-name Qwen/Qwen3-4B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype half \
  --max-model-len 32768 \
  --max-num-seqs 4 \
  --gpu-memory-utilization 0.85 \
  --generation-config vllm \
  --enable-request-id-headers \
  >"${VLLM_RAW_LOG}" 2>&1 &
VLLM_PID="$!"

for _ in $(seq 1 120); do
  if curl --fail --silent http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "vLLM exited during startup" >&2
    tail -100 "${VLLM_RAW_LOG}" >&2
    exit 1
  fi
  sleep 5
done
curl --fail --silent http://127.0.0.1:8000/health >/dev/null

"${UV_BIN}" run --locked cupmem-stale-eval preflight \
  --artifact-root "${ARTIFACT_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --vllm-base-url "${VLLM_BASE_URL}"

RUN_ARGS=(
  --artifact-root "${ARTIFACT_ROOT}"
  --cache-dir "${CACHE_DIR}"
  --vllm-base-url "${VLLM_BASE_URL}"
  --resume
)
JUDGE_ARGS=(--artifact-root "${ARTIFACT_ROOT}" --model gpt-6-luna --resume)
if [[ "${MODE}" == "smoke" ]]; then
  RUN_ARGS+=(--max-per-type 1)
  JUDGE_ARGS+=(--max-scenarios 2)
elif [[ "${MODE}" == "subset" ]]; then
  RUN_ARGS+=(--max-per-type 7)
fi

"${UV_BIN}" run --locked cupmem-stale-eval run "${RUN_ARGS[@]}"
"${UV_BIN}" run --locked cupmem-stale-eval judge "${JUDGE_ARGS[@]}"
"${UV_BIN}" run --locked cupmem-stale-eval diagnose \
  --artifact-root "${ARTIFACT_ROOT}" --model gpt-6-luna --resume
"${UV_BIN}" run --locked cupmem-stale-eval report --artifact-root "${ARTIFACT_ROOT}"
if [[ "${MODE}" == "smoke" || "${MODE}" == "subset" ]]; then
  "${UV_BIN}" run --locked cupmem-stale-eval verify \
    --artifact-root "${ARTIFACT_ROOT}" --allow-incomplete
else
  "${UV_BIN}" run --locked cupmem-stale-eval verify --artifact-root "${ARTIFACT_ROOT}"
fi
sync_artifacts

DRY_RUN_OUTPUT="$(aws s3 sync "${ARTIFACT_ROOT}" "${S3_RUN_URI}" --dryrun --exclude ".*" --exclude "*/.*")"
if [[ -n "${DRY_RUN_OUTPUT}" ]]; then
  echo "S3 consistency check found pending uploads:" >&2
  echo "${DRY_RUN_OUTPUT}" >&2
  exit 1
fi
echo "Artifacts: ${S3_RUN_URI}"
