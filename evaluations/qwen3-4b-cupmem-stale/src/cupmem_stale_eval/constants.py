from __future__ import annotations

PACKAGE_VERSION = "0.1.0"
SCENARIO_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
ENGINE_CHECKPOINT_SCHEMA_VERSION = 1
SELECTION_ALGORITHM_VERSION = "uid-seeded-distractors-v1"

DATASET_ID = "STALEproj/STALE"
DATASET_REVISION = "617c51dc200b5ab09970834144c7e51c77959af0"
DATASET_SPLIT = "train"

TARGET_MODEL_ID = "Qwen/Qwen3-4B"
TARGET_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
JUDGE_MODEL_ID = "gpt-6-luna"

DEFAULT_SEED = 42
DEFAULT_SAMPLES_PER_TYPE = 50
DEFAULT_DISTRACTORS_PER_SCENARIO = 10
SOURCE_SESSION_COUNT = 50
DIMENSION_KEYS = ("dim1_query", "dim2_query", "dim3_query")
DIMENSION_CODES = {
    "dim1_query": "SR",
    "dim2_query": "PR",
    "dim3_query": "IPA",
}
TYPE_NAMES = {"T1": "Co-referential conflict", "T2": "Propagated conflict"}

DEFAULT_RUN_ID = "qwen3-4b-cupmem-stale-seed42-v1"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_S3_BASE_URI = (
    "s3://stale-qwen-evals-589498925064-us-east-1-an/cup-mem"
)

FAILURE_STAGES = (
    "chunking",
    "delta extraction/filtering",
    "update resolution",
    "invalidation/propagation",
    "memory representation",
    "retrieval",
    "premise verification",
    "basis/action grounding",
    "answer composition",
    "model-format/runtime failure",
)
