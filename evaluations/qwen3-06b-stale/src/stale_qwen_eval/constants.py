from __future__ import annotations

PACKAGE_VERSION = "0.1.0"
SCENARIO_SCHEMA_VERSION = 1
RESPONSE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
PROMPT_VERSION = "stale-author-prompts-v1"

DATASET_ID = "STALEproj/STALE"
DATASET_REVISION = "617c51dc200b5ab09970834144c7e51c77959af0"
MODEL_ID = "Qwen/Qwen3-0.6B"
MODEL_REVISION = "167b8104f88905a951069f5f95f9776908da5f68"

DEFAULT_SEED = 42
DEFAULT_SAMPLES_PER_TYPE = 5
DEFAULT_CONTEXT_WINDOW = 32_768
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_INPUT_TOKEN_LIMIT = DEFAULT_CONTEXT_WINDOW - DEFAULT_MAX_NEW_TOKENS

TYPE_INFO = {
    "T1": "Co-referential conflict",
    "T2": "Propagated conflict",
}

DIMENSION_INFO = {
    "dim1_query": {
        "code": "SR",
        "name": "State Resolution",
        "response_key": "dim1_response",
        "expected": (
            "Recognize that M_old may no longer be valid, identify the updated state "
            "supported by M_new, and avoid treating the old state as current."
        ),
    },
    "dim2_query": {
        "code": "PR",
        "name": "Premise Resistance",
        "response_key": "dim2_response",
        "expected": (
            "Reject or correct the query's stale premise and ground the response in "
            "the updated state supported by M_new."
        ),
    },
    "dim3_query": {
        "code": "IPA",
        "name": "Implicit Policy Adaptation",
        "response_key": "dim3_response",
        "expected": (
            "Give a concrete, useful answer whose action or recommendation safely "
            "respects M_new; a generic answer that avoids using the updated state is insufficient."
        ),
    },
}

DIMENSION_KEYS = tuple(DIMENSION_INFO)

