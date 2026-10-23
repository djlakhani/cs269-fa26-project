from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

from .constants import (
    DIMENSION_INFO,
    DIMENSION_KEYS,
    MANIFEST_SCHEMA_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    PACKAGE_VERSION,
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
)
from .inference import TransformersQwenBackend
from .io_utils import atomic_write_json, load_json, sha256_file, utc_now
from .persistence import (
    ArtifactPaths,
    load_response_map,
    rebuild_responses_jsonl,
    write_response_checkpoint,
)
from .prompts import build_messages, format_haystack
from .reporting import load_artifact, refresh_report


def expected_run_config(artifact: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "prompt_version": PROMPT_VERSION,
        "thinking_enabled": False,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": artifact["token_budget"]["max_new_tokens"],
        },
        "context_window": artifact["token_budget"]["context_window"],
        "input_token_limit": artifact["token_budget"]["input_token_limit"],
        "dtype": "float16",
        "attention_implementation": "sdpa",
        "batch_size": 1,
    }


def validate_or_initialize_manifest(
    paths: ArtifactPaths,
    artifact: dict[str, Any],
    runtime_metadata: dict[str, Any],
) -> dict[str, Any]:
    manifest = validate_manifest_compatibility(paths, artifact)
    prior_run = manifest.get("run")
    manifest["run"] = {
        "started_at": prior_run.get("started_at") if prior_run else utc_now(),
        "last_opened_at": utc_now(),
        "config": expected_run_config(artifact),
        "runtime": runtime_metadata,
    }
    atomic_write_json(paths.manifest, manifest)
    return manifest


def validate_manifest_compatibility(
    paths: ArtifactPaths,
    artifact: dict[str, Any],
) -> dict[str, Any]:
    if artifact.get("model_tokenizer") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("Prepared scenarios target a different model/tokenizer revision")
    if artifact.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Prepared scenarios use an incompatible prompt version")
    budget = artifact.get("token_budget", {})
    if budget.get("input_token_limit") != (
        budget.get("context_window", 0) - budget.get("max_new_tokens", 0)
    ):
        raise ValueError("Prepared scenario token budget is internally inconsistent")
    scenario_sha = sha256_file(paths.scenarios)
    manifest = load_json(paths.manifest) if paths.manifest.exists() else {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
    }
    preparation = manifest.setdefault("preparation", {})
    recorded_sha = preparation.get("scenarios_sha256")
    if recorded_sha and recorded_sha != scenario_sha:
        raise ValueError(
            "scenarios.json does not match manifest.json; use a clean artifact root"
        )
    preparation["scenarios_sha256"] = scenario_sha
    run_config = expected_run_config(artifact)
    prior_run = manifest.get("run")
    if prior_run and prior_run.get("config") != run_config:
        raise ValueError(
            "Existing run configuration differs from this invocation; use a new artifact root"
        )
    return manifest


def run_evaluation(
    artifact_root: Path,
    *,
    cache_dir: Path | None = None,
    resume: bool = False,
    max_queries: int | None = None,
    backend: Any | None = None,
) -> int:
    paths = ArtifactPaths(artifact_root)
    artifact = load_artifact(paths)
    existing = load_response_map(paths)
    completed = {key for key, record in existing.items() if record.get("status") == "ok"}
    if completed and not resume:
        raise ValueError(
            f"Found {len(completed)} completed responses. Pass --resume or use a new artifact root."
        )
    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be positive")

    validate_manifest_compatibility(paths, artifact)
    total_queries = len(artifact["data"]) * len(DIMENSION_KEYS)
    if len(completed) == total_queries:
        rebuild_responses_jsonl(paths, artifact, existing)
        refresh_report(paths, artifact)
        return 0

    if backend is None:
        backend = TransformersQwenBackend(
            cache_dir=cache_dir,
            context_window=int(artifact["token_budget"]["context_window"]),
            max_new_tokens=int(artifact["token_budget"]["max_new_tokens"]),
            seed=int(artifact["sampling"]["seed"]),
        )
    runtime_metadata = backend.runtime_metadata()
    validate_or_initialize_manifest(paths, artifact, runtime_metadata)
    refresh_report(paths, artifact)

    generated_count = 0
    for scenario in artifact["data"]:
        history = format_haystack(scenario["sessions"])
        for dimension_key in DIMENSION_KEYS:
            key = (str(scenario["uid"]), dimension_key)
            if key in completed:
                continue
            if max_queries is not None and generated_count >= max_queries:
                rebuild_responses_jsonl(paths, artifact, load_response_map(paths))
                refresh_report(paths, artifact)
                return generated_count

            query = str(scenario["probing_queries"][dimension_key])
            messages = build_messages(history, query, dimension_key)
            base_record: dict[str, Any] = {
                "schema_version": RESPONSE_SCHEMA_VERSION,
                "uid": scenario["uid"],
                "type": scenario["type"],
                "type_name": scenario["type_name"],
                "dimension_key": dimension_key,
                "dimension_code": DIMENSION_INFO[dimension_key]["code"],
                "dimension_name": DIMENSION_INFO[dimension_key]["name"],
                "query": query,
                "generated_at": utc_now(),
            }
            try:
                generated = backend.generate(messages)
                expected_tokens = int(scenario["input_token_counts"][dimension_key])
                if int(generated["input_tokens"]) != expected_tokens:
                    raise ValueError(
                        f"Token count changed for {key}: prepared={expected_tokens}, "
                        f"runtime={generated['input_tokens']}"
                    )
                record = {**base_record, "status": "ok", **generated}
            except Exception as error:
                record = {
                    **base_record,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                write_response_checkpoint(paths, record)
                rebuild_responses_jsonl(paths, artifact, load_response_map(paths))
                refresh_report(paths, artifact)
                raise

            write_response_checkpoint(paths, record)
            existing[key] = record
            completed.add(key)
            generated_count += 1
            rebuild_responses_jsonl(paths, artifact, load_response_map(paths))
            refresh_report(paths, artifact)

    return generated_count

