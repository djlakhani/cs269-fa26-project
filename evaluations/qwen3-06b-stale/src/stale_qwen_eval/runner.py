from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    CONTEXT_VIEWS,
    DEFAULT_ATTENTION_BACKEND,
    DIMENSION_INFO,
    DIMENSION_KEYS,
    MANIFEST_SCHEMA_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    PACKAGE_VERSION,
    PROMPT_VERSION,
    RESPONSE_SCHEMA_VERSION,
    SCENARIO_SCHEMA_VERSION,
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


def expected_run_config(
    artifact: dict[str, Any], context_view: str, attention_backend: str
) -> dict[str, Any]:
    return {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "prompt_version": PROMPT_VERSION,
        "context_view": context_view,
        "thinking_enabled": False,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": artifact["token_budget"]["max_new_tokens"],
        },
        "context_window": artifact["token_budget"]["context_window"],
        "input_token_limit": artifact["token_budget"]["input_token_limit"],
        "dtype": "float16",
        "attention_implementation": "sdpa",
        "enforced_sdpa_backend": attention_backend,
        "batch_size": 1,
    }


def validate_manifest_compatibility(
    paths: ArtifactPaths,
    artifact: dict[str, Any],
    attention_backend: str,
) -> dict[str, Any]:
    if artifact.get("schema_version") != SCENARIO_SCHEMA_VERSION:
        raise ValueError("Prepared scenarios use an incompatible schema")
    if artifact.get("model_tokenizer") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("Prepared scenarios target a different model/tokenizer revision")
    if artifact.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Prepared scenarios use an incompatible prompt version")
    if paths.view not in CONTEXT_VIEWS:
        raise ValueError(f"Unknown context view {paths.view!r}")
    for scenario in artifact.get("data", []):
        if paths.view not in scenario.get("views", {}):
            raise ValueError(
                f"Scenario {scenario.get('uid')} does not contain context view {paths.view!r}"
            )

    budget = artifact.get("token_budget", {})
    if budget.get("input_token_limit") != (
        budget.get("context_window", 0) - budget.get("max_new_tokens", 0)
    ):
        raise ValueError("Prepared scenario token budget is internally inconsistent")

    scenario_sha = sha256_file(paths.scenarios)
    if paths.preparation_manifest.exists():
        preparation = load_json(paths.preparation_manifest)
        recorded_sha = preparation.get("preparation", {}).get("scenarios_sha256")
        if recorded_sha and recorded_sha != scenario_sha:
            raise ValueError("scenarios.json does not match preparation_manifest.json")

    manifest = load_json(paths.manifest) if paths.manifest.exists() else {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "context_view": paths.view,
        "scenarios_sha256": scenario_sha,
    }
    if manifest.get("context_view") != paths.view:
        raise ValueError("Run manifest belongs to a different context view")
    if manifest.get("scenarios_sha256") != scenario_sha:
        raise ValueError("scenarios.json does not match the run manifest")

    run_config = expected_run_config(artifact, paths.view, attention_backend)
    prior_run = manifest.get("run")
    if prior_run and prior_run.get("config") != run_config:
        raise ValueError(
            "Existing run configuration differs from this invocation; use a new run ID"
        )
    return manifest


def validate_or_initialize_manifest(
    paths: ArtifactPaths,
    artifact: dict[str, Any],
    runtime_metadata: dict[str, Any],
    attention_backend: str,
) -> dict[str, Any]:
    manifest = validate_manifest_compatibility(paths, artifact, attention_backend)
    prior_run = manifest.get("run")
    manifest["run"] = {
        "started_at": prior_run.get("started_at") if prior_run else utc_now(),
        "last_opened_at": utc_now(),
        "config": expected_run_config(artifact, paths.view, attention_backend),
        "runtime": runtime_metadata,
    }
    atomic_write_json(paths.manifest, manifest)
    return manifest


def _completed_keys(paths: ArtifactPaths) -> set[tuple[str, str]]:
    return {
        key
        for key, record in load_response_map(paths).items()
        if record.get("status") == "ok"
    }


def _has_pending_work(artifact: dict[str, Any], paths: ArtifactPaths) -> bool:
    return len(_completed_keys(paths)) < len(artifact["data"]) * len(DIMENSION_KEYS)


def run_evaluation(
    artifact_root: Path,
    *,
    context_view: str,
    attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    cache_dir: Path | None = None,
    resume: bool = False,
    max_queries: int | None = None,
    max_scenarios: int | None = None,
    show_progress: bool = True,
    backend: Any | None = None,
) -> int:
    paths = ArtifactPaths(artifact_root, context_view)
    artifact = load_artifact(paths)
    existing = load_response_map(paths)
    completed = {key for key, record in existing.items() if record.get("status") == "ok"}
    if completed and not resume:
        raise ValueError(
            f"Found {len(completed)} completed responses. Pass --resume or use a new run ID."
        )
    if max_queries is not None and max_queries <= 0:
        raise ValueError("max_queries must be positive")
    if max_scenarios is not None and max_scenarios <= 0:
        raise ValueError("max_scenarios must be positive")

    validate_manifest_compatibility(paths, artifact, attention_backend)
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
            attention_backend=attention_backend,
        )
    runtime_metadata = backend.runtime_metadata()
    validate_or_initialize_manifest(paths, artifact, runtime_metadata, attention_backend)
    refresh_report(paths, artifact)

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None
    completed_scenarios = sum(
        all((str(scenario["uid"]), key) in completed for key in DIMENSION_KEYS)
        for scenario in artifact["data"]
    )
    progress = (
        tqdm(
            total=len(artifact["data"]),
            initial=completed_scenarios,
            desc=context_view,
            unit="scenario",
        )
        if show_progress and tqdm is not None
        else None
    )

    generated_count = 0
    attempted_scenarios = 0
    try:
        for scenario in artifact["data"]:
            uid = str(scenario["uid"])
            missing_dimensions = [
                key for key in DIMENSION_KEYS if (uid, key) not in completed
            ]
            if not missing_dimensions:
                continue
            if max_scenarios is not None and attempted_scenarios >= max_scenarios:
                break
            attempted_scenarios += 1

            view = scenario["views"][context_view]
            history = format_haystack(view["sessions"])
            for dimension_key in missing_dimensions:
                if max_queries is not None and generated_count >= max_queries:
                    break
                key = (uid, dimension_key)
                query = str(scenario["probing_queries"][dimension_key])
                messages = build_messages(history, query, dimension_key)
                if progress is not None:
                    progress.set_postfix_str(
                        f"{scenario['type']} {uid[:8]} {DIMENSION_INFO[dimension_key]['code']}"
                    )
                base_record: dict[str, Any] = {
                    "schema_version": RESPONSE_SCHEMA_VERSION,
                    "context_view": context_view,
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
                    expected_tokens = int(view["input_token_counts"][dimension_key])
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

            scenario_complete = all((uid, key) in completed for key in DIMENSION_KEYS)
            if scenario_complete and progress is not None:
                progress.update(1)
            if max_queries is not None and generated_count >= max_queries:
                break
    finally:
        if progress is not None:
            progress.close()

    return generated_count


def run_context_views(
    artifact_root: Path,
    *,
    context_views: Iterable[str],
    attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    cache_dir: Path | None = None,
    resume: bool = False,
    max_scenarios: int | None = None,
    show_progress: bool = True,
) -> dict[str, int]:
    views = list(context_views)
    if not views:
        raise ValueError("At least one context view is required")
    artifact = load_artifact(ArtifactPaths(artifact_root, views[0]))
    for view in views:
        validate_manifest_compatibility(
            ArtifactPaths(artifact_root, view), artifact, attention_backend
        )

    pending_views = [
        view
        for view in views
        if _has_pending_work(artifact, ArtifactPaths(artifact_root, view))
    ]
    if not pending_views:
        return {view: 0 for view in views}

    backend = TransformersQwenBackend(
        cache_dir=cache_dir,
        context_window=int(artifact["token_budget"]["context_window"]),
        max_new_tokens=int(artifact["token_budget"]["max_new_tokens"]),
        seed=int(artifact["sampling"]["seed"]),
        attention_backend=attention_backend,
    )
    counts: dict[str, int] = {}
    for view in views:
        counts[view] = run_evaluation(
            artifact_root,
            context_view=view,
            attention_backend=attention_backend,
            cache_dir=cache_dir,
            resume=resume,
            max_scenarios=max_scenarios,
            show_progress=show_progress,
            backend=backend,
        )
    return counts
