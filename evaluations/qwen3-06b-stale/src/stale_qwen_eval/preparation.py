from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .constants import (
    CONTEXT_VIEWS,
    DATASET_ID,
    DATASET_REVISION,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_SAMPLES_PER_TYPE,
    DEFAULT_SEED,
    MANIFEST_SCHEMA_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    PACKAGE_VERSION,
    PROMPT_VERSION,
    SCENARIO_SCHEMA_VERSION,
)
from .io_utils import atomic_write_json, load_json, sha256_file, utc_now
from .prompts import EXPLICIT_SYSTEM_PROMPT, IMPLICIT_SYSTEM_PROMPT
from .sampling import select_balanced_scenarios
from .truncation import evidence_only_view, truncate_scenario


def build_scenario_artifact(
    records: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    *,
    seed: int = DEFAULT_SEED,
    samples_per_type: int = DEFAULT_SAMPLES_PER_TYPE,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
) -> dict[str, Any]:
    input_token_limit = context_window - max_new_tokens
    if input_token_limit <= 0:
        raise ValueError("max_new_tokens must be smaller than context_window")

    selected = select_balanced_scenarios(
        records, samples_per_type=samples_per_type, seed=seed
    )
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover - dependency validation catches this in production
        tqdm = lambda values, **_: values

    scenarios: list[dict[str, Any]] = []
    original_model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(original_model_max_length, int):
        # Preparation intentionally counts histories before truncating them. Raising
        # this tokenizer-only warning threshold avoids claiming those counts are model
        # inputs; inference still enforces the real 32,256-token input limit.
        tokenizer.model_max_length = max(original_model_max_length, 1_000_000_000)
    try:
        for record in tqdm(selected, desc="Preparing scenarios", unit="scenario"):
            truncated = truncate_scenario(
                record,
                tokenizer,
                input_token_limit=input_token_limit,
                seed=seed,
            )
            evidence_only = evidence_only_view(
                record,
                tokenizer,
                input_token_limit=input_token_limit,
            )
            scenarios.append(
                {
                    "uid": truncated["uid"],
                    "type": truncated["type"],
                    "type_name": truncated["type_name"],
                    "M_old": truncated["M_old"],
                    "M_new": truncated["M_new"],
                    "explanation": truncated["explanation"],
                    "probing_queries": truncated["probing_queries"],
                    "relevant_session_index": truncated["relevant_session_index"],
                    "source_session_count": truncated["source_session_count"],
                    "views": {
                        "truncated": {
                            key: truncated[key]
                            for key in (
                                "kept_session_indices",
                                "dropped_session_indices",
                                "distractor_removal_order",
                                "sessions",
                                "input_token_counts",
                            )
                        },
                        "evidence_only": evidence_only,
                    },
                }
            )
    finally:
        if isinstance(original_model_max_length, int):
            tokenizer.model_max_length = original_model_max_length
    return {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": "train"},
        "model_tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "prompt_version": PROMPT_VERSION,
        "context_views": dict(CONTEXT_VIEWS),
        "sampling": {
            "seed": seed,
            "samples_per_type": samples_per_type,
            "total_scenarios": len(scenarios),
        },
        "token_budget": {
            "context_window": context_window,
            "max_new_tokens": max_new_tokens,
            "input_token_limit": input_token_limit,
        },
        "data": scenarios,
    }


def prepare_from_huggingface(
    artifact_root: Path,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    seed: int = DEFAULT_SEED,
    samples_per_type: int = DEFAULT_SAMPLES_PER_TYPE,
    context_window: int = DEFAULT_CONTEXT_WINDOW,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    run_id: str | None = None,
    description: str | None = None,
) -> Path:
    scenario_path = artifact_root / "scenarios.json"
    manifest_path = artifact_root / "preparation_manifest.json"
    if scenario_path.exists() and not force:
        existing = load_json(scenario_path)
        if existing.get("schema_version") != SCENARIO_SCHEMA_VERSION:
            raise ValueError(
                f"Existing {scenario_path} has unsupported schema; use a new artifact root"
            )
        requested = {
            "seed": seed,
            "samples_per_type": samples_per_type,
            "context_window": context_window,
            "max_new_tokens": max_new_tokens,
        }
        recorded = {
            "seed": existing.get("sampling", {}).get("seed"),
            "samples_per_type": existing.get("sampling", {}).get("samples_per_type"),
            "context_window": existing.get("token_budget", {}).get("context_window"),
            "max_new_tokens": existing.get("token_budget", {}).get("max_new_tokens"),
        }
        if requested != recorded:
            raise ValueError(
                f"Existing scenarios use {recorded}, but this invocation requested {requested}. "
                "Use a new artifact root or --force when no responses exist."
            )
        _write_run_metadata(artifact_root, run_id=run_id, description=description)
        return scenario_path

    views_root = artifact_root / "views"
    has_responses = views_root.exists() and any(views_root.glob("*/checkpoints/*.json"))
    if force and has_responses:
        raise ValueError(
            "Refusing to replace scenarios alongside existing responses. Use a new artifact root."
        )

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Preparation dependencies are missing. Run `uv sync --locked` first."
        ) from error

    cache_text = str(cache_dir) if cache_dir else None
    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split="train",
        cache_dir=cache_text,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=cache_text,
    )
    artifact = build_scenario_artifact(
        dataset,
        tokenizer,
        seed=seed,
        samples_per_type=samples_per_type,
        context_window=context_window,
        max_new_tokens=max_new_tokens,
    )
    atomic_write_json(scenario_path, artifact)

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "preparation": {
            "created_at": artifact["created_at"],
            "dataset": artifact["dataset"],
            "model_tokenizer": artifact["model_tokenizer"],
            "prompt_version": PROMPT_VERSION,
            "prompts": {
                "explicit_system": EXPLICIT_SYSTEM_PROMPT,
                "implicit_system": IMPLICIT_SYSTEM_PROMPT,
                "explicit_user_template": (
                    "[Conversation History]\n{history_text}\n\n[Question]\n{query_text}"
                ),
                "implicit_user_template": (
                    "[Conversation History]\n{history_text}\n\n[Latest Query]\n{query_text}"
                ),
            },
            "context_views": artifact["context_views"],
            "sampling": artifact["sampling"],
            "token_budget": artifact["token_budget"],
            "scenarios_sha256": sha256_file(scenario_path),
        },
    }
    atomic_write_json(manifest_path, manifest)
    _write_run_metadata(artifact_root, run_id=run_id, description=description)
    return scenario_path


def _write_run_metadata(
    artifact_root: Path, *, run_id: str | None, description: str | None
) -> None:
    path = artifact_root / "run_metadata.json"
    requested_id = run_id or artifact_root.name
    if path.exists():
        existing = load_json(path)
        if existing.get("run_id") != requested_id:
            raise ValueError(
                f"Existing run ID {existing.get('run_id')!r} differs from {requested_id!r}"
            )
        if description and existing.get("description") not in {None, description}:
            raise ValueError("Existing run description differs from this invocation")
        if description and not existing.get("description"):
            existing["description"] = description
            atomic_write_json(path, existing)
        return
    atomic_write_json(
        path,
        {
            "run_id": requested_id,
            "description": description or "",
            "created_at": utc_now(),
        },
    )

