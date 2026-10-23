from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .constants import (
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
from .sampling import select_balanced_scenarios
from .truncation import truncate_scenario


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
    scenarios = [
        truncate_scenario(
            record,
            tokenizer,
            input_token_limit=input_token_limit,
            seed=seed,
        )
        for record in selected
    ]
    return {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": "train"},
        "model_tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "prompt_version": PROMPT_VERSION,
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
) -> Path:
    scenario_path = artifact_root / "scenarios.json"
    manifest_path = artifact_root / "manifest.json"
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
        return scenario_path

    if force and ((artifact_root / "checkpoints").exists() or (artifact_root / "responses.jsonl").exists()):
        raise ValueError(
            "Refusing to replace scenarios alongside existing responses. Use a new artifact root."
        )

    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Preparation dependencies are missing. Install requirements.txt first."
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
            "sampling": artifact["sampling"],
            "token_budget": artifact["token_budget"],
            "scenarios_sha256": sha256_file(scenario_path),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return scenario_path

