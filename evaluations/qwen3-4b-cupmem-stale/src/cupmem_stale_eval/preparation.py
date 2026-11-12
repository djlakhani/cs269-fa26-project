from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .constants import (
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    DEFAULT_DISTRACTORS_PER_SCENARIO,
    DEFAULT_SAMPLES_PER_TYPE,
    DEFAULT_SEED,
    DIMENSION_KEYS,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    PACKAGE_VERSION,
    SCENARIO_SCHEMA_VERSION,
    SELECTION_ALGORITHM_VERSION,
    TARGET_MODEL_ID,
    TARGET_MODEL_REVISION,
    TYPE_NAMES,
)
from .io_utils import atomic_write_json, load_json, sha256_file, utc_now
from .sampling import select_balanced_scenarios, select_session_indices


def build_scenario_artifact(
    records: Iterable[Mapping[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    samples_per_type: int = DEFAULT_SAMPLES_PER_TYPE,
    distractors_per_scenario: int = DEFAULT_DISTRACTORS_PER_SCENARIO,
) -> dict[str, Any]:
    selected = select_balanced_scenarios(
        records, samples_per_type=samples_per_type, seed=seed
    )
    scenarios: list[dict[str, Any]] = []
    for record in selected:
        uid = str(record.get("uid", ""))
        queries = record.get("probing_queries")
        if not isinstance(queries, dict) or not all(
            isinstance(queries.get(key), str) for key in DIMENSION_KEYS
        ):
            raise ValueError(f"Scenario {uid} has invalid probing queries")
        timestamps = list(record.get("timestamps", []))
        sessions = list(record.get("haystack_session", []))
        if len(timestamps) != len(sessions):
            raise ValueError(f"Scenario {uid} timestamp/session counts differ")
        for index, session in enumerate(sessions):
            if not isinstance(session, list) or any(
                not isinstance(message, dict)
                or not isinstance(message.get("role"), str)
                or not isinstance(message.get("content"), str)
                for message in session
            ):
                raise ValueError(f"Scenario {uid} session {index} has invalid messages")
        old_state = str(record.get("M_old", record.get("old_info", ""))).strip()
        new_state = str(record.get("M_new", "")).strip()
        explanation = str(record.get("explanation", "")).strip()
        if not old_state or not new_state or not explanation:
            raise ValueError(f"Scenario {uid} has incomplete ground truth")
        selection = select_session_indices(
            record, distractor_count=distractors_per_scenario, seed=seed
        )
        retained = [
            {
                "original_index": index,
                "timestamp": timestamps[index],
                "messages": sessions[index],
                "is_evidence": index in selection["evidence_indices"],
            }
            for index in selection["kept_indices"]
        ]
        scenarios.append(
            {
                "uid": uid,
                "type": str(record["type"]),
                "type_name": TYPE_NAMES[str(record["type"])],
                "M_old": old_state,
                "M_new": new_state,
                "explanation": explanation,
                "probing_queries": {key: str(queries[key]) for key in DIMENSION_KEYS},
                "source_session_count": len(sessions),
                "relevant_session_index": selection["evidence_indices"],
                "scenario_seed": selection["scenario_seed"],
                "distractor_indices": selection["distractor_indices"],
                "kept_session_indices": selection["kept_indices"],
                "dropped_session_indices": selection["dropped_indices"],
                "sessions": retained,
            }
        )
    return {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": DATASET_SPLIT},
        "target_model": {"id": TARGET_MODEL_ID, "revision": TARGET_MODEL_REVISION},
        "embedding_model": {"id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_MODEL_REVISION},
        "sampling": {
            "seed": seed,
            "samples_per_type": samples_per_type,
            "distractors_per_scenario": distractors_per_scenario,
            "sessions_per_scenario": distractors_per_scenario + 2,
            "total_scenarios": len(scenarios),
            "algorithm_version": SELECTION_ALGORITHM_VERSION,
        },
        "data": scenarios,
    }


def prepare_from_huggingface(
    artifact_root: Path,
    *,
    cache_dir: Path,
    seed: int = DEFAULT_SEED,
    samples_per_type: int = DEFAULT_SAMPLES_PER_TYPE,
    distractors_per_scenario: int = DEFAULT_DISTRACTORS_PER_SCENARIO,
    force: bool = False,
) -> Path:
    scenario_path = artifact_root / "scenarios.json"
    manifest_path = artifact_root / "preparation_manifest.json"
    if scenario_path.exists() and not force:
        if not manifest_path.exists():
            raise ValueError("Existing scenarios.json is missing preparation_manifest.json")
        existing = load_json(scenario_path)
        requested = (seed, samples_per_type, distractors_per_scenario)
        recorded_sampling = existing.get("sampling", {})
        recorded = (
            recorded_sampling.get("seed"),
            recorded_sampling.get("samples_per_type"),
            recorded_sampling.get("distractors_per_scenario"),
        )
        if existing.get("schema_version") != SCENARIO_SCHEMA_VERSION or recorded != requested:
            raise ValueError("Existing scenarios use different settings; choose a new artifact root")
        manifest = load_json(manifest_path)
        if manifest.get("scenarios_sha256") != sha256_file(scenario_path):
            raise ValueError("Existing scenarios.json checksum does not match its manifest")
        return scenario_path
    if force:
        checkpoint_exists = (
            (artifact_root / "responses.jsonl").exists()
            or any((artifact_root / "scenarios").glob("*/status.json"))
            or any((artifact_root / "scenarios").glob("*/queries/*/result.json"))
        )
        if checkpoint_exists:
            raise ValueError("Refusing to replace scenarios alongside run checkpoints")

    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(
        DATASET_ID,
        revision=DATASET_REVISION,
        split=DATASET_SPLIT,
        cache_dir=str(cache_dir),
    )
    embedding_path = snapshot_download(
        EMBEDDING_MODEL_ID,
        revision=EMBEDDING_MODEL_REVISION,
        cache_dir=str(cache_dir),
    )
    artifact = build_scenario_artifact(
        dataset,
        seed=seed,
        samples_per_type=samples_per_type,
        distractors_per_scenario=distractors_per_scenario,
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(scenario_path, artifact)
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "package_version": PACKAGE_VERSION,
            "created_at": utc_now(),
            "scenarios_sha256": sha256_file(scenario_path),
            "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": DATASET_SPLIT},
            "target_model": {"id": TARGET_MODEL_ID, "revision": TARGET_MODEL_REVISION},
            "embedding_model": {
                "id": EMBEDDING_MODEL_ID,
                "revision": EMBEDDING_MODEL_REVISION,
            },
            "selection_algorithm_version": SELECTION_ALGORITHM_VERSION,
            "embedding_model_local_path": str(Path(embedding_path).resolve()),
            "configuration": {
                "seed": seed,
                "samples_per_type": samples_per_type,
                "distractors_per_scenario": distractors_per_scenario,
            },
        },
    )
    return scenario_path
