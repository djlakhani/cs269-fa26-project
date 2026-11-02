from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .constants import (
    DIMENSION_KEYS,
    KEY_PAIR_CONTEXT_VIEWS,
    MANIFEST_SCHEMA_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    PACKAGE_VERSION,
    PROMPT_VERSION,
    SCENARIO_SCHEMA_VERSION,
)
from .io_utils import atomic_write_json, load_json, sha256_file, utc_now
from .preparation import _write_run_metadata
from .prompts import (
    EXPLICIT_SYSTEM_PROMPT,
    IMPLICIT_SYSTEM_PROMPT,
    count_prompt_tokens,
)


EXTRACTION_VERSION = "normalized-target-coverage-v1"
MIN_TARGET_TOKEN_COVERAGE = 0.95
MIN_BEST_MATCH_MARGIN = 0.05


class KeyPairExtractionError(ValueError):
    """Raised when an M_old/M_new dialogue pair cannot be identified safely."""


def _normalized_tokens(text: Any) -> list[str]:
    return re.findall(r"[a-z0-9]+", str(text).lower())


def _normalized_text(text: Any) -> str:
    return " ".join(_normalized_tokens(text))


def _target_coverage(target: str, candidate: str) -> float:
    target_counts = Counter(_normalized_tokens(target))
    candidate_counts = Counter(_normalized_tokens(candidate))
    target_total = sum(target_counts.values())
    if target_total == 0:
        return 0.0
    matched = sum(
        min(count, candidate_counts.get(token, 0))
        for token, count in target_counts.items()
    )
    return matched / target_total


def extract_key_dialogue_pair(
    session: Mapping[str, Any], target: str, *, uid: str, state_label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the injected user turn and its immediate assistant response."""
    messages = list(session.get("messages", []))
    target_normalized = _normalized_text(target)
    candidates: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "user":
            continue
        content = str(message.get("content", ""))
        normalized = _normalized_text(content)
        candidates.append(
            {
                "index": index,
                "normalized_contains_target": bool(
                    target_normalized and target_normalized in normalized
                ),
                "target_token_coverage": _target_coverage(target, content),
                "sequence_similarity": SequenceMatcher(
                    None, target_normalized, normalized
                ).ratio(),
            }
        )
    if not candidates:
        raise KeyPairExtractionError(
            f"Scenario {uid} {state_label} session has no user turns"
        )

    candidates.sort(
        key=lambda item: (
            item["normalized_contains_target"],
            item["target_token_coverage"],
            item["sequence_similarity"],
            -item["index"],
        ),
        reverse=True,
    )
    best = candidates[0]
    second_coverage = (
        float(candidates[1]["target_token_coverage"])
        if len(candidates) > 1
        else 0.0
    )
    coverage = float(best["target_token_coverage"])
    margin = coverage - second_coverage
    if coverage < MIN_TARGET_TOKEN_COVERAGE or margin < MIN_BEST_MATCH_MARGIN:
        raise KeyPairExtractionError(
            f"Scenario {uid} {state_label} dialogue match is not reliable: "
            f"coverage={coverage:.3f}, runner_up={second_coverage:.3f}, "
            f"required_coverage={MIN_TARGET_TOKEN_COVERAGE:.2f}, "
            f"required_margin={MIN_BEST_MATCH_MARGIN:.2f}"
        )

    user_index = int(best["index"])
    assistant_index = user_index + 1
    if assistant_index >= len(messages) or messages[assistant_index].get("role") != "assistant":
        raise KeyPairExtractionError(
            f"Scenario {uid} {state_label} matched user turn {user_index}, but it "
            "does not have an immediate assistant response"
        )

    extracted_session = {
        "original_index": int(session["original_index"]),
        "timestamp": session.get("timestamp"),
        "messages": [messages[user_index], messages[assistant_index]],
    }
    metadata = {
        "state": state_label,
        "session_index": int(session["original_index"]),
        "timestamp": session.get("timestamp"),
        "user_turn_index": user_index,
        "assistant_turn_index": assistant_index,
        "normalized_contains_target": bool(best["normalized_contains_target"]),
        "target_token_coverage": coverage,
        "runner_up_target_token_coverage": second_coverage,
        "match_margin": margin,
        "extraction_version": EXTRACTION_VERSION,
    }
    return extracted_session, metadata


def _input_token_counts(
    tokenizer: Any,
    sessions: Sequence[Mapping[str, Any]],
    queries: Mapping[str, str],
) -> dict[str, int]:
    return {
        dimension_key: count_prompt_tokens(
            tokenizer,
            sessions,
            str(queries[dimension_key]),
            dimension_key,
            preserve_input_order=True,
        )
        for dimension_key in DIMENSION_KEYS
    }


def build_key_pair_artifact(
    source: Mapping[str, Any],
    tokenizer: Any,
    *,
    source_sha256: str,
    source_reference: str | None = None,
) -> dict[str, Any]:
    """Derive chronological and reversed key-pair views from frozen scenarios."""
    if source.get("schema_version") != SCENARIO_SCHEMA_VERSION:
        raise ValueError("Source scenarios use an incompatible schema")
    if source.get("model_tokenizer") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("Source scenarios target a different model/tokenizer revision")

    input_limit = int(source["token_budget"]["input_token_limit"])
    prepared: list[dict[str, Any]] = []
    for scenario in source.get("data", []):
        uid = str(scenario["uid"])
        relevant = [int(value) for value in scenario["relevant_session_index"]]
        if len(relevant) != 2 or relevant[0] >= relevant[1]:
            raise ValueError(
                f"Scenario {uid} must contain chronological old/new session indices"
            )
        evidence_sessions = scenario.get("views", {}).get("evidence_only", {}).get(
            "sessions", []
        )
        by_index = {
            int(session["original_index"]): session for session in evidence_sessions
        }
        if set(relevant) != set(by_index):
            raise ValueError(
                f"Scenario {uid} evidence-only sessions do not match relevant indices"
            )

        old_session, old_metadata = extract_key_dialogue_pair(
            by_index[relevant[0]], str(scenario["M_old"]), uid=uid, state_label="M_old"
        )
        new_session, new_metadata = extract_key_dialogue_pair(
            by_index[relevant[1]], str(scenario["M_new"]), uid=uid, state_label="M_new"
        )
        chronological = [old_session, new_session]
        reversed_order = [new_session, old_session]
        queries = dict(scenario["probing_queries"])
        chronological_counts = _input_token_counts(tokenizer, chronological, queries)
        reversed_counts = _input_token_counts(tokenizer, reversed_order, queries)
        if max([*chronological_counts.values(), *reversed_counts.values()]) > input_limit:
            raise ValueError(f"Scenario {uid} key dialogue pairs exceed the input limit")

        common_view = {
            "kept_session_indices": relevant,
            "dropped_session_indices": sorted(
                set(range(int(scenario["source_session_count"]))) - set(relevant)
            ),
            "preserve_input_order": True,
            "message_count": 4,
        }
        prepared.append(
            {
                key: scenario[key]
                for key in (
                    "uid",
                    "type",
                    "type_name",
                    "M_old",
                    "M_new",
                    "explanation",
                    "probing_queries",
                    "relevant_session_index",
                    "source_session_count",
                )
            }
            | {
                "key_pair_extraction": {
                    "M_old": old_metadata,
                    "M_new": new_metadata,
                },
                "views": {
                    "key_pairs_chronological": common_view
                    | {
                        "presentation_session_indices": relevant,
                        "sessions": chronological,
                        "input_token_counts": chronological_counts,
                    },
                    "key_pairs_reversed": common_view
                    | {
                        "presentation_session_indices": list(reversed(relevant)),
                        "sessions": reversed_order,
                        "input_token_counts": reversed_counts,
                    },
                },
            }
        )

    return {
        "schema_version": SCENARIO_SCHEMA_VERSION,
        "created_at": utc_now(),
        "dataset": dict(source["dataset"]),
        "model_tokenizer": dict(source["model_tokenizer"]),
        "prompt_version": source["prompt_version"],
        "context_views": dict(KEY_PAIR_CONTEXT_VIEWS),
        "sampling": dict(source["sampling"]),
        "token_budget": dict(source["token_budget"]),
        "derived_from": {
            "scenarios_sha256": source_sha256,
            "reference": source_reference or "",
            "context_views": list(source.get("context_views", {})),
        },
        "key_pair_extraction": {
            "version": EXTRACTION_VERSION,
            "minimum_target_token_coverage": MIN_TARGET_TOKEN_COVERAGE,
            "minimum_best_match_margin": MIN_BEST_MATCH_MARGIN,
            "messages_per_scenario": 4,
        },
        "data": prepared,
    }


def prepare_key_pair_views(
    source_scenarios: Path,
    artifact_root: Path,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    run_id: str | None = None,
    description: str | None = None,
    source_reference: str | None = None,
) -> Path:
    source_sha = sha256_file(source_scenarios)
    destination = artifact_root / "scenarios.json"
    manifest_path = artifact_root / "preparation_manifest.json"
    if destination.exists() and not force:
        existing = load_json(destination)
        if existing.get("derived_from", {}).get("scenarios_sha256") != source_sha:
            raise ValueError(
                "Existing derived scenarios came from a different source; use a new run ID"
            )
        if existing.get("context_views") != dict(KEY_PAIR_CONTEXT_VIEWS):
            raise ValueError("Existing scenarios do not contain the key-pair experiment")
        if existing.get("key_pair_extraction", {}).get("version") != EXTRACTION_VERSION:
            raise ValueError(
                "Existing scenarios use a different extraction version; use a new run ID"
            )
        _write_run_metadata(artifact_root, run_id=run_id, description=description)
        return destination

    views_root = artifact_root / "views"
    has_responses = views_root.exists() and any(views_root.glob("*/checkpoints/*.json"))
    if force and has_responses:
        raise ValueError(
            "Refusing to replace scenarios alongside existing responses. Use a new run ID."
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "Preparation dependencies are missing. Run `uv sync --locked` first."
        ) from error

    source = load_json(source_scenarios)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(cache_dir) if cache_dir else None,
    )
    artifact = build_key_pair_artifact(
        source,
        tokenizer,
        source_sha256=source_sha,
        source_reference=source_reference,
    )
    atomic_write_json(destination, artifact)
    atomic_write_json(
        manifest_path,
        {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "package_version": PACKAGE_VERSION,
            "preparation": {
                "created_at": artifact["created_at"],
                "mode": "key_dialogue_pairs_order_experiment",
                "source_scenarios_sha256": source_sha,
                "source_reference": source_reference or "",
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
                "key_pair_extraction": artifact["key_pair_extraction"],
                "scenarios_sha256": sha256_file(destination),
            },
        },
    )
    _write_run_metadata(artifact_root, run_id=run_id, description=description)
    return destination
