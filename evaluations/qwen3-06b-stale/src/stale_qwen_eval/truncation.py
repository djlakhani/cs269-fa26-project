from __future__ import annotations

import random
from collections.abc import Mapping
from typing import Any

from .constants import DIMENSION_KEYS, TYPE_INFO
from .prompts import count_prompt_tokens


class EvidenceTooLongError(ValueError):
    """Raised when the mandatory evidence sessions cannot fit the input budget."""


def _indexed_sessions(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_sessions = item.get("haystack_session")
    if not isinstance(raw_sessions, list) or not raw_sessions:
        raise ValueError(f"Scenario {item.get('uid')} has no haystack_session list")
    timestamps = item.get("timestamps") or []
    if timestamps and len(timestamps) != len(raw_sessions):
        raise ValueError(
            f"Scenario {item.get('uid')} has {len(raw_sessions)} sessions but "
            f"{len(timestamps)} timestamps"
        )
    return [
        {
            "original_index": index,
            "timestamp": timestamps[index] if timestamps else None,
            "messages": session,
        }
        for index, session in enumerate(raw_sessions)
    ]


def _token_counts(
    tokenizer: Any,
    sessions: list[dict[str, Any]],
    queries: Mapping[str, str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for dimension_key in DIMENSION_KEYS:
        query = queries.get(dimension_key)
        if not query:
            raise ValueError(f"Missing probing query: {dimension_key}")
        counts[dimension_key] = count_prompt_tokens(
            tokenizer, sessions, str(query), dimension_key
        )
    return counts


def truncate_scenario(
    item: Mapping[str, Any],
    tokenizer: Any,
    *,
    input_token_limit: int,
    seed: int,
) -> dict[str, Any]:
    uid = str(item["uid"])
    conflict_type = str(item["type"])
    if conflict_type not in TYPE_INFO:
        raise ValueError(f"Scenario {uid} has unknown conflict type {conflict_type!r}")
    if input_token_limit <= 0:
        raise ValueError("input_token_limit must be positive")

    evidence_indices = [int(index) for index in item["relevant_session_index"]]
    if len(evidence_indices) != 2 or len(set(evidence_indices)) != 2:
        raise ValueError(f"Scenario {uid} must identify two distinct evidence sessions")

    all_sessions = _indexed_sessions(item)
    all_indices = set(range(len(all_sessions)))
    if not set(evidence_indices).issubset(all_indices):
        raise ValueError(f"Scenario {uid} has out-of-range evidence indices {evidence_indices}")

    queries = dict(item["probing_queries"])
    required_sessions = [all_sessions[index] for index in sorted(evidence_indices)]
    required_counts = _token_counts(tokenizer, required_sessions, queries)
    if max(required_counts.values()) > input_token_limit:
        raise EvidenceTooLongError(
            f"Scenario {uid} mandatory evidence requires up to "
            f"{max(required_counts.values())} input tokens, above limit {input_token_limit}"
        )

    distractor_order = sorted(all_indices - set(evidence_indices))
    random.Random(f"{seed}:{uid}").shuffle(distractor_order)

    dropped: set[int] = set()
    retained_sessions = all_sessions
    token_counts = _token_counts(tokenizer, retained_sessions, queries)
    removed_count = 0
    while max(token_counts.values()) > input_token_limit:
        if removed_count >= len(distractor_order):
            raise EvidenceTooLongError(
                f"Scenario {uid} could not fit after removing every distractor"
            )
        dropped.add(distractor_order[removed_count])
        removed_count += 1
        retained_sessions = [
            session
            for session in all_sessions
            if int(session["original_index"]) not in dropped
        ]
        token_counts = _token_counts(tokenizer, retained_sessions, queries)

    kept_indices = [int(session["original_index"]) for session in retained_sessions]
    if not set(evidence_indices).issubset(kept_indices):
        raise AssertionError(f"Internal error: evidence was removed for scenario {uid}")
    if max(token_counts.values()) > input_token_limit:
        raise AssertionError(f"Internal error: scenario {uid} remains over token budget")

    return {
        "uid": uid,
        "type": conflict_type,
        "type_name": TYPE_INFO[conflict_type],
        "M_old": str(item["M_old"]),
        "M_new": str(item["M_new"]),
        "explanation": str(item["explanation"]),
        "probing_queries": queries,
        "relevant_session_index": evidence_indices,
        "source_session_count": len(all_sessions),
        "kept_session_indices": kept_indices,
        "dropped_session_indices": sorted(dropped),
        "distractor_removal_order": distractor_order[:removed_count],
        "sessions": retained_sessions,
        "input_token_counts": token_counts,
    }

