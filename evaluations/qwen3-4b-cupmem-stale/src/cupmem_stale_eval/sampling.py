from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable, Mapping
from typing import Any

from .constants import SOURCE_SESSION_COUNT, TYPE_NAMES


def select_balanced_scenarios(
    records: Iterable[Mapping[str, Any]], *, samples_per_type: int, seed: int
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in TYPE_NAMES}
    for raw in records:
        record = dict(raw)
        if str(record.get("type")) in grouped:
            grouped[str(record["type"])].append(record)
    selected: list[dict[str, Any]] = []
    for conflict_type in TYPE_NAMES:
        candidates = sorted(grouped[conflict_type], key=lambda item: str(item["uid"]))
        if len(candidates) < samples_per_type:
            raise ValueError(
                f"Need {samples_per_type} {conflict_type} scenarios, found {len(candidates)}"
            )
        selected.extend(random.Random(seed).sample(candidates, samples_per_type))
    return sorted(selected, key=lambda item: (str(item["type"]), str(item["uid"])))


def scenario_seed(global_seed: int, uid: str) -> int:
    digest = hashlib.sha256(f"{global_seed}{uid}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def select_session_indices(
    record: Mapping[str, Any], *, distractor_count: int, seed: int
) -> dict[str, list[int] | int]:
    uid = str(record.get("uid", ""))
    sessions = record.get("haystack_session", [])
    if len(sessions) != SOURCE_SESSION_COUNT:
        raise ValueError(f"Scenario {uid} must contain {SOURCE_SESSION_COUNT} sessions")
    evidence = [int(value) for value in record.get("relevant_session_index", [])]
    if len(evidence) != 2 or len(set(evidence)) != 2:
        raise ValueError(f"Scenario {uid} must identify two distinct evidence sessions")
    if evidence != sorted(evidence):
        raise ValueError(f"Scenario {uid} evidence sessions are not chronological")
    if not all(0 <= index < len(sessions) for index in evidence):
        raise ValueError(f"Scenario {uid} has out-of-range evidence indices")
    distractor_pool = sorted(set(range(len(sessions))) - set(evidence))
    if distractor_count > len(distractor_pool):
        raise ValueError(f"Scenario {uid} does not have enough distractors")
    derived_seed = scenario_seed(seed, uid)
    distractors = sorted(random.Random(derived_seed).sample(distractor_pool, distractor_count))
    kept = sorted(evidence + distractors)
    return {
        "scenario_seed": derived_seed,
        "evidence_indices": evidence,
        "distractor_indices": distractors,
        "kept_indices": kept,
        "dropped_indices": sorted(set(range(len(sessions))) - set(kept)),
    }
