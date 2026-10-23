from __future__ import annotations

import random
from collections.abc import Iterable, Mapping
from typing import Any

from .constants import TYPE_INFO


def select_balanced_scenarios(
    records: Iterable[Mapping[str, Any]],
    *,
    samples_per_type: int,
    seed: int,
) -> list[dict[str, Any]]:
    if samples_per_type <= 0:
        raise ValueError("samples_per_type must be positive")

    grouped: dict[str, list[dict[str, Any]]] = {key: [] for key in TYPE_INFO}
    for raw_record in records:
        record = dict(raw_record)
        conflict_type = str(record.get("type"))
        if conflict_type in grouped:
            grouped[conflict_type].append(record)

    selected: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for conflict_type in TYPE_INFO:
        candidates = sorted(grouped[conflict_type], key=lambda item: str(item["uid"]))
        if len(candidates) < samples_per_type:
            raise ValueError(
                f"Need {samples_per_type} {conflict_type} scenarios, found {len(candidates)}"
            )
        selected.extend(rng.sample(candidates, samples_per_type))

    return sorted(selected, key=lambda item: (str(item["type"]), str(item["uid"])))

