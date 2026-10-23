from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import DIMENSION_KEYS
from .io_utils import atomic_write_json, atomic_write_text, read_jsonl


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    @property
    def scenarios(self) -> Path:
        return self.root / "scenarios.json"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def responses(self) -> Path:
        return self.root / "responses.jsonl"

    @property
    def report(self) -> Path:
        return self.root / "results.md"

    @property
    def manual_scores(self) -> Path:
        return self.root / "manual_scores.csv"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    @property
    def summary_markdown(self) -> Path:
        return self.root / "summary.md"


def response_key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["uid"]), str(record["dimension_key"])


def checkpoint_path(paths: ArtifactPaths, uid: str, dimension_key: str) -> Path:
    safe_uid = "".join(character for character in uid if character.isalnum() or character in "-_")
    if not safe_uid:
        raise ValueError(f"Unsafe or empty UID: {uid!r}")
    if dimension_key not in DIMENSION_KEYS:
        raise ValueError(f"Unknown dimension: {dimension_key}")
    return paths.checkpoints / f"{safe_uid}__{dimension_key}.json"


def load_response_map(paths: ArtifactPaths) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if paths.responses.exists():
        for record in read_jsonl(paths.responses):
            records[response_key(record)] = record
    if paths.checkpoints.exists():
        for path in sorted(paths.checkpoints.glob("*.json")):
            with path.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            records[response_key(record)] = record
    return records


def write_response_checkpoint(paths: ArtifactPaths, record: dict[str, Any]) -> None:
    path = checkpoint_path(paths, str(record["uid"]), str(record["dimension_key"]))
    atomic_write_json(path, record)


def ordered_keys(scenario_artifact: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(scenario["uid"]), dimension_key)
        for scenario in scenario_artifact["data"]
        for dimension_key in DIMENSION_KEYS
    ]


def rebuild_responses_jsonl(
    paths: ArtifactPaths,
    scenario_artifact: dict[str, Any],
    records: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> None:
    records = records if records is not None else load_response_map(paths)
    lines = [
        json.dumps(records[key], ensure_ascii=False)
        for key in ordered_keys(scenario_artifact)
        if key in records
    ]
    atomic_write_text(paths.responses, "\n".join(lines) + ("\n" if lines else ""))

