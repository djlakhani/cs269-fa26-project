from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_jsonl_zst, load_jsonl_zst


class AuditCollector:
    """Thread-safe in-memory audit collector with atomic compressed snapshots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        with self._lock:
            enriched = dict(event)
            enriched["sequence"] = len(self._events) + 1
            if enriched.get("event") in {"json_parse", "json_parse_error"}:
                identity_fields = ("phase", "session_id", "query_label", "lane_name")
                for prior in reversed(self._events):
                    if prior.get("event") != "llm_attempt":
                        continue
                    if any(
                        enriched.get(field) and prior.get(field) != enriched.get(field)
                        for field in identity_fields
                    ):
                        continue
                    if enriched.get("event") == "json_parse":
                        prior["json_parse_status"] = "parsed"
                        prior["parsed_json"] = enriched.get("parsed_json")
                        prior["json_parse_mode"] = enriched.get("parse_mode")
                        prior["json_repair_succeeded"] = bool(enriched.get("retry"))
                    else:
                        prior["json_parse_status"] = "invalid"
                        prior["json_parse_error"] = enriched.get("parse_error")
                        prior["json_repair_will_retry"] = bool(enriched.get("retry"))
                    break
            self._events.append(enriched)

    def reset(self) -> None:
        with self._lock:
            self._events = []

    def load(self, path: Path) -> None:
        events = load_jsonl_zst(path)
        with self._lock:
            self._events = events

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(event) for event in self._events]

    def flush(self, path: Path) -> None:
        atomic_write_jsonl_zst(path, self.snapshot())

    def answer_composer_request(self, query_label: str) -> dict[str, Any]:
        matches = [
            event
            for event in self.snapshot()
            if event.get("event") == "llm_attempt"
            and event.get("phase") == "answer_composer"
            and event.get("query_label") == query_label
        ]
        if not matches:
            return {}
        event = matches[-1]
        return {
            "model": event.get("model"),
            "messages": event.get("messages", []),
            "request_kwargs": event.get("request_kwargs", {}),
            "raw_response": event.get("raw_response", ""),
            "usage": event.get("usage", {}),
            "audit_sequence": event.get("sequence"),
        }
