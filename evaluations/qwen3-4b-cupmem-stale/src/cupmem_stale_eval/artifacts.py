from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io_utils import safe_uid


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def scenarios_file(self) -> Path:
        return self.root / "scenarios.json"

    @property
    def preparation_manifest(self) -> Path:
        return self.root / "preparation_manifest.json"

    @property
    def run_manifest(self) -> Path:
        return self.root / "run_manifest.json"

    @property
    def progress(self) -> Path:
        return self.root / "progress.json"

    @property
    def responses(self) -> Path:
        return self.root / "responses.jsonl"

    def scenario(self, uid: str) -> "ScenarioPaths":
        return ScenarioPaths(self.root / "scenarios" / safe_uid(uid))

    @property
    def judge_root(self) -> Path:
        return self.root / "judge"

    @property
    def diagnostics_root(self) -> Path:
        return self.root / "diagnostics"

    @property
    def review_root(self) -> Path:
        return self.root / "review"

    @property
    def reports_root(self) -> Path:
        return self.root / "reports"


@dataclass(frozen=True)
class ScenarioPaths:
    root: Path

    @property
    def input(self) -> Path:
        return self.root / "input.json"

    @property
    def status(self) -> Path:
        return self.root / "status.json"

    @property
    def engine_state(self) -> Path:
        return self.root / "engine_state.json.zst"

    @property
    def llm_calls(self) -> Path:
        return self.root / "llm_calls.jsonl.zst"

    @property
    def memory_root(self) -> Path:
        return self.root / "memory"

    @property
    def final_profile(self) -> Path:
        return self.memory_root / "final_profile.json"

    @property
    def memory_markdown(self) -> Path:
        return self.memory_root / "memory.md"

    @property
    def session_traces(self) -> Path:
        return self.memory_root / "session_traces.jsonl.zst"

    def query_root(self, dimension_key: str) -> Path:
        return self.root / "queries" / dimension_key.replace("_query", "")
