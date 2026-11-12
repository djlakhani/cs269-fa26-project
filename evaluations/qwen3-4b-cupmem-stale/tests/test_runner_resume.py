from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cupmem_stale_eval.artifacts import RunPaths
from cupmem_stale_eval.constants import DIMENSION_KEYS
from cupmem_stale_eval.io_utils import atomic_write_json, load_json, sha256_file
from cupmem_stale_eval.preparation import build_scenario_artifact
from cupmem_stale_eval.runner import run_evaluation

from conftest import make_record


class FakeThresholds:
    latent_invalidation_temperature = 0.2


class FakeTraceConfig:
    def __init__(self, **_kwargs) -> None:
        pass


class FakeStore:
    def __init__(self, engine: "FakeEngine") -> None:
        self.engine = engine

    def to_snapshot(self) -> dict:
        return {
            "active_profile": {
                "test/state": [
                    {
                        "item_id": "pi_00001",
                        "value": f"processed {len(self.engine.processed)} sessions",
                        "confidence": 1.0,
                        "active_strength": "STRONG",
                        "evidence_chunk_ids": [],
                        "source_delta_ids": [],
                    }
                ]
            },
            "stale_archive": [],
            "unknown_current": [],
            "stale_support_links": [],
        }


class FakeLLM:
    def __init__(self, **kwargs) -> None:
        self.audit_sink = kwargs["audit_sink"]
        self.cache_dir = kwargs.get("cache_dir")
        self.records: list[dict] = []

    def reset_usage_tracking(self) -> None:
        self.records = []

    def get_call_records(self) -> list[dict]:
        return list(self.records)


class FakeEngine:
    fail_session_once: int | None = None
    fail_query_once: str | None = None

    def __init__(self, *, llm: FakeLLM, **_kwargs) -> None:
        self.llm = llm
        self.processed: list[int] = []
        self.store = FakeStore(self)

    def reset(self) -> None:
        self.processed = []

    def process_session(self, *, session, session_index: int, session_time: str) -> dict:
        del session, session_time
        if FakeEngine.fail_session_once == session_index:
            FakeEngine.fail_session_once = None
            raise RuntimeError("simulated session interruption")
        self.processed.append(session_index)
        self.llm.audit_sink(
            {
                "event": "llm_attempt",
                "phase": "session_chunker",
                "session_id": f"s_{session_index:05d}",
                "messages": [{"role": "user", "content": f"session {session_index}"}],
                "request_kwargs": {"temperature": 0},
                "raw_response": "{}",
            }
        )
        return {"session_index": session_index}

    def export_state(self) -> dict:
        return {
            "schema_version": 1,
            "store": {"processed": list(self.processed)},
            "chunk_bank": [],
            "delta_store": [],
            "counters": {"chunk": len(self.processed), "delta": 0, "proposal": 0},
        }

    def restore_state(self, state: dict) -> None:
        self.processed = list(state["store"]["processed"])

    def answer_query(self, *, query_label: str, query_text: str) -> dict:
        if FakeEngine.fail_query_once == query_label:
            FakeEngine.fail_query_once = None
            raise RuntimeError("simulated query interruption")
        response = f"answer for {query_label} after {len(self.processed)} sessions"
        event = {
            "event": "llm_attempt",
            "phase": "answer_composer",
            "query_label": query_label,
            "messages": [
                {"role": "system", "content": "compose from memory"},
                {"role": "user", "content": query_text},
            ],
            "request_kwargs": {"temperature": 0},
            "raw_response": response,
            "usage": {"total_tokens": 2},
        }
        self.llm.audit_sink(event)
        self.llm.records.append({"phase": "answer_composer", "query_label": query_label})
        return {
            "answer": {"answer": response},
            "verdict": {"status": "SUPPORTED"},
            "debug_trace": {},
        }


def fake_symbols() -> dict:
    return {
        "CupMemEngine": FakeEngine,
        "LLMClient": FakeLLM,
        "PipelineThresholds": FakeThresholds,
        "TraceConfig": FakeTraceConfig,
        "summarize_call_records": lambda records: {"total_calls": len(records)},
    }


def write_artifact(root: Path) -> None:
    artifact = build_scenario_artifact(
        [make_record("resume-one", "T1"), make_record("resume-two", "T2")],
        samples_per_type=1,
        distractors_per_scenario=1,
    )
    paths = RunPaths(root)
    atomic_write_json(paths.scenarios_file, artifact)
    atomic_write_json(
        paths.preparation_manifest,
        {"scenarios_sha256": sha256_file(paths.scenarios_file), "embedding_model_local_path": "/unused"},
    )


class RunnerResumeTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeEngine.fail_session_once = None
        FakeEngine.fail_query_once = None

    def _run(self, root: Path, *, resume: bool = True) -> int:
        with (
            patch("cupmem_stale_eval.runner.load_cupmem_symbols", return_value=fake_symbols()),
            patch("cupmem_stale_eval.runner._resolve_embedding_path", return_value=Path("/unused")),
        ):
            return run_evaluation(
                root,
                cache_dir=root / "cache",
                resume=resume,
                max_per_type=1,
                show_progress=False,
            )

    def test_resume_after_interrupted_session(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            first_scenario = load_json(RunPaths(root).scenarios_file)["data"][0]
            fail_index = first_scenario["sessions"][1]["original_index"]
            FakeEngine.fail_session_once = fail_index
            with self.assertRaisesRegex(RuntimeError, "session interruption"):
                self._run(root)
            status_path = RunPaths(root).scenario(first_scenario["uid"]).status
            interrupted = load_json(status_path)
            self.assertEqual(interrupted["completed_session_indices"], [first_scenario["sessions"][0]["original_index"]])

            generated = self._run(root)
            self.assertEqual(generated, 2)
            completed = load_json(status_path)
            self.assertEqual(completed["phase"], "completed")
            self.assertEqual(completed["completed_session_indices"], first_scenario["kept_session_indices"])

    def test_resume_after_interrupted_query_preserves_completed_result(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            artifact = load_json(RunPaths(root).scenarios_file)
            first = artifact["data"][0]
            FakeEngine.fail_query_once = "dim2_query"
            with self.assertRaisesRegex(RuntimeError, "query interruption"):
                self._run(root)
            first_result = RunPaths(root).scenario(first["uid"]).query_root("dim1_query") / "result.json"
            before = first_result.read_bytes()

            self._run(root)
            self.assertEqual(first_result.read_bytes(), before)
            status = load_json(RunPaths(root).scenario(first["uid"]).status)
            self.assertEqual(status["completed_queries"], list(DIMENSION_KEYS))
            for key in DIMENSION_KEYS:
                result = load_json(RunPaths(root).scenario(first["uid"]).query_root(key) / "result.json")
                self.assertEqual(result["memory_hash_before"], result["memory_hash_after"])

    def test_any_existing_checkpoint_requires_resume(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_artifact(root)
            FakeEngine.fail_session_once = load_json(RunPaths(root).scenarios_file)["data"][0]["sessions"][1]["original_index"]
            with self.assertRaises(RuntimeError):
                self._run(root)
            with self.assertRaisesRegex(ValueError, "--resume"):
                self._run(root, resume=False)


if __name__ == "__main__":
    unittest.main()
