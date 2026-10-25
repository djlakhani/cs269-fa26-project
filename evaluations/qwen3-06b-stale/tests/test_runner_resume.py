from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from stale_qwen_eval.constants import (
    DATASET_ID,
    DATASET_REVISION,
    DIMENSION_KEYS,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_VERSION,
)
from stale_qwen_eval.io_utils import atomic_write_json
from stale_qwen_eval.persistence import ArtifactPaths, load_response_map
from stale_qwen_eval.runner import run_evaluation

from conftest import make_record


class FakeBackend:
    def __init__(self, input_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.calls = 0

    def runtime_metadata(self) -> dict[str, Any]:
        return {"backend": "fake"}

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        self.calls += 1
        return {
            "response": f"answer-{self.calls}",
            "input_tokens": self.input_tokens,
            "output_tokens": 2,
            "elapsed_seconds": 0.01,
            "hit_token_limit": False,
            "thinking_marker_detected": False,
        }


def make_artifact() -> dict[str, Any]:
    raw = make_record("resume-one", "T1", session_count=6)
    sessions = [
        {
            "original_index": index,
            "timestamp": raw["timestamps"][index],
            "messages": session,
        }
        for index, session in enumerate(raw["haystack_session"])
    ]
    return {
        "schema_version": 2,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": "train"},
        "model_tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "prompt_version": PROMPT_VERSION,
        "sampling": {"seed": 42, "samples_per_type": 1, "total_scenarios": 1},
        "token_budget": {
            "context_window": 100,
            "max_new_tokens": 10,
            "input_token_limit": 90,
        },
        "data": [
            {
                "uid": raw["uid"],
                "type": raw["type"],
                "type_name": "Co-referential conflict",
                "M_old": raw["M_old"],
                "M_new": raw["M_new"],
                "explanation": raw["explanation"],
                "probing_queries": raw["probing_queries"],
                "relevant_session_index": raw["relevant_session_index"],
                "source_session_count": len(sessions),
                "views": {
                    "truncated": {
                        "kept_session_indices": list(range(len(sessions))),
                        "dropped_session_indices": [],
                        "distractor_removal_order": [],
                        "sessions": sessions,
                        "input_token_counts": {key: 50 for key in DIMENSION_KEYS},
                    },
                    "evidence_only": {
                        "kept_session_indices": raw["relevant_session_index"],
                        "dropped_session_indices": [0, 1, 3, 4],
                        "sessions": [sessions[2], sessions[5]],
                        "input_token_counts": {key: 50 for key in DIMENSION_KEYS},
                    },
                },
            }
        ],
    }


class RunnerResumeTests(unittest.TestCase):
    def test_smoke_completes_one_scenario_in_each_view_independently(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            atomic_write_json(ArtifactPaths(root, "truncated").scenarios, make_artifact())

            truncated_backend = FakeBackend(50)
            generated = run_evaluation(
                root,
                context_view="truncated",
                resume=True,
                max_scenarios=1,
                show_progress=False,
                backend=truncated_backend,
            )
            self.assertEqual(generated, len(DIMENSION_KEYS))
            self.assertEqual(truncated_backend.calls, len(DIMENSION_KEYS))

            evidence_backend = FakeBackend(50)
            generated = run_evaluation(
                root,
                context_view="evidence_only",
                resume=True,
                max_scenarios=1,
                show_progress=False,
                backend=evidence_backend,
            )
            self.assertEqual(generated, len(DIMENSION_KEYS))
            self.assertEqual(evidence_backend.calls, len(DIMENSION_KEYS))
            self.assertEqual(
                len(load_response_map(ArtifactPaths(root, "truncated"))), 3
            )
            self.assertEqual(
                len(load_response_map(ArtifactPaths(root, "evidence_only"))), 3
            )

    def test_interrupted_run_resumes_without_duplicates(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = ArtifactPaths(root, "truncated")
            atomic_write_json(paths.scenarios, make_artifact())

            first_backend = FakeBackend(50)
            self.assertEqual(
                run_evaluation(
                    root,
                    context_view="truncated",
                    resume=True,
                    max_queries=2,
                    show_progress=False,
                    backend=first_backend,
                ),
                2,
            )
            self.assertEqual(first_backend.calls, 2)

            second_backend = FakeBackend(50)
            self.assertEqual(
                run_evaluation(
                    root,
                    context_view="truncated",
                    resume=True,
                    show_progress=False,
                    backend=second_backend,
                ),
                1,
            )
            self.assertEqual(second_backend.calls, 1)

            responses = load_response_map(paths)
            self.assertEqual(len(responses), 3)
            self.assertEqual(
                len(paths.responses.read_text(encoding="utf-8").splitlines()), 3
            )
            self.assertTrue(paths.report.exists())
            self.assertTrue(paths.manual_scores.exists())

    def test_existing_responses_require_resume(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = ArtifactPaths(root, "truncated")
            atomic_write_json(paths.scenarios, make_artifact())
            run_evaluation(
                root,
                context_view="truncated",
                resume=True,
                max_queries=1,
                show_progress=False,
                backend=FakeBackend(50),
            )
            with self.assertRaisesRegex(ValueError, "--resume"):
                run_evaluation(
                    root,
                    context_view="truncated",
                    resume=False,
                    show_progress=False,
                    backend=FakeBackend(50),
                )

    def test_manifest_detects_scenario_changes_before_generation(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = ArtifactPaths(root, "truncated")
            artifact = make_artifact()
            atomic_write_json(paths.scenarios, artifact)
            run_evaluation(
                root,
                context_view="truncated",
                resume=True,
                max_queries=1,
                show_progress=False,
                backend=FakeBackend(50),
            )

            artifact["data"][0]["M_new"] = "changed after preparation"
            atomic_write_json(paths.scenarios, artifact)
            backend = FakeBackend(50)
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_evaluation(
                    root,
                    context_view="truncated",
                    resume=True,
                    show_progress=False,
                    backend=backend,
                )
            self.assertEqual(backend.calls, 0)

