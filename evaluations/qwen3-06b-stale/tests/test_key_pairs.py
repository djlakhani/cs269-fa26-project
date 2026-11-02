from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from stale_qwen_eval.constants import (
    DATASET_ID,
    DATASET_REVISION,
    MODEL_ID,
    MODEL_REVISION,
    PROMPT_VERSION,
)
from stale_qwen_eval.key_pairs import (
    KeyPairExtractionError,
    build_key_pair_artifact,
    extract_key_dialogue_pair,
)
from stale_qwen_eval.io_utils import atomic_write_json
from stale_qwen_eval.persistence import ArtifactPaths
from stale_qwen_eval.prompts import format_haystack
from stale_qwen_eval.reporting import build_report
from stale_qwen_eval.runner import run_evaluation

from conftest import FakeTokenizer, make_record


def source_artifact() -> dict:
    raw = make_record("key-pair-one", "T1", session_count=6)
    old_index, new_index = raw["relevant_session_index"]
    raw["haystack_session"][old_index] = [
        {"role": "user", "content": "Let us discuss moving."},
        {"role": "assistant", "content": "Sure."},
        {
            "role": "user",
            "content": f"For context, {raw['M_old']} What should I plan?",
        },
        {"role": "assistant", "content": "Plan around Seattle for now."},
        {"role": "user", "content": "Anything else?"},
        {"role": "assistant", "content": "No."},
    ]
    raw["haystack_session"][new_index] = [
        {"role": "user", "content": "I have an update."},
        {"role": "assistant", "content": "What changed?"},
        {
            "role": "user",
            "content": (
                "I actually need to mention this: the user is unpacking in an "
                "Austin apartment. What should happen next?"
            ),
        },
        {"role": "assistant", "content": "Plan around Austin now."},
    ]
    sessions = [
        {
            "original_index": index,
            "timestamp": raw["timestamps"][index],
            "messages": raw["haystack_session"][index],
        }
        for index in (old_index, new_index)
    ]
    return {
        "schema_version": 2,
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": "train"},
        "model_tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "prompt_version": PROMPT_VERSION,
        "context_views": {"evidence_only": "source evidence"},
        "sampling": {"seed": 42, "samples_per_type": 1, "total_scenarios": 1},
        "token_budget": {
            "context_window": 100_000,
            "max_new_tokens": 512,
            "input_token_limit": 99_488,
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
                "source_session_count": len(raw["haystack_session"]),
                "views": {"evidence_only": {"sessions": sessions}},
            }
        ],
    }


class KeyPairTests(unittest.TestCase):
    def test_builds_four_message_chronological_and_reversed_views(self) -> None:
        artifact = build_key_pair_artifact(
            source_artifact(), FakeTokenizer(), source_sha256="source-sha"
        )
        scenario = artifact["data"][0]
        chronological = scenario["views"]["key_pairs_chronological"]
        reversed_view = scenario["views"]["key_pairs_reversed"]

        self.assertEqual(chronological["presentation_session_indices"], [2, 5])
        self.assertEqual(reversed_view["presentation_session_indices"], [5, 2])
        self.assertEqual(sum(len(s["messages"]) for s in chronological["sessions"]), 4)
        self.assertEqual(sum(len(s["messages"]) for s in reversed_view["sessions"]), 4)
        self.assertEqual(
            [session["timestamp"] for session in reversed_view["sessions"]],
            ["2025-01-06T12:00:00", "2025-01-03T12:00:00"],
        )
        rendered = format_haystack(
            reversed_view["sessions"], preserve_input_order=True
        )
        self.assertLess(rendered.index("Session 6"), rendered.index("Session 3"))
        self.assertIn("Austin apartment", rendered)
        self.assertIn("lives in Seattle", rendered)
        self.assertTrue(
            scenario["key_pair_extraction"]["M_new"]["normalized_contains_target"]
        )
        report = build_report(artifact, {}, "key_pairs_reversed")
        context = report.split("**Presented key-dialogue context**", 1)[1]
        self.assertLess(context.index("Session 6"), context.index("Session 3"))

    def test_extraction_accepts_small_generated_leadin(self) -> None:
        session = {
            "original_index": 4,
            "timestamp": "later",
            "messages": [
                {
                    "role": "user",
                    "content": "I actually signed the paperwork at the clinic today.",
                },
                {"role": "assistant", "content": "Understood."},
            ],
        }
        extracted, metadata = extract_key_dialogue_pair(
            session,
            "I signed the paperwork at the clinic today.",
            uid="fuzzy",
            state_label="M_new",
        )
        self.assertEqual(len(extracted["messages"]), 2)
        self.assertGreaterEqual(metadata["target_token_coverage"], 0.95)
        self.assertFalse(metadata["normalized_contains_target"])

    def test_extraction_fails_on_ambiguous_repeated_fact(self) -> None:
        session = {
            "original_index": 1,
            "timestamp": "time",
            "messages": [
                {"role": "user", "content": "I moved to Austin. What now?"},
                {"role": "assistant", "content": "First answer."},
                {"role": "user", "content": "I moved to Austin. Any other ideas?"},
                {"role": "assistant", "content": "Second answer."},
            ],
        }
        with self.assertRaisesRegex(KeyPairExtractionError, "not reliable"):
            extract_key_dialogue_pair(
                session,
                "I moved to Austin.",
                uid="ambiguous",
                state_label="M_new",
            )

    def test_runner_preserves_reversed_pair_order_in_model_prompt(self) -> None:
        class RecordingBackend:
            def __init__(self) -> None:
                self.messages: list[list[dict[str, str]]] = []

            def runtime_metadata(self) -> dict[str, Any]:
                return {"backend": "recording"}

            def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
                self.messages.append(messages)
                rendered = "".join(
                    f"<{message['role']}>{message['content']}</{message['role']}>"
                    for message in messages
                ) + "<assistant>"
                return {
                    "response": "answer",
                    "input_tokens": len(rendered),
                    "output_tokens": 1,
                    "elapsed_seconds": 0.01,
                    "hit_token_limit": False,
                    "thinking_marker_detected": False,
                }

        artifact = build_key_pair_artifact(
            source_artifact(), FakeTokenizer(), source_sha256="source-sha"
        )
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            atomic_write_json(
                ArtifactPaths(root, "key_pairs_reversed").scenarios, artifact
            )
            backend = RecordingBackend()
            generated = run_evaluation(
                root,
                context_view="key_pairs_reversed",
                resume=True,
                max_queries=1,
                show_progress=False,
                backend=backend,
            )
            self.assertEqual(generated, 1)
            prompt = backend.messages[0][1]["content"]
            self.assertLess(prompt.index("Session 6"), prompt.index("Session 3"))
            self.assertIn("Time: 2025-01-06T12:00:00", prompt)
            self.assertIn("Time: 2025-01-03T12:00:00", prompt)
