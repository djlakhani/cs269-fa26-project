from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from conftest import FakeTokenizer
from stale_qwen_eval.constants import MODEL_ID, MODEL_REVISION
from stale_qwen_eval.io_utils import atomic_write_json, sha256_file
from stale_qwen_eval.meme import (
    MEME_PROMPT_VERSION,
    MEME_SCHEMA_VERSION,
    _build_filler_view,
    build_probes,
    format_sessions,
)
from stale_qwen_eval.meme_runner import (
    MemePaths,
    refresh_manual_scores,
    run_meme_view,
    summarize,
)


def make_episode() -> dict[str, Any]:
    sessions = []
    for index in range(6):
        content = (
            f"evidence-{index}"
            if index in {0, 2, 5}
            else f"filler-{index}-" + ("x" * 500)
        )
        sessions.append({
            "session_id": f"s{index}",
            "type": "evidence" if index in {0, 2, 5} else "filler",
            "timestamp": f"2026/01/0{index + 1}",
            "conversation": [
                {"role": "user", "content": content},
                {"role": "assistant", "content": "ack"},
            ],
        })
    before = [
        {
            "task_type": "Cas",
            "entity": ["medicine"],
            "entity_values": {"medicine": "oldmed"},
            "question": "What medicine?",
            "expected_answer": "oldmed",
            "hop": 1,
        },
        {
            "task_type": "Abs",
            "entity": ["appointment"],
            "entity_values": {"appointment": "weekly"},
            "question": "What appointment?",
            "expected_answer": "weekly",
            "hop": 2,
        },
        {
            "task_type": "Del",
            "entity": ["unused"],
            "entity_values": {"unused": "ignore"},
            "question": "Ignored?",
            "expected_answer": "ignore",
        },
    ]
    after = [
        {
            "task_type": "Cas",
            "entity": ["medicine"],
            "entity_values": {"medicine": "newmed"},
            "question": "What medicine?",
            "gold_answer": "newmed",
            "hop": 1,
        },
        {
            "task_type": "Abs",
            "entity": ["appointment"],
            "entity_values": {"appointment": "weekly"},
            "question": "What appointment?",
            "gold_answer": "Uncertain",
            "hop": 2,
        },
        {
            "task_type": "ER",
            "entity": ["unused"],
            "entity_values": {"unused": "ignore"},
            "question": "Ignored?",
            "gold_answer": "ignore",
        },
    ]
    return {
        "episode_id": "pl_001",
        "domain": "personal_life",
        "root": "condition",
        "root_change": {"before": "old", "after": "new"},
        "has_2hop": True,
        "evidence_session_indices": [0, 2, 5],
        "sessions": sessions,
        "before_questions": {
            "position_after_session": 2,
            "questions": before,
        },
        "after_questions": {
            "position_after_session": 5,
            "questions": after,
        },
    }


class FakeBackend:
    def __init__(self, counts: list[int]) -> None:
        self.counts = counts
        self.calls = 0

    def runtime_metadata(self) -> dict[str, Any]:
        return {"backend": "fake"}

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        count = self.counts[self.calls]
        self.calls += 1
        return {
            "response": f"answer-{self.calls}",
            "input_tokens": count,
            "output_tokens": 2,
            "elapsed_seconds": 0.01,
            "hit_token_limit": False,
            "thinking_marker_detected": False,
        }


class MemeTests(unittest.TestCase):
    def test_author_format_and_task_filtering(self) -> None:
        episode = make_episode()
        probes = build_probes(episode)
        self.assertEqual(len(probes), 4)
        self.assertEqual({probe["task_type"] for probe in probes}, {"Cas", "Abs"})
        after = [probe for probe in probes if probe["phase"] == "after"]
        self.assertTrue(all(probe["prerequisite_probe_id"] for probe in after))
        transcript = format_sessions(
            [{"timestamp": "T", "conversation": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]}],
            1,
        )
        self.assertEqual(
            transcript, "[Session: T]\nUser: hello\nAssistant: world"
        )

    def test_filler_truncation_preserves_evidence_and_whole_sessions(self) -> None:
        episode = make_episode()
        tokenizer = FakeTokenizer()
        view = _build_filler_view(
            episode,
            build_probes(episode),
            tokenizer,
            seed=42,
            input_target=1400,
        )
        self.assertTrue({0, 2, 5}.issubset(view["kept_session_indices"]))
        self.assertLessEqual(view["maximum_input_tokens"], 1400)
        self.assertEqual(
            view["kept_session_indices"], sorted(view["kept_session_indices"])
        )
        self.assertTrue(view["dropped_session_indices"])
        self.assertTrue(all(
            len(session["conversation"]) == 2 for session in view["sessions"]
        ))
        self.assertTrue(all(
            value is False for value in tokenizer.enable_thinking_values
        ))

    def test_resume_artifacts_and_manual_trivial_pass_summary(self) -> None:
        episode = make_episode()
        probes = build_probes(episode)
        sessions = [
            {"original_index": index, **session}
            for index, session in enumerate(episode["sessions"])
            if index in {0, 2, 5}
        ]
        tokenizer = FakeTokenizer()
        positions = {"before": 2, "after": 3}
        counts = {
            probe["probe_id"]: len(tokenizer.apply_chat_template(
                [
                    {
                        "role": "user",
                        "content": (
                            "Answer the user's question based ONLY on the context "
                            "provided below."
                        ),
                    }
                ],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            ))
            for probe in probes
        }
        # The fake backend validates persistence/resume; use a uniform prepared
        # count rather than duplicating tokenizer rendering details in this test.
        counts = {probe["probe_id"]: 50 for probe in probes}
        artifact = {
            "schema_version": MEME_SCHEMA_VERSION,
            "benchmark": "MEME",
            "dataset": {"id": "meme-benchmark/MEME", "revision": "test"},
            "model_tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "prompt_version": MEME_PROMPT_VERSION,
            "context_views": {"nofiller": "test"},
            "selection": {
                "seed": 42,
                "after_question_count": 2,
                "before_prerequisite_count": 2,
            },
            "token_budget": {
                "context_window": 32768,
                "max_new_tokens": 500,
                "hard_input_limit": 32268,
                "filler_input_target": 29000,
            },
            "data": [{
                "episode_id": episode["episode_id"],
                "domain": episode["domain"],
                "root": episode["root"],
                "root_change": episode["root_change"],
                "has_2hop": True,
                "probes": probes,
                "views": {
                    "nofiller": {
                        "source_config": "nofiller",
                        "source_session_count": 3,
                        "kept_session_indices": [0, 1, 2],
                        "dropped_session_indices": [],
                        "distractor_removal_order": [],
                        "sessions": sessions,
                        "phase_positions": positions,
                        "input_token_counts": counts,
                        "maximum_input_tokens": 50,
                    }
                },
            }],
        }
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = MemePaths(root, "nofiller")
            atomic_write_json(paths.episodes, artifact)
            atomic_write_json(paths.preparation_manifest, {
                "preparation": {
                    "scenarios_sha256": sha256_file(paths.episodes)
                }
            })
            backend = FakeBackend([50] * 4)
            generated = run_meme_view(
                root,
                view="nofiller",
                resume=True,
                show_progress=False,
                backend=backend,
            )
            self.assertEqual(generated, 4)
            self.assertEqual(backend.calls, 4)
            self.assertEqual(len(paths.responses.read_text().splitlines()), 4)
            self.assertTrue(paths.report.exists())

            refresh_manual_scores(paths, artifact)
            with paths.manual_scores.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["before_score"] = "pass"
            rows[0]["after_score"] = "pass"
            rows[1]["before_score"] = "fail"
            rows[1]["after_score"] = "pass"
            with paths.manual_scores.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            result = summarize(paths, artifact)
            self.assertTrue(result["complete"])
            self.assertEqual(result["metrics"]["overall"]["passed"], 1)
            self.assertEqual(result["outcomes"]["real_pass"], 1)
            self.assertEqual(result["outcomes"]["trivial_pass_filtered"], 1)


if __name__ == "__main__":
    unittest.main()
