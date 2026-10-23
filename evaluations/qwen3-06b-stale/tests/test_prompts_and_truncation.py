from __future__ import annotations

import unittest

from stale_qwen_eval.prompts import (
    EXPLICIT_SYSTEM_PROMPT,
    IMPLICIT_SYSTEM_PROMPT,
    build_messages,
    count_prompt_tokens,
    format_haystack,
)
from stale_qwen_eval.truncation import EvidenceTooLongError, truncate_scenario

from conftest import FakeTokenizer, make_record


class PromptAndTruncationTests(unittest.TestCase):
    def test_author_prompt_shapes_and_thinking_disabled(self) -> None:
        explicit = build_messages("history", "question", "dim1_query")
        implicit = build_messages("history", "request", "dim3_query")
        self.assertEqual(explicit[0]["content"], EXPLICIT_SYSTEM_PROMPT)
        self.assertIn("[Question]\nquestion", explicit[1]["content"])
        self.assertEqual(implicit[0]["content"], IMPLICIT_SYSTEM_PROMPT)
        self.assertIn("[Latest Query]\nrequest", implicit[1]["content"])

        tokenizer = FakeTokenizer()
        count_prompt_tokens(
            tokenizer,
            [{"original_index": 0, "timestamp": None, "messages": []}],
            "question",
            "dim1_query",
        )
        self.assertEqual(tokenizer.enable_thinking_values, [False])

    def test_truncation_preserves_evidence_order_timestamps_and_budget(self) -> None:
        tokenizer = FakeTokenizer()
        record = make_record("scenario-a", "T1")
        full_sessions = [
            {
                "original_index": index,
                "timestamp": record["timestamps"][index],
                "messages": session,
            }
            for index, session in enumerate(record["haystack_session"])
        ]
        full_max = max(
            count_prompt_tokens(tokenizer, full_sessions, query, dimension)
            for dimension, query in record["probing_queries"].items()
        )

        prepared = truncate_scenario(
            record,
            tokenizer,
            input_token_limit=full_max - 200,
            seed=42,
        )

        self.assertTrue(
            set(record["relevant_session_index"]).issubset(prepared["kept_session_indices"])
        )
        self.assertEqual(
            prepared["kept_session_indices"], sorted(prepared["kept_session_indices"])
        )
        self.assertTrue(prepared["dropped_session_indices"])
        self.assertLessEqual(max(prepared["input_token_counts"].values()), full_max - 200)
        for session in prepared["sessions"]:
            index = session["original_index"]
            self.assertEqual(session["timestamp"], record["timestamps"][index])
            self.assertEqual(session["messages"], record["haystack_session"][index])

        formatted = format_haystack(prepared["sessions"])
        self.assertIn("=== Session 3 [Time:", formatted)
        self.assertIn("=== Session 6 [Time:", formatted)

    def test_truncation_is_deterministic(self) -> None:
        record = make_record("scenario-b", "T2")
        first = truncate_scenario(record, FakeTokenizer(), input_token_limit=850, seed=42)
        second = truncate_scenario(record, FakeTokenizer(), input_token_limit=850, seed=42)
        self.assertEqual(first["kept_session_indices"], second["kept_session_indices"])
        self.assertEqual(
            first["distractor_removal_order"], second["distractor_removal_order"]
        )

    def test_mandatory_evidence_overflow_fails(self) -> None:
        with self.assertRaisesRegex(EvidenceTooLongError, "mandatory evidence"):
            truncate_scenario(
                make_record("scenario-c", "T1"),
                FakeTokenizer(),
                input_token_limit=10,
                seed=42,
            )

