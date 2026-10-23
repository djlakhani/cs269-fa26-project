from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from stale_qwen_eval.io_utils import atomic_write_json
from stale_qwen_eval.preparation import prepare_from_huggingface


class PreparationTests(unittest.TestCase):
    def test_existing_frozen_scenarios_must_match_requested_settings(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scenario_path = root / "scenarios.json"
            atomic_write_json(
                scenario_path,
                {
                    "schema_version": 1,
                    "sampling": {"seed": 42, "samples_per_type": 5},
                    "token_budget": {
                        "context_window": 32_768,
                        "max_new_tokens": 512,
                    },
                },
            )
            self.assertEqual(prepare_from_huggingface(root), scenario_path)
            with self.assertRaisesRegex(ValueError, "different|requested"):
                prepare_from_huggingface(root, seed=7)

