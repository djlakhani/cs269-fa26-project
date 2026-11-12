from __future__ import annotations

import random
import unittest

from cupmem_stale_eval.preparation import build_scenario_artifact
from cupmem_stale_eval.runner import _restricted_scenario
from cupmem_stale_eval.sampling import (
    scenario_seed,
    select_balanced_scenarios,
    select_session_indices,
)

from conftest import make_record


class SamplingAndPreparationTests(unittest.TestCase):
    def test_balanced_selection_sorts_then_uses_fresh_seed_per_type(self) -> None:
        records = [make_record(f"{kind}-{index:03d}", kind) for kind in ("T1", "T2") for index in range(60)]
        random.Random(99).shuffle(records)
        selected = select_balanced_scenarios(records, samples_per_type=5, seed=42)
        for kind in ("T1", "T2"):
            candidates = sorted(
                [record for record in records if record["type"] == kind],
                key=lambda record: record["uid"],
            )
            expected = sorted(item["uid"] for item in random.Random(42).sample(candidates, 5))
            actual = [item["uid"] for item in selected if item["type"] == kind]
            self.assertEqual(actual, expected)

    def test_uid_seeded_sessions_retain_evidence_and_chronology(self) -> None:
        record = make_record("stable-uid", "T1")
        first = select_session_indices(record, distractor_count=10, seed=42)
        second = select_session_indices(record, distractor_count=10, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(first["scenario_seed"], scenario_seed(42, "stable-uid"))
        self.assertEqual(len(first["kept_indices"]), 12)
        self.assertEqual(first["kept_indices"], sorted(first["kept_indices"]))
        self.assertTrue({7, 41}.issubset(first["kept_indices"]))

    def test_artifact_preserves_source_indices_and_isolates_ground_truth(self) -> None:
        artifact = build_scenario_artifact(
            [make_record("one", "T1"), make_record("two", "T2")],
            samples_per_type=1,
            distractors_per_scenario=10,
        )
        scenario = artifact["data"][0]
        self.assertEqual(len(scenario["sessions"]), 12)
        self.assertEqual(
            [item["original_index"] for item in scenario["sessions"]],
            scenario["kept_session_indices"],
        )
        restricted = _restricted_scenario(scenario)
        for forbidden in ("M_old", "M_new", "explanation"):
            self.assertNotIn(forbidden, restricted)
        self.assertEqual(restricted["probing_queries"], scenario["probing_queries"])

    def test_invalid_source_shapes_are_rejected(self) -> None:
        record = make_record("bad", "T1")
        record["haystack_session"].pop()
        with self.assertRaisesRegex(ValueError, "50 sessions"):
            select_session_indices(record, distractor_count=10, seed=42)


if __name__ == "__main__":
    unittest.main()
