from __future__ import annotations

import unittest

from stale_qwen_eval.sampling import select_balanced_scenarios

from conftest import make_record


class SamplingTests(unittest.TestCase):
    def test_balanced_sampling_is_deterministic(self) -> None:
        records = [make_record(f"t1-{index:02d}", "T1") for index in range(12)]
        records += [make_record(f"t2-{index:02d}", "T2") for index in range(12)]

        first = select_balanced_scenarios(records, samples_per_type=5, seed=42)
        second = select_balanced_scenarios(reversed(records), samples_per_type=5, seed=42)

        self.assertEqual(
            [record["uid"] for record in first],
            [record["uid"] for record in second],
        )
        self.assertEqual(sum(record["type"] == "T1" for record in first), 5)
        self.assertEqual(sum(record["type"] == "T2" for record in first), 5)

    def test_sampling_rejects_missing_type_population(self) -> None:
        records = [make_record(f"t1-{index}", "T1") for index in range(5)]
        with self.assertRaisesRegex(ValueError, "T2"):
            select_balanced_scenarios(records, samples_per_type=5, seed=42)

