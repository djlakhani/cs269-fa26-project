from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from stale_qwen_eval.constants import DIMENSION_KEYS
from stale_qwen_eval.persistence import ArtifactPaths, write_response_checkpoint
from stale_qwen_eval.reporting import refresh_manual_scores, summarize_manual_scores


def artifact_with_one_scenario() -> dict:
    return {
        "data": [
            {
                "uid": "one",
                "type": "T1",
                "type_name": "Co-referential conflict",
            }
        ]
    }


class ReportingTests(unittest.TestCase):
    def test_manual_scores_are_preserved_and_summarized(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = ArtifactPaths(Path(temporary_directory))
            artifact = artifact_with_one_scenario()
            refresh_manual_scores(paths, artifact)

            with paths.manual_scores.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["score"] = "pass"
            rows[0]["notes"] = "kept"
            rows[1]["score"] = "fail"
            rows[2]["score"] = "pass"
            with paths.manual_scores.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            for dimension_key in DIMENSION_KEYS:
                write_response_checkpoint(
                    paths,
                    {
                        "uid": "one",
                        "dimension_key": dimension_key,
                        "status": "ok",
                    },
                )

            refresh_manual_scores(paths, artifact)
            with paths.manual_scores.open("r", encoding="utf-8", newline="") as handle:
                preserved = list(csv.DictReader(handle))
            self.assertEqual(preserved[0]["score"], "pass")
            self.assertEqual(preserved[0]["notes"], "kept")

            summary = summarize_manual_scores(paths, artifact)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["metrics"]["overall"]["passed"], 2)
            self.assertEqual(
                summary["metrics"]["overall"]["total"], len(DIMENSION_KEYS)
            )
            self.assertEqual(summary["response_coverage"]["completed"], 3)
            self.assertTrue(paths.summary.exists())
            self.assertTrue(paths.summary_markdown.exists())

    def test_incomplete_scores_do_not_emit_metrics(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            paths = ArtifactPaths(Path(temporary_directory))
            artifact = artifact_with_one_scenario()
            refresh_manual_scores(paths, artifact)
            summary = summarize_manual_scores(paths, artifact)
            self.assertFalse(summary["complete"])
            self.assertNotIn("metrics", summary)
            self.assertEqual(len(summary["missing"]), 3)

