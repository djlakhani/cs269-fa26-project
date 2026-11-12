from __future__ import annotations

import csv
import hashlib
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from cupmem_stale_eval.artifacts import RunPaths
from cupmem_stale_eval.constants import DIMENSION_CODES, DIMENSION_KEYS
from cupmem_stale_eval.cupmem_bridge import canonical_judge_system_prompt
from cupmem_stale_eval.diagnostics import run_diagnostics
from cupmem_stale_eval.io_utils import (
    atomic_write_json,
    atomic_write_json_zst,
    atomic_write_jsonl,
    atomic_write_jsonl_zst,
    sha256_file,
)
from cupmem_stale_eval.judging import _parse_json_object, official_judge_user_prompt, run_judge
from cupmem_stale_eval.preparation import build_scenario_artifact, prepare_from_huggingface
from cupmem_stale_eval.reporting import REVIEW_FIELDS, refresh_reports
from cupmem_stale_eval.verification import verify_run

from conftest import make_record


EVALUATOR_ROOT = Path(__file__).resolve().parents[1]


class FlakyCompletions:
    should_fail = True
    payload = ""

    def create(self, **_kwargs):
        if FlakyCompletions.should_fail:
            FlakyCompletions.should_fail = False
            raise RuntimeError("transient provider failure")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=FlakyCompletions.payload))],
            usage=None,
        )


class FlakyOpenAI:
    def __init__(self, **_kwargs) -> None:
        self.chat = SimpleNamespace(completions=FlakyCompletions())


def small_artifact() -> dict:
    return build_scenario_artifact(
        [make_record("report-one", "T1"), make_record("report-two", "T2")],
        samples_per_type=1,
        distractors_per_scenario=1,
    )


class JudgeReportingAndWrapperTests(unittest.TestCase):
    def test_canonical_system_prompt_snapshot_and_exact_user_template(self) -> None:
        prompt = canonical_judge_system_prompt()
        self.assertEqual(
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "2d2d0a52992980120bde7e297197f03853255fa6f36fa3c70df7215476fbe669",
        )
        scenario = make_record("prompt", "T1")
        responses = {
            "dim1_response": "one",
            "dim2_response": "two",
            "dim3_response": "three",
        }
        expected = f"""
[Ground Truth Context]
- M_old: \"{scenario['M_old']}\"
- M_new: \"{scenario['M_new']}\"
- Hidden Logic: {scenario['explanation']}

--------------------------------------------------
[Dimension 1: Explicit Probing]
Question 1: {scenario['probing_queries']['dim1_query']}
Target Model Response 1: one

--------------------------------------------------
[Dimension 2: Adversarial Robustness]
Question 2: {scenario['probing_queries']['dim2_query']}
Target Model Response 2: two

--------------------------------------------------
[Dimension 3: Implicit Task]
Question 3: {scenario['probing_queries']['dim3_query']}
Target Model Response 3: three
"""
        self.assertEqual(official_judge_user_prompt(scenario, responses), expected)

    def test_judge_parser_requires_three_boolean_labels(self) -> None:
        valid = {
            f"dim{index}_eval": {"reasoning": "reason", "pass": index != 2}
            for index in range(1, 4)
        }
        self.assertEqual(_parse_json_object(str(valid).replace("'", '"').replace("True", "true").replace("False", "false")), valid)
        with self.assertRaisesRegex(ValueError, "dim3_eval.pass"):
            _parse_json_object(
                '{"dim1_eval":{"pass":true,"reasoning":"r"},'
                '"dim2_eval":{"pass":true,"reasoning":"r"}}'
            )

    def test_reports_preserve_human_columns(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RunPaths(root)
            artifact = small_artifact()
            atomic_write_json(paths.scenarios_file, artifact)
            score_records = []
            for scenario in artifact["data"]:
                scenario_paths = paths.scenario(scenario["uid"])
                for dimension_key in DIMENSION_KEYS:
                    atomic_write_json(
                        scenario_paths.query_root(dimension_key) / "result.json",
                        {"response": f"response {scenario['uid']} {dimension_key}"},
                    )
                score_records.append(
                    {
                        "uid": scenario["uid"],
                        "evaluation": {
                            f"dim{index}_eval": {"pass": not (index == 2 and scenario["type"] == "T1"), "reasoning": "r"}
                            for index in range(1, 4)
                        },
                    }
                )
            atomic_write_jsonl(paths.judge_root / "scores.jsonl", score_records)
            atomic_write_jsonl(
                paths.diagnostics_root / "failures.jsonl",
                [
                    {
                        "uid": artifact["data"][0]["uid"],
                        "dimension_key": "dim2_query",
                        "status": "ok",
                        "packet_path": "diagnostics/packets/failure.json",
                        "diagnosis": {"primary_stage": "retrieval", "confidence": 0.8},
                    }
                ],
            )
            metrics = refresh_reports(root)
            self.assertEqual(metrics["review_rows"], 6)

            review_path = paths.review_root / "all_300_answers.csv"
            with review_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["human_label"] = "pass"
            rows[0]["human_notes"] = "confirmed manually"
            with review_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS)
                writer.writeheader()
                writer.writerows(rows)

            refresh_reports(root)
            with review_path.open("r", encoding="utf-8", newline="") as handle:
                refreshed = list(csv.DictReader(handle))
            self.assertEqual(refreshed[0]["human_label"], "pass")
            self.assertEqual(refreshed[0]["human_notes"], "confirmed manually")
            failure_report = (paths.reports_root / "failure_report.md").read_text(encoding="utf-8")
            self.assertIn("retrieval", failure_report)

    def test_prepare_is_immutable_after_checkpoints(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RunPaths(root)
            artifact = small_artifact()
            atomic_write_json(paths.scenarios_file, artifact)
            atomic_write_json(
                paths.preparation_manifest,
                {"scenarios_sha256": sha256_file(paths.scenarios_file)},
            )
            returned = prepare_from_huggingface(
                root,
                cache_dir=root / "cache",
                samples_per_type=1,
                distractors_per_scenario=1,
            )
            self.assertEqual(returned, paths.scenarios_file)
            atomic_write_json(paths.scenario(artifact["data"][0]["uid"]).status, {"phase": "memory"})
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                prepare_from_huggingface(
                    root,
                    cache_dir=root / "cache",
                    samples_per_type=1,
                    distractors_per_scenario=1,
                    force=True,
                )

    def test_incomplete_verification_accepts_prepared_balanced_artifact(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RunPaths(root)
            artifact = small_artifact()
            atomic_write_json(paths.scenarios_file, artifact)
            atomic_write_json(
                paths.preparation_manifest,
                {"scenarios_sha256": sha256_file(paths.scenarios_file)},
            )
            result = verify_run(root, require_complete=False)
            self.assertTrue(result["ok"])
            self.assertEqual(result["type_counts"], {"T1": 1, "T2": 1})

    def test_runpod_wrapper_validates_arguments_and_never_syncs_with_delete(self) -> None:
        run_script = EVALUATOR_ROOT / "scripts" / "run_runpod.sh"
        setup_script = EVALUATOR_ROOT / "scripts" / "setup_runpod.sh"
        subprocess.run(["bash", "-n", str(run_script)], check=True)
        subprocess.run(["bash", "-n", str(setup_script)], check=True)
        text = run_script.read_text(encoding="utf-8")
        self.assertNotIn("--delete", text)
        self.assertIn("--dryrun", text)
        self.assertIn("sleep \"${SYNC_INTERVAL}\"", text)
        invalid = subprocess.run(
            ["bash", str(run_script), "../bad", "smoke"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("invalid character", invalid.stderr)

    def test_judge_and_diagnostic_resume_after_provider_errors(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = RunPaths(root)
            artifact = small_artifact()
            scenario = artifact["data"][0]
            atomic_write_json(paths.scenarios_file, artifact)
            scenario_paths = paths.scenario(scenario["uid"])
            for dimension_key in DIMENSION_KEYS:
                atomic_write_json(
                    scenario_paths.query_root(dimension_key) / "result.json",
                    {
                        "response": f"answer {dimension_key}",
                        "memory_hash_before": "frozen",
                        "memory_hash_after": "frozen",
                    },
                )

            FlakyCompletions.should_fail = True
            FlakyCompletions.payload = (
                '{"dim1_eval":{"reasoning":"bad","pass":false},'
                '"dim2_eval":{"reasoning":"ok","pass":true},'
                '"dim3_eval":{"reasoning":"ok","pass":true}}'
            )
            with (
                patch("cupmem_stale_eval.judging.OpenAI", FlakyOpenAI),
                patch.dict("os.environ", {"JUDGE_OPENAI_API_KEY": "test"}),
            ):
                with self.assertRaisesRegex(RuntimeError, "transient"):
                    run_judge(root, max_scenarios=1)
                self.assertEqual(run_judge(root, resume=True, max_scenarios=1), 1)

            atomic_write_json(scenario_paths.final_profile, {"active_profile": {}})
            atomic_write_jsonl_zst(
                scenario_paths.session_traces,
                [{"is_evidence": True, "original_index": scenario["relevant_session_index"][0]}],
            )
            query_root = scenario_paths.query_root("dim1_query")
            atomic_write_json_zst(query_root / "trace.json.zst", {"parsed_query": {}})
            atomic_write_json(query_root / "answer_composer_request.json", {"messages": []})
            atomic_write_jsonl_zst(query_root / "llm_calls.jsonl.zst", [])

            FlakyCompletions.should_fail = True
            FlakyCompletions.payload = (
                '{"primary_stage":"retrieval","secondary_stages":[],"confidence":0.8,'
                '"rationale":"missing update","supporting_artifact_ids":[],"remediation":"improve recall"}'
            )
            with (
                patch("cupmem_stale_eval.diagnostics.OpenAI", FlakyOpenAI),
                patch.dict("os.environ", {"JUDGE_OPENAI_API_KEY": "test"}),
            ):
                with self.assertRaisesRegex(RuntimeError, "transient"):
                    run_diagnostics(root, max_failures=1)
                self.assertEqual(run_diagnostics(root, resume=True, max_failures=1), 1)


if __name__ == "__main__":
    unittest.main()
