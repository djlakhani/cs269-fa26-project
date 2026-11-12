from __future__ import annotations

from pathlib import Path
from typing import Any
import json

from .artifacts import RunPaths
from .constants import (
    DEFAULT_DISTRACTORS_PER_SCENARIO,
    DEFAULT_SAMPLES_PER_TYPE,
    DATASET_ID,
    DATASET_REVISION,
    DIMENSION_KEYS,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    SCENARIO_SCHEMA_VERSION,
    SOURCE_SESSION_COUNT,
    TARGET_MODEL_ID,
    TARGET_MODEL_REVISION,
)
from .diagnostics import _failure_dimensions
from .io_utils import (
    load_json,
    load_jsonl,
    load_jsonl_zst,
    load_json_zst,
    safe_uid,
    sha256_file,
    sha256_json,
)


def verify_run(artifact_root: Path, *, require_complete: bool = True) -> dict[str, Any]:
    paths = RunPaths(artifact_root)
    errors: list[str] = []
    artifact = load_json(paths.scenarios_file)
    if artifact.get("schema_version") != SCENARIO_SCHEMA_VERSION:
        errors.append("Unsupported scenarios schema")
    expected_metadata = {
        "dataset": {"id": DATASET_ID, "revision": DATASET_REVISION, "split": "train"},
        "target_model": {"id": TARGET_MODEL_ID, "revision": TARGET_MODEL_REVISION},
        "embedding_model": {"id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_MODEL_REVISION},
    }
    for field, expected in expected_metadata.items():
        if artifact.get(field) != expected:
            errors.append(f"scenarios.json {field} metadata does not match pinned configuration")
    preparation = load_json(paths.preparation_manifest)
    if preparation.get("scenarios_sha256") != sha256_file(paths.scenarios_file):
        errors.append("scenarios.json checksum mismatch")
    if paths.run_manifest.exists():
        run_manifest = load_json(paths.run_manifest)
        if run_manifest.get("scenarios_sha256") != sha256_file(paths.scenarios_file):
            errors.append("run_manifest.json scenarios checksum mismatch")
    elif require_complete:
        errors.append("Missing run_manifest.json")
    data = artifact.get("data", [])
    uids = [str(item.get("uid", "")) for item in data]
    if len(uids) != len(set(uids)) or any(not uid for uid in uids):
        errors.append("Scenario UIDs are empty or duplicated")
    type_counts = {key: sum(item.get("type") == key for item in data) for key in ("T1", "T2")}
    expected_per_type = int(artifact.get("sampling", {}).get("samples_per_type", 0))
    if any(value != expected_per_type for value in type_counts.values()):
        errors.append(f"Unbalanced scenario counts: {type_counts}")
    expected_sessions = int(artifact.get("sampling", {}).get("sessions_per_scenario", 0))
    if require_complete:
        if expected_per_type != DEFAULT_SAMPLES_PER_TYPE:
            errors.append(
                f"Full evaluation requires {DEFAULT_SAMPLES_PER_TYPE} scenarios per type"
            )
        if expected_sessions != DEFAULT_DISTRACTORS_PER_SCENARIO + 2:
            errors.append(
                f"Full evaluation requires {DEFAULT_DISTRACTORS_PER_SCENARIO + 2} sessions per scenario"
            )
    completed_writes = 0
    completed_results = 0
    for scenario in data:
        uid = str(scenario["uid"])
        if int(scenario.get("source_session_count", 0)) != SOURCE_SESSION_COUNT:
            errors.append(f"{uid}: source session count is not {SOURCE_SESSION_COUNT}")
        if len(scenario.get("sessions", [])) != expected_sessions:
            errors.append(f"{uid}: incorrect retained session count")
        original_indices = [int(item["original_index"]) for item in scenario.get("sessions", [])]
        if original_indices != sorted(original_indices) or len(set(original_indices)) != len(original_indices):
            errors.append(f"{uid}: retained sessions are not unique and chronological")
        if not set(scenario.get("relevant_session_index", [])).issubset(
            set(scenario.get("kept_session_indices", []))
        ):
            errors.append(f"{uid}: evidence session missing")
        evidence_indices = [int(value) for value in scenario.get("relevant_session_index", [])]
        if len(evidence_indices) != 2 or len(set(evidence_indices)) != 2:
            errors.append(f"{uid}: expected two distinct evidence sessions")
        scenario_paths = paths.scenario(uid)
        if not scenario_paths.status.exists():
            if require_complete:
                errors.append(f"{uid}: missing status")
            continue
        status = load_json(scenario_paths.status)
        if require_complete and status.get("phase") != "completed":
            errors.append(f"{uid}: target phase is {status.get('phase')}")
        completed_indices = [int(value) for value in status.get("completed_session_indices", [])]
        completed_writes += len(completed_indices)
        if status.get("phase") == "completed" and completed_indices != original_indices:
            errors.append(f"{uid}: session checkpoint set does not match retained sessions")
        if status.get("phase") == "completed" and status.get("completed_queries") != list(DIMENSION_KEYS):
            errors.append(f"{uid}: completed query checkpoint list is invalid")
        if status.get("phase") == "completed":
            required = [
                scenario_paths.engine_state,
                scenario_paths.llm_calls,
                scenario_paths.final_profile,
                scenario_paths.memory_markdown,
                scenario_paths.session_traces,
            ]
            for required_path in required:
                if not required_path.exists():
                    errors.append(f"{uid}: missing {required_path.relative_to(paths.root)}")
            if scenario_paths.engine_state.exists():
                state_hash = sha256_json(load_json_zst(scenario_paths.engine_state))
                if state_hash != status.get("memory_hash"):
                    errors.append(f"{uid}: frozen engine-state hash mismatch")
            if scenario_paths.session_traces.exists():
                traces = load_jsonl_zst(scenario_paths.session_traces)
                if len(traces) != len(completed_indices):
                    errors.append(f"{uid}: session trace count mismatch")
            if scenario_paths.llm_calls.exists():
                for event in load_jsonl_zst(scenario_paths.llm_calls):
                    if event.get("event") not in {"llm_attempt", "llm_attempt_error"}:
                        continue
                    serialized_messages = json.dumps(event.get("messages", []), ensure_ascii=False)
                    forbidden = ('"M_old"', '"M_new"', '"explanation"', "[Ground Truth Context]")
                    if any(marker in serialized_messages for marker in forbidden):
                        errors.append(f"{uid}: target request contains benchmark ground-truth metadata")
                        break
                    raw_response = str(event.get("raw_response", "")).lower()
                    if "<think>" in raw_response or "</think>" in raw_response:
                        errors.append(f"{uid}: target response contains a thinking marker")
                        break
        for dimension_key in DIMENSION_KEYS:
            result_path = scenario_paths.query_root(dimension_key) / "result.json"
            if not result_path.exists():
                if require_complete:
                    errors.append(f"{uid}/{dimension_key}: missing result")
                continue
            result = load_json(result_path)
            completed_results += 1
            if result.get("memory_hash_before") != result.get("memory_hash_after"):
                errors.append(f"{uid}/{dimension_key}: memory changed during query")
            if status.get("memory_hash") and result.get("memory_hash_before") != status.get("memory_hash"):
                errors.append(f"{uid}/{dimension_key}: result does not use frozen scenario memory")
            query_root = scenario_paths.query_root(dimension_key)
            for required_path in (
                query_root / "answer_composer_request.json",
                query_root / "trace.json.zst",
                query_root / "llm_calls.jsonl.zst",
            ):
                if not required_path.exists():
                    errors.append(f"{uid}/{dimension_key}: missing {required_path.name}")

    if require_complete and completed_writes != len(data) * expected_sessions:
        errors.append(
            f"Expected {len(data) * expected_sessions} session writes, found {completed_writes}"
        )
    if require_complete and completed_results != len(data) * len(DIMENSION_KEYS):
        errors.append(f"Expected {len(data) * 3} query results, found {completed_results}")
    response_records = load_jsonl(paths.responses)
    if require_complete and len(response_records) != len(data):
        errors.append(f"Expected {len(data)} aggregate responses, found {len(response_records)}")

    score_records = load_jsonl(paths.judge_root / "scores.jsonl")
    if require_complete and len(score_records) != len(data):
        errors.append(f"Expected {len(data)} judge records, found {len(score_records)}")
    official_calls_path = paths.judge_root / "official_calls.jsonl.zst"
    official_calls = load_jsonl_zst(official_calls_path) if official_calls_path.exists() else []
    if require_complete and len(official_calls) != len(data):
        errors.append(f"Expected {len(data)} official call audits, found {len(official_calls)}")
    for record in score_records:
        for index in range(1, 4):
            value = record.get("evaluation", {}).get(f"dim{index}_eval", {}).get("pass")
            if not isinstance(value, bool):
                errors.append(f"{record.get('uid')}: judge dim{index} pass is not Boolean")
    if require_complete:
        for uid in uids:
            checkpoint = paths.judge_root / "official_checkpoints" / f"{safe_uid(uid)}.json"
            if not checkpoint.exists() or load_json(checkpoint).get("status") != "ok":
                errors.append(f"{uid}: missing successful official judge checkpoint")
    clean_judge_inputs = load_jsonl(paths.judge_root / "judge_inputs.jsonl")
    if require_complete and len(clean_judge_inputs) != len(data) * len(DIMENSION_KEYS):
        errors.append(
            f"Expected {len(data) * 3} clean judge inputs, found {len(clean_judge_inputs)}"
        )
    expected_failures = sum(len(_failure_dimensions(record)) for record in score_records)
    diagnostic_records = [
        record
        for record in load_jsonl(paths.diagnostics_root / "failures.jsonl")
        if record.get("status") == "ok"
    ]
    if require_complete and len(diagnostic_records) != expected_failures:
        errors.append(
            f"Expected {expected_failures} failure diagnoses, found {len(diagnostic_records)}"
        )
    expected_failure_keys = {
        (str(record.get("uid")), dimension)
        for record in score_records
        for dimension in _failure_dimensions(record)
    }
    diagnostic_keys = {
        (str(record.get("uid")), str(record.get("dimension_key")))
        for record in diagnostic_records
    }
    if require_complete and diagnostic_keys != expected_failure_keys:
        errors.append("Diagnostic checkpoints do not match the official failure set")
    if require_complete:
        for uid, dimension_key in expected_failure_keys:
            checkpoint = (
                paths.diagnostics_root
                / "checkpoints"
                / f"{safe_uid(uid)}__{dimension_key}.json"
            )
            if not checkpoint.exists() or load_json(checkpoint).get("status") != "ok":
                errors.append(f"{uid}/{dimension_key}: missing successful diagnostic checkpoint")
    review_path = paths.review_root / "all_300_answers.csv"
    review_rows = 0
    if review_path.exists():
        import csv

        with review_path.open("r", encoding="utf-8", newline="") as handle:
            review_rows = sum(1 for _ in csv.DictReader(handle))
    if require_complete and review_rows != len(data) * len(DIMENSION_KEYS):
        errors.append(f"Expected {len(data) * 3} review rows, found {review_rows}")
    if require_complete:
        for relative_path in (
            "reports/metrics.json",
            "reports/metrics.md",
            "reports/failure_report.md",
            "reports/timing_and_usage.json",
        ):
            if not (paths.root / relative_path).exists():
                errors.append(f"Missing {relative_path}")
    result = {
        "ok": not errors,
        "errors": errors,
        "scenario_count": len(data),
        "type_counts": type_counts,
        "judge_records": len(score_records),
        "diagnostic_records": len(diagnostic_records),
        "review_rows": review_rows,
        "completed_session_writes": completed_writes,
        "completed_query_results": completed_results,
        "require_complete": require_complete,
    }
    if errors:
        raise ValueError("Verification failed:\n- " + "\n- ".join(errors))
    return result
