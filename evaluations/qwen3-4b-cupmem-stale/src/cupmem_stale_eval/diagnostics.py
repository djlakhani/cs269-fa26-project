from __future__ import annotations

import json
import hashlib
import os
import time
import traceback
from pathlib import Path
from typing import Any

from openai import OpenAI

from .artifacts import RunPaths
from .constants import DIMENSION_CODES, DIMENSION_KEYS, FAILURE_STAGES, JUDGE_MODEL_ID
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    load_json,
    load_jsonl_zst,
    load_json_zst,
    safe_uid,
    utc_now,
)


DIAGNOSTIC_SYSTEM_PROMPT = """You are diagnosing a structured long-term-memory pipeline.
The official correctness label is already fixed and must not be changed. Identify the earliest
pipeline stage that materially caused the failed answer. Use only the supplied trace evidence.
Return one JSON object with primary_stage, secondary_stages, confidence, rationale,
supporting_artifact_ids, and remediation. primary_stage and secondary_stages must use only
the allowed stage names supplied in the input."""


def _failure_dimensions(judge_record: dict[str, Any]) -> list[str]:
    result: list[str] = []
    evaluation = judge_record.get("evaluation", {})
    for index, dimension_key in enumerate(DIMENSION_KEYS, start=1):
        if evaluation.get(f"dim{index}_eval", {}).get("pass") is False:
            result.append(dimension_key)
    return result


def build_diagnostic_packet(
    paths: RunPaths,
    scenario: dict[str, Any],
    judge_record: dict[str, Any],
    dimension_key: str,
) -> dict[str, Any]:
    uid = str(scenario["uid"])
    scenario_paths = paths.scenario(uid)
    traces = load_json_zst(scenario_paths.query_root(dimension_key) / "trace.json.zst")
    session_traces = []
    for trace in load_jsonl_zst(scenario_paths.session_traces):
        if trace.get("is_evidence"):
            session_traces.append(trace)
    result = load_json(scenario_paths.query_root(dimension_key) / "result.json")
    answer_request = load_json(
        scenario_paths.query_root(dimension_key) / "answer_composer_request.json"
    )
    query_calls = load_jsonl_zst(
        scenario_paths.query_root(dimension_key) / "llm_calls.jsonl.zst"
    )
    evidence_session_ids = {
        f"s_{int(index):03d}" for index in scenario["relevant_session_index"]
    }
    evidence_calls = [
        event
        for event in load_jsonl_zst(scenario_paths.llm_calls)
        if event.get("session_id") in evidence_session_ids
    ]
    judge_index = DIMENSION_KEYS.index(dimension_key) + 1
    return {
        "uid": uid,
        "type": scenario["type"],
        "dimension_key": dimension_key,
        "dimension_code": DIMENSION_CODES[dimension_key],
        "ground_truth": {
            "M_old": scenario["M_old"],
            "M_new": scenario["M_new"],
            "explanation": scenario["explanation"],
        },
        "question": scenario["probing_queries"][dimension_key],
        "response": result["response"],
        "official_judge": judge_record["evaluation"][f"dim{judge_index}_eval"],
        "evidence_session_traces": session_traces,
        "final_memory": load_json(scenario_paths.final_profile),
        "query_trace": traces,
        "answer_composer_request": answer_request,
        "relevant_qwen_call_ids": {
            "evidence_writes": [
                {
                    "audit_sequence": event.get("sequence"),
                    "session_id": event.get("session_id"),
                    "phase": event.get("phase"),
                    "event": event.get("event"),
                    "attempt": event.get("attempt"),
                }
                for event in evidence_calls
            ],
            "query": [
                {
                    "audit_sequence": event.get("sequence"),
                    "phase": event.get("phase"),
                    "event": event.get("event"),
                    "attempt": event.get("attempt"),
                }
                for event in query_calls
            ],
        },
        "artifact_paths": {
            "all_qwen_calls": str(scenario_paths.llm_calls.relative_to(paths.root)),
            "query_qwen_calls": str(
                (scenario_paths.query_root(dimension_key) / "llm_calls.jsonl.zst").relative_to(
                    paths.root
                )
            ),
            "query_trace": str(
                (scenario_paths.query_root(dimension_key) / "trace.json.zst").relative_to(
                    paths.root
                )
            ),
        },
        "allowed_failure_stages": list(FAILURE_STAGES),
    }


def _parse_diagnostic(text: str) -> dict[str, Any]:
    parsed = json.loads(str(text or "").strip())
    if not isinstance(parsed, dict):
        raise ValueError("Diagnostic response must be an object")
    if parsed.get("primary_stage") not in FAILURE_STAGES:
        raise ValueError("Diagnostic primary_stage is invalid")
    secondary = parsed.get("secondary_stages", [])
    if not isinstance(secondary, list) or any(item not in FAILURE_STAGES for item in secondary):
        raise ValueError("Diagnostic secondary_stages are invalid")
    confidence = parsed.get("confidence")
    if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= confidence <= 1:
        raise ValueError("Diagnostic confidence must be between 0 and 1")
    if not isinstance(parsed.get("rationale"), str) or not parsed["rationale"].strip():
        raise ValueError("Diagnostic rationale is required")
    if not isinstance(parsed.get("supporting_artifact_ids"), list):
        raise ValueError("Diagnostic supporting_artifact_ids must be a list")
    return parsed


def run_diagnostics(
    artifact_root: Path,
    *,
    model: str = JUDGE_MODEL_ID,
    resume: bool = False,
    max_failures: int | None = None,
) -> int:
    paths = RunPaths(artifact_root)
    artifact = load_json(paths.scenarios_file)
    api_key = os.environ.get("JUDGE_OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("JUDGE_OPENAI_API_KEY is required")
    client = OpenAI(
        api_key=api_key,
        base_url=os.environ.get("JUDGE_OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )
    generated = 0
    failure_records: list[dict[str, Any]] = []
    for scenario in artifact["data"]:
        uid = str(scenario["uid"])
        judge_path = paths.judge_root / "official_checkpoints" / f"{safe_uid(uid)}.json"
        if not judge_path.exists():
            continue
        judge_record = load_json(judge_path)
        if judge_record.get("status") != "ok":
            continue
        for dimension_key in _failure_dimensions(judge_record):
            checkpoint = (
                paths.diagnostics_root
                / "checkpoints"
                / f"{safe_uid(uid)}__{dimension_key}.json"
            )
            packet_path = (
                paths.diagnostics_root
                / "packets"
                / f"{safe_uid(uid)}__{dimension_key}.json"
            )
            if checkpoint.exists() and load_json(checkpoint).get("status") == "ok":
                failure_records.append(load_json(checkpoint))
                continue
            if checkpoint.exists() and not resume:
                raise ValueError(f"Existing diagnostic checkpoint {checkpoint}; pass --resume")
            if max_failures is not None and generated >= max_failures:
                atomic_write_jsonl(paths.diagnostics_root / "failures.jsonl", failure_records)
                return generated
            packet = build_diagnostic_packet(paths, scenario, judge_record, dimension_key)
            atomic_write_json(packet_path, packet)
            user_payload = json.dumps(packet, ensure_ascii=False)
            messages = [
                {"role": "system", "content": DIAGNOSTIC_SYSTEM_PROMPT},
                {"role": "user", "content": user_payload},
            ]
            started = time.perf_counter()
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    reasoning_effort="none",
                    response_format={"type": "json_object"},
                )
                raw = response.choices[0].message.content or ""
                diagnosis = _parse_diagnostic(raw)
                usage_obj = getattr(response, "usage", None)
                usage = usage_obj.model_dump(mode="json") if usage_obj is not None else {}
                record = {
                    "uid": uid,
                    "type": scenario["type"],
                    "dimension_key": dimension_key,
                    "dimension_code": DIMENSION_CODES[dimension_key],
                    "status": "ok",
                    "model": model,
                    "diagnosis": diagnosis,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "usage": usage,
                    "packet_path": str(packet_path.relative_to(paths.root)),
                    "call": {
                        "system_prompt": DIAGNOSTIC_SYSTEM_PROMPT,
                        "user_payload_sha256": hashlib.sha256(
                            user_payload.encode("utf-8")
                        ).hexdigest(),
                        "request": {
                            "temperature": 0,
                            "reasoning_effort": "none",
                            "response_format": {"type": "json_object"},
                        },
                        "raw_response": raw,
                    },
                    "generated_at": utc_now(),
                }
            except Exception as error:
                record = {
                    "uid": uid,
                    "dimension_key": dimension_key,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                }
                atomic_write_json(checkpoint, record)
                raise
            atomic_write_json(checkpoint, record)
            failure_records.append(record)
            generated += 1
            atomic_write_jsonl(paths.diagnostics_root / "failures.jsonl", failure_records)
    atomic_write_jsonl(paths.diagnostics_root / "failures.jsonl", failure_records)
    return generated
