from __future__ import annotations

import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Any

from openai import OpenAI

from .artifacts import RunPaths
from .constants import DIMENSION_CODES, DIMENSION_KEYS, JUDGE_MODEL_ID
from .cupmem_bridge import canonical_judge_system_prompt
from .io_utils import (
    atomic_write_json,
    atomic_write_jsonl,
    atomic_write_jsonl_zst,
    load_json,
    load_jsonl_zst,
    safe_uid,
    utc_now,
)


def official_judge_user_prompt(scenario: dict[str, Any], responses: dict[str, str]) -> str:
    return f"""
[Ground Truth Context]
- M_old: \"{scenario['M_old']}\"
- M_new: \"{scenario['M_new']}\"
- Hidden Logic: {scenario['explanation']}

--------------------------------------------------
[Dimension 1: Explicit Probing]
Question 1: {scenario['probing_queries']['dim1_query']}
Target Model Response 1: {responses['dim1_response']}

--------------------------------------------------
[Dimension 2: Adversarial Robustness]
Question 2: {scenario['probing_queries']['dim2_query']}
Target Model Response 2: {responses['dim2_response']}

--------------------------------------------------
[Dimension 3: Implicit Task]
Question 3: {scenario['probing_queries']['dim3_query']}
Target Model Response 3: {responses['dim3_response']}
"""


def _parse_json_object(text: str) -> dict[str, Any]:
    content = str(text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
    if fenced:
        content = fenced.group(1)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("Judge response must be a JSON object")
    for index in range(1, 4):
        value = parsed.get(f"dim{index}_eval")
        if not isinstance(value, dict) or not isinstance(value.get("pass"), bool):
            raise ValueError(f"Judge response is missing dim{index}_eval.pass")
        if not isinstance(value.get("reasoning"), str):
            raise ValueError(f"Judge response is missing dim{index}_eval.reasoning")
    return parsed


def _scenario_responses(paths: RunPaths, scenario: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    scenario_paths = paths.scenario(str(scenario["uid"]))
    for dimension_key in DIMENSION_KEYS:
        query_result = load_json(scenario_paths.query_root(dimension_key) / "result.json")
        result[dimension_key.replace("_query", "_response")] = str(query_result["response"])
    return result


def _rebuild_judge_outputs(paths: RunPaths, artifact: dict[str, Any]) -> None:
    scores: list[dict[str, Any]] = []
    clean_inputs: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    for scenario in artifact["data"]:
        uid = str(scenario["uid"])
        checkpoint = paths.judge_root / "official_checkpoints" / f"{safe_uid(uid)}.json"
        scenario_paths = paths.scenario(uid)
        if all(
            (scenario_paths.query_root(key) / "result.json").exists()
            for key in DIMENSION_KEYS
        ):
            responses = _scenario_responses(paths, scenario)
            for dimension_key in DIMENSION_KEYS:
                clean_inputs.append(
                    {
                        "uid": uid,
                        "type": scenario["type"],
                        "dimension": DIMENSION_CODES[dimension_key],
                        "question": scenario["probing_queries"][dimension_key],
                        "response": responses[dimension_key.replace("_query", "_response")],
                        "M_old": scenario["M_old"],
                        "M_new": scenario["M_new"],
                    }
                )
        if checkpoint.exists():
            record = load_json(checkpoint)
            if record.get("status") == "ok":
                scores.append(record)
                calls.append(record.get("call", {}))
    atomic_write_jsonl(paths.judge_root / "judge_inputs.jsonl", clean_inputs)
    atomic_write_jsonl(paths.judge_root / "scores.jsonl", scores)
    atomic_write_jsonl_zst(paths.judge_root / "official_calls.jsonl.zst", calls)


def run_judge(
    artifact_root: Path,
    *,
    model: str = JUDGE_MODEL_ID,
    resume: bool = False,
    max_scenarios: int | None = None,
) -> int:
    paths = RunPaths(artifact_root)
    artifact = load_json(paths.scenarios_file)
    api_key = os.environ.get("JUDGE_OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("JUDGE_OPENAI_API_KEY is required")
    base_url = os.environ.get("JUDGE_OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
    client = OpenAI(api_key=api_key, base_url=base_url)
    system_prompt = canonical_judge_system_prompt()
    completed = 0
    generated = 0
    for scenario in artifact["data"]:
        uid = str(scenario["uid"])
        checkpoint = paths.judge_root / "official_checkpoints" / f"{safe_uid(uid)}.json"
        if checkpoint.exists() and load_json(checkpoint).get("status") == "ok":
            completed += 1
            continue
        if checkpoint.exists() and not resume:
            raise ValueError(f"Existing judge checkpoint for {uid}; pass --resume")
        if max_scenarios is not None and generated >= max_scenarios:
            break
        scenario_paths = paths.scenario(uid)
        if not all(
            (scenario_paths.query_root(key) / "result.json").exists()
            for key in DIMENSION_KEYS
        ):
            continue
        responses = _scenario_responses(paths, scenario)
        user_prompt = official_judge_user_prompt(scenario, responses)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        started = time.perf_counter()
        base_record: dict[str, Any] = {
            "uid": uid,
            "type": scenario["type"],
            "model": model,
            "generated_at": utc_now(),
        }
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                reasoning_effort="none",
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or ""
            evaluation = _parse_json_object(raw)
            usage_obj = getattr(response, "usage", None)
            usage = usage_obj.model_dump(mode="json") if usage_obj is not None else {}
            record = {
                **base_record,
                "status": "ok",
                "evaluation": evaluation,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "usage": usage,
                "call": {
                    "model": model,
                    "messages": messages,
                    "request": {
                        "temperature": 0,
                        "reasoning_effort": "none",
                        "response_format": {"type": "json_object"},
                    },
                    "raw_response": raw,
                    "usage": usage,
                },
            }
        except Exception as error:
            record = {
                **base_record,
                "status": "error",
                "error": f"{type(error).__name__}: {error}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
            atomic_write_json(checkpoint, record)
            _rebuild_judge_outputs(paths, artifact)
            raise
        atomic_write_json(checkpoint, record)
        generated += 1
        completed += 1
        _rebuild_judge_outputs(paths, artifact)
        atomic_write_json(
            paths.progress,
            {
                "phase": "official_judging",
                "updated_at": utc_now(),
                "completed_scenarios": completed,
                "total_scenarios": len(artifact["data"]),
                "current_uid": uid,
            },
        )
    _rebuild_judge_outputs(paths, artifact)
    return generated
