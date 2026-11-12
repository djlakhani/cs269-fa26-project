from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .artifacts import RunPaths
from .constants import DIMENSION_CODES, DIMENSION_KEYS
from .io_utils import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    load_json,
    load_jsonl,
    safe_uid,
    utc_now,
)


REVIEW_FIELDS = [
    "uid",
    "type",
    "dimension",
    "question",
    "response",
    "M_old",
    "M_new",
    "explanation",
    "luna_pass",
    "luna_reasoning",
    "diagnostic_primary_stage",
    "diagnostic_secondary_stages",
    "diagnostic_confidence",
    "diagnostic_rationale",
    "diagnostic_remediation",
    "human_label",
    "human_notes",
    "result_artifact",
    "memory_artifact",
    "diagnostic_artifact",
]


def _existing_review(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            (row.get("uid", ""), row.get("dimension", "")): row
            for row in csv.DictReader(handle)
        }


def _score_maps(paths: RunPaths) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    scores: dict[tuple[str, str], dict[str, Any]] = {}
    for record in load_jsonl(paths.judge_root / "scores.jsonl"):
        for index, dimension_key in enumerate(DIMENSION_KEYS, start=1):
            scores[(str(record["uid"]), dimension_key)] = record.get("evaluation", {}).get(
                f"dim{index}_eval", {}
            )
    diagnostics = {
        (str(record["uid"]), str(record["dimension_key"])): record
        for record in load_jsonl(paths.diagnostics_root / "failures.jsonl")
        if record.get("status") == "ok"
    }
    return scores, diagnostics


def rebuild_review(paths: RunPaths, artifact: dict[str, Any]) -> list[dict[str, Any]]:
    review_path = paths.review_root / "all_300_answers.csv"
    prior = _existing_review(review_path)
    scores, diagnostics = _score_maps(paths)
    rows: list[dict[str, Any]] = []
    for scenario in artifact["data"]:
        uid = str(scenario["uid"])
        scenario_paths = paths.scenario(uid)
        for dimension_key in DIMENSION_KEYS:
            result_path = scenario_paths.query_root(dimension_key) / "result.json"
            if not result_path.exists():
                continue
            result = load_json(result_path)
            dimension = DIMENSION_CODES[dimension_key]
            score = scores.get((uid, dimension_key), {})
            diagnostic = diagnostics.get((uid, dimension_key), {})
            diagnosis = diagnostic.get("diagnosis", {})
            previous = prior.get((uid, dimension), {})
            diagnostic_path = diagnostic.get("packet_path", "")
            rows.append(
                {
                    "uid": uid,
                    "type": scenario["type"],
                    "dimension": dimension,
                    "question": scenario["probing_queries"][dimension_key],
                    "response": result.get("response", ""),
                    "M_old": scenario["M_old"],
                    "M_new": scenario["M_new"],
                    "explanation": scenario["explanation"],
                    "luna_pass": score.get("pass", ""),
                    "luna_reasoning": score.get("reasoning", ""),
                    "diagnostic_primary_stage": diagnosis.get("primary_stage", ""),
                    "diagnostic_secondary_stages": "; ".join(
                        str(item) for item in diagnosis.get("secondary_stages", [])
                    ),
                    "diagnostic_confidence": diagnosis.get("confidence", ""),
                    "diagnostic_rationale": diagnosis.get("rationale", ""),
                    "diagnostic_remediation": diagnosis.get("remediation", ""),
                    "human_label": previous.get("human_label", ""),
                    "human_notes": previous.get("human_notes", ""),
                    "result_artifact": str(
                        (scenario_paths.query_root(dimension_key) / "result.json").relative_to(paths.root)
                    ),
                    "memory_artifact": str(scenario_paths.final_profile.relative_to(paths.root)),
                    "diagnostic_artifact": diagnostic_path,
                }
            )
    atomic_write_csv(review_path, REVIEW_FIELDS, rows)
    return rows


def _accuracy(rows: list[dict[str, Any]], label_field: str) -> dict[str, Any]:
    groups: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
    for row in rows:
        raw = row.get(label_field)
        if raw in {True, "true", "pass", "PASS", "Pass"}:
            passed = True
        elif raw in {False, "false", "fail", "FAIL", "Fail"}:
            passed = False
        else:
            continue
        keys = ["overall", str(row["type"]), str(row["dimension"]), f"{row['type']}/{row['dimension']}"]
        for key in keys:
            groups[key]["total"] += 1
            groups[key]["correct"] += int(passed)
    return {
        key: {
            **counts,
            "accuracy": counts["correct"] / counts["total"] if counts["total"] else None,
        }
        for key, counts in sorted(groups.items())
    }


def _timing_usage(paths: RunPaths, artifact: dict[str, Any]) -> dict[str, Any]:
    scenario_times: list[float] = []
    qwen_calls = 0
    for scenario in artifact["data"]:
        status_path = paths.scenario(str(scenario["uid"])).status
        if not status_path.exists():
            continue
        status = load_json(status_path)
        if status.get("phase") == "completed":
            scenario_times.append(float(status.get("elapsed_seconds", 0.0)))
            qwen_calls += int(status.get("call_count", 0))
    judge_records = load_jsonl(paths.judge_root / "scores.jsonl")
    diagnostic_records = load_jsonl(paths.diagnostics_root / "failures.jsonl")
    return {
        "completed_scenarios": len(scenario_times),
        "scenario_seconds_total": sum(scenario_times),
        "scenario_seconds_mean": sum(scenario_times) / len(scenario_times) if scenario_times else None,
        "qwen_call_count": qwen_calls,
        "judge_call_count": len(judge_records),
        "diagnostic_call_count": len([r for r in diagnostic_records if r.get("status") == "ok"]),
        "judge_usage": [record.get("usage", {}) for record in judge_records],
        "diagnostic_usage": [record.get("usage", {}) for record in diagnostic_records],
    }


def refresh_reports(artifact_root: Path) -> dict[str, Any]:
    paths = RunPaths(artifact_root)
    artifact = load_json(paths.scenarios_file)
    rows = rebuild_review(paths, artifact)
    automated = _accuracy(rows, "luna_pass")
    human = _accuracy(rows, "human_label")
    diagnostics = Counter(
        str(row["diagnostic_primary_stage"])
        for row in rows
        if row.get("diagnostic_primary_stage")
    )
    metrics = {
        "generated_at": utc_now(),
        "automated_luna": automated,
        "human_confirmed": human,
        "review_rows": len(rows),
        "human_reviewed_rows": sum(bool(str(row.get("human_label", "")).strip()) for row in rows),
        "failure_stage_counts": dict(sorted(diagnostics.items())),
    }
    paths.reports_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(paths.reports_root / "metrics.json", metrics)
    lines = ["# CUP-Mem STALE metrics", "", "Automated labels use GPT-6 Luna with the official STALE prompt.", ""]
    lines.extend(["## Automated accuracy", ""])
    for key, value in automated.items():
        accuracy = value["accuracy"]
        rendered = f"{accuracy:.1%}" if accuracy is not None else "n/a"
        lines.append(f"- **{key}:** {value['correct']}/{value['total']} ({rendered})")
    lines.extend(["", "## Human-confirmed accuracy", ""])
    if human:
        for key, value in human.items():
            accuracy = value["accuracy"]
            rendered = f"{accuracy:.1%}" if accuracy is not None else "n/a"
            lines.append(f"- **{key}:** {value['correct']}/{value['total']} ({rendered})")
    else:
        lines.append("_No human labels have been entered._")
    atomic_write_text(paths.reports_root / "metrics.md", "\n".join(lines) + "\n")

    failure_lines = ["# CUP-Mem failure report", "", "## Automated primary-stage counts", ""]
    if diagnostics:
        for stage, count in sorted(diagnostics.items()):
            failure_lines.append(f"- **{stage}:** {count}")
    else:
        failure_lines.append("_No completed failure diagnoses._")
    failure_lines.extend(["", "## Failure index", ""])
    for row in rows:
        if row.get("luna_pass") is False or str(row.get("luna_pass")).lower() == "false":
            failure_lines.append(
                f"- `{row['uid']}` {row['type']}/{row['dimension']}: "
                f"{row.get('diagnostic_primary_stage') or 'diagnosis pending'} — "
                f"[{row['result_artifact']}](../{row['result_artifact']})"
            )
            failure_lines.append(
                f"  - official judge: {str(row.get('luna_reasoning') or 'no reasoning').strip()}"
            )
            if row.get("diagnostic_rationale"):
                failure_lines.append(
                    f"  - diagnosis: {str(row['diagnostic_rationale']).strip()}"
                )
            if row.get("diagnostic_remediation"):
                failure_lines.append(
                    f"  - remediation: {str(row['diagnostic_remediation']).strip()}"
                )
    atomic_write_text(
        paths.reports_root / "failure_report.md", "\n".join(failure_lines) + "\n"
    )
    atomic_write_json(paths.reports_root / "timing_and_usage.json", _timing_usage(paths, artifact))
    return metrics
