from __future__ import annotations

import csv
import io
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from .constants import CONTEXT_VIEWS, DIMENSION_INFO, DIMENSION_KEYS, TYPE_INFO
from .io_utils import atomic_write_json, atomic_write_text, load_json, utc_now
from .persistence import ArtifactPaths, load_response_map


SCORE_FIELDS = [
    "context_view",
    "uid",
    "type",
    "type_name",
    "dimension",
    "dimension_name",
    "score",
    "notes",
]


def _fenced(text: Any) -> str:
    value = "" if text is None else str(text)
    longest = max((len(match) for match in re.findall(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{value}\n{fence}"


def expected_behavior(scenario: dict[str, Any], dimension_key: str) -> str:
    info = DIMENSION_INFO[dimension_key]
    return (
        f"{info['expected']}\n\n"
        f"- Outdated state (`M_old`): {scenario['M_old']}\n"
        f"- Updated evidence (`M_new`): {scenario['M_new']}\n"
        f"- Hidden state logic: {scenario['explanation']}"
    )


def _load_existing_scores(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("uid", ""), row.get("dimension", ""))
        if key in result:
            raise ValueError(f"Duplicate manual-score row: {key}")
        result[key] = row
    return result


def refresh_manual_scores(paths: ArtifactPaths, artifact: dict[str, Any]) -> None:
    existing = _load_existing_scores(paths.manual_scores)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SCORE_FIELDS)
    writer.writeheader()
    for scenario in artifact["data"]:
        for dimension_key in DIMENSION_KEYS:
            key = (str(scenario["uid"]), dimension_key)
            prior = existing.get(key, {})
            info = DIMENSION_INFO[dimension_key]
            writer.writerow(
                {
                    "uid": scenario["uid"],
                    "context_view": paths.view,
                    "type": scenario["type"],
                    "type_name": TYPE_INFO[scenario["type"]],
                    "dimension": dimension_key,
                    "dimension_name": f"{info['code']} - {info['name']}",
                    "score": prior.get("score", ""),
                    "notes": prior.get("notes", ""),
                }
            )
    atomic_write_text(paths.manual_scores, output.getvalue())


def build_report(
    artifact: dict[str, Any],
    responses: dict[tuple[str, str], dict[str, Any]],
    context_view: str,
) -> str:
    meta = artifact
    type_counts = Counter(str(scenario["type"]) for scenario in meta["data"])
    lines = [
        "# Qwen3-0.6B on STALE",
        "",
        f"- Context view: `{context_view}` — {CONTEXT_VIEWS[context_view]}",
        (
            f"- Scenarios: {len(meta['data'])} "
            f"({type_counts.get('T1', 0)} Type I, {type_counts.get('T2', 0)} Type II)"
        ),
        f"- Queries: {len(meta['data']) * len(DIMENSION_KEYS)}",
        f"- Dataset: `{meta['dataset']['id']}@{meta['dataset']['revision']}`",
        f"- Model: `{meta['model_tokenizer']['id']}@{meta['model_tokenizer']['revision']}`",
        f"- Prompt version: `{meta['prompt_version']}`",
        f"- Input limit: {meta['token_budget']['input_token_limit']:,} tokens",
        f"- Reserved generation: {meta['token_budget']['max_new_tokens']:,} tokens",
        "- Scoring: manual; this report does not contain LLM-judge decisions.",
        "",
    ]
    for scenario_number, scenario in enumerate(meta["data"], start=1):
        view = scenario["views"][context_view]
        lines.extend(
            [
                f"## {scenario_number}. {scenario['type']} — {scenario['type_name']}",
                "",
                f"UID: `{scenario['uid']}`  ",
                f"Sessions retained: {len(view['kept_session_indices'])}/{scenario['source_session_count']}  ",
                f"Evidence sessions: {', '.join(str(index + 1) for index in scenario['relevant_session_index'])}",
                "",
            ]
        )
        for dimension_key in DIMENSION_KEYS:
            info = DIMENSION_INFO[dimension_key]
            record = responses.get((str(scenario["uid"]), dimension_key))
            lines.extend(
                [
                    f"### {info['code']} — {info['name']}",
                    "",
                    "**Query**",
                    "",
                    _fenced(scenario["probing_queries"][dimension_key]),
                    "",
                    "**Model response**",
                    "",
                ]
            )
            if record is None:
                lines.append("_Pending._")
            elif record.get("status") != "ok":
                lines.append(f"_Generation error: {record.get('error', 'unknown error')}_")
            else:
                lines.extend(
                    [
                        _fenced(record.get("response", "")),
                        "",
                        (
                            f"Input/output tokens: {record.get('input_tokens', 'n/a')}/"
                            f"{record.get('output_tokens', 'n/a')} · "
                            f"Elapsed: {record.get('elapsed_seconds', 0):.3f}s · "
                            f"Hit cap: {record.get('hit_token_limit', False)} · "
                            f"Thinking marker: {record.get('thinking_marker_detected', False)}"
                        ),
                    ]
                )
            lines.extend(
                [
                    "",
                    "**Expected ground-truth criteria**",
                    "",
                    expected_behavior(scenario, dimension_key),
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


def refresh_report(paths: ArtifactPaths, artifact: dict[str, Any]) -> None:
    responses = load_response_map(paths)
    atomic_write_text(paths.report, build_report(artifact, responses, paths.view))
    refresh_manual_scores(paths, artifact)


def _normalized_score(raw: str) -> bool | None:
    value = raw.strip().lower()
    if value in {"pass", "1", "true", "yes"}:
        return True
    if value in {"fail", "0", "false", "no"}:
        return False
    if not value:
        return None
    raise ValueError(f"Invalid score {raw!r}; use pass or fail")


def summarize_manual_scores(paths: ArtifactPaths, artifact: dict[str, Any]) -> dict[str, Any]:
    existing = _load_existing_scores(paths.manual_scores)
    responses = load_response_map(paths)
    expected_keys = [
        (str(scenario["uid"]), dimension_key)
        for scenario in artifact["data"]
        for dimension_key in DIMENSION_KEYS
    ]
    scenario_by_uid = {str(scenario["uid"]): scenario for scenario in artifact["data"]}
    scored: list[tuple[dict[str, Any], str, bool]] = []
    missing: list[dict[str, str]] = []
    missing_responses: list[dict[str, str]] = []
    for uid, dimension_key in expected_keys:
        response = responses.get((uid, dimension_key))
        if not response or response.get("status") != "ok":
            missing_responses.append({"uid": uid, "dimension": dimension_key})
        row = existing.get((uid, dimension_key))
        score = _normalized_score(row.get("score", "") if row else "")
        if score is None:
            missing.append({"uid": uid, "dimension": dimension_key})
        else:
            scored.append((scenario_by_uid[uid], dimension_key, score))

    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "complete": not missing and not missing_responses,
        "coverage": {"scored": len(scored), "total": len(expected_keys)},
        "missing": missing,
        "response_coverage": {
            "completed": len(expected_keys) - len(missing_responses),
            "total": len(expected_keys),
        },
        "missing_responses": missing_responses,
    }
    if result["complete"]:
        buckets: dict[str, list[bool]] = defaultdict(list)
        for scenario, dimension_key, score in scored:
            type_code = str(scenario["type"])
            dim_code = str(DIMENSION_INFO[dimension_key]["code"])
            buckets["overall"].append(score)
            buckets[f"type:{type_code}"].append(score)
            buckets[f"dimension:{dim_code}"].append(score)
            buckets[f"cell:{type_code}:{dim_code}"].append(score)
        result["metrics"] = {
            key: {
                "passed": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values),
                "accuracy_percent": 100.0 * sum(values) / len(values),
            }
            for key, values in sorted(buckets.items())
        }

    atomic_write_json(paths.summary, result)
    lines = [
        "# Manual STALE score summary",
        "",
        f"Coverage: {len(scored)}/{len(expected_keys)}",
        f"Successful responses: {len(expected_keys) - len(missing_responses)}/{len(expected_keys)}",
        "",
    ]
    if not result["complete"]:
        lines.append(
            "The evaluation is incomplete; generate every response and fill every score in "
            "`manual_scores.csv`."
        )
    else:
        lines.extend(["| Group | Passed | Total | Accuracy |", "|---|---:|---:|---:|"])
        for key, metric in result["metrics"].items():
            lines.append(
                f"| {key} | {metric['passed']} | {metric['total']} | "
                f"{metric['accuracy_percent']:.1f}% |"
            )
    atomic_write_text(paths.summary_markdown, "\n".join(lines) + "\n")
    return result


def summarize_context_comparison(
    artifact_root: Path,
    artifact: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Write paired accuracy deltas after both context views are fully scored."""
    output: dict[str, Any] = {
        "generated_at": utc_now(),
        "complete": all(summary.get("complete") for summary in summaries.values()),
        "views": list(CONTEXT_VIEWS),
    }
    comparison_json = artifact_root / "comparison_summary.json"
    comparison_markdown = artifact_root / "comparison_summary.md"
    if not output["complete"]:
        output["message"] = "Complete both views' responses and manual scores first."
        atomic_write_json(comparison_json, output)
        atomic_write_text(
            comparison_markdown,
            "# Context-view comparison\n\n"
            "Incomplete: generate and manually score every response in both views.\n",
        )
        return output

    metrics: dict[str, dict[str, Any]] = {}
    for group in summaries["truncated"]["metrics"]:
        truncated = summaries["truncated"]["metrics"][group]
        evidence = summaries["evidence_only"]["metrics"][group]
        metrics[group] = {
            "truncated_accuracy": truncated["accuracy"],
            "evidence_only_accuracy": evidence["accuracy"],
            "delta_evidence_only_minus_truncated": (
                evidence["accuracy"] - truncated["accuracy"]
            ),
            "total": truncated["total"],
        }

    score_maps = {
        view: _load_existing_scores(ArtifactPaths(artifact_root, view).manual_scores)
        for view in CONTEXT_VIEWS
    }
    paired_counts = Counter(
        {
            "both_pass": 0,
            "both_fail": 0,
            "improved_without_distractors": 0,
            "regressed_without_distractors": 0,
        }
    )
    for scenario in artifact["data"]:
        for dimension_key in DIMENSION_KEYS:
            key = (str(scenario["uid"]), dimension_key)
            truncated_score = _normalized_score(score_maps["truncated"][key]["score"])
            evidence_score = _normalized_score(score_maps["evidence_only"][key]["score"])
            pair = (bool(truncated_score), bool(evidence_score))
            paired_counts[
                {
                    (True, True): "both_pass",
                    (False, False): "both_fail",
                    (False, True): "improved_without_distractors",
                    (True, False): "regressed_without_distractors",
                }[pair]
            ] += 1

    output["metrics"] = metrics
    output["paired_outcomes"] = dict(sorted(paired_counts.items()))
    atomic_write_json(comparison_json, output)

    lines = [
        "# Context-view comparison",
        "",
        "Positive deltas mean evidence-only context scored higher.",
        "",
        "| Group | Truncated | Evidence only | Delta (pp) | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for group, metric in metrics.items():
        lines.append(
            f"| {group} | {100 * metric['truncated_accuracy']:.1f}% | "
            f"{100 * metric['evidence_only_accuracy']:.1f}% | "
            f"{100 * metric['delta_evidence_only_minus_truncated']:+.1f} | "
            f"{metric['total']} |"
        )
    lines.extend(["", "## Paired probe outcomes", ""])
    for name, count in sorted(paired_counts.items()):
        lines.append(f"- {name}: {count}")
    atomic_write_text(comparison_markdown, "\n".join(lines) + "\n")
    return output


def load_artifact(paths: ArtifactPaths) -> dict[str, Any]:
    if not paths.scenarios.exists():
        raise FileNotFoundError(f"Missing prepared scenarios: {paths.scenarios}")
    return load_json(paths.scenarios)

