from __future__ import annotations

import json
import math
import statistics
import time
import traceback
from pathlib import Path
from typing import Any

from .artifacts import RunPaths, ScenarioPaths
from .audit import AuditCollector
from .constants import (
    DEFAULT_SEED,
    DEFAULT_VLLM_BASE_URL,
    DIMENSION_CODES,
    DIMENSION_KEYS,
    EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION,
    PACKAGE_VERSION,
    RUN_SCHEMA_VERSION,
    TARGET_MODEL_ID,
    TARGET_MODEL_REVISION,
)
from .cupmem_bridge import load_cupmem_symbols
from .io_utils import (
    atomic_write_json,
    atomic_write_json_zst,
    atomic_write_jsonl,
    atomic_write_jsonl_zst,
    atomic_write_text,
    load_json,
    load_json_zst,
    load_jsonl_zst,
    sha256_file,
    sha256_json,
    utc_now,
)


def _initial_status(uid: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "uid": uid,
        "phase": "not_started",
        "completed_session_indices": [],
        "completed_queries": [],
        "memory_hash": "",
        "call_count": 0,
        "started_at": None,
        "updated_at": utc_now(),
        "elapsed_seconds": 0.0,
    }


def _load_status(paths: ScenarioPaths, uid: str) -> dict[str, Any]:
    return load_json(paths.status) if paths.status.exists() else _initial_status(uid)


def _memory_markdown(snapshot: dict[str, Any]) -> str:
    lines = ["# CUP-Mem profile", "", "## Active profile", ""]
    active = snapshot.get("active_profile", {})
    if not active:
        lines.append("_No active items._")
    for track, items in sorted(active.items()):
        lines.append(f"### {track}")
        lines.append("")
        for item in items:
            lines.append(
                f"- `{item.get('item_id')}` — **{item.get('status', 'ACTIVE')}** — "
                f"{item.get('value')}"
            )
            lines.append(
                f"  - confidence: {item.get('confidence')}; strength: "
                f"{item.get('active_strength', 'n/a')}"
            )
            lines.append(
                f"  - timestamps: first_seen={item.get('first_seen', '')}; "
                f"last_updated={item.get('last_updated', '')}; "
                f"created_session={item.get('created_session_id', '')} "
                f"(index={item.get('created_session_index', '')}, "
                f"time={item.get('created_session_time', '')})"
            )
            lines.append(
                f"  - provenance: evidence={', '.join(item.get('evidence_chunk_ids', [])) or 'none'}; "
                f"deltas={', '.join(item.get('source_delta_ids', [])) or 'none'}"
            )
            revisions = item.get("revision_history", [])
            lines.append(
                "  - revisions: "
                + (json.dumps(revisions, ensure_ascii=False) if revisions else "none")
            )
        lines.append("")
    lines.extend(["## Stale archive", ""])
    stale = snapshot.get("stale_archive", [])
    if not stale:
        lines.append("_No stale items._")
    stale_by_track: dict[str, list[dict[str, Any]]] = {}
    for item in stale:
        track = f"{item.get('bucket')}/{item.get('local_track')}"
        stale_by_track.setdefault(track, []).append(item)
    for track, items in sorted(stale_by_track.items()):
        lines.extend([f"### {track}", ""])
        for item in items:
            lines.append(
                f"- `{item.get('item_id')}` — **{item.get('status', 'STALE')}** — "
                f"{item.get('value')}"
            )
            lines.append(
                f"  - confidence: {item.get('confidence')}; first_seen={item.get('first_seen', '')}; "
                f"last_updated={item.get('last_updated', '')}"
            )
            lines.append(
                f"  - provenance: evidence={', '.join(item.get('evidence_chunk_ids', [])) or 'none'}; "
                f"deltas={', '.join(item.get('source_delta_ids', [])) or 'none'}"
            )
            revisions = item.get("revision_history", [])
            lines.append(
                "  - revisions: "
                + (json.dumps(revisions, ensure_ascii=False) if revisions else "none")
            )
        lines.append("")
    lines.extend(["", "## Unknown current", ""])
    unknown = snapshot.get("unknown_current", [])
    if not unknown:
        lines.append("_No unknown-current tracks._")
    for item in unknown:
        lines.append(
            f"- `{item.get('bucket')}/{item.get('local_track')}` — "
            f"**{item.get('status', 'UNKNOWN_CURRENT')}** — {item.get('reason')}"
        )
        lines.append(
            f"  - invalidated_by={item.get('invalidated_by_delta_id', '')}; "
            f"last_updated={item.get('last_updated', '')}; "
            f"created_session={item.get('created_session_id', '')} "
            f"(index={item.get('created_session_index', '')}, "
            f"time={item.get('created_session_time', '')})"
        )
    lines.extend(["", "## Stale-support links", ""])
    links = snapshot.get("stale_support_links", [])
    if not links:
        lines.append("_No stale-support links._")
    for link in links:
        lines.append(
            f"- `{link.get('stale_item_id')}` ← `{link.get('linking_delta_id')}` "
            f"({link.get('link_type')}): {link.get('reason')}"
        )
    return "\n".join(lines).rstrip() + "\n"


def _resolve_embedding_path(cache_dir: Path, preparation_manifest: dict[str, Any]) -> Path:
    recorded = Path(str(preparation_manifest.get("embedding_model_local_path", "")))
    if recorded.exists():
        return recorded
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            EMBEDDING_MODEL_ID,
            revision=EMBEDDING_MODEL_REVISION,
            cache_dir=str(cache_dir),
        )
    ).resolve()


def _restricted_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Return only fields permitted to reach target inference."""
    return {
        "uid": scenario["uid"],
        "type": scenario["type"],
        "sessions": scenario["sessions"],
        "probing_queries": scenario["probing_queries"],
    }


def _validate_run_manifest(paths: RunPaths) -> None:
    manifest = load_json(paths.run_manifest)
    expected = {
        "scenarios_sha256": sha256_file(paths.scenarios_file),
        "target_model": {"id": TARGET_MODEL_ID, "revision": TARGET_MODEL_REVISION},
        "embedding_model": {"id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_MODEL_REVISION},
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"run_manifest.json {field} does not match this evaluation")


def _aggregate_responses(paths: RunPaths, artifact: dict[str, Any]) -> None:
    records: list[dict[str, Any]] = []
    for scenario in artifact["data"]:
        scenario_paths = paths.scenario(str(scenario["uid"]))
        responses: dict[str, str] = {}
        meta: dict[str, Any] = {}
        complete = True
        for dimension_key in DIMENSION_KEYS:
            result_path = scenario_paths.query_root(dimension_key) / "result.json"
            if not result_path.exists():
                complete = False
                break
            result = load_json(result_path)
            response_key = dimension_key.replace("_query", "_response")
            responses[response_key] = str(result.get("response", ""))
            meta[dimension_key.replace("_query", "_meta")] = {
                "elapsed_seconds": result.get("elapsed_seconds"),
                "usage": result.get("usage", {}),
            }
        if complete:
            records.append(
                {
                    "uid": scenario["uid"],
                    "type": scenario["type"],
                    "target_model_responses": responses,
                    "target_model_meta": meta,
                }
            )
    atomic_write_jsonl(paths.responses, records)


def _progress_payload(
    *,
    completed: int,
    total_scenarios: int,
    scenario: dict[str, Any] | None,
    status: dict[str, Any] | None,
    durations: list[float],
    started_at: float,
    total_qwen_calls: int,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started_at
    median = statistics.median(durations) if durations else None
    ordered = sorted(durations)
    p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)] if ordered else None
    remaining = total_scenarios - completed
    return {
        "phase": "target_inference",
        "updated_at": utc_now(),
        "completed_scenarios": completed,
        "total_scenarios": total_scenarios,
        "current_uid": scenario.get("uid") if scenario else None,
        "current_type": scenario.get("type") if scenario else None,
        "completed_sessions_current": len(status.get("completed_session_indices", [])) if status else 0,
        "sessions_current_total": len(scenario.get("sessions", [])) if scenario else 0,
        "completed_queries_current": len(status.get("completed_queries", [])) if status else 0,
        "queries_current_total": len(DIMENSION_KEYS),
        "qwen_calls_current": status.get("call_count", 0) if status else 0,
        "qwen_calls_total": total_qwen_calls,
        "elapsed_seconds": round(elapsed, 3),
        "median_scenario_seconds": round(median, 3) if median is not None else None,
        "p90_scenario_seconds": round(p90, 3) if p90 is not None else None,
        "eta_seconds": round(median * remaining, 3) if median is not None else None,
    }


def _progress_postfix(
    status: dict[str, Any],
    durations: list[float],
    *,
    completed: int,
    total: int,
    total_qwen_calls: int,
) -> dict[str, str | int]:
    ordered = sorted(durations)
    median = statistics.median(ordered) if ordered else None
    p90 = ordered[max(0, math.ceil(0.9 * len(ordered)) - 1)] if ordered else None
    remaining = max(0, total - completed)
    return {
        "phase": str(status.get("phase", "")),
        "session": f"{len(status.get('completed_session_indices', []))}/12",
        "query": f"{len(status.get('completed_queries', []))}/3",
        "calls": total_qwen_calls,
        "median_s": f"{median:.1f}" if median is not None else "n/a",
        "p90_s": f"{p90:.1f}" if p90 is not None else "n/a",
        "eta_s": f"{median * remaining:.0f}" if median is not None else "n/a",
    }


def run_evaluation(
    artifact_root: Path,
    *,
    cache_dir: Path,
    vllm_base_url: str = DEFAULT_VLLM_BASE_URL,
    resume: bool = False,
    max_per_type: int | None = None,
    show_progress: bool = True,
) -> int:
    paths = RunPaths(artifact_root)
    artifact = load_json(paths.scenarios_file)
    preparation = load_json(paths.preparation_manifest)
    if preparation.get("scenarios_sha256") != sha256_file(paths.scenarios_file):
        raise ValueError("scenarios.json does not match preparation_manifest.json")

    selected: list[dict[str, Any]] = []
    type_counts: dict[str, int] = {}
    for scenario in artifact["data"]:
        conflict_type = str(scenario["type"])
        if max_per_type is not None and type_counts.get(conflict_type, 0) >= max_per_type:
            continue
        type_counts[conflict_type] = type_counts.get(conflict_type, 0) + 1
        selected.append(scenario)

    symbols = load_cupmem_symbols()
    collector = AuditCollector()
    embedding_path = _resolve_embedding_path(cache_dir, preparation)
    local_call_cache = cache_dir / "cupmem-call-cache"
    llm = symbols["LLMClient"](
        model=TARGET_MODEL_ID,
        api_key="EMPTY",
        base_url=vllm_base_url,
        cache_dir=local_call_cache,
        audit_sink=collector,
        request_overrides={
            "temperature": 0,
            "max_tokens": 4096,
            "seed": DEFAULT_SEED,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )
    thresholds = symbols["PipelineThresholds"]()
    engine = symbols["CupMemEngine"](
        llm=llm,
        embedding_model_path=str(embedding_path),
        embedding_device="cpu",
        thresholds=thresholds,
        trace_config=symbols["TraceConfig"](enable_debug_trace=True),
    )

    if not paths.run_manifest.exists():
        atomic_write_json(
            paths.run_manifest,
            {
                "schema_version": RUN_SCHEMA_VERSION,
                "package_version": PACKAGE_VERSION,
                "created_at": utc_now(),
                "scenarios_sha256": sha256_file(paths.scenarios_file),
                "target_model": {"id": TARGET_MODEL_ID, "revision": TARGET_MODEL_REVISION},
                "embedding_model": {"id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_MODEL_REVISION},
                "target_inference": {
                    "base_url": vllm_base_url,
                    "thinking_enabled": False,
                    "temperature": 0,
                    "seed": DEFAULT_SEED,
                    "max_tokens": 4096,
                    "cup_mem_default_thresholds_unchanged": True,
                },
            },
        )
    else:
        _validate_run_manifest(paths)

    checkpointed_before = sum(
        paths.scenario(str(item["uid"])).status.exists() for item in selected
    )
    completed_before = sum(
        _load_status(paths.scenario(str(item["uid"])), str(item["uid"])).get("phase") == "completed"
        for item in selected
    )
    completed_call_count = sum(
        int(_load_status(paths.scenario(str(item["uid"])), str(item["uid"])).get("call_count", 0))
        for item in selected
        if _load_status(paths.scenario(str(item["uid"])), str(item["uid"])).get("phase")
        == "completed"
    )
    completed_durations = [
        float(_load_status(paths.scenario(str(item["uid"])), str(item["uid"])).get("elapsed_seconds", 0.0))
        for item in selected
        if _load_status(paths.scenario(str(item["uid"])), str(item["uid"])).get("phase")
        == "completed"
    ]
    if checkpointed_before and not resume:
        raise ValueError(f"Found {checkpointed_before} checkpointed scenarios; pass --resume")

    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None
    progress = (
        tqdm(total=len(selected), initial=completed_before, desc="CUP-Mem", unit="scenario")
        if show_progress and tqdm is not None
        else None
    )
    generated = 0
    durations: list[float] = completed_durations
    overall_started = time.perf_counter()
    completed = completed_before

    for scenario in selected:
        uid = str(scenario["uid"])
        scenario_paths = paths.scenario(uid)
        status = _load_status(scenario_paths, uid)
        if status.get("phase") == "completed":
            continue
        scenario_started = time.perf_counter()
        status["started_at"] = status.get("started_at") or utc_now()
        atomic_write_json(scenario_paths.input, scenario)
        restricted = _restricted_scenario(scenario)
        if any(key in restricted for key in ("M_old", "M_new", "explanation")):
            raise AssertionError("Ground truth leaked into target scenario view")

        collector.reset()
        if scenario_paths.llm_calls.exists():
            collector.load(scenario_paths.llm_calls)
        llm.reset_usage_tracking()
        llm.cache_dir = local_call_cache / uid
        llm.cache_dir.mkdir(parents=True, exist_ok=True)
        if status.get("completed_session_indices"):
            if not scenario_paths.engine_state.exists():
                raise ValueError(f"Scenario {uid} has session status but no engine checkpoint")
            engine.restore_state(load_json_zst(scenario_paths.engine_state))
            session_traces = load_jsonl_zst(scenario_paths.session_traces)
        else:
            engine.reset()
            session_traces = []

        try:
            status["phase"] = "memory"
            for retained in restricted["sessions"]:
                original_index = int(retained["original_index"])
                if original_index in status["completed_session_indices"]:
                    continue
                if progress is not None:
                    progress.set_postfix_str(
                        f"{scenario['type']} {uid[:8]} memory {len(status['completed_session_indices']) + 1}/12"
                    )
                session_started = time.perf_counter()
                trace = engine.process_session(
                    session=retained["messages"],
                    session_index=original_index,
                    session_time=str(retained["timestamp"]),
                )
                trace["original_index"] = original_index
                trace["is_evidence"] = bool(retained["is_evidence"])
                trace["elapsed_seconds_runner"] = round(time.perf_counter() - session_started, 3)
                session_traces.append(trace)
                status["completed_session_indices"].append(original_index)
                status["call_count"] = sum(
                    event.get("event") == "llm_attempt" for event in collector.snapshot()
                )
                status["updated_at"] = utc_now()
                atomic_write_json_zst(scenario_paths.engine_state, engine.export_state())
                atomic_write_jsonl_zst(scenario_paths.session_traces, session_traces)
                collector.flush(scenario_paths.llm_calls)
                atomic_write_json(scenario_paths.status, status)
                atomic_write_json(
                    paths.progress,
                    _progress_payload(
                        completed=completed,
                        total_scenarios=len(selected),
                        scenario=scenario,
                        status=status,
                        durations=durations,
                        started_at=overall_started,
                        total_qwen_calls=completed_call_count + int(status["call_count"]),
                    ),
                )
                if progress is not None:
                    progress.set_postfix(
                        _progress_postfix(
                            status,
                            durations,
                            completed=completed,
                            total=len(selected),
                            total_qwen_calls=completed_call_count + int(status["call_count"]),
                        ),
                        refresh=True,
                    )

            memory_state = engine.export_state()
            memory_hash = sha256_json(memory_state)
            snapshot = engine.store.to_snapshot()
            status["memory_hash"] = memory_hash
            atomic_write_json(scenario_paths.final_profile, snapshot)
            atomic_write_text(scenario_paths.memory_markdown, _memory_markdown(snapshot))

            status["phase"] = "queries"
            for dimension_key in DIMENSION_KEYS:
                if dimension_key in status["completed_queries"]:
                    continue
                if progress is not None:
                    progress.set_postfix_str(
                        f"{scenario['type']} {uid[:8]} {DIMENSION_CODES[dimension_key]}"
                    )
                query_text = str(restricted["probing_queries"][dimension_key])
                query_started = time.perf_counter()
                query_log = engine.answer_query(
                    query_label=dimension_key,
                    query_text=query_text,
                )
                elapsed = time.perf_counter() - query_started
                after_hash = sha256_json(engine.export_state())
                if after_hash != memory_hash:
                    raise AssertionError(f"CUP-Mem state changed while answering {uid}/{dimension_key}")
                query_root = scenario_paths.query_root(dimension_key)
                answer = str(query_log.get("answer", {}).get("answer", ""))
                query_events = [
                    event
                    for event in collector.snapshot()
                    if event.get("query_label") == dimension_key
                ]
                usage = symbols["summarize_call_records"](
                    [record for record in llm.get_call_records() if record.get("query_label") == dimension_key]
                )
                atomic_write_json(
                    query_root / "result.json",
                    {
                        "uid": uid,
                        "type": scenario["type"],
                        "dimension_key": dimension_key,
                        "dimension_code": DIMENSION_CODES[dimension_key],
                        "query": query_text,
                        "response": answer,
                        "verdict": query_log.get("verdict", {}),
                        "elapsed_seconds": round(elapsed, 3),
                        "usage": usage,
                        "memory_hash_before": memory_hash,
                        "memory_hash_after": after_hash,
                    },
                )
                atomic_write_json(
                    query_root / "answer_composer_request.json",
                    collector.answer_composer_request(dimension_key),
                )
                atomic_write_json_zst(query_root / "trace.json.zst", query_log)
                atomic_write_jsonl_zst(query_root / "llm_calls.jsonl.zst", query_events)
                status["completed_queries"].append(dimension_key)
                status["call_count"] = sum(
                    event.get("event") == "llm_attempt" for event in collector.snapshot()
                )
                status["updated_at"] = utc_now()
                collector.flush(scenario_paths.llm_calls)
                atomic_write_json(scenario_paths.status, status)
                _aggregate_responses(paths, artifact)
                if progress is not None:
                    progress.set_postfix(
                        _progress_postfix(
                            status,
                            durations,
                            completed=completed,
                            total=len(selected),
                            total_qwen_calls=completed_call_count + int(status["call_count"]),
                        ),
                        refresh=True,
                    )

            status["phase"] = "completed"
            status["updated_at"] = utc_now()
            status["elapsed_seconds"] = round(time.perf_counter() - scenario_started, 3)
            atomic_write_json(scenario_paths.status, status)
            collector.flush(scenario_paths.llm_calls)
            generated += 1
            completed += 1
            durations.append(float(status["elapsed_seconds"]))
            completed_call_count += int(status["call_count"])
            if progress is not None:
                progress.update(1)
                progress.set_postfix(
                    _progress_postfix(
                        status,
                        durations,
                        completed=completed,
                        total=len(selected),
                        total_qwen_calls=completed_call_count,
                    ),
                    refresh=True,
                )
        except Exception as error:
            status["phase"] = "error"
            status["updated_at"] = utc_now()
            status["error"] = f"{type(error).__name__}: {error}"
            status["traceback"] = traceback.format_exc()
            atomic_write_json(scenario_paths.status, status)
            collector.flush(scenario_paths.llm_calls)
            raise
        finally:
            atomic_write_json(
                paths.progress,
                _progress_payload(
                    completed=completed,
                    total_scenarios=len(selected),
                    scenario=scenario,
                    status=status,
                    durations=durations,
                    started_at=overall_started,
                    total_qwen_calls=(
                        completed_call_count
                        if status.get("phase") == "completed"
                        else completed_call_count + int(status.get("call_count", 0))
                    ),
                ),
            )

    if progress is not None:
        progress.close()
    _aggregate_responses(paths, artifact)
    return generated
