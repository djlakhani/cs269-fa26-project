from __future__ import annotations

import csv
import io
import json
import re
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    DEFAULT_ATTENTION_BACKEND,
    MANIFEST_SCHEMA_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    PACKAGE_VERSION,
)
from .inference import TransformersQwenBackend
from .io_utils import (
    atomic_write_json,
    atomic_write_text,
    load_json,
    read_jsonl,
    sha256_file,
    utc_now,
)
from .meme import (
    MEME_PROMPT_VERSION,
    MEME_RESPONSE_SCHEMA_VERSION,
    MEME_SCHEMA_VERSION,
    MEME_TASK_NAMES,
    MEME_VIEWS,
    build_messages,
    format_sessions,
)


@dataclass(frozen=True)
class MemePaths:
    root: Path
    view: str

    @property
    def episodes(self) -> Path:
        return self.root / "episodes.json"

    @property
    def preparation_manifest(self) -> Path:
        return self.root / "preparation_manifest.json"

    @property
    def view_root(self) -> Path:
        return self.root / "views" / self.view

    @property
    def checkpoints(self) -> Path:
        return self.view_root / "checkpoints"

    @property
    def responses(self) -> Path:
        return self.view_root / "responses.jsonl"

    @property
    def manifest(self) -> Path:
        return self.view_root / "manifest.json"

    @property
    def report(self) -> Path:
        return self.view_root / "results.md"

    @property
    def manual_scores(self) -> Path:
        return self.view_root / "manual_scores.csv"

    @property
    def summary(self) -> Path:
        return self.view_root / "summary.json"

    @property
    def summary_markdown(self) -> Path:
        return self.view_root / "summary.md"


def _key(record: dict[str, Any]) -> tuple[str, str]:
    return str(record["episode_id"]), str(record["probe_id"])


def _safe(value: str) -> str:
    result = "".join(
        character for character in value
        if character.isalnum() or character in "-_"
    )
    if not result:
        raise ValueError(f"Unsafe artifact key: {value!r}")
    return result


def checkpoint_path(paths: MemePaths, episode_id: str, probe_id: str) -> Path:
    return paths.checkpoints / f"{_safe(episode_id)}__{_safe(probe_id)}.json"


def load_responses(paths: MemePaths) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if paths.responses.exists():
        for record in read_jsonl(paths.responses):
            records[_key(record)] = record
    if paths.checkpoints.exists():
        for path in sorted(paths.checkpoints.glob("*.json")):
            record = load_json(path)
            records[_key(record)] = record
    return records


def write_checkpoint(paths: MemePaths, record: dict[str, Any]) -> None:
    atomic_write_json(
        checkpoint_path(
            paths, str(record["episode_id"]), str(record["probe_id"])
        ),
        record,
    )


def ordered_keys(artifact: dict[str, Any]) -> list[tuple[str, str]]:
    return [
        (str(episode["episode_id"]), str(probe["probe_id"]))
        for episode in artifact["data"]
        for probe in episode["probes"]
    ]


def rebuild_jsonl(
    paths: MemePaths,
    artifact: dict[str, Any],
    records: dict[tuple[str, str], dict[str, Any]] | None = None,
) -> None:
    records = records if records is not None else load_responses(paths)
    lines = [
        json.dumps(records[key], ensure_ascii=False)
        for key in ordered_keys(artifact) if key in records
    ]
    atomic_write_text(
        paths.responses, "\n".join(lines) + ("\n" if lines else "")
    )


def _fenced(value: Any) -> str:
    text = "" if value is None else str(value)
    longest = max((len(match) for match in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{text}\n{fence}"


def _response_block(record: dict[str, Any] | None) -> list[str]:
    if record is None:
        return ["_Pending._"]
    if record.get("status") != "ok":
        return [f"_Generation error: {record.get('error', 'unknown error')}_"]
    return [
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


def build_report(
    artifact: dict[str, Any],
    view: str,
    records: dict[tuple[str, str], dict[str, Any]],
) -> str:
    after_count = int(artifact["selection"]["after_question_count"])
    before_count = int(artifact["selection"]["before_prerequisite_count"])
    lines = [
        "# Qwen3-4B on MEME",
        "",
        f"- Context view: `{view}` — {MEME_VIEWS[view]}",
        f"- Episodes: {len(artifact['data'])}",
        f"- Scored after-change questions: {after_count}",
        f"- Required pre-change checks: {before_count}",
        (
            f"- Dataset: `{artifact['dataset']['id']}@"
            f"{artifact['dataset']['revision']}`"
        ),
        (
            f"- Model: `{artifact['model_tokenizer']['id']}@"
            f"{artifact['model_tokenizer']['revision']}`"
        ),
        f"- Prompt version: `{artifact['prompt_version']}`",
        "- Task selection: Cascade and Absence only.",
        (
            "- Scoring: manual. Official pass requires both the after-change "
            "answer and its matching pre-change prerequisite to pass."
        ),
        (
            "- Known v1.0 Absence label issues are preserved and visibly flagged; "
            "this pipeline does not silently repair benchmark gold labels."
        ),
        "",
    ]
    for episode_number, episode in enumerate(artifact["data"], start=1):
        view_data = episode["views"][view]
        probes = {
            str(probe["probe_id"]): probe for probe in episode["probes"]
        }
        after_probes = [
            probe for probe in episode["probes"]
            if probe["phase"] == "after"
        ]
        lines.extend([
            f"## {episode_number}. {episode['episode_id']} — {episode['domain']}",
            "",
            (
                f"Sessions retained: {len(view_data['kept_session_indices'])}/"
                f"{view_data['source_session_count']}  "
            ),
            (
                f"Maximum prepared input: "
                f"{view_data['maximum_input_tokens']:,} tokens  "
            ),
            (
                f"Root change: {episode['root_change']['before']} → "
                f"{episode['root_change']['after']}"
            ),
            "",
        ])
        for probe in after_probes:
            before = probes[str(probe["prerequisite_probe_id"])]
            before_record = records.get(
                (str(episode["episode_id"]), str(before["probe_id"]))
            )
            after_record = records.get(
                (str(episode["episode_id"]), str(probe["probe_id"]))
            )
            hop = f", hop {probe['hop']}" if probe.get("hop") is not None else ""
            lines.extend([
                (
                    f"### {probe['task_name']} "
                    f"(`{probe['probe_id']}`{hop})"
                ),
                "",
                *(
                    {
                        True: [
                            "> **Known dataset issue:** the MEME authors flag this "
                            "episode's released Absence gold label as potentially incorrect.",
                            "",
                        ],
                        False: [],
                    }[bool(probe.get("known_label_issue"))]
                ),
                "**Pre-change prerequisite question**",
                "",
                _fenced(before["question"]),
                "",
                "**Expected pre-change value**",
                "",
                _fenced(before["gold_answer"]),
                "",
                "**Pre-change model response**",
                "",
                *_response_block(before_record),
                "",
                "**After-change scored question**",
                "",
                _fenced(probe["question"]),
                "",
                "**Released gold answer**",
                "",
                _fenced(probe["gold_answer"]),
                "",
                "**After-change model response**",
                "",
                *_response_block(after_record),
                "",
            ])
    return "\n".join(lines).rstrip() + "\n"


SCORE_FIELDS = [
    "context_view",
    "episode_id",
    "domain",
    "task_type",
    "task_name",
    "hop",
    "after_probe_id",
    "before_probe_id",
    "before_score",
    "after_score",
    "notes",
]


def _read_scores(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (row.get("episode_id", ""), row.get("after_probe_id", ""))
        if key in result:
            raise ValueError(f"Duplicate MEME manual-score row: {key}")
        result[key] = row
    return result


def refresh_manual_scores(paths: MemePaths, artifact: dict[str, Any]) -> None:
    existing = _read_scores(paths.manual_scores)
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=SCORE_FIELDS)
    writer.writeheader()
    for episode in artifact["data"]:
        for probe in episode["probes"]:
            if probe["phase"] != "after":
                continue
            key = (str(episode["episode_id"]), str(probe["probe_id"]))
            prior = existing.get(key, {})
            writer.writerow({
                "context_view": paths.view,
                "episode_id": episode["episode_id"],
                "domain": episode["domain"],
                "task_type": probe["task_type"],
                "task_name": probe["task_name"],
                "hop": probe.get("hop", ""),
                "after_probe_id": probe["probe_id"],
                "before_probe_id": probe["prerequisite_probe_id"],
                "before_score": prior.get("before_score", ""),
                "after_score": prior.get("after_score", ""),
                "notes": prior.get("notes", ""),
            })
    atomic_write_text(paths.manual_scores, output.getvalue())


def refresh_report(paths: MemePaths, artifact: dict[str, Any]) -> None:
    records = load_responses(paths)
    atomic_write_text(paths.report, build_report(artifact, paths.view, records))
    refresh_manual_scores(paths, artifact)


def _run_config(
    artifact: dict[str, Any], view: str, attention_backend: str
) -> dict[str, Any]:
    return {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "benchmark": "MEME",
        "prompt_version": MEME_PROMPT_VERSION,
        "context_view": view,
        "thinking_enabled": False,
        "decoding": {
            "do_sample": False,
            "max_new_tokens": artifact["token_budget"]["max_new_tokens"],
        },
        "context_window": artifact["token_budget"]["context_window"],
        "hard_input_limit": artifact["token_budget"]["hard_input_limit"],
        "filler_input_target": artifact["token_budget"]["filler_input_target"],
        "dtype": "float16",
        "attention_implementation": "sdpa",
        "enforced_sdpa_backend": attention_backend,
        "batch_size": 1,
    }


def validate(
    paths: MemePaths,
    artifact: dict[str, Any],
    attention_backend: str,
) -> dict[str, Any]:
    if artifact.get("schema_version") != MEME_SCHEMA_VERSION:
        raise ValueError("Prepared MEME episodes use an incompatible schema")
    if artifact.get("model_tokenizer") != {
        "id": MODEL_ID, "revision": MODEL_REVISION
    }:
        raise ValueError("Prepared MEME episodes target another model revision")
    if artifact.get("prompt_version") != MEME_PROMPT_VERSION:
        raise ValueError("Prepared MEME episodes use another prompt version")
    if paths.view not in artifact.get("context_views", {}):
        raise ValueError(f"MEME artifact does not define view {paths.view!r}")
    scenario_sha = sha256_file(paths.episodes)
    preparation = load_json(paths.preparation_manifest)
    if preparation["preparation"].get("scenarios_sha256") != scenario_sha:
        raise ValueError("episodes.json does not match preparation_manifest.json")
    manifest = load_json(paths.manifest) if paths.manifest.exists() else {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "context_view": paths.view,
        "episodes_sha256": scenario_sha,
    }
    if manifest.get("context_view") != paths.view:
        raise ValueError("MEME run manifest belongs to another context view")
    if manifest.get("episodes_sha256") != scenario_sha:
        raise ValueError("episodes.json does not match the MEME run manifest")
    prior = manifest.get("run")
    if prior and prior.get("config") != _run_config(
        artifact, paths.view, attention_backend
    ):
        raise ValueError("Existing MEME run configuration differs")
    return manifest


def initialize_manifest(
    paths: MemePaths,
    artifact: dict[str, Any],
    runtime: dict[str, Any],
    attention_backend: str,
) -> None:
    manifest = validate(paths, artifact, attention_backend)
    prior = manifest.get("run")
    manifest["run"] = {
        "started_at": prior.get("started_at") if prior else utc_now(),
        "last_opened_at": utc_now(),
        "config": _run_config(artifact, paths.view, attention_backend),
        "runtime": runtime,
    }
    atomic_write_json(paths.manifest, manifest)


def run_meme_view(
    artifact_root: Path,
    *,
    view: str,
    attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    cache_dir: Path | None = None,
    resume: bool = False,
    max_episodes: int | None = None,
    show_progress: bool = True,
    backend: Any | None = None,
) -> int:
    paths = MemePaths(artifact_root, view)
    artifact = load_json(paths.episodes)
    validate(paths, artifact, attention_backend)
    records = load_responses(paths)
    completed = {
        key for key, record in records.items() if record.get("status") == "ok"
    }
    if completed and not resume:
        raise ValueError("Existing MEME responses found; pass --resume")
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")
    expected = ordered_keys(artifact)
    if all(key in completed for key in expected):
        rebuild_jsonl(paths, artifact, records)
        refresh_report(paths, artifact)
        return 0
    if backend is None:
        backend = TransformersQwenBackend(
            cache_dir=cache_dir,
            context_window=int(artifact["token_budget"]["context_window"]),
            max_new_tokens=int(artifact["token_budget"]["max_new_tokens"]),
            seed=int(artifact["selection"]["seed"]),
            attention_backend=attention_backend,
        )
    initialize_manifest(
        paths, artifact, backend.runtime_metadata(), attention_backend
    )
    refresh_report(paths, artifact)
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None
    completed_episodes = sum(
        all(
            (str(episode["episode_id"]), str(probe["probe_id"])) in completed
            for probe in episode["probes"]
        )
        for episode in artifact["data"]
    )
    progress = (
        tqdm(
            total=len(artifact["data"]),
            initial=completed_episodes,
            desc=f"MEME {view}",
            unit="episode",
        )
        if show_progress and tqdm is not None else None
    )
    generated = 0
    attempted = 0
    try:
        for episode in artifact["data"]:
            episode_id = str(episode["episode_id"])
            missing = [
                probe for probe in episode["probes"]
                if (episode_id, str(probe["probe_id"])) not in completed
            ]
            if not missing:
                continue
            if max_episodes is not None and attempted >= max_episodes:
                break
            attempted += 1
            view_data = episode["views"][view]
            transcripts = {
                phase: format_sessions(
                    view_data["sessions"],
                    int(view_data["phase_positions"][phase]),
                )
                for phase in ("before", "after")
            }
            for probe in missing:
                probe_id = str(probe["probe_id"])
                key = (episode_id, probe_id)
                if progress is not None:
                    progress.set_postfix_str(
                        f"{episode_id} {probe['phase']} {probe['task_type']}"
                    )
                base = {
                    "schema_version": MEME_RESPONSE_SCHEMA_VERSION,
                    "context_view": view,
                    "episode_id": episode_id,
                    "domain": episode["domain"],
                    "probe_id": probe_id,
                    "phase": probe["phase"],
                    "task_type": probe["task_type"],
                    "task_name": probe["task_name"],
                    "hop": probe.get("hop"),
                    "question": probe["question"],
                    "gold_answer": probe["gold_answer"],
                    "prerequisite_probe_id": probe.get("prerequisite_probe_id"),
                    "known_label_issue": bool(probe.get("known_label_issue")),
                    "generated_at": utc_now(),
                }
                messages = build_messages(
                    transcripts[str(probe["phase"])],
                    str(probe["question"]),
                )
                try:
                    output = backend.generate(messages)
                    expected_tokens = int(
                        view_data["input_token_counts"][probe_id]
                    )
                    if int(output["input_tokens"]) != expected_tokens:
                        raise ValueError(
                            f"Token count changed for {key}: "
                            f"prepared={expected_tokens}, "
                            f"runtime={output['input_tokens']}"
                        )
                    record = {**base, "status": "ok", **output}
                except Exception as error:
                    record = {
                        **base,
                        "status": "error",
                        "error": f"{type(error).__name__}: {error}",
                        "traceback": traceback.format_exc(),
                    }
                    write_checkpoint(paths, record)
                    rebuild_jsonl(paths, artifact)
                    refresh_report(paths, artifact)
                    raise
                write_checkpoint(paths, record)
                records[key] = record
                completed.add(key)
                generated += 1
                rebuild_jsonl(paths, artifact, records)
            refresh_report(paths, artifact)
            if progress is not None:
                progress.update(1)
    finally:
        if progress is not None:
            progress.close()
        rebuild_jsonl(paths, artifact)
        refresh_report(paths, artifact)
    return generated


def run_meme_views(
    artifact_root: Path,
    *,
    views: Iterable[str],
    attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    cache_dir: Path | None = None,
    resume: bool = False,
    max_episodes: int | None = None,
    show_progress: bool = True,
) -> dict[str, int]:
    selected = list(views)
    if not selected:
        raise ValueError("At least one MEME view is required")
    artifact = load_json(artifact_root / "episodes.json")
    for view in selected:
        validate(MemePaths(artifact_root, view), artifact, attention_backend)
    pending = []
    expected_count = len(ordered_keys(artifact))
    for view in selected:
        complete = sum(
            record.get("status") == "ok"
            for record in load_responses(MemePaths(artifact_root, view)).values()
        )
        if complete < expected_count:
            pending.append(view)
    if not pending:
        return {view: 0 for view in selected}
    backend = TransformersQwenBackend(
        cache_dir=cache_dir,
        context_window=int(artifact["token_budget"]["context_window"]),
        max_new_tokens=int(artifact["token_budget"]["max_new_tokens"]),
        seed=int(artifact["selection"]["seed"]),
        attention_backend=attention_backend,
    )
    return {
        view: run_meme_view(
            artifact_root,
            view=view,
            attention_backend=attention_backend,
            cache_dir=cache_dir,
            resume=resume,
            max_episodes=max_episodes,
            show_progress=show_progress,
            backend=backend,
        )
        for view in selected
    }


def _score(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"pass", "1", "true", "yes"}:
        return True
    if normalized in {"fail", "0", "false", "no"}:
        return False
    if not normalized:
        return None
    raise ValueError(f"Invalid score {value!r}; use pass or fail")


def summarize(paths: MemePaths, artifact: dict[str, Any]) -> dict[str, Any]:
    rows = _read_scores(paths.manual_scores)
    records = load_responses(paths)
    evaluated: list[tuple[dict[str, Any], dict[str, Any], bool, bool]] = []
    missing_scores: list[dict[str, str]] = []
    missing_responses: list[dict[str, str]] = []
    for episode in artifact["data"]:
        for probe in episode["probes"]:
            if probe["phase"] != "after":
                continue
            key = (str(episode["episode_id"]), str(probe["probe_id"]))
            row = rows.get(key, {})
            before = _score(row.get("before_score", ""))
            after = _score(row.get("after_score", ""))
            if before is None or after is None:
                missing_scores.append({
                    "episode_id": key[0], "after_probe_id": key[1]
                })
            else:
                evaluated.append((episode, probe, before, after))
            response_keys = [
                key,
                (key[0], str(probe["prerequisite_probe_id"])),
            ]
            if any(
                not records.get(response_key)
                or records[response_key].get("status") != "ok"
                for response_key in response_keys
            ):
                missing_responses.append({
                    "episode_id": key[0], "after_probe_id": key[1]
                })
    total = int(artifact["selection"]["after_question_count"])
    result: dict[str, Any] = {
        "generated_at": utc_now(),
        "complete": not missing_scores and not missing_responses,
        "coverage": {"scored": len(evaluated), "total": total},
        "missing_scores": missing_scores,
        "missing_responses": missing_responses,
        "metric_definition": (
            "official_pass = before_score AND after_score "
            "(MEME trivial-pass filter)"
        ),
    }
    if result["complete"]:
        buckets: dict[str, list[bool]] = defaultdict(list)
        outcomes = Counter()
        for episode, probe, before, after in evaluated:
            official = before and after
            buckets["overall"].append(official)
            buckets[f"task:{probe['task_type']}"].append(official)
            buckets[f"domain:{episode['domain']}"].append(official)
            buckets[f"hop:{probe['hop']}"].append(official)
            buckets[f"task_hop:{probe['task_type']}:{probe['hop']}"].append(official)
            outcomes[
                {
                    (True, True): "real_pass",
                    (False, True): "trivial_pass_filtered",
                    (True, False): "knew_but_failed",
                    (False, False): "never_knew",
                }[(before, after)]
            ] += 1
        result["metrics"] = {
            key: {
                "passed": sum(values),
                "total": len(values),
                "accuracy": sum(values) / len(values),
                "accuracy_percent": 100 * sum(values) / len(values),
            }
            for key, values in sorted(buckets.items())
        }
        result["outcomes"] = dict(sorted(outcomes.items()))
    atomic_write_json(paths.summary, result)
    lines = [
        f"# MEME manual score summary — {paths.view}",
        "",
        f"Coverage: {len(evaluated)}/{total}",
        "",
    ]
    if not result["complete"]:
        lines.append(
            "Incomplete: generate every response and fill both score columns "
            "in manual_scores.csv."
        )
    else:
        lines.extend([
            "| Group | Passed | Total | Accuracy |",
            "|---|---:|---:|---:|",
        ])
        for key, metric in result["metrics"].items():
            lines.append(
                f"| {key} | {metric['passed']} | {metric['total']} | "
                f"{metric['accuracy_percent']:.1f}% |"
            )
        lines.extend(["", "## Trivial-pass outcomes", ""])
        for name, count in result["outcomes"].items():
            lines.append(f"- {name}: {count}")
    atomic_write_text(paths.summary_markdown, "\n".join(lines) + "\n")
    return result
