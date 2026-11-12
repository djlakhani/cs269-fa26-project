from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .constants import MANIFEST_SCHEMA_VERSION, MODEL_ID, MODEL_REVISION, PACKAGE_VERSION
from .io_utils import atomic_write_json, load_json, sha256_file, utc_now
from .preparation import _write_run_metadata
from .prompts import chat_template_token_ids


MEME_SCHEMA_VERSION = 1
MEME_RESPONSE_SCHEMA_VERSION = 1
MEME_DATASET_ID = "meme-benchmark/MEME"
MEME_DATASET_REVISION = "e14c4e685772014deb96f91dc7eddfbeb21d90d8"
MEME_NOFILLER_FILE = "meme_nofiller.json"
MEME_FILLER_FILE = "meme_filler32k.json"
MEME_PROMPT_VERSION = "meme-author-unified-answer-v1"
MEME_TASKS = ("Cas", "Abs")
MEME_TASK_NAMES = {"Cas": "Cascade", "Abs": "Absence"}
MEME_VIEWS = {
    "nofiller": "Official MEME nofiller episodes (evidence sessions only)",
    "filler_29k": (
        "Official filler32k episodes with only whole filler sessions removed "
        "deterministically to fit a 29,000-token Qwen input target"
    ),
}
MEME_CONTEXT_WINDOW = 32_768
MEME_MAX_NEW_TOKENS = 500
MEME_FILLER_INPUT_TARGET = 29_000
MEME_SEED = 42

# Documented by the benchmark authors for dataset release v1.0. Preserve the
# released gold labels and flag affected rows rather than silently changing them.
KNOWN_ABSENCE_LABEL_ISSUES = {
    "pl_009", "pl_041", "pl_044", "pl_047", "sw_013",
    "sw_016", "sw_020", "sw_026", "sw_040", "sw_048",
}

UNIFIED_ANSWER_PROMPT = """Answer the user's question based ONLY on the context provided below.
If the information is not in the context, say you don't have that information.
Answer with ONLY the value. Do not explain or add context.

Context:
{context}

Question: {question}"""


def task_base(question: Mapping[str, Any]) -> str:
    return str(question.get("task_type", "")).split(" (", 1)[0]


def format_sessions(sessions: Sequence[Mapping[str, Any]], up_to: int) -> str:
    """Match the authors' in_context_baseline._flatten_sessions serialization."""
    parts: list[str] = []
    for index, session in enumerate(sessions[:up_to]):
        timestamp = session.get("timestamp", f"session_{index}")
        parts.append(f"[Session: {timestamp}]")
        for turn in session.get("conversation", []):
            role = "User" if turn.get("role") == "user" else "Assistant"
            parts.append(f"{role}: {turn.get('content', '')}")
    return "\n".join(parts)


def build_messages(transcript: str, question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": UNIFIED_ANSWER_PROMPT.format(
        context=transcript, question=question
    )}]


def count_tokens(
    tokenizer: Any,
    sessions: Sequence[Mapping[str, Any]],
    up_to: int,
    question: str,
) -> int:
    return len(chat_template_token_ids(
        tokenizer, build_messages(format_sessions(sessions, up_to), question)
    ))


def _question_key(question: Mapping[str, Any]) -> tuple[str, tuple[str, ...], int | None]:
    entities = question.get("entity")
    if not isinstance(entities, list):
        entities = list((question.get("entity_values") or {}).keys())
    return (
        task_base(question),
        tuple(str(value) for value in entities),
        int(question["hop"]) if question.get("hop") is not None else None,
    )


def build_probes(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    before_questions = [
        (index, question)
        for index, question in enumerate(episode["before_questions"]["questions"])
        if task_base(question) in MEME_TASKS
    ]
    after_questions = [
        (index, question)
        for index, question in enumerate(episode["after_questions"]["questions"])
        if task_base(question) in MEME_TASKS
    ]
    before_by_key: dict[tuple[str, tuple[str, ...], int | None], str] = {}
    probes: list[dict[str, Any]] = []
    for source_index, question in before_questions:
        key = _question_key(question)
        if key in before_by_key:
            raise ValueError(
                f"Episode {episode['episode_id']} has duplicate before-question key {key}"
            )
        probe_id = f"before_{source_index:03d}_{key[0]}"
        before_by_key[key] = probe_id
        probes.append({
            "probe_id": probe_id,
            "phase": "before",
            "source_question_index": source_index,
            "task_type": key[0],
            "task_name": MEME_TASK_NAMES[key[0]],
            "hop": question.get("hop"),
            "entity": list(question.get("entity", [])),
            "entity_values": dict(question["entity_values"]),
            "question": str(question["question"]),
            "gold_answer": str(question["expected_answer"]),
            "prerequisite_probe_id": None,
            "known_label_issue": False,
        })
    for source_index, question in after_questions:
        key = _question_key(question)
        prerequisite = before_by_key.get(key)
        if prerequisite is None:
            raise ValueError(
                f"Episode {episode['episode_id']} after-question has no matching "
                f"before-question: {key}"
            )
        probes.append({
            "probe_id": f"after_{source_index:03d}_{key[0]}",
            "phase": "after",
            "source_question_index": source_index,
            "task_type": key[0],
            "task_name": MEME_TASK_NAMES[key[0]],
            "hop": question.get("hop"),
            "entity": list(question.get("entity", [])),
            "entity_values": dict(question["entity_values"]),
            "question": str(question["question"]),
            "gold_answer": str(question["gold_answer"]),
            "prerequisite_probe_id": prerequisite,
            "known_label_issue": (
                key[0] == "Abs"
                and str(episode["episode_id"]) in KNOWN_ABSENCE_LABEL_ISSUES
            ),
        })
    return probes


def _indexed_sessions(episode: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"original_index": index, **dict(session)}
        for index, session in enumerate(episode["sessions"])
    ]


def _phase_position(
    episode: Mapping[str, Any],
    retained_sessions: Sequence[Mapping[str, Any]],
    phase: str,
) -> int:
    original_position = int(
        episode[f"{phase}_questions"]["position_after_session"]
    ) + 1
    return sum(
        int(session["original_index"]) < original_position
        for session in retained_sessions
    )


def _view_token_counts(
    tokenizer: Any,
    episode: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
    sessions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, int], dict[str, int]]:
    positions = {
        phase: _phase_position(episode, sessions, phase)
        for phase in ("before", "after")
    }
    counts = {
        str(probe["probe_id"]): count_tokens(
            tokenizer,
            sessions,
            positions[str(probe["phase"])],
            str(probe["question"]),
        )
        for probe in probes
    }
    return positions, counts


def _build_nofiller_view(
    episode: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
    tokenizer: Any,
) -> dict[str, Any]:
    sessions = _indexed_sessions(episode)
    positions, counts = _view_token_counts(tokenizer, episode, probes, sessions)
    limit = MEME_CONTEXT_WINDOW - MEME_MAX_NEW_TOKENS
    if max(counts.values()) > limit:
        raise ValueError(
            f"Episode {episode['episode_id']} nofiller input exceeds {limit} tokens"
        )
    return {
        "source_config": "nofiller",
        "source_session_count": len(sessions),
        "kept_session_indices": list(range(len(sessions))),
        "dropped_session_indices": [],
        "distractor_removal_order": [],
        "sessions": sessions,
        "phase_positions": positions,
        "input_token_counts": counts,
        "maximum_input_tokens": max(counts.values()),
    }


def _build_filler_view(
    episode: Mapping[str, Any],
    probes: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    seed: int,
    input_target: int,
) -> dict[str, Any]:
    all_sessions = _indexed_sessions(episode)
    evidence = {int(value) for value in episode["evidence_session_indices"]}
    all_indices = set(range(len(all_sessions)))
    if not evidence or not evidence.issubset(all_indices):
        raise ValueError(
            f"Episode {episode['episode_id']} has invalid evidence_session_indices"
        )
    removal_order = sorted(all_indices - evidence)
    random.Random(f"{seed}:{episode['episode_id']}").shuffle(removal_order)
    cache: dict[int, tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]] = {}

    def retained_after(
        removal_count: int,
    ) -> tuple[list[dict[str, Any]], dict[str, int], dict[str, int]]:
        cached = cache.get(removal_count)
        if cached is not None:
            return cached
        dropped = set(removal_order[:removal_count])
        retained = [
            session for session in all_sessions
            if int(session["original_index"]) not in dropped
        ]
        positions, counts = _view_token_counts(
            tokenizer, episode, probes, retained
        )
        result = retained, positions, counts
        cache[removal_count] = result
        return result

    evidence_only = [
        session for session in all_sessions
        if int(session["original_index"]) in evidence
    ]
    _, evidence_counts = _view_token_counts(
        tokenizer, episode, probes, evidence_only
    )
    if max(evidence_counts.values()) > input_target:
        raise ValueError(
            f"Episode {episode['episode_id']} evidence alone requires "
            f"{max(evidence_counts.values())} tokens, above target {input_target}"
        )

    low, high = 0, len(removal_order)
    while low < high:
        middle = (low + high) // 2
        _, _, counts = retained_after(middle)
        if max(counts.values()) <= input_target:
            high = middle
        else:
            low = middle + 1
    removed_count = low
    sessions, positions, counts = retained_after(removed_count)
    dropped = set(removal_order[:removed_count])

    # Binary search finds the shortest member of the seeded prefix-removal
    # family that fits. A large session early in that prefix can cause an
    # unnecessary undershoot, so deterministically try to pack each removed
    # whole session back. This never slices or reorders a session.
    for candidate_index in reversed(removal_order[:removed_count]):
        trial_dropped = dropped - {candidate_index}
        trial_sessions = [
            session for session in all_sessions
            if int(session["original_index"]) not in trial_dropped
        ]
        trial_positions, trial_counts = _view_token_counts(
            tokenizer, episode, probes, trial_sessions
        )
        if max(trial_counts.values()) <= input_target:
            dropped = trial_dropped
            sessions = trial_sessions
            positions = trial_positions
            counts = trial_counts

    kept = [int(session["original_index"]) for session in sessions]
    if not evidence.issubset(kept):
        raise AssertionError("Internal error: MEME evidence session was removed")
    if max(counts.values()) > input_target:
        raise AssertionError("Internal error: MEME filler view remains over target")
    return {
        "source_config": "filler32k",
        "source_session_count": len(all_sessions),
        "kept_session_indices": kept,
        "dropped_session_indices": sorted(dropped),
        "distractor_removal_order": [
            index for index in removal_order if index in dropped
        ],
        "sessions": sessions,
        "phase_positions": positions,
        "input_token_counts": counts,
        "maximum_input_tokens": max(counts.values()),
    }


def _episode_semantics(episode: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "episode_id": episode["episode_id"],
        "domain": episode["domain"],
        "root": episode["root"],
        "root_change": episode["root_change"],
        "tasks": episode["tasks"],
        "before_questions": episode["before_questions"]["questions"],
        "after_questions": episode["after_questions"]["questions"],
    }


def build_meme_artifact(
    nofiller_episodes: Sequence[Mapping[str, Any]],
    filler_episodes: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    seed: int = MEME_SEED,
    filler_input_target: int = MEME_FILLER_INPUT_TARGET,
) -> dict[str, Any]:
    nofiller_by_id = {str(item["episode_id"]): item for item in nofiller_episodes}
    filler_by_id = {str(item["episode_id"]): item for item in filler_episodes}
    if len(nofiller_by_id) != 100 or set(nofiller_by_id) != set(filler_by_id):
        raise ValueError("Expected the same 100 unique episodes in both MEME variants")
    if not 0 < filler_input_target <= MEME_CONTEXT_WINDOW - MEME_MAX_NEW_TOKENS:
        raise ValueError("Invalid MEME filler input target")
    try:
        from tqdm.auto import tqdm
    except ImportError:  # pragma: no cover
        tqdm = lambda values, **_: values

    original_model_max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(original_model_max_length, int):
        tokenizer.model_max_length = max(original_model_max_length, 1_000_000_000)
    prepared: list[dict[str, Any]] = []
    try:
        for episode_id in tqdm(
            sorted(nofiller_by_id),
            desc="Preparing MEME episodes",
            unit="episode",
        ):
            nofiller = nofiller_by_id[episode_id]
            filler = filler_by_id[episode_id]
            if _episode_semantics(nofiller) != _episode_semantics(filler):
                raise ValueError(
                    f"MEME variants disagree on episode semantics for {episode_id}"
                )
            probes = build_probes(nofiller)
            prepared.append({
                "episode_id": episode_id,
                "domain": str(nofiller["domain"]),
                "root": str(nofiller["root"]),
                "root_change": dict(nofiller["root_change"]),
                "has_2hop": bool(nofiller["has_2hop"]),
                "probes": probes,
                "views": {
                    "nofiller": _build_nofiller_view(
                        nofiller, probes, tokenizer
                    ),
                    "filler_29k": _build_filler_view(
                        filler, probes, tokenizer,
                        seed=seed, input_target=filler_input_target,
                    ),
                },
            })
    finally:
        if isinstance(original_model_max_length, int):
            tokenizer.model_max_length = original_model_max_length

    return {
        "schema_version": MEME_SCHEMA_VERSION,
        "benchmark": "MEME",
        "created_at": utc_now(),
        "dataset": {
            "id": MEME_DATASET_ID,
            "revision": MEME_DATASET_REVISION,
            "files": {
                "nofiller": MEME_NOFILLER_FILE,
                "filler32k": MEME_FILLER_FILE,
            },
        },
        "model_tokenizer": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "prompt_version": MEME_PROMPT_VERSION,
        "context_views": dict(MEME_VIEWS),
        "selection": {
            "task_types": list(MEME_TASKS),
            "episode_count": len(prepared),
            "after_question_count": sum(
                probe["phase"] == "after"
                for episode in prepared for probe in episode["probes"]
            ),
            "before_prerequisite_count": sum(
                probe["phase"] == "before"
                for episode in prepared for probe in episode["probes"]
            ),
            "seed": seed,
        },
        "token_budget": {
            "context_window": MEME_CONTEXT_WINDOW,
            "max_new_tokens": MEME_MAX_NEW_TOKENS,
            "hard_input_limit": MEME_CONTEXT_WINDOW - MEME_MAX_NEW_TOKENS,
            "filler_input_target": filler_input_target,
        },
        "known_dataset_issues": {
            "absence_v1_episode_ids": sorted(KNOWN_ABSENCE_LABEL_ISSUES),
            "policy": "preserve released gold labels and flag affected rows",
        },
        "data": prepared,
    }


def prepare_meme(
    artifact_root: Path,
    *,
    cache_dir: Path | None = None,
    force: bool = False,
    seed: int = MEME_SEED,
    filler_input_target: int = MEME_FILLER_INPUT_TARGET,
    run_id: str | None = None,
    description: str | None = None,
) -> Path:
    scenario_path = artifact_root / "episodes.json"
    manifest_path = artifact_root / "preparation_manifest.json"
    if scenario_path.exists() and not force:
        existing = load_json(scenario_path)
        requested = {"seed": seed, "filler_input_target": filler_input_target}
        recorded = {
            "seed": existing.get("selection", {}).get("seed"),
            "filler_input_target": existing.get("token_budget", {}).get(
                "filler_input_target"
            ),
        }
        if existing.get("schema_version") != MEME_SCHEMA_VERSION or requested != recorded:
            raise ValueError(
                f"Existing MEME artifact is incompatible ({recorded}); use a new run ID"
            )
        _write_run_metadata(artifact_root, run_id=run_id, description=description)
        return scenario_path
    if force and any((artifact_root / "views").glob("*/checkpoints/*.json")):
        raise ValueError(
            "Refusing to replace MEME episodes alongside existing responses"
        )
    try:
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Run uv sync --locked before preparation") from error

    cache_text = str(cache_dir) if cache_dir else None
    paths = {
        name: hf_hub_download(
            MEME_DATASET_ID,
            filename,
            repo_type="dataset",
            revision=MEME_DATASET_REVISION,
            cache_dir=cache_text,
        )
        for name, filename in {
            "nofiller": MEME_NOFILLER_FILE,
            "filler32k": MEME_FILLER_FILE,
        }.items()
    }
    with open(paths["nofiller"], encoding="utf-8") as handle:
        nofiller = json.load(handle)
    with open(paths["filler32k"], encoding="utf-8") as handle:
        filler = json.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=cache_text
    )
    artifact = build_meme_artifact(
        nofiller, filler, tokenizer,
        seed=seed, filler_input_target=filler_input_target,
    )
    atomic_write_json(scenario_path, artifact)
    atomic_write_json(manifest_path, {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "package_version": PACKAGE_VERSION,
        "preparation": {
            "created_at": artifact["created_at"],
            "dataset": artifact["dataset"],
            "model_tokenizer": artifact["model_tokenizer"],
            "prompt_version": MEME_PROMPT_VERSION,
            "unified_answer_prompt": UNIFIED_ANSWER_PROMPT,
            "context_views": artifact["context_views"],
            "selection": artifact["selection"],
            "token_budget": artifact["token_budget"],
            "scenarios_sha256": sha256_file(scenario_path),
        },
    })
    _write_run_metadata(
        artifact_root, run_id=run_id, description=description
    )
    return scenario_path
