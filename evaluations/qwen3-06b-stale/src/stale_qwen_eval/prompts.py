from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .constants import DIMENSION_KEYS


EXPLICIT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Review the following conversation history "
    "with the user, then accurately answer the question."
)

IMPLICIT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Review the following conversation history "
    "with the user, then respond to the user's latest query directly."
)


def format_haystack(
    sessions: Iterable[Mapping[str, Any]], *, preserve_input_order: bool = False
) -> str:
    """Serialize sessions with their original numbers and optional presentation order."""
    chunks: list[str] = []
    ordered_sessions = list(sessions)
    if not preserve_input_order:
        ordered_sessions.sort(key=lambda value: int(value["original_index"]))
    for session in ordered_sessions:
        index = int(session["original_index"])
        timestamp = session.get("timestamp")
        time_text = f" [Time: {timestamp}]" if timestamp else ""
        lines = [f"\n=== Session {index + 1}{time_text} ==="]
        for turn in session["messages"]:
            role = "User" if turn["role"] == "user" else "Assistant"
            lines.append(f"{role}: {turn['content']}")
        chunks.append("\n".join(lines) + "\n")
    return "".join(chunks)


def build_messages(history_text: str, query_text: str, dimension_key: str) -> list[dict[str, str]]:
    if dimension_key not in DIMENSION_KEYS:
        raise ValueError(f"Unknown STALE dimension: {dimension_key}")
    if dimension_key == "dim3_query":
        system_prompt = IMPLICIT_SYSTEM_PROMPT
        user_prompt = f"[Conversation History]\n{history_text}\n\n[Latest Query]\n{query_text}"
    else:
        system_prompt = EXPLICIT_SYSTEM_PROMPT
        user_prompt = f"[Conversation History]\n{history_text}\n\n[Question]\n{query_text}"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def chat_template_token_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    token_ids = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    return list(token_ids)


def count_prompt_tokens(
    tokenizer: Any,
    sessions: Iterable[Mapping[str, Any]],
    query_text: str,
    dimension_key: str,
    *,
    preserve_input_order: bool = False,
) -> int:
    history = format_haystack(sessions, preserve_input_order=preserve_input_order)
    messages = build_messages(history, query_text, dimension_key)
    return len(chat_template_token_ids(tokenizer, messages))

