from __future__ import annotations

from typing import Any


class FakeTokenizer:
    """Character-counting tokenizer with a chat-template-shaped interface."""

    def __init__(self) -> None:
        self.enable_thinking_values: list[bool] = []

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
        **_: Any,
    ) -> list[int]:
        assert tokenize is True
        assert add_generation_prompt is True
        self.enable_thinking_values.append(enable_thinking)
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        rendered += "<assistant>"
        return list(range(len(rendered)))


def make_record(uid: str, conflict_type: str, *, session_count: int = 8) -> dict[str, Any]:
    sessions = []
    for index in range(session_count):
        sessions.append(
            [
                {"role": "user", "content": f"user-{index}-" + ("x" * 40)},
                {"role": "assistant", "content": f"assistant-{index}-" + ("y" * 30)},
            ]
        )
    return {
        "uid": uid,
        "type": conflict_type,
        "M_old": "The user lives in Seattle.",
        "M_new": "The user is unpacking in an Austin apartment.",
        "explanation": "The user's current city is now Austin.",
        "probing_queries": {
            "dim1_query": "Does the user still live in Seattle?",
            "dim2_query": "Recommend a Seattle neighborhood for the user.",
            "dim3_query": "Where should I shop for groceries this week?",
        },
        "relevant_session_index": [2, 5],
        "timestamps": [f"2025-01-{index + 1:02d}T12:00:00" for index in range(session_count)],
        "haystack_session": sessions,
    }

