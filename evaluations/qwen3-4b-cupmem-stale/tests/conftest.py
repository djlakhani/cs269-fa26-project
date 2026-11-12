from __future__ import annotations

from typing import Any


def make_record(uid: str, conflict_type: str) -> dict[str, Any]:
    sessions = [
        [
            {"role": "user", "content": f"user state {uid} session {index}"},
            {"role": "assistant", "content": "acknowledged"},
        ]
        for index in range(50)
    ]
    return {
        "uid": uid,
        "type": conflict_type,
        "M_old": f"old state {uid}",
        "M_new": f"new state {uid}",
        "explanation": f"transition for {uid}",
        "haystack_session": sessions,
        "timestamps": [f"2025-01-{(index % 28) + 1:02d}T00:00:{index:02d}Z" for index in range(50)],
        "relevant_session_index": [7, 41],
        "probing_queries": {
            "dim1_query": f"SR question {uid}",
            "dim2_query": f"PR question {uid}",
            "dim3_query": f"IPA question {uid}",
        },
    }
