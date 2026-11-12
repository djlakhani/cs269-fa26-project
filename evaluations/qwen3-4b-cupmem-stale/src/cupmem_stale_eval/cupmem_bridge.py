from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def stale_repository_root() -> Path:
    return project_root() / "STALE"


def load_cupmem_symbols() -> dict[str, Any]:
    stale_root = stale_repository_root()
    if not (stale_root / "cup_mem").exists():
        raise FileNotFoundError(f"CUP-Mem source not found at {stale_root / 'cup_mem'}")
    if str(stale_root) not in sys.path:
        sys.path.insert(0, str(stale_root))
    from cup_mem import CupMemEngine, LLMClient, PipelineThresholds, TraceConfig
    from cup_mem.llm_layer.client import summarize_call_records

    return {
        "CupMemEngine": CupMemEngine,
        "LLMClient": LLMClient,
        "PipelineThresholds": PipelineThresholds,
        "TraceConfig": TraceConfig,
        "summarize_call_records": summarize_call_records,
    }


def canonical_judge_system_prompt() -> str:
    evaluation_root = stale_repository_root() / "STALE" / "Evaluation"
    if str(evaluation_root) not in sys.path:
        sys.path.insert(0, str(evaluation_root))
    from judge_prompts import SYSTEM_PROMPT_ALL_IN_ONE_JUDGE

    return SYSTEM_PROMPT_ALL_IN_ONE_JUDGE
