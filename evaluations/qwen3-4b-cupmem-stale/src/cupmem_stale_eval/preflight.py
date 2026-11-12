from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import zstandard
from openai import OpenAI

from .constants import DEFAULT_VLLM_BASE_URL, JUDGE_MODEL_ID, TARGET_MODEL_ID
from .cupmem_bridge import load_cupmem_symbols
from .io_utils import atomic_write_bytes, load_json
from .runner import _resolve_embedding_path


def run_preflight(
    artifact_root: Path,
    *,
    cache_dir: Path,
    vllm_base_url: str = DEFAULT_VLLM_BASE_URL,
    check_judge: bool = True,
) -> dict[str, Any]:
    preparation = load_json(artifact_root / "preparation_manifest.json")
    embedding_path = _resolve_embedding_path(cache_dir, preparation)
    symbols = load_cupmem_symbols()
    from cup_mem.retrieval.embedding import build_retriever

    embedding = build_retriever(str(embedding_path), device="cpu")
    candidates = ["bike commute", "baking bread"]
    first = embedding.rank(
        query_text="cycling to work", candidates=candidates, text_getter=str, top_k=2
    )
    second = embedding.rank(
        query_text="cycling to work", candidates=candidates, text_getter=str, top_k=2
    )
    if [item["item"] for item in first] != [item["item"] for item in second]:
        raise RuntimeError("MiniLM ranking is not deterministic")

    local = OpenAI(api_key="EMPTY", base_url=vllm_base_url)
    served_models = {item.id for item in local.models.list().data}
    if TARGET_MODEL_ID not in served_models:
        raise RuntimeError(
            f"vLLM does not advertise {TARGET_MODEL_ID}; found {sorted(served_models)}"
        )
    target_response = local.chat.completions.create(
        model=TARGET_MODEL_ID,
        messages=[
            {"role": "system", "content": "Return one JSON object only."},
            {"role": "user", "content": "Return {\"ok\": true}."},
        ],
        temperature=0,
        max_tokens=64,
        seed=42,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    target_text = target_response.choices[0].message.content or ""
    if "<think>" in target_text.lower() or "</think>" in target_text.lower():
        raise RuntimeError("Qwen thinking marker detected")
    parsed_target = json.loads(target_text)
    if parsed_target.get("ok") is not True:
        raise RuntimeError("Qwen JSON preflight failed")

    cup_client = symbols["LLMClient"](
        model=TARGET_MODEL_ID,
        api_key="EMPTY",
        base_url=vllm_base_url,
        request_overrides={
            "temperature": 0,
            "max_tokens": 4096,
            "seed": 42,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        },
    )
    from cup_mem.prompt_lib.templates import SESSION_CHUNKER_PROMPT

    cup_payload = cup_client.call_json(
        SESSION_CHUNKER_PROMPT,
        json.dumps(
            {
                "session_id": "s_00000",
                "bucket_track_schema": "personal_details.location",
                "user_turns": [{"turn_id": "t_0", "content": "I now live in Los Angeles."}],
            }
        ),
        max_retries=1,
    )
    if not isinstance(cup_payload.get("chunks"), list):
        raise RuntimeError("Qwen CUP-Mem JSON preflight did not return a chunks list")

    result: dict[str, Any] = {
        "minilm": {"path": str(embedding_path), "ranking": [item["item"] for item in first]},
        "qwen": {
            "model": TARGET_MODEL_ID,
            "served_models": sorted(served_models),
            "response": parsed_target,
            "cup_mem_json_valid": True,
        },
    }
    if check_judge:
        api_key = os.environ.get("JUDGE_OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("JUDGE_OPENAI_API_KEY is required for judge preflight")
        judge = OpenAI(
            api_key=api_key,
            base_url=os.environ.get("JUDGE_OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        judge_response = judge.chat.completions.create(
            model=JUDGE_MODEL_ID,
            messages=[
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": "Return {\"ok\": true}."},
            ],
            temperature=0,
            reasoning_effort="none",
            response_format={"type": "json_object"},
        )
        judge_text = judge_response.choices[0].message.content or ""
        parsed_judge = json.loads(judge_text)
        if parsed_judge.get("ok") is not True:
            raise RuntimeError("GPT-6 Luna structured-output preflight failed")
        result["judge"] = {"model": JUDGE_MODEL_ID, "response": parsed_judge}
    return result


def archive_log(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(
        destination,
        zstandard.ZstdCompressor(level=8).compress(source.read_bytes()),
    )
