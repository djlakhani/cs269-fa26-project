from __future__ import annotations

import platform
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from .constants import MODEL_ID, MODEL_REVISION


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class TransformersQwenBackend:
    """Single-example Transformers backend configured for a Colab T4."""

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
        cache_dir: Path | None = None,
        context_window: int,
        max_new_tokens: int,
        seed: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Inference dependencies are missing. Install requirements.txt first."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this evaluation; no CUDA device was detected")

        self.torch = torch
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.input_token_limit = context_window - max_new_tokens
        self.model_id = model_id
        self.model_revision = model_revision
        cache_text = str(cache_dir) if cache_dir else None

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=model_revision,
            cache_dir=cache_text,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=model_revision,
            cache_dir=cache_text,
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        ).to("cuda")
        self.model.eval()
        self.model.generation_config.do_sample = False
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

        configured_window = int(getattr(self.model.config, "max_position_embeddings", 0) or 0)
        if configured_window and context_window > configured_window:
            raise ValueError(
                f"Requested context window {context_window} exceeds model config "
                f"max_position_embeddings={configured_window}"
            )

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        torch = self.torch
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
            return_dict=True,
        )
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens > self.input_token_limit:
            raise ValueError(
                f"Prepared prompt has {input_tokens} tokens, exceeding hard input limit "
                f"{self.input_token_limit}"
            )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                pad_token_id=pad_token_id,
            )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started

        generated_ids = output[0, input_tokens:]
        output_tokens = int(generated_ids.shape[-1])
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        lowered = response.lower()
        return {
            "response": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_seconds": elapsed,
            "hit_token_limit": output_tokens >= self.max_new_tokens,
            "thinking_marker_detected": "<think>" in lowered or "</think>" in lowered,
        }

    def runtime_metadata(self) -> dict[str, Any]:
        torch = self.torch
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                "torch": _package_version("torch"),
                "transformers": _package_version("transformers"),
                "datasets": _package_version("datasets"),
            },
            "cuda": {
                "available": torch.cuda.is_available(),
                "torch_cuda_version": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(0),
                "device_capability": list(torch.cuda.get_device_capability(0)),
            },
            "dtype": "float16",
            "attention_implementation": "sdpa",
            "batch_size": 1,
        }

