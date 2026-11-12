from __future__ import annotations

import platform
import time
from importlib import metadata
from pathlib import Path
from typing import Any

from .constants import ATTENTION_BACKENDS, DEFAULT_ATTENTION_BACKEND, MODEL_ID, MODEL_REVISION


def _package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


class TransformersQwenBackend:
    """Single-example Transformers backend with an explicitly enforced SDPA kernel."""

    def __init__(
        self,
        *,
        model_id: str = MODEL_ID,
        model_revision: str = MODEL_REVISION,
        cache_dir: Path | None = None,
        context_window: int,
        max_new_tokens: int,
        seed: int,
        attention_backend: str = DEFAULT_ATTENTION_BACKEND,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Inference dependencies are missing. Run `uv sync --locked` first."
            ) from error

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this evaluation; no CUDA device was detected")

        if attention_backend not in ATTENTION_BACKENDS:
            raise ValueError(
                f"Unknown attention backend {attention_backend!r}; choose from {ATTENTION_BACKENDS}"
            )

        self.torch = torch
        self.context_window = context_window
        self.max_new_tokens = max_new_tokens
        self.input_token_limit = context_window - max_new_tokens
        self.model_id = model_id
        self.model_revision = model_revision
        self.attention_backend = attention_backend
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
            dtype=torch.float16,
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
        if attention_backend == "flash_sdpa":
            capability = torch.cuda.get_device_capability(0)
            if capability[0] < 8:
                raise RuntimeError(
                    "Flash SDPA requires an Ampere-or-newer NVIDIA GPU; "
                    f"detected compute capability {capability}."
                )
            self._verify_flash_sdpa()

    def _attention_context(self) -> Any:
        if self.attention_backend == "flash_sdpa":
            from torch.nn.attention import SDPBackend, sdpa_kernel

            # Fail closed: never allow a long prompt to fall back to quadratic math attention.
            return sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        from contextlib import nullcontext

        return nullcontext()

    def _verify_flash_sdpa(self) -> None:
        torch = self.torch
        query = torch.zeros((1, 16, 16, 128), device="cuda", dtype=torch.float16)
        key = torch.zeros((1, 8, 16, 128), device="cuda", dtype=torch.float16)
        value = torch.zeros_like(key)
        try:
            with self._attention_context(), torch.inference_mode():
                torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    is_causal=True,
                    enable_gqa=True,
                )
        except Exception as error:
            raise RuntimeError(
                "The requested Flash SDPA backend failed its L4/GQA preflight. "
                "Check the NVIDIA driver and locked CUDA-enabled PyTorch build."
            ) from error
        finally:
            del query, key, value
            torch.cuda.empty_cache()

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
        with torch.inference_mode(), self._attention_context():
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
            "enforced_sdpa_backend": self.attention_backend,
            "batch_size": 1,
        }

