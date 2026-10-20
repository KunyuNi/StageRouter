"""Injectable LLM boundary for RAG operators."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol


@dataclass(frozen=True)
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class LLMAdapter(Protocol):
    def call(self, model: str, prompt: str, **options: Any) -> LLMResult:
        ...


def _request_options(model: str, options: dict[str, Any]) -> dict[str, Any]:
    values = dict(options)
    request_options = {
        "temperature": values.pop("temperature", 0.0),
        "max_tokens": values.pop("max_tokens", 4096),
        "top_p": values.pop("top_p", 0.95),
    }
    if model == "qwen3.5-flash":
        request_options["extra_body"] = {"enable_thinking": False}
    request_options.update(values)
    return request_options


class QwenAdapter:
    """Qwen client using the DashScope OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        sdk_max_retries: int = 1,
    ) -> None:
        self._load_runtime_env()
        self.api_key = (
            api_key
            or os.getenv("RAG_LLM_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        self.base_url = (
            base_url
            or os.getenv("RAG_LLM_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        ).rstrip("/")
        self.sdk_max_retries = int(sdk_max_retries)
        if self.sdk_max_retries not in {0, 1}:
            raise ValueError("sdk_max_retries must be 0 or 1")

    @staticmethod
    def _load_runtime_env() -> None:
        env_path = Path(__file__).resolve().parents[3] / ".env"
        if not env_path.is_file():
            return
        try:
            from dotenv import load_dotenv
        except ImportError:
            return
        load_dotenv(env_path, override=False)

    def call(self, model: str, prompt: str, **options: Any) -> LLMResult:
        if not self.api_key:
            raise RuntimeError(
                "RAG LLM credentials are missing; configure RAG_LLM_API_KEY, "
                "LLM_API_KEY, DASHSCOPE_API_KEY, or OPENAI_API_KEY"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is required for RAG LLM calls") from exc

        request_options = _request_options(model, options)

        started = time.perf_counter()
        client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            max_retries=self.sdk_max_retries,
            timeout=120.0,
        )
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **request_options,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        usage = response.usage
        return LLMResult(
            text=response.choices[0].message.content or "",
            input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            latency_ms=latency_ms,
        )


_adapter: Optional[LLMAdapter] = None


def get_adapter() -> LLMAdapter:
    global _adapter
    if _adapter is None:
        _adapter = QwenAdapter()
    return _adapter


def set_adapter(adapter: LLMAdapter) -> None:
    global _adapter
    _adapter = adapter


def record_llm_result(payload, stage: str, model: str, result: LLMResult) -> None:
    """Record one adapter result on a payload's cost tracker."""
    payload.cost_tracker.record_llm_call(
        stage=stage,
        model=model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
    )
