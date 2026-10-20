"""
Qwen API 适配器 + 缺失的 LLM 调用函数

关键修复: 原有代码从 llm.models 导入 llm_chain / async_llm_chain_call / get_llm_chain,
但这些函数在仓库中不存在。本模块提供完整的 Qwen 适配实现。

支持两种调用模式:
  1. DashScope API (models centralized in _shared/model_config.py)
  2. OpenAI 兼容模式 (备选)
"""

from __future__ import annotations

import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from .model_config import DEFAULT_LLM_MODEL

# ============================================================
# Thread-local cost tracking context
# ============================================================
_cost_ctx = threading.local()

def start_cost_context(tracker, stage_name: str):
    """设置 cost 追踪上下文。此后同一线程内所有 adapter.call() 自动记录成本。"""
    _cost_ctx.tracker = tracker
    _cost_ctx.stage = stage_name

def end_cost_context():
    """清除 cost 追踪上下文。"""
    _cost_ctx.tracker = None
    _cost_ctx.stage = None

def current_cost_context():
    """返回当前线程内的 cost tracker 和 stage name。"""
    return getattr(_cost_ctx, "tracker", None), getattr(_cost_ctx, "stage", None)

def run_with_cost_context(tracker, stage_name: str | None, func, *args, **kwargs):
    """在 worker 线程中临时恢复主线程传入的 cost context。"""
    if tracker is None or stage_name is None:
        return func(*args, **kwargs)
    start_cost_context(tracker, stage_name)
    try:
        return func(*args, **kwargs)
    finally:
        end_cost_context()

def _auto_record_cost(model: str, input_tokens: int, output_tokens: int, latency_ms: float):
    """若当前有追踪上下文, 自动记录成本。"""
    tracker, stage = current_cost_context()
    if tracker is not None and stage is not None:
        tracker.record_llm_call(
            stage=stage, model=model,
            input_tokens=input_tokens, output_tokens=output_tokens,
            latency_ms=latency_ms,
        )


# ============================================================
# Decorator: auto-track all LLM costs in a tier function
# ============================================================
import functools

def track_cost(stage_name: str):
    """装饰器: 自动追踪函数内所有 adapter.call() 的成本。

    Usage:
        @track_cost("keyword_extraction")
        def keyword_extraction_tier1(payload: StatePayload, ...) -> StatePayload:
            ...
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(payload, *args, **kwargs):
            start_cost_context(payload.cost_tracker, stage_name)
            try:
                return func(payload, *args, **kwargs)
            finally:
                end_cost_context()
        return wrapper
    return decorator

# ============================================================
# QwenAdapter — 统一的 Qwen API 调用接口
# ============================================================

class QwenAdapter:
    """
    阿里云 DashScope Qwen API 适配器。
    支持 _shared.model_config 中定义的模型。

    备选方案: 若 DASHSCOPE_API_KEY 不存在，回退到 OpenAI 兼容 endpoint。
    """

    def __init__(self, api_key: Optional[str] = None, use_openai_compat: bool = False):
        self.api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        self.use_openai_compat = use_openai_compat or not self.api_key

        if self.use_openai_compat:
            # OpenAI 兼容模式 (本地 vLLM 或代理)
            self.openai_base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
            self.openai_api_key = os.environ.get("OPENAI_API_KEY", "not-needed")
        else:
            import dashscope
            self.dashscope = dashscope

    def resolve_model(self, model_name: str) -> str:
        """Return the configured model name unchanged."""
        return model_name

    def call(
        self,
        model: str,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """
        调用 Qwen API，返回 {"text": str, "input_tokens": int, "output_tokens": int, "latency_ms": float}
        """
        physical_model = self.resolve_model(model)
        start = time.time()

        if self.use_openai_compat:
            result = self._call_openai_compat(physical_model, prompt, system_prompt,
                                              temperature, top_p)
        else:
            result = self._call_dashscope(physical_model, prompt, system_prompt,
                                          temperature, top_p)

        latency_ms = (time.time() - start) * 1000.0
        result["latency_ms"] = latency_ms
        result["model"] = model
        result["physical_model"] = physical_model
        _auto_record_cost(physical_model, result["input_tokens"], result["output_tokens"], latency_ms)
        return result

    def _call_dashscope(
        self, model: str, prompt: str, system_prompt: Optional[str],
        temperature: float, top_p: float,
    ) -> Dict[str, Any]:
        """通过 DashScope SDK 调用。"""
        from dashscope import Generation

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = Generation.call(
            api_key=self.api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            result_format="message",
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"DashScope API error: code={response.status_code}, "
                f"message={response.message}"
            )

        return {
            "text": response.output.choices[0].message.content,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    def _call_openai_compat(
        self, model: str, prompt: str, system_prompt: Optional[str],
        temperature: float, top_p: float,
    ) -> Dict[str, Any]:
        """OpenAI 兼容模式 (vLLM 或代理)。"""
        from openai import OpenAI

        client = OpenAI(base_url=self.openai_base_url, api_key=self.openai_api_key)

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )

        return {
            "text": response.choices[0].message.content,
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
        }


# ============================================================
# 全局适配器实例
# ============================================================
_adapter: Optional[QwenAdapter] = None


def get_adapter() -> QwenAdapter:
    global _adapter
    if _adapter is None:
        _adapter = QwenAdapter()
    return _adapter


def set_adapter(adapter: QwenAdapter):
    global _adapter
    _adapter = adapter


# ============================================================
# 缺失的 llm_chain / async_llm_chain_call / get_llm_chain
# ============================================================

def llm_chain(
    prompt: Any,            # ChatPromptTemplate (LangChain) 或 str
    engine: Any,            # LLM 引擎标识 (str model_name 或 dict)
    parser: Any,            # BaseOutputParser 或 callable
    request_kwargs: Dict[str, Any],
) -> Any:
    """
    同步单次 LLM 调用。

    替代原有缺失的 llm.models.llm_chain。
    支持 LangChain ChatPromptTemplate 和纯字符串 prompt。

    Args:
        prompt: ChatPromptTemplate 或普通字符串
        engine: 模型名字符串 (如 "qwen-plus") 或 {"model": "...", "temperature": ...}
        parser: 输出解析器 (BaseOutputParser 或 callable(text) -> parsed)
        request_kwargs: 模板变量 (如 {"HINT": "...", "QUESTION": "..."})

    Returns:
        解析后的输出
    """
    adapter = get_adapter()

    # 解析 engine 参数
    if isinstance(engine, dict):
        model = engine.get("model", DEFAULT_LLM_MODEL)
        temperature = engine.get("temperature", 0.0)
    elif isinstance(engine, str):
        model = engine
        temperature = 0.0
    else:
        # LangChain LLM 实例 — 尝试直接调用
        model = DEFAULT_LLM_MODEL
        temperature = 0.0

    # 构造 prompt 文本
    if hasattr(prompt, "format"):
        # LangChain ChatPromptTemplate
        try:
            prompt_text = prompt.format(**request_kwargs)
        except Exception:
            prompt_text = str(prompt)
    elif isinstance(prompt, str):
        prompt_text = prompt
    else:
        prompt_text = str(prompt)

    # 调用 LLM
    result = adapter.call(
        model=model,
        prompt=prompt_text,
        temperature=temperature,
    )

    raw_text = result["text"]

    # 解析输出
    if parser is not None:
        try:
            if hasattr(parser, "parse"):
                parsed = parser.parse(raw_text)
            elif callable(parser):
                parsed = parser(raw_text)
            else:
                parsed = raw_text
        except Exception:
            # 解析失败 → 返回原始文本
            parsed = raw_text
    else:
        parsed = raw_text

    return parsed


def async_llm_chain_call(
    prompt: Any,
    engine: Any,
    parser: Any,
    request_list: List[Dict[str, Any]],
    step: str = "",
    sampling_count: int = 1,
) -> List[List[Any]]:
    """
    异步批量 LLM 调用 (顺序执行，每个 request 重复 sampling_count 次)。

    替代原有缺失的 llm.models.async_llm_chain_call。

    Args:
        prompt: 模板 (同 llm_chain)
        engine: 模型配置 (同 llm_chain)
        parser: 解析器 (同 llm_chain)
        request_list: 请求列表，每个元素是一个 request_kwargs dict
        step: 阶段名 (仅用于日志)
        sampling_count: 每个 request 的采样次数

    Returns:
        List[List[Any]]: 外层对 request_list，内层对 sampling_count
    """
    adapter = get_adapter()

    if isinstance(engine, dict):
        model = engine.get("model", DEFAULT_LLM_MODEL)
        temperature = engine.get("temperature", 0.0)
    elif isinstance(engine, str):
        model = engine
        temperature = 0.0
    else:
        model = DEFAULT_LLM_MODEL
        temperature = 0.0

    all_results = []
    for req_kwargs in request_list:
        # 构造 prompt
        if hasattr(prompt, "format"):
            try:
                prompt_text = prompt.format(**req_kwargs)
            except Exception:
                prompt_text = str(prompt)
        else:
            prompt_text = str(prompt)

        sample_results = []
        for _ in range(sampling_count):
            result = adapter.call(
                model=model,
                prompt=prompt_text,
                temperature=temperature,
            )
            raw_text = result["text"]

            # 解析
            if parser is not None:
                try:
                    if hasattr(parser, "parse"):
                        parsed = parser.parse(raw_text)
                    elif callable(parser):
                        parsed = parser(raw_text)
                    else:
                        parsed = raw_text
                except Exception:
                    parsed = raw_text
            else:
                parsed = raw_text

            sample_results.append(parsed)

        all_results.append(sample_results)

    return all_results


def get_llm_chain(engine: str, temperature: float = 0.0) -> Dict[str, Any]:
    """
    构造 LLM 引擎配置 (替代原缺失的 get_llm_chain)。

    返回 dict 而非 LangChain 实例，由 llm_chain 内部消费。
    """
    return {
        "model": engine,
        "temperature": temperature,
    }
