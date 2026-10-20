"""
LLM model utilities - provides get_llm_chain, llm_chain, async_llm_chain_call, gemini_api_call_with_config
Uses ChatOpenAI via OpenAI-compatible API (DashScope / DeepSeek).
"""
import json
import logging
import os
import re
import time
from typing import Any, Dict, Mapping, Optional

try:
    from _shared.model_config import EMBEDDING_MODEL
except ModuleNotFoundError:
    try:
        from env._shared.model_config import EMBEDDING_MODEL
    except ModuleNotFoundError:
        from nl2sql.env._shared.model_config import EMBEDDING_MODEL

from .engine_configs import ENGINE_CONFIGS

logger = logging.getLogger(__name__)

_QWEN35_FLASH_TIMEOUT_SECONDS = 300
_QWEN_FLASH_TIMEOUT_SECONDS = 20 * 60
_QWEN_PLUS_TIMEOUT_SECONDS = 300


def _json_loads_relaxed(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # LLMs sometimes escape characters that JSON does not allow, e.g. SQL
        # apostrophes as \'. Preserve valid JSON escapes and remove only the
        # invalid backslash marker.
        repaired = re.sub(r'\\([^"\\/bfnrtu])', r'\1', text)
        return json.loads(repaired)


def _robust_parse_json(content: str, parser=None):
    """
    兜底 JSON 解析：当 LangChain parser.parse() 失败时，
    从 LLM 返回的混合文本中提取 JSON。

    解析策略（按优先级）：
    1. 原始 parser.parse()
    2. 提取 ``` 代码块（支持有无 json 标识）
    3. 字符串匹配最外层 {...} 或 [...]
    """
    # 策略 1：原始 parser
    if parser:
        try:
            return parser.parse(content)
        except Exception:
            pass

    # 策略 2：提取 ``` 代码块 (兼容带有或不带有 json 标识的情况)
    json_blocks = re.findall(r'```(?:json)?\s*\n?(.*?)\n?\s*```', content, re.DOTALL)
    for block in json_blocks:
        try:
            return _json_loads_relaxed(block.strip())
        except json.JSONDecodeError:
            continue

    # 策略 3：使用字符串匹配寻找最外层的 {} 或 []
    # 尝试提取 JSON 对象 {...}
    start_obj = content.find('{')
    end_obj = content.rfind('}')
    if start_obj != -1 and end_obj != -1 and start_obj < end_obj:
        try:
            return _json_loads_relaxed(content[start_obj:end_obj + 1].strip())
        except json.JSONDecodeError:
            pass
            
    # 尝试提取 JSON 数组 [...]
    start_arr = content.find('[')
    end_arr = content.rfind(']')
    if start_arr != -1 and end_arr != -1 and start_arr < end_arr:
        try:
            return _json_loads_relaxed(content[start_arr:end_arr + 1].strip())
        except json.JSONDecodeError:
            pass

    # 策略 4: 提取 ```sql 代码块 (LLM 返回推理文本 + SQL 代码块，非 JSON)
    sql_blocks = re.findall(r'```sql\s*\n?(.*?)\n?\s*```', content, re.DOTALL)
    if sql_blocks:
        return {"SQL": sql_blocks[0].strip()}

    # 全部失败，返回 None
    return None


def _default_extra_body(engine: str) -> Dict[str, Any]:
    if engine.lower().startswith("qwen3"):
        return {"enable_thinking": False}
    return {}


def get_llm_chain(engine: str, temperature: float = 0.0,
                  extra_body: Optional[Mapping[str, Any]] = None) -> Any:
    config = ENGINE_CONFIGS.get(engine)
    if not config:
        raise ValueError(f"Unknown engine: {engine}. Available: {list(ENGINE_CONFIGS.keys())}")
    params = config["params"].copy()
    controlled_retries = os.getenv("NL2SQL_SDK_MAX_RETRIES")
    if controlled_retries is not None:
        retries = int(controlled_retries)
        if retries not in {0, 1}:
            raise ValueError("audited SDK max_retries must be 0 or 1")
        params["max_retries"] = retries
    if engine == "qwen3.5-flash":
        params["timeout"] = _QWEN35_FLASH_TIMEOUT_SECONDS
    elif engine == "qwen-flash":
        params["timeout"] = _QWEN_FLASH_TIMEOUT_SECONDS
    elif engine == "qwen-plus":
        params["timeout"] = _QWEN_PLUS_TIMEOUT_SECONDS
    params["temperature"] = temperature
    merged_extra_body = _default_extra_body(engine)
    if extra_body is not None:
        merged_extra_body.update(dict(extra_body))
    if merged_extra_body:
        params["extra_body"] = merged_extra_body
    return config["constructor"](**params)


def _extract_token_usage(result) -> Dict[str, int]:
    """从 LangChain AIMessage 中提取 token 使用信息"""
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    if hasattr(result, "response_metadata"):
        meta = result.response_metadata
        token_info = meta.get("token_usage", meta.get("usage", {}))
        if isinstance(token_info, dict):
            usage["prompt_tokens"] = token_info.get("prompt_tokens",
                                        token_info.get("input_tokens", 0))
            usage["completion_tokens"] = token_info.get("completion_tokens",
                                            token_info.get("output_tokens", 0))
    if hasattr(result, "usage_metadata") and result.usage_metadata:
        um = result.usage_metadata
        if hasattr(um, "input_tokens"):
            usage["prompt_tokens"] = um.input_tokens or 0
        if hasattr(um, "output_tokens"):
            usage["completion_tokens"] = um.output_tokens or 0
    return usage


def _llm_chain_unclosed(prompt, engine, parser,
                        request_kwargs: Dict[str, Any]) -> Any:
    chain = prompt | engine
    result = chain.invoke(request_kwargs)
    content = result.content if hasattr(result, "content") else str(result)
    token_usage = _extract_token_usage(result)
    # Attach model name for cost tracking
    model_name = getattr(engine, 'model_name', getattr(engine, 'model', ''))
    token_usage["model_name"] = model_name
    try:
        parsed = parser.parse(content) if parser else content
    except Exception:
        # 原始 parser 失败，尝试兜底 JSON 提取
        logger.warning(f"Parser failed, trying robust JSON extraction...")
        fallback = _robust_parse_json(content)
        if fallback is not None:
            logger.info(f"Robust JSON extraction succeeded")
            parsed = fallback
        else:
            # 兜底也失败，抛出异常并附带 token_usage
            e = ValueError(f"All JSON parse strategies failed for content: {content[:200]}...")
            e.token_usage = token_usage
            raise e
    return parsed, token_usage


def _owned_llm_client(engine: Any, root_name: str, resource_name: str) -> Any:
    client = getattr(engine, root_name, None)
    if client is not None:
        return client
    resource = getattr(engine, resource_name, None)
    return getattr(resource, "_client", None)


def _close_sync_llm_client(engine: Any) -> None:
    """Release the sync transport owned by a one-shot LangChain engine."""
    client = _owned_llm_client(engine, "root_client", "client")
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logger.warning("Failed to close synchronous LLM client", exc_info=True)


def llm_chain(prompt, engine, parser, request_kwargs: Dict[str, Any]) -> Any:
    try:
        return _llm_chain_unclosed(prompt, engine, parser, request_kwargs)
    finally:
        _close_sync_llm_client(engine)


async def _async_llm_chain_call_unclosed(
        prompt, engine, parser,
        request_kwargs: Optional[Dict[str, Any]] = None,
        request_list: Optional[list] = None,
        **kwargs) -> Any:
    chain = prompt | engine
    # Attach model name for cost tracking
    model_name = getattr(engine, 'model_name', getattr(engine, 'model', ''))
    if request_list is not None:
        # This batch is intentionally serial. Running it through the sync
        # transport avoids creating a fresh async HTTP pool for every S4
        # invocation and then closing that pool as asyncio.run() exits.
        results = chain.batch(request_list, config={"max_concurrency": 1})
        parsed = []
        aggregated_usage = {"prompt_tokens": 0, "completion_tokens": 0, "model_name": model_name}
        for result in results:
            content = result.content if hasattr(result, "content") else str(result)
            token_usage = _extract_token_usage(result)
            aggregated_usage["prompt_tokens"] += token_usage["prompt_tokens"]
            aggregated_usage["completion_tokens"] += token_usage["completion_tokens"]
            if parser:
                try:
                    parsed.append(parser.parse(content))
                except Exception:
                    fallback = _robust_parse_json(content)
                    if fallback is not None:
                        parsed.append(fallback)
                    else:
                        e = ValueError(f"All JSON parse strategies failed")
                        e.token_usage = aggregated_usage
                        raise e
            else:
                parsed.append(content)
        return parsed, aggregated_usage
    else:
        result = await chain.ainvoke(request_kwargs)
        content = result.content if hasattr(result, "content") else str(result)
        token_usage = _extract_token_usage(result)
        token_usage["model_name"] = model_name
        try:
            parsed = parser.parse(content) if parser else content
        except Exception:
            logger.warning(f"Async parser failed, trying robust JSON extraction...")
            fallback = _robust_parse_json(content)
            if fallback is not None:
                parsed = fallback
            else:
                e = ValueError(f"All JSON parse strategies failed")
                e.token_usage = token_usage
                raise e
        return parsed, token_usage


async def _close_async_llm_client(engine: Any) -> None:
    """Release the async transport before asyncio.run() closes its loop."""
    client = _owned_llm_client(
        engine, "root_async_client", "async_client"
    )
    close = getattr(client, "close", None)
    if callable(close):
        try:
            await close()
        except Exception:
            logger.warning("Failed to close asynchronous LLM client", exc_info=True)


async def async_llm_chain_call(
        prompt, engine, parser,
        request_kwargs: Optional[Dict[str, Any]] = None,
        request_list: Optional[list] = None,
        **kwargs) -> Any:
    try:
        return await _async_llm_chain_call_unclosed(
            prompt=prompt,
            engine=engine,
            parser=parser,
            request_kwargs=request_kwargs,
            request_list=request_list,
            **kwargs,
        )
    finally:
        if request_list is not None:
            _close_sync_llm_client(engine)
        else:
            await _close_async_llm_client(engine)


def gemini_api_call_with_config(model_name: str, prompt: str) -> str:
    from openai import OpenAI
    import os

    if model_name not in ENGINE_CONFIGS:
        raise ValueError(f"Model {model_name} not found in ENGINE_CONFIGS")

    params = ENGINE_CONFIGS[model_name]["params"]

    client = OpenAI(
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
    )

    response = client.chat.completions.create(
        model=params["model"],
        messages=[{"role": "user", "content": prompt}],
    )

    return response.choices[0].message.content


from langchain_core.embeddings import Embeddings
import threading

EMBEDDING_SEMAPHORE = threading.Semaphore(4)
EMBEDDING_BATCH_SIZE = 10


def _estimate_embedding_tokens(texts: list[str]) -> int:
    total = 0
    for text in texts:
        normalized = str(text)
        if not normalized.strip():
            normalized = "empty_string"
        total += max(1, (len(normalized) + 3) // 4)
    return total


def _embedding_input_tokens(response, batch: list[str]) -> int:
    usage = getattr(response, "usage", None)
    if usage is not None:
        tokens = int(
            getattr(usage, "prompt_tokens", 0)
            or getattr(usage, "input_tokens", 0)
            or getattr(usage, "total_tokens", 0)
            or 0
        )
        if tokens > 0:
            return tokens
    return _estimate_embedding_tokens(batch)


class SafeOpenAIEmbeddings(Embeddings):
    def __init__(self, model=EMBEDDING_MODEL, api_key=None, base_url=None):
        from openai import OpenAI
        import os
        controlled_retries = os.getenv("NL2SQL_SDK_MAX_RETRIES")
        embedding_max_retries = 15 if controlled_retries is None else int(controlled_retries)
        if controlled_retries is not None and embedding_max_retries not in {0, 1}:
            raise ValueError("audited SDK max_retries must be 0 or 1")
        self.client = OpenAI(api_key=api_key or os.getenv("LLM_API_KEY"), 
                             base_url=base_url or os.getenv("LLM_BASE_URL"),
                             max_retries=embedding_max_retries,
                             timeout=120.0)
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        # DashScope will throw "InvalidParameter" if it receives empty or whitespace-only strings.
        clean_texts = []
        for t in texts:
            t_str = str(t)
            clean_texts.append(t_str if t_str.strip() else "empty_string")
            
        embeddings = []
        for i in range(0, len(clean_texts), EMBEDDING_BATCH_SIZE):
            batch = clean_texts[i:i + EMBEDDING_BATCH_SIZE]
            start = time.time()
            with EMBEDDING_SEMAPHORE:
                res = self.client.embeddings.create(input=batch, model=self.model)
            latency_ms = (time.time() - start) * 1000.0
            embeddings.extend(d.embedding for d in res.data)

            try:
                from _shared.llm_adapter import current_cost_context

                tracker, stage = current_cost_context()
                if tracker is not None and stage is not None:
                    input_tokens = _embedding_input_tokens(res, batch)
                    tracker.record_embedding_call(
                        stage=stage,
                        model=self.model,
                        input_tokens=input_tokens,
                        latency_ms=latency_ms,
                    )
            except Exception:
                logger.debug("Embedding cost tracking skipped", exc_info=True)
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
