from threading import Lock
from typing import Any, Dict, Tuple, Optional

from .prompts import get_prompt
from .models import get_llm_chain
from .parsers import get_parser
try:
    from _shared.model_config import DEFAULT_LLM_MODEL
except ModuleNotFoundError:
    try:
        from env._shared.model_config import DEFAULT_LLM_MODEL
    except ModuleNotFoundError:
        from nl2sql.env._shared.model_config import DEFAULT_LLM_MODEL


def get_prompt_engine_parser(node_name: str,
                             llm_config: Optional[Dict[str, Any]] = None,
                             **kwargs: Any) -> Tuple[Any, Any, Any]:
    """
        Retrieves the prompt, engine, and parser for the current node based on the pipeline setup.

        Args:
            llm_config: llm model, temperature, and prompt template.
        Returns:
            Tuple[Any, Any, Any]: The prompt, engine, and parser instances.

        Raises:
            ValueError: If the engine is not specified for the node.
        """

    llm_config = llm_config or {}
    engine_name = llm_config.get("model", DEFAULT_LLM_MODEL)
    template_name = llm_config.get("template_name", node_name)
    prompt = get_prompt(template_name, **kwargs)

    temperature = llm_config.get("temperature", 0)
    extra_body = llm_config.get("extra_body")
    engine = get_llm_chain(
        engine=engine_name,
        temperature=temperature,
        extra_body=extra_body,
    )

    parser_name = llm_config.get("parser_name", node_name)
    if parser_name == '':
        parser_name = node_name

    # parser_name = get_parser_name(node_name)
    parser = get_parser(parser_name)

    return prompt, engine, parser


def get_parser_name(node_name: str) -> str:
    """
    Determines the appropriate parser name for the given node.

    Args:
        node_name (str): The name of the node.

    Returns:
        str: The parser name.
    """
    # if node_name == "candidate_generation":
    #     engine_name = self.candidate_generation.get("engine", None)
    #     if engine_name == "finetuned_nl2sql":
    #         return "finetuned_candidate_generation"
    return node_name


def get_template_name(self, node_name: str) -> str:
    """
    Determines the appropriate template name for the given node.

    Args:
        node_name (str): The name of the node.

    Returns:
        str: The template name.
    """
    if node_name == "column_filtering":
        engine_name = self.column_filtering.get("engine", None)
        if engine_name and "llama" in engine_name.lower():
            return "column_filtering_with_examples_llama"
        else:
            return "column_filtering_with_examples"
    elif node_name == "candidate_generation":
        engine_name = self.candidate_generation.get("engine", None)
        if engine_name == "finetuned_nl2sql":
            return "finetuned_candidate_generation"
    return node_name
