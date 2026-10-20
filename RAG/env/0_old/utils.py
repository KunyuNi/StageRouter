import re
import logging
from typing import Dict
from functools import wraps
from typing import Dict, List, Any, Callable
import dspy


def search_wikipedia(query: str, k=3, max_retries: int = 10) -> list[str]:
    for attempt in range(max_retries):
        try:
            results = dspy.ColBERTv2(
                url="http://20.102.90.50:2017/wiki17_abstracts")(query, k=k)
            return [x["text"] for x in results]

        except Exception as e:
            # If this wasn't the last attempt, wait before retrying.
            if attempt >= max_retries - 1:
                logging.error(
                    f"All {max_retries} attempts failed for query: '{query}'.")

    # If the loop completes without a successful return, return an empty list.
    return []


def node_decorator() -> Callable:
    """
    A decorator to add logging and error handling to pipeline node functions.

    Args:
        check_schema_status (bool, optional): Whether to check the schema status. Defaults to False.

    Returns:
        Callable: The decorated function.
    """

    def decorator(func: Callable) -> Callable:

        @wraps(func)
        def wrapper(
            state: Dict[str, Any],
            config: Dict[str, Any],
        ) -> Dict[str, Any]:
            node_name = func.__name__
            result = {"node_type": node_name}

            try:
                task = state["keys"]["task"]
                execution_history = state["keys"]["execution_history"]

                output = func(task, execution_history, config)

                result.update(output)

                result["status"] = "success"
            except Exception as e:
                result.update({
                    "status": "error",
                    "error": f"{type(e)}: <{e}>",
                })

            execution_history.append(result)

            return state

        return wrapper

    return decorator


def get_last_node_result(
        execution_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    if len(execution_history) == 0:
        return {}

    return execution_history[-1]


def get_node_result(execution_history: List[Dict[str, Any]],
                    node_type: str) -> Dict[str, Any]:
    """
    Retrieves the last result for a specific node type from the execution history.

    Args:
        execution_history (List[Dict[str, Any]]): The execution history.
        node_type (str): The type of node to look for.

    Returns:
        Dict[str, Any]: The result of the last node of the specified type, or None if not found.
    """
    for node in reversed(execution_history):
        if node["node_type"] == node_type:
            return node
    return None


def parse_vql_from_string(response: str):
    vql_matches = re.findall(r'Visualize\s+.*', response,
                             re.IGNORECASE | re.MULTILINE)
    if vql_matches:
        return vql_matches[-1].strip()
    else:
        return ''


def parse_query_from_string(response: str):
    match = re.search(r"search query:\s*(.*)", response)
    if match:
        # The .strip() is still useful if the captured group has trailing whitespace
        query = match.group(1).strip()
        return query
    return response


def parse_answer_from_string(response: str):
    match = re.search(r"answer:\s*(.*)", response)
    if match:
        # The .strip() is still useful if the captured group has trailing whitespace
        answer = match.group(1).strip()
        return answer
    return response


if __name__ == "__main__":
    t = search_wikipedia("Hello World", 3)
    print(t)
