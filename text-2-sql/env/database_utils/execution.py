import sqlite3
import random
import logging
import os
import time
from pathlib import Path
from typing import Any, Union, List, Dict


_RETRYABLE_SQLITE_PRIMARY_CODES = frozenset({5, 6, 10, 13, 14})
_RETRYABLE_SQLITE_MESSAGES = (
    "database is busy",
    "database is locked",
    "database table is locked",
    "disk i/o error",
    "unable to open database file",
    "database or disk is full",
)
_DENIED_SQLITE_ACTIONS = frozenset(
    action
    for action in (
        getattr(sqlite3, name, None)
        for name in (
            "SQLITE_ALTER_TABLE",
            "SQLITE_ANALYZE",
            "SQLITE_ATTACH",
            "SQLITE_CREATE_INDEX",
            "SQLITE_CREATE_TABLE",
            "SQLITE_CREATE_TEMP_INDEX",
            "SQLITE_CREATE_TEMP_TABLE",
            "SQLITE_CREATE_TEMP_TRIGGER",
            "SQLITE_CREATE_TEMP_VIEW",
            "SQLITE_CREATE_TRIGGER",
            "SQLITE_CREATE_VIEW",
            "SQLITE_CREATE_VTABLE",
            "SQLITE_DELETE",
            "SQLITE_DETACH",
            "SQLITE_DROP_INDEX",
            "SQLITE_DROP_TABLE",
            "SQLITE_DROP_TEMP_INDEX",
            "SQLITE_DROP_TEMP_TABLE",
            "SQLITE_DROP_TEMP_TRIGGER",
            "SQLITE_DROP_TEMP_VIEW",
            "SQLITE_DROP_TRIGGER",
            "SQLITE_DROP_VIEW",
            "SQLITE_DROP_VTABLE",
            "SQLITE_INSERT",
            "SQLITE_REINDEX",
            "SQLITE_SAVEPOINT",
            "SQLITE_TRANSACTION",
            "SQLITE_UPDATE",
        )
    )
    if action is not None
)
_READ_ONLY_ARGUMENT_PRAGMAS = frozenset(
    {
        "foreign_key_check",
        "foreign_key_list",
        "index_info",
        "index_list",
        "index_xinfo",
        "integrity_check",
        "quick_check",
        "table_info",
        "table_xinfo",
    }
)


class SQLExecutionDeadlineExceeded(RuntimeError):
    """The active SQLite VM was interrupted at its comparison deadline."""


def is_retryable_sqlite_infrastructure_error(exc: BaseException) -> bool:
    """Return whether SQLite reported a transient storage/connection failure."""
    if not isinstance(exc, sqlite3.Error):
        return False
    error_code = getattr(exc, "sqlite_errorcode", None)
    if type(error_code) is int:
        return (error_code & 0xFF) in _RETRYABLE_SQLITE_PRIMARY_CODES
    message = str(exc).lower()
    return any(marker in message for marker in _RETRYABLE_SQLITE_MESSAGES)


def _read_only_database_uri(db_path: str | os.PathLike[str]) -> str:
    try:
        path_string = os.fspath(db_path)
    except TypeError as exc:
        raise ValueError("db_path must be a non-empty path string") from exc
    if type(path_string) is not str or not path_string:
        raise ValueError("db_path must be a non-empty path string")
    return Path(path_string).resolve().as_uri() + "?mode=ro"


def _open_read_only_connection(
    db_path: str | os.PathLike[str],
    *,
    deadline: float | None,
) -> tuple[sqlite3.Connection, list[bool]]:
    connection = sqlite3.connect(_read_only_database_uri(db_path), uri=True)
    deadline_hit = [False]
    try:
        connection.execute("PRAGMA query_only=ON")

        def authorize(
            action: int,
            argument_one: str | None,
            argument_two: str | None,
            _database: str | None,
            _trigger: str | None,
        ) -> int:
            if action in _DENIED_SQLITE_ACTIONS:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_PRAGMA and argument_two is not None:
                pragma = str(argument_one or "").lower()
                if pragma not in _READ_ONLY_ARGUMENT_PRAGMAS:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorize)
        if deadline is not None:

            def interrupt_at_deadline() -> int:
                if time.monotonic() >= deadline:
                    deadline_hit[0] = True
                    return 1
                return 0

            connection.set_progress_handler(interrupt_at_deadline, 1_000)
        return connection, deadline_hit
    except BaseException:
        connection.close()
        raise

def _clean_sql(sql: str) -> str:
    """
    Cleans the SQL query by removing unwanted characters and whitespace.
    
    Args:
        sql (str): The SQL query string.
        
    Returns:
        str: The cleaned SQL query string.
    """
    return sql.replace('\n', ' ').replace('"', "'").strip("`.")

def execute_sql(
    db_path: str,
    sql: str,
    fetch: Union[str, int] = "all",
    *,
    _deadline: float | None = None,
) -> Any:
    """
    Executes an SQL query on a database and fetches results.
    
    Args:
        db_path (str): The path to the database file.
        sql (str): The SQL query to execute.
        fetch (Union[str, int]): How to fetch the results. Options are "all", "one", "random", or an integer.
        
    Returns:
        Any: The fetched results based on the fetch argument.
    
    Raises:
        Exception: If an error occurs during SQL execution.
    """
    connection = None
    deadline_hit = [False]
    try:
        connection, deadline_hit = _open_read_only_connection(
            db_path,
            deadline=_deadline,
        )
        cursor = connection.cursor()
        try:
            cursor.execute(sql)
            if fetch == "all":
                return cursor.fetchall()
            elif fetch == "one":
                return cursor.fetchone()
            elif fetch == "random":
                samples = cursor.fetchmany(10)
                return random.choice(samples) if samples else []
            elif isinstance(fetch, int):
                return cursor.fetchmany(fetch)
            else:
                raise ValueError("Invalid fetch argument. Must be 'all', 'one', 'random', or an integer.")
        except sqlite3.OperationalError as exc:
            if deadline_hit[0]:
                raise SQLExecutionDeadlineExceeded(
                    "SQLite comparison deadline exceeded"
                ) from exc
            raise
        except sqlite3.DatabaseError as exc:
            if "not authorized" in str(exc).lower():
                raise sqlite3.OperationalError(
                    "write-capable SQL is forbidden during evaluation"
                ) from exc
            raise
    except Exception as e:
        logging.error(f"Error in execute_sql: {e}\nSQL: {sql}")
        raise
    finally:
        if connection is not None:
            connection.set_progress_handler(None, 0)
            connection.set_authorizer(None)
            connection.close()

def _compare_sqls_outcomes(
    db_path: str,
    predicted_sql: str,
    ground_truth_sql: str,
    deadline: float | None = None,
) -> int:
    """
    Compares the outcomes of two SQL queries to check for equivalence.
    
    Args:
        db_path (str): The path to the database file.
        predicted_sql (str): The predicted SQL query.
        ground_truth_sql (str): The ground truth SQL query.
        
    Returns:
        int: 1 if the outcomes are equivalent, 0 otherwise.
    
    Raises:
        Exception: If an error occurs during SQL execution.
    """
    try:
        predicted_res = execute_sql(db_path, predicted_sql, _deadline=deadline)
        ground_truth_res = execute_sql(
            db_path,
            ground_truth_sql,
            _deadline=deadline,
        )
        return int(set(predicted_res) == set(ground_truth_res))
    except Exception as e:
        logging.critical(f"Error comparing SQL outcomes: {e}")
        raise e

def compare_sqls(db_path: str, predicted_sql: str, ground_truth_sql: str, meta_time_out: float | None = 30) -> Dict[str, Union[int, str]]:
    """
    Compares predicted SQL with ground truth SQL within a timeout.
    
    Args:
        db_path (str): The path to the database file.
        predicted_sql (str): The predicted SQL query.
        ground_truth_sql (str): The ground truth SQL query.
        meta_time_out (float | None): Timeout in seconds, or None for no deadline.
        
    Returns:
        dict: A dictionary with the comparison result and any error message.
    """
    predicted_sql = _clean_sql(predicted_sql)
    if meta_time_out is None:
        deadline = None
    else:
        if (
            isinstance(meta_time_out, bool)
            or not isinstance(meta_time_out, (int, float))
            or not 0.0 < float(meta_time_out) < float("inf")
        ):
            raise ValueError("meta_time_out must be a positive finite number or None")
        deadline = time.monotonic() + float(meta_time_out)
    try:
        res = _compare_sqls_outcomes(
            db_path,
            predicted_sql,
            ground_truth_sql,
            deadline,
        )
        error = "incorrect answer" if res == 0 else "--"
    except SQLExecutionDeadlineExceeded:
        logging.warning("Comparison timed out.")
        res = 0
        error = "timeout"
    except sqlite3.OperationalError as exc:
        if is_retryable_sqlite_infrastructure_error(exc):
            raise
        logging.info("Invalid SQL during comparison: %s", exc)
        error = str(exc)
        res = 0
    return {'exec_res': res, 'exec_err': error}

def validate_sql_query(db_path: str, sql: str, max_returned_rows: int = 30) -> Dict[str, Union[str, Any]]:
    """
    Validates an SQL query by executing it and returning the result.
    
    Args:
        db_path (str): The path to the database file.
        sql (str): The SQL query to validate.
        max_returned_rows (int): The maximum number of rows to return.
        
    Returns:
        dict: A dictionary with the SQL query, result, and status.
    """
    try:
        result = execute_sql(db_path, sql, fetch=max_returned_rows)
        return {"SQL": sql, "RESULT": result, "STATUS": "OK"}
    except Exception as e:
        logging.error(f"Error in validate_sql_query: {e}")
        return {"SQL": sql, "RESULT": str(e), "STATUS": "ERROR"}

def aggregate_sqls(db_path: str, sqls: List[str]) -> str:
    """
    Aggregates multiple SQL queries by validating them and clustering based on result sets.
    
    Args:
        db_path (str): The path to the database file.
        sqls (List[str]): A list of SQL queries to aggregate.
        
    Returns:
        str: The shortest SQL query from the largest cluster of equivalent queries.
    """
    results = [validate_sql_query(db_path, sql) for sql in sqls]
    clusters = {}

    # Group queries by unique result sets
    for result in results:
        if result['STATUS'] == 'OK':
            # Using a frozenset as the key to handle unhashable types like lists
            key = frozenset(tuple(row) for row in result['RESULT'])
            if key in clusters:
                clusters[key].append(result['SQL'])
            else:
                clusters[key] = [result['SQL']]
    
    if clusters:
        # Find the largest cluster
        largest_cluster = max(clusters.values(), key=len, default=[])
        # Select the shortest SQL query from the largest cluster
        if largest_cluster:
            return min(largest_cluster, key=len)
    
    logging.warning("No valid SQL clusters found. Returning the first SQL query.")
    return sqls[0]
