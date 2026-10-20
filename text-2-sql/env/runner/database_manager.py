import os
import pickle
import logging
import time
from concurrent.futures import (
    CancelledError,
    ThreadPoolExecutor,
    TimeoutError as FutureTimeoutError,
)
from contextlib import contextmanager
from threading import Lock
from pathlib import Path
from dotenv import load_dotenv
from langchain_chroma import Chroma
from typing import Callable, Dict, List, Any

from database_utils.schema import DatabaseSchema
from database_utils.schema_generator import DatabaseSchemaGenerator
from database_utils.execution import execute_sql, compare_sqls, validate_sql_query, aggregate_sqls
from database_utils.db_info import get_db_all_tables, get_table_all_columns, get_db_schema
from database_utils.sql_parser import get_sql_tables, get_sql_columns_dict, get_sql_condition_literals
from database_utils.db_values.search import query_lsh
from database_utils.db_catalog.search import query_vector_db
from database_utils.db_catalog.preprocess import EMBEDDING_FUNCTION
from database_utils.db_catalog.csv_utils import load_tables_description

try:
    from rl.api_guard import ApiInfrastructureError
except ImportError:  # Compatibility with older standalone environments.
    ApiInfrastructureError = RuntimeError

# load_dotenv(override=False)


DEFAULT_DB_ROOT_PATH = Path(__file__).resolve().parents[2] / "data" / "train"
DEFAULT_DB_MODE = "train"
# Large dev LSH pickles can require roughly 90 seconds for a single cold load.
# Keep this deadline below the process watchdog while allowing one legitimate
# initializer to finish and wake all same-database waiters.
DEFAULT_RESOURCE_INIT_LOCK_TIMEOUT_SECONDS = 180.0
DEFAULT_CHROMA_QUERY_TIMEOUT_SECONDS = 300.0
CHROMA_EXECUTOR_WORKERS = 4

# Chroma 0.5.x keeps one SQLite connection per calling thread.  Route all
# Chroma construction and queries through this bounded pool so 64 collection
# workers cannot multiply persistent SQLite descriptors across every db_id.
_CHROMA_EXECUTOR = ThreadPoolExecutor(
    max_workers=CHROMA_EXECUTOR_WORKERS,
    thread_name_prefix="nl2sql-chroma",
)
_CHROMA_EXECUTOR_GENERATION = 0
_CHROMA_EXECUTOR_LOCK = Lock()

logger = logging.getLogger(__name__)


class StageDeadlineExceeded(ApiInfrastructureError):
    """Raised when local resource initialization cannot acquire its lock in time."""

    def __init__(self, *, db_id: str, resource: str, timeout_seconds: float):
        self.db_id = str(db_id)
        self.resource = str(resource)
        self.timeout_seconds = float(timeout_seconds)
        self.original_error_type = "local_lock_timeout"
        super().__init__(
            f"Timed out after {self.timeout_seconds:.1f}s waiting for "
            f"{self.resource} initialization for db_id={self.db_id}"
        )


class ChromaQueryTimeout(ApiInfrastructureError):
    """Raised when a Chroma operation exceeds its caller-visible deadline."""

    def __init__(
        self,
        *,
        db_id: str,
        operation: str,
        timeout_seconds: float,
        cause_type: str,
    ):
        self.db_id = str(db_id)
        self.operation = str(operation)
        self.timeout_seconds = float(timeout_seconds)
        self.original_error_type = "chroma_query_timeout"
        self.cause_type = str(cause_type)
        super().__init__(
            f"Chroma {self.operation} exceeded {self.timeout_seconds:.1f}s "
            f"for db_id={self.db_id} ({self.cause_type})"
        )


def _submit_chroma_operation(callable_, *args):
    """Submit against a stable executor generation."""
    with _CHROMA_EXECUTOR_LOCK:
        executor = _CHROMA_EXECUTOR
        generation = _CHROMA_EXECUTOR_GENERATION
        future = executor.submit(callable_, *args)
    return executor, generation, future


def _rotate_chroma_executor(stalled_executor, stalled_generation: int) -> bool:
    """Replace one stalled generation so new requests are not head-of-line blocked."""
    global _CHROMA_EXECUTOR, _CHROMA_EXECUTOR_GENERATION

    with _CHROMA_EXECUTOR_LOCK:
        if (
            _CHROMA_EXECUTOR is not stalled_executor
            or _CHROMA_EXECUTOR_GENERATION != stalled_generation
        ):
            return False
        old_executor = _CHROMA_EXECUTOR
        _CHROMA_EXECUTOR = ThreadPoolExecutor(
            max_workers=CHROMA_EXECUTOR_WORKERS,
            thread_name_prefix="nl2sql-chroma",
        )
        _CHROMA_EXECUTOR_GENERATION += 1
    # Running Python threads cannot be killed safely. Cancel queued work and
    # let the bounded API deadline drain running calls in the old generation.
    old_executor.shutdown(wait=False, cancel_futures=True)
    return True


def _current_db_root_path() -> Path:
    return Path(os.getenv("NL2SQL_DB_ROOT", DEFAULT_DB_ROOT_PATH))


def _current_db_mode() -> str:
    return os.getenv("NL2SQL_DB_MODE", DEFAULT_DB_MODE)


class DatabaseManager:
    """
    A per-database singleton used to manage database-local resources.

    Instances are isolated by ``(db_mode, db_id)`` so concurrent workers can
    never rewrite another database's paths, LSH data, or Chroma client.
    """
    _instance = None
    _instances = {}
    _instances_lock = Lock()

    def __new__(cls, db_mode=None, db_id=None):
        if db_mode is None:
            db_mode = _current_db_mode()
        if (db_mode is not None) and (db_id is not None):
            key = (str(db_mode), str(db_id))
            timeout = float(
                os.getenv(
                    "NL2SQL_RESOURCE_INIT_LOCK_TIMEOUT_SECONDS",
                    DEFAULT_RESOURCE_INIT_LOCK_TIMEOUT_SECONDS,
                )
            )
            if not cls._instances_lock.acquire(timeout=timeout):
                raise StageDeadlineExceeded(
                    db_id=str(db_id),
                    resource="database-manager registry",
                    timeout_seconds=timeout,
                )
            try:
                instance = cls._instances.get(key)
                if instance is None:
                    instance = super(DatabaseManager, cls).__new__(cls)
                    instance._init(str(db_mode), str(db_id))
                    cls._instances[key] = instance
                # Retain the legacy no-argument fallback without using it for
                # keyed lookups. Production callers always provide db_id.
                cls._instance = instance
                return instance
            finally:
                cls._instances_lock.release()
        else:
            if cls._instance is None:
                raise ValueError(
                    "DatabaseManager instance has not been initialized yet.")
            return cls._instance

    def _init(self, db_mode: str, db_id: str):
        """
        Initializes the DatabaseManager instance.

        Args:
            db_mode (str): The mode of the database (e.g., 'train', 'test').
            db_id (str): The database identifier.
        """
        self.db_mode = db_mode
        self.db_id = db_id
        self._set_paths()
        self.lsh = None
        self.minhashes = None
        self.vector_db = None
        self._lsh_init_lock = Lock()
        self._vector_init_lock = Lock()
        self.resource_init_lock_timeout_seconds = float(
            os.getenv(
                "NL2SQL_RESOURCE_INIT_LOCK_TIMEOUT_SECONDS",
                DEFAULT_RESOURCE_INIT_LOCK_TIMEOUT_SECONDS,
            )
        )
        self.chroma_query_timeout_seconds = float(
            os.getenv(
                "NL2SQL_CHROMA_QUERY_TIMEOUT_SECONDS",
                DEFAULT_CHROMA_QUERY_TIMEOUT_SECONDS,
            )
        )

    def _set_paths(self):
        """Sets the paths for the database files and directories."""
        db_root_path = _current_db_root_path()
        self.db_path = db_root_path / f"{self.db_mode}_databases" / self.db_id / f"{self.db_id}.sqlite"
        self.db_directory_path = db_root_path / f"{self.db_mode}_databases" / self.db_id
        # logger
        print("[DB DEBUG]", "mode=", self.db_mode, "db_id=", self.db_id, "path=", self.db_path, "exists=",
              self.db_path.exists())

    @contextmanager
    def _bounded_init_lock(self, lock: Lock, resource: str):
        timeout = float(self.resource_init_lock_timeout_seconds)
        if timeout <= 0:
            raise ValueError("resource initialization lock timeout must be positive")
        if not lock.acquire(timeout=timeout):
            raise StageDeadlineExceeded(
                db_id=self.db_id,
                resource=resource,
                timeout_seconds=timeout,
            )
        try:
            yield
        finally:
            lock.release()

    def set_lsh(self) -> str:
        """Sets the LSH and minhashes attributes by loading from pickle files."""
        if self.lsh is not None:
            return "error" if self.lsh == "error" else "success"
        with self._bounded_init_lock(self._lsh_init_lock, "LSH"):
            if self.lsh is None:
                print(f"Loading LSH for {self.db_id}")
                started = time.monotonic()
                logger.info("LSH initialization started: db_id=%s", self.db_id)
                try:
                    with (self.db_directory_path / "preprocessed" /
                          f"{self.db_id}_lsh.pkl").open("rb") as file:
                        lsh = pickle.load(file)
                    with (self.db_directory_path / "preprocessed" /
                          f"{self.db_id}_minhashes.pkl").open("rb") as file:
                        minhashes = pickle.load(file)
                    self.lsh = lsh
                    self.minhashes = minhashes
                    logger.info(
                        "LSH initialization finished: db_id=%s elapsed_seconds=%.3f",
                        self.db_id,
                        time.monotonic() - started,
                    )
                    return "success"
                except Exception as e:
                    self.lsh = "error"
                    self.minhashes = "error"
                    logger.exception(
                        "LSH initialization failed: db_id=%s elapsed_seconds=%.3f",
                        self.db_id,
                        time.monotonic() - started,
                    )
                    print(f"Error loading LSH for {self.db_id}: {e}")
                    return "error"
            elif self.lsh == "error":
                return "error"
            else:
                return "success"

    def _set_vector_db_on_chroma_thread(self) -> str:
        """Initializes Chroma from one of the bounded Chroma worker threads."""
        if self.vector_db is not None:
            return "error" if self.vector_db == "error" else "success"
        with self._bounded_init_lock(self._vector_init_lock, "Chroma"):
            if self.vector_db is None:
                started = time.monotonic()
                logger.info("Chroma initialization started: db_id=%s", self.db_id)
                try:
                    vector_db_path = self.db_directory_path / "context_vector_db"
                    vector_db = Chroma(
                        persist_directory=str(vector_db_path),
                        embedding_function=EMBEDDING_FUNCTION,
                    )
                    self.vector_db = vector_db
                    logger.info(
                        "Chroma initialization finished: db_id=%s elapsed_seconds=%.3f",
                        self.db_id,
                        time.monotonic() - started,
                    )
                    return "success"
                except Exception as e:
                    self.vector_db = "error"
                    logger.exception(
                        "Chroma initialization failed: db_id=%s elapsed_seconds=%.3f",
                        self.db_id,
                        time.monotonic() - started,
                    )
                    print(f"Error loading Vector DB for {self.db_id}: {e}")
                    return "error"
            return "error" if self.vector_db == "error" else "success"

    def set_vector_db(self) -> str:
        """Sets the vector_db attribute using the bounded Chroma executor."""
        return self._run_chroma_operation(
            "initialization",
            self._set_vector_db_on_chroma_thread,
        )

    def query_lsh(self,
                  keyword: str,
                  signature_size: int = 20,
                  n_gram: int = 3,
                  top_n: int = 10) -> Dict[str, List[str]]:
        """
        Queries the LSH for similar values to the given keyword.

        Args:
            keyword (str): The keyword to search for.
            signature_size (int, optional): The size of the MinHash signature. Defaults to 20.
            n_gram (int, optional): The n-gram size for the MinHash. Defaults to 3.
            top_n (int, optional): The number of top results to return. Defaults to 10.

        Returns:
            Dict[str, List[str]]: The dictionary of similar values.
        """
        lsh_status = self.set_lsh()
        if lsh_status == "success":
            return query_lsh(self.lsh, self.minhashes, keyword, signature_size,
                             n_gram, top_n)
        else:
            raise Exception(f"Error loading LSH for {self.db_id}")

    def _query_vector_db_on_chroma_thread(
            self, keyword: str,
            top_k: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
        vector_db_status = self._set_vector_db_on_chroma_thread()
        if vector_db_status == "success":
            return query_vector_db(self.vector_db, keyword, top_k)
        raise Exception(f"Error loading Vector DB for {self.db_id}")

    def query_vector_db(self, keyword: str,
                        top_k: int) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Queries the vector database for similar values to the given keyword.

        Args:
            keyword (str): The keyword to search for.
            top_k (int): The number of top results to return.

        Returns:
            Dict[str, Dict[str, Dict[str, Any]]]: The dictionary of similar values.
        """
        return self._run_chroma_operation(
            "query",
            self._query_vector_db_on_chroma_thread,
            keyword,
            top_k,
        )

    def _run_chroma_operation(self, operation: str, callable_, *args):
        """Run one Chroma operation with progress logs and pool recovery."""
        timeout = float(self.chroma_query_timeout_seconds)
        if timeout <= 0:
            raise ValueError("Chroma query timeout must be positive")

        def instrumented_call():
            started = time.monotonic()
            logger.info(
                "Chroma query progress: started db_id=%s operation=%s",
                self.db_id,
                operation,
            )
            try:
                result = callable_(*args)
            except BaseException:
                logger.exception(
                    "Chroma query progress: failed db_id=%s operation=%s "
                    "elapsed_seconds=%.3f",
                    self.db_id,
                    operation,
                    time.monotonic() - started,
                )
                raise
            logger.info(
                "Chroma query progress: finished db_id=%s operation=%s "
                "elapsed_seconds=%.3f",
                self.db_id,
                operation,
                time.monotonic() - started,
            )
            return result

        executor, generation, future = _submit_chroma_operation(instrumented_call)
        try:
            return future.result(timeout=timeout)
        except CancelledError as exc:
            logger.error(
                "Chroma query progress: cancelled db_id=%s operation=%s generation=%d",
                self.db_id,
                operation,
                generation,
            )
            raise ChromaQueryTimeout(
                db_id=self.db_id,
                operation=operation,
                timeout_seconds=timeout,
                cause_type="executor_generation_rotated",
            ) from exc
        except FutureTimeoutError as exc:
            # A TimeoutError raised by the callable itself is already complete
            # and must retain its original API error semantics.
            if future.done():
                raise
            rotated = _rotate_chroma_executor(executor, generation)
            logger.error(
                "Chroma query progress: timeout db_id=%s operation=%s "
                "timeout_seconds=%.1f generation=%d executor_rotated=%s",
                self.db_id,
                operation,
                timeout,
                generation,
                rotated,
            )
            raise ChromaQueryTimeout(
                db_id=self.db_id,
                operation=operation,
                timeout_seconds=timeout,
                cause_type="operation_deadline",
            ) from exc

    def get_column_profiles(
            self, schema_with_examples: Dict[str, List[str]],
            use_value_description: bool, with_keys: bool,
            with_references: bool) -> Dict[str, Dict[str, str]]:
        """
        Generates column profiles for the schema.

        Args:
            schema_with_examples (Dict[str, List[str]]): Schema with example values.
            use_value_description (bool): Whether to use value descriptions.
            with_keys (bool): Whether to include keys.
            with_references (bool): Whether to include references.

        Returns:
            Dict[str, Dict[str, str]]: The dictionary of column profiles.
        """
        schema_with_descriptions = load_tables_description(
            self.db_directory_path, use_value_description)
        database_schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(
                self.get_db_schema()),
            schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(
                schema_with_examples) if schema_with_examples else None,
            schema_with_descriptions=DatabaseSchema.
            from_schema_dict_with_descriptions(schema_with_descriptions),
            db_id=self.db_id,
            db_path=self.db_path,
            add_examples=True,
        )

        column_profiles = database_schema_generator.get_column_profiles(
            with_keys, with_references)
        return column_profiles

    def get_database_schema_string(self, tentative_schema: Dict[str,
                                                                List[str]],
                                   schema_with_examples: Dict[str, List[str]],
                                   schema_with_descriptions: Dict[str, Dict[
                                       str, Dict[str, Any]]],
                                   include_value_description: bool) -> str:
        """
        Generates a schema string for the database.

        Args:
            tentative_schema (Dict[str, List[str]]): The tentative schema.
            schema_with_examples (Dict[str, List[str]]): Schema with example values.
            schema_with_descriptions (Dict[str, Dict[str, Dict[str, Any]]]): Schema with descriptions.
            include_value_description (bool): Whether to include value descriptions.

        Returns:
            str: The generated schema string.
        """
        schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
            schema_with_examples=DatabaseSchema.from_schema_dict_with_examples(
                schema_with_examples) if schema_with_examples else None,
            schema_with_descriptions=DatabaseSchema.
            from_schema_dict_with_descriptions(schema_with_descriptions)
            if schema_with_descriptions else None,
            db_id=self.db_id,
            db_path=self.db_path,
        )
        schema_string = schema_generator.generate_schema_string(
            include_value_description=include_value_description)
        return schema_string

    def add_connections_to_tentative_schema(
            self, tentative_schema: Dict[str,
                                         List[str]]) -> Dict[str, List[str]]:
        """
        Adds connections to the tentative schema.

        Args:
            tentative_schema (Dict[str, List[str]]): The tentative schema.

        Returns:
            Dict[str, List[str]]: The updated schema with connections.
        """
        schema_generator = DatabaseSchemaGenerator(
            tentative_schema=DatabaseSchema.from_schema_dict(tentative_schema),
            db_id=self.db_id,
            db_path=self.db_path,
        )
        return schema_generator.get_schema_with_connections()

    @staticmethod
    def with_db_path(func: Callable):
        """
        Decorator to inject db_path as the first argument to the function.
        """

        def wrapper(self, *args, **kwargs):
            return func(os.fspath(self.db_path), *args, **kwargs)

        return wrapper

    @classmethod
    def add_methods_to_class(cls, funcs: List[Callable]):
        """
        Adds methods to the class with db_path automatically provided.

        Args:
            funcs (List[Callable]): List of functions to be added as methods.
        """
        for func in funcs:
            method = cls.with_db_path(func)
            setattr(cls, func.__name__, method)


# List of functions to be added to the class
functions_to_add = [
    execute_sql, compare_sqls, validate_sql_query, aggregate_sqls,
    get_db_all_tables, get_table_all_columns, get_db_schema, get_sql_tables,
    get_sql_columns_dict, get_sql_condition_literals
]

# Adding methods to the class
DatabaseManager.add_methods_to_class(functions_to_add)
