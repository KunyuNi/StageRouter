"""Runtime guards for expensive API collection jobs."""

from __future__ import annotations

import errno
import sqlite3
import subprocess
from typing import Optional


class ApiQuotaExhaustedError(RuntimeError):
    """Raised when the upstream API reports quota/balance exhaustion."""


class ApiRateLimitError(RuntimeError):
    """Raised when the provider rejects concurrency or request rate."""


class ApiInfrastructureError(RuntimeError):
    """Raised for transport and provider infrastructure failures."""


_QUOTA_ERROR_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "quota exceeded",
    "exceeded your current quota",
    "free allocated quota",
    "balance",
    "billing",
    "credit",
    "arrear",
    "arrears",
    "arrearage",
    "good standing",
    "overdue-payment",
    "余额不足",
    "账户余额",
    "账号余额",
    "欠费",
    "额度不足",
    "额度已用完",
    "计费",
    "充值",
)

_RATE_LIMIT_MARKERS = (
    "rate limit",
    "rate_limit",
    "ratelimit",
    "too many requests",
    "throttl",
    "concurrency limit",
    "并发限制",
    "限流",
)

_API_PROVIDER_MODULES = frozenset(
    {"dashscope", "httpcore", "httpx", "openai", "requests"}
)

_API_INFRASTRUCTURE_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ConnectTimeout",
        "ConnectionError",
        "ReadError",
        "ReadTimeout",
        "RemoteProtocolError",
        "Timeout",
        "TimeoutException",
    }
)

_TRANSIENT_OS_ERRNOS = frozenset(
    value
    for value in (
        getattr(errno, "EAGAIN", None),
        getattr(errno, "EBUSY", None),
        getattr(errno, "ECONNABORTED", None),
        getattr(errno, "ECONNREFUSED", None),
        getattr(errno, "ECONNRESET", None),
        getattr(errno, "EHOSTUNREACH", None),
        getattr(errno, "EINTR", None),
        getattr(errno, "EIO", None),
        getattr(errno, "ENETDOWN", None),
        getattr(errno, "ENETUNREACH", None),
        getattr(errno, "ENOSPC", None),
        getattr(errno, "ESTALE", None),
        getattr(errno, "ETIMEDOUT", None),
    )
    if value is not None
)

_SQLITE_INFRASTRUCTURE_CODES = frozenset(
    value
    for value in (
        getattr(sqlite3, "SQLITE_BUSY", None),
        getattr(sqlite3, "SQLITE_CANTOPEN", None),
        getattr(sqlite3, "SQLITE_FULL", None),
        getattr(sqlite3, "SQLITE_IOERR", None),
        getattr(sqlite3, "SQLITE_LOCKED", None),
        getattr(sqlite3, "SQLITE_PROTOCOL", None),
    )
    if value is not None
)

_SQLITE_INFRASTRUCTURE_MESSAGES = frozenset(
    {
        "database is busy",
        "database is locked",
        "database table is locked",
        "disk i/o error",
        "unable to open database file",
    }
)


def _status_code(error: object) -> int | None:
    for candidate in (error, getattr(error, "response", None)):
        value = getattr(candidate, "status_code", None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    text = str(error)
    dashscope_prefix = "DashScope API error: code="
    if isinstance(error, RuntimeError) and text.startswith(dashscope_prefix):
        value, separator, _ = text[len(dashscope_prefix) :].partition(",")
        if separator:
            try:
                return int(value)
            except ValueError:
                pass
    return None


def _is_provider_exception(error: BaseException) -> bool:
    return type(error).__module__.split(".", 1)[0] in _API_PROVIDER_MODULES


def _is_sqlite_infrastructure_error(error: BaseException) -> bool:
    if not isinstance(error, sqlite3.Error):
        return False
    error_code = getattr(error, "sqlite_errorcode", None)
    if type(error_code) is int:
        return (error_code & 0xFF) in _SQLITE_INFRASTRUCTURE_CODES
    return str(error).strip().lower() in _SQLITE_INFRASTRUCTURE_MESSAGES


def is_api_quota_error(error: object) -> bool:
    text = str(error).lower()
    return any(marker in text for marker in _QUOTA_ERROR_MARKERS)


def as_api_quota_error(error: BaseException) -> Optional[ApiQuotaExhaustedError]:
    if isinstance(error, ApiQuotaExhaustedError):
        return error
    if not is_api_quota_error(error):
        return None
    return ApiQuotaExhaustedError(
        f"API quota/balance appears exhausted; stopping collection. Original error: {error}"
    )


def as_api_rate_limit_error(error: BaseException) -> Optional[ApiRateLimitError]:
    if isinstance(error, ApiRateLimitError):
        return error
    status = _status_code(error)
    text = str(error).lower()
    if status != 429 and not any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return None
    return ApiRateLimitError(
        f"API rate/concurrency limit encountered ({type(error).__name__})"
    )


def as_api_infrastructure_error(error: BaseException) -> Optional[ApiInfrastructureError]:
    if isinstance(error, ApiInfrastructureError):
        return error
    status = _status_code(error)
    provider_exception = _is_provider_exception(error)
    is_transient_os_error = (
        isinstance(error, OSError) and error.errno in _TRANSIENT_OS_ERRNOS
    )
    if not any(
        (
            isinstance(error, (ConnectionError, TimeoutError, subprocess.TimeoutExpired)),
            _is_sqlite_infrastructure_error(error),
            is_transient_os_error,
            provider_exception
            and type(error).__name__ in _API_INFRASTRUCTURE_EXCEPTION_NAMES,
            status in {408, 409},
            status is not None and status >= 500,
        )
    ):
        return None
    return ApiInfrastructureError(
        f"API infrastructure failure ({type(error).__name__})"
    )


def classify_api_error(error: BaseException) -> BaseException:
    """Return a credential-free structured provider error when applicable."""
    return (
        as_api_quota_error(error)
        or as_api_rate_limit_error(error)
        or as_api_infrastructure_error(error)
        or error
    )


def as_fatal_api_error(error: BaseException) -> Optional[BaseException]:
    """Return provider conditions that collection must never downgrade to a miss."""
    if isinstance(
        error, (ApiQuotaExhaustedError, ApiRateLimitError, ApiInfrastructureError)
    ):
        return error
    if _status_code(error) == 429:
        return as_api_rate_limit_error(error)
    infrastructure_error = as_api_infrastructure_error(error)
    if infrastructure_error is not None:
        return infrastructure_error
    if _is_provider_exception(error):
        return as_api_quota_error(error) or as_api_rate_limit_error(error)
    return None
