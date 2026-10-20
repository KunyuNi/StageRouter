from __future__ import annotations

import hashlib
import re
import sqlite3


_CONTENT_SAFETY_MARKERS = (
    "data_inspection_failed",
    "may contain inappropriate content",
)

_RATE_LIMIT_MARKERS = (
    "http 429",
    "http error 429",
    "limit_requests",
    "rate limit",
    "rate_limit",
    "too many requests",
    "限流",
)

_QUOTA_OR_AUTH_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "quota exceeded",
    "balance is exhausted",
    "balance insufficient",
    "余额不足",
    "欠费",
    "invalid api key",
    "incorrect api key",
    "authentication failed",
    "unauthorized",
    "forbidden",
    "credentials are missing",
)

_FATAL_SHARED_STATE_MARKERS = (
    "non-finite",
    "negative route cost",
    "database disk image is malformed",
    "file is not a database",
    "readonly database",
    "read-only database",
    "no space left on device",
    "disk full",
    "disk i/o error",
    "workflow identity mismatch",
    "index identity mismatch",
    "embedding dimension mismatch",
    "route profile mismatch",
    "writer conflict",
)

_TRANSIENT_SQLITE_MARKERS = (
    "database is busy",
    "database is locked",
    "database table is locked",
)

_INFRASTRUCTURE_MARKERS = (
    "timed out",
    "timeout",
    "connection error",
    "connection reset",
    "connection aborted",
    "remote disconnected",
    "temporary provider failure",
)


def is_content_safety_error(value: object) -> bool:
    message = str(value).lower()
    return any(marker in message for marker in _CONTENT_SAFETY_MARKERS)


def _status_code(error: object) -> int | None:
    for candidate in (error, getattr(error, "response", None)):
        for name in ("status_code", "status", "code"):
            value = getattr(candidate, name, None)
            try:
                if value is not None:
                    return int(value)
            except (TypeError, ValueError):
                pass
    match = re.search(r"\b(?:http(?: error)?\s*)?(\d{3})\b", str(error), re.I)
    return int(match.group(1)) if match else None


def is_rate_limit_error(error: object) -> bool:
    message = str(error).lower()
    return _status_code(error) == 429 or any(
        marker in message for marker in _RATE_LIMIT_MARKERS
    )


def is_quota_or_auth_error(error: object) -> bool:
    status = _status_code(error)
    message = str(error).lower()
    return status in {401, 402, 403} or any(
        marker in message for marker in _QUOTA_OR_AUTH_MARKERS
    )


def is_transient_sqlite_error(error: object) -> bool:
    return isinstance(error, sqlite3.OperationalError) and any(
        marker in str(error).strip().lower() for marker in _TRANSIENT_SQLITE_MARKERS
    )


def is_fatal_collection_error(error: BaseException) -> bool:
    if is_quota_or_auth_error(error):
        return True
    if isinstance(error, sqlite3.Error) and not is_transient_sqlite_error(error):
        return True
    message = str(error).lower()
    if getattr(error, "stage_index", None) in {1, 3} and any(
        marker in message
        for marker in (
            "connection refused",
            "retrieval index is not loaded",
            "failed to load retrieval index",
        )
    ):
        return True
    return any(marker in message for marker in _FATAL_SHARED_STATE_MARKERS)


def pair_error_category(error: object) -> str:
    if is_content_safety_error(error):
        return "content_safety"
    if is_rate_limit_error(error):
        return "rate_limit"
    if is_transient_sqlite_error(error):
        return "transient_storage"
    status = _status_code(error)
    message = str(error).lower()
    if (
        status in {408, 409}
        or (status is not None and status >= 500)
        or any(marker in message for marker in _INFRASTRUCTURE_MARKERS)
        or isinstance(error, (ConnectionError, TimeoutError))
    ):
        return "infrastructure"
    return "pair_exception"


def pair_error_fingerprint(
    error: object,
    *,
    error_type: str = "",
    stage_index: int | None = None,
) -> str:
    message = str(error).strip().lower()
    message = re.sub(
        r"\b[0-9a-f]{8}-[0-9a-f-]{27,}\b", "<request-id>", message, flags=re.I
    )
    message = re.sub(r"\s+", " ", message)
    canonical = f"{error_type}|{stage_index}|{message}"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "is_content_safety_error",
    "is_fatal_collection_error",
    "is_quota_or_auth_error",
    "is_rate_limit_error",
    "is_transient_sqlite_error",
    "pair_error_category",
    "pair_error_fingerprint",
]
