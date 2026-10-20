"""Semantic encoder defaults for the NL2SQL router."""

from __future__ import annotations

import os

DEFAULT_SEMANTIC_MODEL = os.getenv(
    "NL2SQL_SEMANTIC_MODEL",
    "models/gte-large-en-v1-5",
)

DEFAULT_SEMANTIC_TRUST_REMOTE_CODE = (
    os.getenv("NL2SQL_SEMANTIC_TRUST_REMOTE_CODE", "true").strip().lower()
    not in {"0", "false", "no", "off"}
)
