"""Typed protocol envelopes shared by MCP and A2A adapters (Step 33)."""
from __future__ import annotations

from .envelopes import (
    OUTCOME_STATUSES,
    ProtocolEnvelope,
    ProtocolError,
    ProtocolOutcome,
    ProtocolStatus,
)

__all__ = (
    "OUTCOME_STATUSES",
    "ProtocolEnvelope",
    "ProtocolError",
    "ProtocolOutcome",
    "ProtocolStatus",
)
