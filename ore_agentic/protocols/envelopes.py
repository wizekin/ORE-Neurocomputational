"""Typed protocol envelopes and normalized outcomes (Step 33).

Every protocol call -- MCP or A2A -- is wrapped in a typed envelope carrying
request ID, caller, capability, schema version, arguments, authorization
context, deadline, idempotency key and trace ID.  Responses are normalized to
success, structured error, timeout or unknown outcome; unknown outcomes are
first-class states, never silently converted to success or failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..contracts import new_id, utc_now

__all__ = (
    "ProtocolEnvelope",
    "ProtocolError",
    "ProtocolOutcome",
    "ProtocolStatus",
    "OUTCOME_STATUSES",
)


class ProtocolStatus(str, Enum):
    """The normalized outcome statuses of a protocol call (Step 33)."""

    SUCCESS = "success"
    STRUCTURED_ERROR = "structured_error"
    TIMEOUT = "timeout"
    UNKNOWN_OUTCOME = "unknown_outcome"


OUTCOME_STATUSES = tuple(status.value for status in ProtocolStatus)


@dataclass(frozen=True, slots=True)
class ProtocolEnvelope:
    """The typed envelope of one protocol call (Step 33)."""

    request_id: str
    caller_id: str
    capability: str
    schema_version: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    authorization_context: Mapping[str, Any] = field(default_factory=dict)
    deadline_at: str = ""
    idempotency_key: str = ""
    trace_id: str = ""
    issued_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate the envelope identity fields."""
        if not self.caller_id or not self.caller_id.strip():
            raise ValueError("caller_id must be a non-empty string")
        if not self.capability or not self.capability.strip():
            raise ValueError("capability must be a non-empty string")
        if not self.schema_version or not self.schema_version.strip():
            raise ValueError("schema_version must be a non-empty string")

    @classmethod
    def create(cls, caller_id: str, capability: str, schema_version: str,
               arguments: Mapping[str, Any] | None = None,
               authorization_context: Mapping[str, Any] | None = None,
               deadline_at: str = "", idempotency_key: str = "",
               trace_id: str = "") -> ProtocolEnvelope:
        """Build an envelope with a fresh request ID (Step 33)."""
        return cls(
            request_id=new_id("preq"), caller_id=caller_id, capability=capability,
            schema_version=schema_version, arguments=dict(arguments or {}),
            authorization_context=dict(authorization_context or {}),
            deadline_at=deadline_at,
            idempotency_key=idempotency_key or new_id("pkey"), trace_id=trace_id)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the envelope for the wire (credentials never included)."""
        return {
            "request_id": self.request_id,
            "caller_id": self.caller_id,
            "capability": self.capability,
            "schema_version": self.schema_version,
            "arguments": dict(self.arguments),
            "authorization_context": dict(self.authorization_context),
            "deadline_at": self.deadline_at,
            "idempotency_key": self.idempotency_key,
            "trace_id": self.trace_id,
            "issued_at": self.issued_at,
        }


@dataclass(frozen=True, slots=True)
class ProtocolError:
    """A structured error with a stable code (Step 33)."""

    code: str
    message: str
    retriable: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the error."""
        return {"code": self.code, "message": self.message,
                "retriable": self.retriable}


@dataclass(frozen=True, slots=True)
class ProtocolOutcome:
    """The normalized outcome of one protocol call (Step 33)."""

    status: ProtocolStatus
    payload: Mapping[str, Any] = field(default_factory=dict)
    error: ProtocolError | None = None
    raw_reference: str = ""
    duration_ms: float = 0.0
    request_id: str = ""

    @property
    def ok(self) -> bool:
        """Return whether the call succeeded."""
        return self.status is ProtocolStatus.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        """Serialize the outcome."""
        return {
            "status": self.status.value,
            "payload": dict(self.payload),
            "error": self.error.to_dict() if self.error else None,
            "raw_reference": self.raw_reference,
            "duration_ms": self.duration_ms,
            "request_id": self.request_id,
        }
