"""Consuming external MCP servers through adapters (Steps 32-34).

External servers are untrusted providers: identity, protocol version,
schemas, scopes and output size are validated on every call, allowlists are
enforced client-side, and tool descriptions never authorize execution.
Responses are normalized to the typed outcome set, including the explicit
unknown-outcome state.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Protocol as TypingProtocol

from ...contracts import utc_now
from ..envelopes import ProtocolEnvelope, ProtocolError, ProtocolOutcome, ProtocolStatus

__all__ = (
    "CapabilityNotAllowedError",
    "MalformedResponseError",
    "MCPClientAdapter",
    "MCPEndpoint",
    "normalize_response",
)


class CapabilityNotAllowedError(PermissionError):
    """The capability is not on the client allowlist (Step 32)."""


class MalformedResponseError(ValueError):
    """The external response failed structure or size validation (Step 32)."""


class MCPEndpoint(TypingProtocol):
    """The untrusted external server boundary (Step 32)."""

    identity: str

    def send(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


def normalize_response(raw: Any, *, max_response_bytes: int = 262_144,
                       expected_schema: Mapping[str, Any] | None = None
                       ) -> ProtocolOutcome:
    """Normalize a raw external response into a typed outcome (Step 33).

    A response must be a mapping with a recognized ``status``; anything else
    -- wrong shape, missing status, oversized payload, or an explicit
    ``"unknown"`` status -- becomes a structured error or an unknown outcome,
    never a silent success.
    """
    if not isinstance(raw, Mapping):
        raise MalformedResponseError(
            f"response is not an object: {type(raw).__name__}")
    serialized = str(dict(raw))
    if len(serialized.encode("utf-8")) > max_response_bytes:
        raise MalformedResponseError(
            f"response exceeds {max_response_bytes} bytes")
    status = raw.get("status")
    if status not in ("success", "error", "timeout", "unknown"):
        raise MalformedResponseError(f"unrecognized response status: {status!r}")
    if status == "success":
        payload = raw.get("payload", {})
        if not isinstance(payload, Mapping):
            raise MalformedResponseError("success payload is not an object")
        if expected_schema:
            for field_name in expected_schema.get("required", ()) or ():
                if field_name not in payload:
                    raise MalformedResponseError(
                        f"response payload missing required field: {field_name}")
        return ProtocolOutcome(status=ProtocolStatus.SUCCESS, payload=dict(payload))
    if status == "timeout":
        return ProtocolOutcome(
            status=ProtocolStatus.TIMEOUT,
            error=ProtocolError(code="TIMEOUT", message="external call timed out",
                                retriable=True))
    if status == "unknown":
        return ProtocolOutcome(
            status=ProtocolStatus.UNKNOWN_OUTCOME,
            error=ProtocolError(
                code="UNKNOWN_OUTCOME",
                message="external server reported an indeterminate outcome",
                retriable=False))
    error = raw.get("error") or {}
    return ProtocolOutcome(
        status=ProtocolStatus.STRUCTURED_ERROR,
        error=ProtocolError(code=str(error.get("code", "EXTERNAL_ERROR")),
                            message=str(error.get("message", "external error")),
                            retriable=bool(error.get("retriable", False))))


class MCPClientAdapter:
    """A controlled client for one external MCP server (Steps 32-34)."""

    def __init__(self, endpoint: MCPEndpoint, allowlist: tuple[str, ...],
                 *, max_response_bytes: int = 262_144,
                 schema_version: str = "1.0.0",
                 audit: Callable[[Mapping[str, Any]], None] | None = None) -> None:
        """Bind the adapter to an endpoint and an explicit capability allowlist."""
        if not allowlist:
            raise ValueError("allowlist must not be empty (deny by construction)")
        self._endpoint = endpoint
        self._allowlist = tuple(allowlist)
        self._max_response_bytes = max_response_bytes
        self._schema_version = schema_version
        self._audit = audit
        self.calls: list[dict[str, Any]] = []

    @property
    def allowlist(self) -> tuple[str, ...]:
        """Return the enforced capability allowlist (Step 32)."""
        return self._allowlist

    def call(self, capability: str, arguments: Mapping[str, Any],
             caller_id: str = "ore_agentic",
             authorization_context: Mapping[str, Any] | None = None,
             expected_schema: Mapping[str, Any] | None = None,
             trace_id: str = "") -> tuple[ProtocolEnvelope, ProtocolOutcome]:
        """Place one allowlisted, enveloped, normalized call (Steps 32-33)."""
        if capability not in self._allowlist:
            raise CapabilityNotAllowedError(
                f"capability {capability!r} is not on the allowlist "
                f"{list(self._allowlist)}")
        envelope = ProtocolEnvelope.create(
            caller_id=caller_id, capability=capability,
            schema_version=self._schema_version, arguments=arguments,
            authorization_context=authorization_context, trace_id=trace_id)
        raw = self._endpoint.send(envelope.to_dict())
        try:
            outcome = normalize_response(raw, max_response_bytes=self._max_response_bytes,
                                         expected_schema=expected_schema)
        except MalformedResponseError as exc:
            outcome = ProtocolOutcome(
                status=ProtocolStatus.STRUCTURED_ERROR,
                error=ProtocolError(code="MALFORMED_RESPONSE", message=str(exc),
                                    retriable=False),
                raw_reference=envelope.request_id,
                request_id=envelope.request_id)
        record = {"request_id": envelope.request_id, "capability": capability,
                  "endpoint": getattr(self._endpoint, "identity", "unknown"),
                  "outcome": outcome.status.value, "at": utc_now()}
        self.calls.append(record)
        if self._audit is not None:
            self._audit({"kind": "mcp_external_call", **record})
        return envelope, outcome
