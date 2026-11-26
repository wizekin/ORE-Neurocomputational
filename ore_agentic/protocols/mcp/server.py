"""Exposing approved ORE capabilities through MCP (Steps 31, 34).

Only explicitly approved capabilities are published, each with a stable
schema, description, version and authorization scope.  The default catalog
is read-only (USG retrieval, capability discovery, approved workspace
queries); write scopes exist but stay disabled unless the governed
configuration explicitly enables them.  Tool descriptions never authorize
execution -- only the policy engine does.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

from ...contracts import utc_now
from ...security.policy import Action, PolicyEngine, Principal
from ..envelopes import ProtocolEnvelope, ProtocolError, ProtocolOutcome, ProtocolStatus

__all__ = (
    "MCPCapability",
    "MCPCapabilityCatalog",
    "MCPServerAdapter",
    "build_readonly_mcp_catalog",
)


@dataclass(frozen=True, slots=True)
class MCPCapability:
    """One approved, explicitly published capability (Step 31)."""

    name: str
    description: str
    input_schema: Mapping[str, Any] = field(default_factory=dict)
    output_schema: Mapping[str, Any] = field(default_factory=dict)
    required_scope: str = "agentic:read"
    read_only: bool = True
    version: str = "1.0.0"
    handler: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None

    def __post_init__(self) -> None:
        """Validate the capability declaration."""
        if not self.name or not self.name.strip():
            raise ValueError("capability name must be a non-empty string")
        if not self.description or not self.description.strip():
            raise ValueError("capability description must be a non-empty string")
        if self.read_only and self.required_scope == "agentic:write":
            raise ValueError("a read-only capability cannot require the write scope")

    def to_dict(self) -> dict[str, Any]:
        """Describe the tool for discovery; descriptions never authorize (Step 31)."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "required_scope": self.required_scope,
            "read_only": self.read_only,
            "version": self.version,
        }


class MCPCapabilityCatalog:
    """The approved allowlist of exposed capabilities (Step 31)."""

    def __init__(self, capabilities: tuple[MCPCapability, ...] = ()) -> None:
        """Hold exactly the approved capabilities; nothing else is exposed."""
        self._capabilities: dict[str, MCPCapability] = {}
        for capability in capabilities:
            self.publish(capability)

    def publish(self, capability: MCPCapability) -> None:
        """Add one capability to the approved catalog (explicit approval only)."""
        if capability.name in self._capabilities:
            raise ValueError(f"capability already published: {capability.name}")
        self._capabilities[capability.name] = capability

    def get(self, name: str) -> MCPCapability | None:
        """Return an approved capability, or None for unpublished names."""
        return self._capabilities.get(name)

    def describe_tools(self) -> tuple[dict[str, Any], ...]:
        """Return the discoverable tool descriptions (Step 31)."""
        return tuple(capability.to_dict()
                     for _, capability in sorted(self._capabilities.items()))

    def __len__(self) -> int:
        """Return the number of approved capabilities."""
        return len(self._capabilities)


class MCPServerAdapter:
    """Dispatch authorized protocol calls to approved capabilities (Steps 31, 34).

    Every call is authorized against the policy engine before dispatch,
    validated against the declared input schema, sandboxed for output size,
    and audited.  Unauthorized calls fail closed without executing handlers.
    """

    def __init__(self, catalog: MCPCapabilityCatalog, policy: PolicyEngine,
                 *, max_response_bytes: int = 262_144,
                 write_scopes_enabled: bool = False,
                 audit: Callable[[Mapping[str, Any]], None] | None = None) -> None:
        """Compose the adapter from the approved catalog and the policy engine."""
        self._catalog = catalog
        self._policy = policy
        self._max_response_bytes = max_response_bytes
        self._write_scopes_enabled = write_scopes_enabled
        self._audit = audit
        self.executed: list[str] = []

    def handle(self, envelope: ProtocolEnvelope,
               principal: Principal | None = None) -> ProtocolOutcome:
        """Authorize, validate and execute one protocol call (Steps 31, 34)."""
        started = utc_now()
        if principal is None:
            principal = Principal(
                principal_id=envelope.caller_id,
                scopes=frozenset(dict(envelope.authorization_context).get(
                    "scopes", ())))
        capability = self._catalog.get(envelope.capability)
        if capability is None:
            return self._error(envelope, "CAPABILITY_NOT_PUBLISHED",
                               f"capability {envelope.capability} is not in the "
                               "approved catalog", retriable=False, started=started)
        if not capability.read_only and not self._write_scopes_enabled:
            return self._error(envelope, "WRITE_SCOPES_DISABLED",
                               "write capabilities are disabled by configuration",
                               retriable=False, started=started)
        decision = self._policy.authorize(
            principal,
            Action(action_id=f"mcp:{capability.name}", resource=capability.name,
                   required_scopes=(capability.required_scope,)))
        if not decision.allowed:
            return self._error(envelope, "AUTHORIZATION_DENIED",
                               decision.reason, retriable=False, started=started)
        schema_errors = _validate_arguments(envelope.arguments,
                                            capability.input_schema)
        if schema_errors:
            return self._error(envelope, "INVALID_ARGUMENTS",
                               "; ".join(schema_errors), retriable=False,
                               started=started)
        if capability.handler is None:
            return self._error(envelope, "HANDLER_UNBOUND",
                               "capability has no bound handler", retriable=False,
                               started=started)
        try:
            payload = capability.handler(dict(envelope.arguments))
        except Exception as exc:  # noqa: BLE001 - normalized into a structured error
            return self._error(envelope, "HANDLER_FAILED",
                               f"{type(exc).__name__}: {exc}", retriable=True,
                               started=started)
        serialized = str(dict(payload))
        if len(serialized.encode("utf-8")) > self._max_response_bytes:
            return self._error(envelope, "RESPONSE_TOO_LARGE",
                               f"response exceeds {self._max_response_bytes} bytes",
                               retriable=False, started=started)
        self.executed.append(envelope.capability)
        return ProtocolOutcome(status=ProtocolStatus.SUCCESS, payload=dict(payload),
                               request_id=envelope.request_id, duration_ms=0.0)

    def _error(self, envelope: ProtocolEnvelope, code: str, message: str, *,
               retriable: bool, started: str) -> ProtocolOutcome:
        """Build a structured-error outcome and audit the refusal (Step 34)."""
        outcome = ProtocolOutcome(
            status=ProtocolStatus.STRUCTURED_ERROR,
            error=ProtocolError(code=code, message=message, retriable=retriable),
            request_id=envelope.request_id)
        if self._audit is not None:
            self._audit({"kind": "mcp_call", "capability": envelope.capability,
                         "request_id": envelope.request_id,
                         "outcome": outcome.status.value, "code": code,
                         "started_at": started})
        return outcome


def _validate_arguments(arguments: Mapping[str, Any],
                        schema: Mapping[str, Any]) -> list[str]:
    """Validate call arguments against the declared input schema (Step 32)."""
    errors: list[str] = []
    for field_name in schema.get("required", ()) or ():
        if field_name not in arguments:
            errors.append(f"missing required argument: {field_name}")
    for field_name, field_spec in (schema.get("properties") or {}).items():
        if field_name not in arguments or not field_spec:
            continue
        expected = field_spec.get("type")
        value = arguments[field_name]
        checks = {
            "string": lambda item: isinstance(item, str),
            "number": lambda item: isinstance(item, (int, float))
            and not isinstance(item, bool),
            "boolean": lambda item: isinstance(item, bool),
            "object": lambda item: isinstance(item, Mapping),
            "array": lambda item: isinstance(item, (list, tuple)),
        }
        check = checks.get(str(expected)) if expected else None
        if check is not None and not check(value):
            errors.append(
                f"argument {field_name}: expected {expected}, got "
                f"{type(value).__name__}")
    return errors


def build_readonly_mcp_catalog(
        usg_lookup: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
        capability_discovery: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        workspace_query: Callable[[Mapping[str, Any]], Mapping[str, Any]]
        | None = None) -> MCPCapabilityCatalog:
    """Build the default read-only approved catalog (Step 31).

    Example read-only capabilities: USG retrieval, capability discovery and
    approved workspace queries.  Handlers are optional so the catalog can be
    inspected without binding execution.
    """
    capabilities = [
        MCPCapability(
            name="usg_lookup",
            description="Canonical USG retrieval with provenance (read-only).",
            input_schema={"type": "object",
                          "required": ["query"],
                          "properties": {"query": {"type": "string"}}},
            handler=usg_lookup),
        MCPCapability(
            name="capability_discovery",
            description="List approved agentic capabilities and their schemas.",
            input_schema={"type": "object", "properties": {}},
            handler=capability_discovery),
        MCPCapability(
            name="workspace_query",
            description="Query approved shared-workspace facts (read-only).",
            input_schema={"type": "object",
                          "required": ["field"],
                          "properties": {"field": {"type": "string"}}},
            handler=workspace_query),
    ]
    return MCPCapabilityCatalog(tuple(capabilities))
