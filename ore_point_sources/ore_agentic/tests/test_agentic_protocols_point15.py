"""Acceptance tests for MCP and A2A protocol interoperability (Point 15).

Covers specification acceptance tests 31-33: MCP capability availability
under authorization, malformed/oversized external response rejection, and
external A2A results remaining untrusted until locally validated.
"""
from __future__ import annotations

import pytest

from ore_agentic import (
    MCPCapability,
    MCPCapabilityCatalog,
    MCPServerAdapter,
    PolicyEngine,
    PolicyRule,
    ProtocolEnvelope,
    ProtocolStatus,
    build_readonly_mcp_catalog,
    normalize_response,
)
from ore_agentic.protocols.mcp.client import (
    CapabilityNotAllowedError,
    MalformedResponseError,
    MCPClientAdapter,
)
from ore_agentic import (
    A2AAgentCard,
    A2AMessage,
    A2AMessageKind,
    A2AParticipant,
)


# ------------------------------------------------------------- MCP server (31)

def test_mcp_tool_is_unavailable_without_authorization():
    """Acceptance 31: MCP tool is unavailable without authorization."""
    catalog = build_readonly_mcp_catalog(usg_lookup=lambda args: {"items": []})
    policy = PolicyEngine(())  # default deny, no allow rules
    adapter = MCPServerAdapter(catalog, policy)
    envelope = ProtocolEnvelope.create(
        caller_id="external-caller", capability="usg_lookup",
        schema_version="1.0.0", arguments={"query": "attention"},
        authorization_context={"scopes": ()})
    outcome = adapter.handle(envelope)
    assert outcome.status is ProtocolStatus.STRUCTURED_ERROR
    assert outcome.error.code == "AUTHORIZATION_DENIED"
    assert adapter.executed == []  # the handler never ran
    # With an explicit allow rule and the required scope, the call succeeds.
    allowed_policy = PolicyEngine((
        PolicyRule(principal_id="external-caller", action_id="mcp:usg_lookup",
                   effect="allow", priority=1, description="read-only grant"),))
    adapter = MCPServerAdapter(catalog, allowed_policy)
    scoped = ProtocolEnvelope.create(
        caller_id="external-caller", capability="usg_lookup",
        schema_version="1.0.0", arguments={"query": "attention"},
        authorization_context={"scopes": ("agentic:read",)})
    outcome = adapter.handle(scoped)
    assert outcome.status is ProtocolStatus.SUCCESS
    assert adapter.executed == ["usg_lookup"]


def test_mcp_catalog_publishes_only_approved_readonly_capabilities():
    """Acceptance 31b: only explicitly approved capabilities are exposed (Step 31)."""
    catalog = build_readonly_mcp_catalog()
    names = {tool["name"] for tool in catalog.describe_tools()}
    assert names == {"usg_lookup", "capability_discovery", "workspace_query"}
    for tool in catalog.describe_tools():
        assert tool["read_only"] is True  # read/write separation
        assert tool["required_scope"] == "agentic:read"
    assert catalog.get("undeclared_tool") is None  # unpublished names fail closed
    with pytest.raises(ValueError):
        MCPCapabilityCatalog((MCPCapability(
            name="evil", description="write scope on read-only",
            required_scope="agentic:write"),))  # read-only cannot demand write
    with pytest.raises(ValueError):
        catalog.publish(MCPCapability(name="usg_lookup",
                                      description="duplicate"))


def test_mcp_server_validates_arguments_and_response_size():
    """MCP calls validate arguments and cap response size (Steps 31-32, 34)."""
    catalog = MCPCapabilityCatalog((MCPCapability(
        name="echo", description="echo the argument",
        input_schema={"type": "object", "required": ["value"],
                      "properties": {"value": {"type": "string"}}},
        handler=lambda args: {"value": args["value"]}),))
    policy = PolicyEngine((PolicyRule(
        principal_id="*", action_id="mcp:echo", effect="allow"),))
    adapter = MCPServerAdapter(catalog, policy, max_response_bytes=64)
    bad_args = ProtocolEnvelope.create(
        "caller", "echo", "1.0.0", arguments={"wrong": 1},
        authorization_context={"scopes": ("agentic:read",)})
    outcome = adapter.handle(bad_args)
    assert outcome.error.code == "INVALID_ARGUMENTS"
    oversized = ProtocolEnvelope.create(
        "caller", "echo", "1.0.0", arguments={"value": "x" * 200},
        authorization_context={"scopes": ("agentic:read",)})
    outcome = adapter.handle(oversized)
    assert outcome.error.code == "RESPONSE_TOO_LARGE"
    write_disabled = MCPCapabilityCatalog((MCPCapability(
        name="write_thing", description="a write capability",
        read_only=False, required_scope="agentic:write",
        handler=lambda args: {}),))
    adapter = MCPServerAdapter(write_disabled, policy)
    outcome = adapter.handle(ProtocolEnvelope.create("caller", "write_thing",
                                                     "1.0.0"))
    assert outcome.error.code == "WRITE_SCOPES_DISABLED"


# ------------------------------------------------------------- MCP client (32)

class FakeEndpoint:
    """An untrusted external MCP server endpoint for tests (Step 32)."""

    def __init__(self, responses: list) -> None:
        self.identity = "external-mcp: test"
        self._responses = list(responses)
        self.received: list[dict] = []

    def send(self, request: dict) -> dict:
        """Return the next scripted raw response."""
        self.received.append(request)
        return self._responses.pop(0)


def test_malformed_or_oversized_mcp_response_is_rejected():
    """Acceptance 32: malformed/oversized MCP response is rejected."""
    with pytest.raises(MalformedResponseError):
        normalize_response(["not", "an", "object"])
    with pytest.raises(MalformedResponseError):
        normalize_response({"status": "bogus"})
    with pytest.raises(MalformedResponseError):
        normalize_response({"status": "success", "payload": "not-an-object"})
    with pytest.raises(MalformedResponseError):
        normalize_response({"status": "success", "payload": {"x": "y" * 500}},
                           max_response_bytes=64)
    # Unknown outcome is a first-class normalized state (Step 33).
    outcome = normalize_response({"status": "unknown"})
    assert outcome.status is ProtocolStatus.UNKNOWN_OUTCOME
    assert not outcome.ok
    timeout = normalize_response({"status": "timeout"})
    assert timeout.status is ProtocolStatus.TIMEOUT
    # The client adapter wraps malformed endpoints into structured errors.
    endpoint = FakeEndpoint([{"status": "garbage"}])
    adapter = MCPClientAdapter(endpoint, ("usg_lookup",))
    _, outcome = adapter.call("usg_lookup", {"query": "x"})
    assert outcome.status is ProtocolStatus.STRUCTURED_ERROR
    assert outcome.error.code == "MALFORMED_RESPONSE"


def test_external_mcp_client_enforces_allowlist():
    """The client allowlist is enforced for external capabilities (Step 32)."""
    endpoint = FakeEndpoint([{"status": "success", "payload": {"answer": 42}}])
    adapter = MCPClientAdapter(endpoint, ("approved_tool",))
    with pytest.raises(CapabilityNotAllowedError):
        adapter.call("unapproved_tool", {})
    envelope, outcome = adapter.call("approved_tool", {"q": 1})
    assert outcome.status is ProtocolStatus.SUCCESS
    assert outcome.payload == {"answer": 42}
    sent = endpoint.received[0]
    # The wire request is a typed envelope (Step 33).
    assert sent["request_id"] == envelope.request_id
    assert sent["capability"] == "approved_tool"
    assert sent["schema_version"] == "1.0.0"
    assert sent["idempotency_key"]
    assert "credentials" not in str(sent).lower()


# ----------------------------------------------------------------- A2A (33)

def test_external_a2a_result_remains_untrusted_until_validated():
    """Acceptance 33: external A2A result remains untrusted until validated."""
    participant = A2AParticipant(
        A2AAgentCard(agent_id="ore-agent", name="ORE", capabilities=("local",)),
        enabled=True,
        expected_schema={"required": ("answer",)},
        policy_check=lambda payload: payload.get("answer") != "unverified")
    peer = A2AAgentCard(agent_id="peer-agent", name="Peer",
                        capabilities=("external_analysis",), trust_level=0.8)
    delegation = participant.delegate(peer, "external_analysis",
                                      "analyze the dataset")
    message = A2AMessage(
        message_id="m-1", agent_id="peer-agent",
        delegation_id=delegation.delegation_id, sequence_no=0,
        kind=A2AMessageKind.RESULT,
        payload={"answer": "unverified"})
    assert participant.receive(message) == "stored_untrusted"
    record = participant.results()[0]
    assert not record.trusted  # untrusted proposal until locally validated
    checked = participant.validate_result("m-1")  # local policy rejects
    assert not checked.validated and not checked.trusted
    good = A2AMessage(
        message_id="m-2", agent_id="peer-agent",
        delegation_id=delegation.delegation_id, sequence_no=1,
        kind=A2AMessageKind.RESULT,
        payload={"answer": "verified result"})
    assert participant.receive(good) == "stored_untrusted"
    validated = participant.validate_result("m-2")
    assert validated.trusted
    assert validated.validated


def test_a2a_duplicate_and_out_of_order_messages_are_handled():
    """Duplicate, delayed and out-of-order A2A messages are handled (Step 36)."""
    participant = A2AParticipant(
        A2AAgentCard(agent_id="ore-agent", name="ORE", capabilities=("c",)),
        enabled=True)
    peer = A2AAgentCard(agent_id="peer", name="Peer", capabilities=("c",))
    delegation = participant.delegate(peer, "c", "instruction")
    first = A2AMessage(message_id="m-1", agent_id="peer",
                       delegation_id=delegation.delegation_id, sequence_no=0,
                       kind=A2AMessageKind.RESULT, payload={"a": 1})
    assert participant.receive(first) == "stored_untrusted"
    assert participant.receive(first) == "duplicate"  # dropped
    assert len(participant.results()) == 1
    skipped = A2AMessage(message_id="m-3", agent_id="peer",
                         delegation_id=delegation.delegation_id, sequence_no=2,
                         kind=A2AMessageKind.RESULT, payload={"a": 3})
    assert participant.receive(skipped) == "out_of_order"  # buffered
    second = A2AMessage(message_id="m-2", agent_id="peer",
                        delegation_id=delegation.delegation_id, sequence_no=1,
                        kind=A2AMessageKind.RESULT, payload={"a": 2})
    assert participant.receive(second) == "stored_untrusted"
    wrong_agent = A2AMessage(message_id="m-9", agent_id="other-agent",
                             delegation_id=delegation.delegation_id,
                             sequence_no=3, kind=A2AMessageKind.RESULT,
                             payload={})
    assert participant.receive(wrong_agent) == "wrong_agent"  # identity check


def test_a2a_contradictory_results_are_both_preserved():
    """Contradictory external results are preserved and flagged (Step 36)."""
    participant = A2AParticipant(
        A2AAgentCard(agent_id="ore-agent", name="ORE", capabilities=("c",)),
        enabled=True, expected_schema={})
    peer = A2AAgentCard(agent_id="peer", name="Peer", capabilities=("c",))
    delegation = participant.delegate(peer, "c", "instruction")
    first = A2AMessage(message_id="m-1", agent_id="peer",
                       delegation_id=delegation.delegation_id, sequence_no=0,
                       kind=A2AMessageKind.RESULT, payload={"verdict": "yes"})
    second = A2AMessage(message_id="m-2", agent_id="peer",
                        delegation_id=delegation.delegation_id, sequence_no=1,
                        kind=A2AMessageKind.RESULT, payload={"verdict": "no"})
    participant.receive(first)
    participant.receive(second)
    records = participant.results(delegation.delegation_id)
    assert len(records) == 2  # contradiction preserved, never averaged
    assert records[0].contradiction_with == ("m-2",)
    assert records[1].contradiction_with == ("m-1",)


def test_a2a_requires_justification_and_declared_capabilities():
    """A2A is used only for justified, independently operated agents (Step 35)."""
    disabled = A2AParticipant(
        A2AAgentCard(agent_id="ore-agent", name="ORE", capabilities=("c",)))
    peer = A2AAgentCard(agent_id="peer", name="Peer", capabilities=("c",))
    with pytest.raises(PermissionError):
        disabled.delegate(peer, "c", "instruction")  # not justified
    participant = A2AParticipant(
        A2AAgentCard(agent_id="ore-agent", name="ORE", capabilities=("c",)),
        enabled=True)
    untrusted = A2AAgentCard(agent_id="zero-trust", name="Zero",
                             capabilities=("c",), trust_level=0.0)
    with pytest.raises(PermissionError):
        participant.delegate(untrusted, "c", "instruction")
    with pytest.raises(PermissionError):
        participant.delegate(peer, "undeclared_capability", "instruction")


def test_protocol_envelope_carries_identity_and_cause():
    """Typed envelopes carry identity, deadlines and trace propagation (Step 33)."""
    envelope = ProtocolEnvelope.create(
        caller_id="caller-1", capability="usg_lookup", schema_version="1.0.0",
        arguments={"query": "x"}, authorization_context={"scopes": ("agentic:read",)},
        deadline_at="2026-01-01T00:00:00Z", trace_id="trace-1")
    serialized = envelope.to_dict()
    assert serialized["caller_id"] == "caller-1"
    assert serialized["capability"] == "usg_lookup"
    assert serialized["deadline_at"] == "2026-01-01T00:00:00Z"
    assert serialized["idempotency_key"]
    assert serialized["trace_id"] == "trace-1"
    assert serialized["authorization_context"]["scopes"] == ("agentic:read",)
    with pytest.raises(ValueError):
        ProtocolEnvelope.create(caller_id="", capability="x", schema_version="1")
