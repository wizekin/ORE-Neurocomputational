"""Acceptance tests: identity, authentication, authorization and transport (Phases A-B, step 15)."""
from __future__ import annotations

import pytest

from ore_security.contracts import AuthorizationRequest, PrincipalKind
from ore_security.exceptions import (
    AuthenticationError,
    ConfigurationRejectedError,
    CredentialExpiredError,
    CredentialRevokedError,
    PeerValidationError,
    PolicyEngineFailure,
    ReplayError,
    TransportError,
)
from ore_security.identity import IdentityService, RequestAuthenticator
from ore_security.policy import DataClassification, PolicyEngine, PolicyRule, ResourceDescriptor, threat_model_catalog
from ore_security.transport import SecureChannelEndpoint, TransportPolicy


class FrozenClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now


@pytest.fixture()
def clock():
    return FrozenClock()


@pytest.fixture()
def identity(clock):
    service = IdentityService(clock=clock)
    service.register_identity("svc-reader", PrincipalKind.SERVICE, roles=("READER",))
    service.register_identity("svc-analyst", PrincipalKind.SERVICE, roles=("ANALYST",))
    service.register_identity("svc-writer", PrincipalKind.SERVICE, roles=("WRITER",))
    return service


@pytest.fixture()
def engine(clock):
    policy = PolicyEngine(clock=clock)
    policy.resource_registry.register(ResourceDescriptor(
        "usg://records", DataClassification.CONFIDENTIAL, "usg-owner",
        allowed_purposes=frozenset({"record-ingest", "record-read"}),
    ))
    policy.add_rule(PolicyRule(
        "allow-analyst-read", "role:ANALYST", frozenset({"usg.record.read"}),
        ("usg://records",), DataClassification.CONFIDENTIAL,
        required_scopes=frozenset({"usg:read"}),
    ))
    return policy


def _request(principal_id: str, action: str = "usg.record.read",
             classification: str = "CONFIDENTIAL") -> AuthorizationRequest:
    return AuthorizationRequest(
        principal_id=principal_id, action=action, resource="usg://records/entry-1",
        purpose="record-read", data_classification=classification,
    )


def test_unauthenticated_service_request_is_rejected(engine, identity, clock):
    """Acceptance: unauthenticated service request is rejected."""
    token, _ = identity.issue_credential("svc-analyst", "svc-keeper", ("usg:read",), ttl=600)
    principal = identity.authenticate_token(token, expected_audience="svc-keeper")
    unauthenticated = type(principal)(
        principal_id="svc-analyst", kind=PrincipalKind.SERVICE, issuer="x",
        credential_id="none", authenticated=False,
    )
    decision = engine.decide(_request("svc-analyst"), unauthenticated)
    assert not decision.allowed
    assert decision.reason_code == "UNAUTHENTICATED"
    assert not decision.identity_ok


def test_authenticated_identity_without_required_scope_is_denied(engine, identity, clock):
    """Acceptance: authenticated identity without required scope is denied."""
    token, _ = identity.issue_credential("svc-analyst", "svc-keeper", ("other:scope",), ttl=600)
    principal = identity.authenticate_token(token, expected_audience="svc-keeper")
    decision = engine.decide(_request("svc-analyst"), principal)
    assert not decision.allowed
    assert "SCOPE" in decision.reason_code


def test_authenticated_identity_with_scope_is_allowed(engine, identity, clock):
    """Counterpart: correct role plus scope allows the operation."""
    token, _ = identity.issue_credential("svc-analyst", "svc-keeper", ("usg:read",), ttl=600)
    principal = identity.authenticate_token(token, expected_audience="svc-keeper")
    decision = engine.decide(_request("svc-analyst"), principal)
    assert decision.allowed
    assert decision.identity_ok and decision.policy_ok
    assert decision.classification_ok and decision.context_ok


def test_expired_credential_is_rejected(engine, identity, clock):
    """Acceptance: expired credentials are rejected."""
    token, _ = identity.issue_credential("svc-analyst", "svc-keeper", ("usg:read",), ttl=600)
    clock.now += 601.0
    with pytest.raises(CredentialExpiredError):
        identity.authenticate_token(token, expected_audience="svc-keeper")


def test_revoked_credential_is_rejected(identity):
    """Acceptance: revoked credentials are rejected."""
    token, record = identity.issue_credential("svc-analyst", "svc-keeper", ("usg:read",), ttl=600)
    identity.revoke_credential(record.credential_id, reason="rotation")
    with pytest.raises(CredentialRevokedError):
        identity.authenticate_token(token, expected_audience="svc-keeper")


def test_revoked_identity_blocks_all_credentials(identity):
    """Revoking an identity revokes every credential issued to it."""
    token, _ = identity.issue_credential("svc-analyst", "svc-keeper", ("usg:read",), ttl=600)
    identity.revoke_identity("svc-analyst", reason="compromise")
    with pytest.raises((CredentialRevokedError, AuthenticationError)):
        identity.authenticate_token(token, expected_audience="svc-keeper")


def test_credential_audience_binding_is_enforced(identity):
    """A credential presented to the wrong audience service is rejected."""
    token, _ = identity.issue_credential("svc-analyst", "svc-keeper", ("usg:read",), ttl=600)
    with pytest.raises(AuthenticationError, match="audience"):
        identity.authenticate_token(token, expected_audience="some-other-service")


def test_service_cannot_access_resources_outside_authorization(engine, identity):
    """Acceptance: service cannot access resources outside explicit authorization."""
    token, _ = identity.issue_credential("svc-analyst", "svc-keeper", ("usg:read",), ttl=600)
    principal = identity.authenticate_token(token, expected_audience="svc-keeper")
    request = AuthorizationRequest(
        principal_id="svc-analyst", action="usg.record.read",
        resource="usg://unregistered/other", purpose="record-read",
        data_classification="CONFIDENTIAL",
    )
    decision = engine.decide(request, principal)
    assert not decision.allowed
    assert decision.reason_code in {"NO_MATCHING_POLICY", "RESOURCE_NOT_CLASSIFIED"}


def test_classification_exceeding_rule_is_denied(engine, identity):
    """Requests above the rule's classification ceiling are denied (D(r) fails)."""
    token, _ = identity.issue_credential("svc-analyst", "svc-keeper", ("usg:read",), ttl=600)
    principal = identity.authenticate_token(token, expected_audience="svc-keeper")
    decision = engine.decide(_request("svc-analyst", classification="HIGHLY_SENSITIVE"),
                             principal)
    assert not decision.allowed
    assert "CLASSIFICATION" in decision.reason_code


def test_authorization_service_failure_does_not_silently_allow(engine, identity):
    """Acceptance: authorization-service failure does not silently allow protected access."""
    token, _ = identity.issue_credential("svc-analyst", "svc-keeper", ("usg:read",), ttl=600)
    principal = identity.authenticate_token(token, expected_audience="svc-keeper")
    engine.simulate_engine_failure(True)
    with pytest.raises(PolicyEngineFailure):
        engine.decide(_request("svc-analyst"), principal)
    engine.simulate_engine_failure(False)


def test_missing_classification_data_means_deny(engine, identity):
    """Unknown classification strings are denied rather than guessed."""
    token, _ = identity.issue_credential("svc-analyst", "svc-keeper", ("usg:read",), ttl=600)
    principal = identity.authenticate_token(token, expected_audience="svc-keeper")
    decision = engine.decide(_request("svc-analyst", classification="WHO-KNOWS"), principal)
    assert not decision.allowed
    assert "CLASSIFICATION_UNKNOWN" in decision.reason_code


def test_replay_of_accepted_request_is_rejected(identity, clock):
    """Acceptance: replay of an accepted sensitive request is rejected."""
    authenticator = RequestAuthenticator(identity, clock=clock)
    token, _ = identity.issue_credential("svc-reader", "svc-keeper", ("usg:read",), ttl=600)
    body = b'{"record": "entry-1", "action": "read"}'
    request = authenticator.build_authenticated_request(
        "svc-reader", method="GET", destination="usg://records/entry-1",
        body=body, token=token,
    )
    principal = authenticator.verify_request(
        request, expected_audience="svc-keeper", body=body,
    )
    assert principal.principal_id == "svc-reader"
    with pytest.raises(ReplayError):
        authenticator.verify_request(request, expected_audience="svc-keeper", body=body)


def test_request_substitution_by_body_swap_is_rejected(identity, clock):
    """Swapping the request body after signing is detected via the digest binding."""
    authenticator = RequestAuthenticator(identity, clock=clock)
    token, _ = identity.issue_credential("svc-reader", "svc-keeper", ("usg:read",), ttl=600)
    request = authenticator.build_authenticated_request(
        "svc-reader", method="GET", destination="usg://records/entry-1",
        body=b"original-body", token=token,
    )
    with pytest.raises(AuthenticationError, match="digest"):
        authenticator.verify_request(
            request, expected_audience="svc-keeper", body=b"swapped-body",
        )


def test_request_timestamp_outside_clock_skew_is_rejected(identity, clock):
    """Stale request timestamps beyond the allowed skew window are rejected."""
    authenticator = RequestAuthenticator(identity, clock=clock, max_clock_skew=30.0)
    token, _ = identity.issue_credential("svc-reader", "svc-keeper", ("usg:read",), ttl=600)
    request = authenticator.build_authenticated_request(
        "svc-reader", method="GET", destination="usg://records/entry-1",
        body=b"payload", token=token,
    )
    clock.now += 120.0
    with pytest.raises(AuthenticationError, match="skew"):
        authenticator.verify_request(request, expected_audience="svc-keeper", body=b"payload")


def test_secure_transport_validates_intended_peer_identity(identity, clock):
    """Acceptance: secure transport validates intended peer identity."""
    identity.register_identity("svc-imposter", PrincipalKind.SERVICE, roles=("PEER",))
    keeper_token, _ = identity.issue_credential("svc-writer", "svc-reader", ("chan",), ttl=600)
    peer_token, _ = identity.issue_credential("svc-reader", "svc-writer", ("chan",), ttl=600)
    imposter_token, _ = identity.issue_credential("svc-imposter", "svc-writer", ("chan",), ttl=600)

    keeper = SecureChannelEndpoint("svc-writer", identity, keeper_token, clock=clock)
    imposter = SecureChannelEndpoint("svc-imposter", identity, imposter_token, clock=clock)
    with pytest.raises(PeerValidationError):
        keeper.establish(imposter.handshake_material(),
                         expected_peer_service="svc-reader", local_is_initiator=True)

    peer = SecureChannelEndpoint("svc-reader", identity, peer_token, clock=clock)
    keeper.establish(peer.handshake_material(),
                     expected_peer_service="svc-reader", local_is_initiator=True)
    peer.establish(keeper.handshake_material(),
                   expected_peer_service="svc-writer", local_is_initiator=False)
    frame = peer.send(b"authenticated payload")
    assert keeper.receive(frame) == b"authenticated payload"
    with pytest.raises(TransportError):
        keeper.receive(frame)  # channel-level replay


def test_obsolete_protocol_and_cipher_configurations_are_rejected():
    """Acceptance: obsolete protocol/cipher configurations are rejected."""
    policy = TransportPolicy()
    with pytest.raises(ConfigurationRejectedError, match="obsolete"):
        policy.validate_connection_request("1.0")
    with pytest.raises(ConfigurationRejectedError, match="obsolete"):
        policy.validate_connection_request("1.1")
    with pytest.raises(ConfigurationRejectedError):
        policy.validate_connection_request("1.3", ("TLS_RSA_WITH_RC4_128_MD5",))
    with pytest.raises(ConfigurationRejectedError):
        policy.validate_connection_request("1.2", ("TLS_ECDHE_RSA_WITH_3DES_EDE_CBC_SHA",))
    assert policy.validate_connection_request("1.3", ("TLS_AES_256_GCM_SHA384",))["accepted"]


def test_short_lived_credentials_enforced(identity):
    """Credentials are bounded by the service maximum TTL and cannot exceed it."""
    with pytest.raises(ValueError):
        identity.issue_credential("svc-reader", "svc-keeper", ("usg:read",), ttl=10**9)
    _, record = identity.issue_credential("svc-reader", "svc-keeper", ("usg:read",), ttl=1.0)
    assert record.expires_at - record.issued_at == pytest.approx(1.0)


def test_threat_model_catalog_is_populated():
    """The step-2 threat catalog documents adversaries and mapped controls."""
    catalog = threat_model_catalog()
    assert len(catalog) >= 10
    assert all({"scenario", "capability", "controls"} <= set(item) for item in catalog)
