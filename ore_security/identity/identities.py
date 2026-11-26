"""Identity registry, short-lived scoped credentials, and authentication (steps 6-9).

The Identity and Credential Service assigns stable identities to ORE nodes, services,
modules, users and external agents, issues audience-bound Ed25519-signed credential
tokens with explicit scopes and expiry, and authenticates them with issuer, audience,
expiry and revocation checks. A valid identity never implies broad access: role
resolution here is separate from the policy engine's per-resource decision.
"""
from __future__ import annotations

import uuid
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ..canonical import canonical_bytes, digest_structure
from ..crypto import ed25519_sign, ed25519_verify, generate_ed25519_keypair, random_bytes
from ..exceptions import (
    AuthenticationError,
    CredentialExpiredError,
    CredentialRevokedError,
    ReplayError,
)
from ..contracts import Principal, PrincipalKind

Clock = Callable[[], float]

IDENTITY_ISSUER = "ore-security-identity-service"
DEFAULT_MAX_CREDENTIAL_TTL = 3600.0


@dataclass(frozen=True, slots=True)
class NodeIdentity:
    """One registered ORE component identity with status and role set."""

    identity_id: str
    kind: PrincipalKind
    roles: frozenset[str]
    created_at: float
    status: str = "ACTIVE"
    display_name: str = ""

    def __post_init__(self) -> None:
        if not self.identity_id.strip():
            raise ValueError("identity_id must be non-empty")
        if self.status not in {"ACTIVE", "SUSPENDED", "REVOKED"}:
            raise ValueError("status must be ACTIVE, SUSPENDED or REVOKED")


@dataclass(frozen=True, slots=True)
class CredentialRecord:
    """Metadata for one issued credential token (no signature material)."""

    credential_id: str
    subject: str
    audience: str
    scopes: frozenset[str]
    issued_at: float
    expires_at: float
    status: str = "ACTIVE"

    def __post_init__(self) -> None:
        if self.expires_at <= self.issued_at:
            raise ValueError("credential expiry must be after issuance")
        if self.status not in {"ACTIVE", "REVOKED"}:
            raise ValueError("credential status must be ACTIVE or REVOKED")


class IdentityService:
    """Issue and authenticate short-lived, scoped, revocable service credentials."""

    def __init__(self, *, clock: Clock = time.time,
                 max_credential_ttl: float = DEFAULT_MAX_CREDENTIAL_TTL) -> None:
        self.clock = clock
        self.max_credential_ttl = float(max_credential_ttl)
        if self.max_credential_ttl <= 0:
            raise ValueError("max_credential_ttl must be positive")
        self._identities: dict[str, NodeIdentity] = {}
        self._credentials: dict[str, CredentialRecord] = {}
        self._revoked_credentials: set[str] = set()
        self._subject_keys: dict[str, tuple[bytes, bytes]] = {}
        issuer_private, issuer_public = generate_ed25519_keypair()
        self._issuer_private = issuer_private
        self.issuer_public = issuer_public
        self._events: list[dict[str, Any]] = []

    # -- identity management -------------------------------------------------

    def register_identity(self, identity_id: str, kind: PrincipalKind | str,
                          roles: Mapping[str, Any] | tuple[str, ...] = (),
                          *, display_name: str = "") -> NodeIdentity:
        """Register one component identity and generate its subject keypair."""
        if identity_id in self._identities:
            raise ValueError(f"identity already registered: {identity_id}")
        resolved = PrincipalKind(kind) if isinstance(kind, str) else kind
        role_set = frozenset(roles)
        private, public = generate_ed25519_keypair()
        self._subject_keys[identity_id] = (private, public)
        identity = NodeIdentity(identity_id, resolved, role_set, self.clock(), display_name=display_name)
        self._identities[identity_id] = identity
        self._events.append({"event": "identity_registered", "identity_id": identity_id,
                             "kind": resolved.value, "at": self.clock()})
        return identity

    def get_identity(self, identity_id: str) -> NodeIdentity | None:
        return self._identities.get(identity_id)

    def public_key(self, identity_id: str) -> bytes:
        """Return the subject's public key for request-signature verification."""
        pair = self._subject_keys.get(identity_id)
        if pair is None:
            raise AuthenticationError(f"unknown identity: {identity_id}")
        return pair[1]

    def revoke_identity(self, identity_id: str, *, reason: str = "unspecified") -> None:
        """Revoke an identity and every credential issued to it."""
        identity = self._identities.get(identity_id)
        if identity is None:
            raise ValueError(f"unknown identity: {identity_id}")
        self._identities[identity_id] = NodeIdentity(
            identity.identity_id, identity.kind, identity.roles, identity.created_at,
            status="REVOKED", display_name=identity.display_name,
        )
        for record in self._credentials.values():
            if record.subject == identity_id and record.status == "ACTIVE":
                self._revoke_credential_record(record)
        self._events.append({"event": "identity_revoked", "identity_id": identity_id,
                             "reason": reason, "at": self.clock()})

    # -- credential lifecycle ------------------------------------------------

    def issue_credential(self, subject: str, audience: str, scopes: tuple[str, ...],
                         *, ttl: float | None = None) -> tuple[str, CredentialRecord]:
        """Issue a signed, audience-bound, short-lived credential token."""
        if subject not in self._identities:
            raise ValueError(f"unknown subject identity: {subject}")
        if not audience.strip():
            raise ValueError("audience must be non-empty")
        effective_ttl = self.max_credential_ttl if ttl is None else float(ttl)
        if effective_ttl <= 0 or effective_ttl > self.max_credential_ttl:
            raise ValueError(f"credential ttl must be in (0, {self.max_credential_ttl}] seconds")
        now = self.clock()
        credential_id = "cred-" + uuid.uuid4().hex
        payload = {
            "iss": IDENTITY_ISSUER,
            "sub": subject,
            "aud": audience,
            "scopes": sorted(scopes),
            "iat": now,
            "exp": now + effective_ttl,
            "jti": credential_id,
        }
        signature = ed25519_sign(self._issuer_private, canonical_bytes(payload))
        token = "ore-cred." + canonical_bytes(payload).hex() + "." + signature.hex()
        record = CredentialRecord(credential_id, subject, audience, frozenset(scopes),
                                  now, now + effective_ttl)
        self._credentials[credential_id] = record
        self._events.append({"event": "credential_issued", "credential_id": credential_id,
                             "subject": subject, "audience": audience, "scopes": sorted(scopes),
                             "expires_at": now + effective_ttl})
        return token, record

    def revoke_credential(self, credential_id: str, *, reason: str = "unspecified") -> None:
        """Revoke one credential by its identifier (jti)."""
        record = self._credentials.get(credential_id)
        if record is None:
            raise ValueError(f"unknown credential: {credential_id}")
        self._revoke_credential_record(record)
        self._events.append({"event": "credential_revoked", "credential_id": credential_id,
                             "reason": reason, "at": self.clock()})

    def _revoke_credential_record(self, record: CredentialRecord) -> None:
        self._credentials[record.credential_id] = CredentialRecord(
            record.credential_id, record.subject, record.audience, record.scopes,
            record.issued_at, record.expires_at, status="REVOKED",
        )
        self._revoked_credentials.add(record.credential_id)

    # -- authentication ------------------------------------------------------

    def authenticate_token(self, token: str, *, expected_audience: str) -> Principal:
        """Validate a credential token and return the authenticated principal.

        Checks, in order: token structure, issuer signature, audience, expiry with
        bounded clock skew, credential revocation, and subject identity status.
        """
        try:
            scheme, payload_hex, signature_hex = token.split(".")
        except (AttributeError, ValueError) as exc:
            raise AuthenticationError("malformed credential token") from exc
        if scheme != "ore-cred":
            raise AuthenticationError("unsupported credential token scheme")
        try:
            payload_bytes = bytes.fromhex(payload_hex)
            signature = bytes.fromhex(signature_hex)
        except ValueError as exc:
            raise AuthenticationError("credential token contains non-hexadecimal data") from exc
        import json
        try:
            payload = json.loads(payload_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuthenticationError("credential token payload is not canonical JSON") from exc
        if not ed25519_verify(self.issuer_public, payload_bytes, signature):
            raise AuthenticationError("credential token signature verification failed")
        if payload.get("iss") != IDENTITY_ISSUER:
            raise AuthenticationError("credential token issuer mismatch")
        if payload.get("aud") != expected_audience:
            raise AuthenticationError(
                f"credential audience mismatch: token issued for {payload.get('aud')!r}"
            )
        now = self.clock()
        if now >= float(payload["exp"]):
            raise CredentialExpiredError(
                f"credential expired at {float(payload['exp'])}; now {now}"
            )
        jti = payload.get("jti", "")
        record = self._credentials.get(jti)
        if record is None or jti in self._revoked_credentials or record.status != "ACTIVE":
            raise CredentialRevokedError(f"credential {jti} is revoked or unknown")
        identity = self._identities.get(payload["sub"])
        if identity is None or identity.status != "ACTIVE":
            raise CredentialRevokedError(f"subject identity {payload['sub']!r} is not active")
        return Principal(
            principal_id=payload["sub"],
            kind=identity.kind,
            issuer=IDENTITY_ISSUER,
            credential_id=jti,
            scopes=frozenset(payload.get("scopes", ())),
            roles=identity.roles,
            expires_at=float(payload["exp"]),
            authenticated=True,
        )

    def sign_request(self, subject: str, *, method: str, destination: str,
                     body_digest: str, nonce: str, timestamp: float) -> str:
        """Sign request-binding material with the subject key (step 11).

        The signature covers method, destination, body digest, nonce and a timestamp
        so an accepted request cannot be replayed or substituted for another call.
        """
        pair = self._subject_keys.get(subject)
        if pair is None:
            raise AuthenticationError(f"unknown identity: {subject}")
        material = {
            "method": method, "destination": destination, "body_digest": body_digest,
            "nonce": nonce, "timestamp": float(timestamp),
        }
        return ed25519_sign(pair[0], canonical_bytes(material)).hex()

    def service_events(self) -> tuple[dict[str, Any], ...]:
        """Return identity-service lifecycle events for audit ingestion."""
        return tuple(self._events)


class RequestAuthenticator:
    """Authenticate signed service requests with replay protection (steps 7, 11)."""

    def __init__(self, identity_service: IdentityService, *, clock: Clock = time.time,
                 max_clock_skew: float = 30.0) -> None:
        self.identity_service = identity_service
        self.clock = clock
        self.max_clock_skew = float(max_clock_skew)
        if self.max_clock_skew < 0:
            raise ValueError("max_clock_skew must be non-negative")
        self._seen_nonces: set[tuple[str, str]] = set()

    def build_authenticated_request(self, subject: str, *, method: str, destination: str,
                                    body: bytes | str, token: str) -> dict[str, str]:
        """Create the authenticated request envelope a caller sends to a service."""
        if isinstance(body, str):
            body = body.encode("utf-8")
        body_digest = digest_structure(body.hex())
        nonce = random_bytes(16).hex()
        timestamp = self.clock()
        signature = self.identity_service.sign_request(
            subject, method=method, destination=destination,
            body_digest=body_digest, nonce=nonce, timestamp=timestamp,
        )
        return {
            "token": token,
            "method": method,
            "destination": destination,
            "body_digest": body_digest,
            "nonce": nonce,
            "timestamp": repr(float(timestamp)),
            "request_signature": signature,
        }

    def verify_request(self, request: Mapping[str, str], *, expected_audience: str,
                       body: bytes | str) -> Principal:
        """Authenticate a request envelope or raise; replays are rejected."""
        if not isinstance(request, Mapping):
            raise AuthenticationError("request envelope must be a mapping")
        required = ("token", "method", "destination", "body_digest", "nonce",
                    "timestamp", "request_signature")
        for key in required:
            if key not in request or not str(request[key]).strip():
                raise AuthenticationError(f"request envelope is missing {key}")
        if isinstance(body, str):
            body = body.encode("utf-8")
        principal = self.identity_service.authenticate_token(
            request["token"], expected_audience=expected_audience,
        )
        expected_digest = digest_structure(body.hex())
        if request["body_digest"] != expected_digest:
            raise AuthenticationError("request body digest mismatch: request was substituted")
        try:
            timestamp = float(request["timestamp"])
        except (TypeError, ValueError) as exc:
            raise AuthenticationError("request timestamp is not a number") from exc
        if abs(self.clock() - timestamp) > self.max_clock_skew:
            raise AuthenticationError(
                "request timestamp outside the allowed clock-skew window"
            )
        material = {
            "method": request["method"],
            "destination": request["destination"],
            "body_digest": request["body_digest"],
            "nonce": request["nonce"],
            "timestamp": timestamp,
        }
        public_key = self.identity_service.public_key(principal.principal_id)
        if not ed25519_verify(public_key, canonical_bytes(material),
                              bytes.fromhex(request["request_signature"])):
            raise AuthenticationError("request signature verification failed")
        nonce_key = (principal.principal_id, request["nonce"])
        if nonce_key in self._seen_nonces:
            raise ReplayError(
                f"replayed request nonce detected for {principal.principal_id}"
            )
        self._seen_nonces.add(nonce_key)
        return principal
