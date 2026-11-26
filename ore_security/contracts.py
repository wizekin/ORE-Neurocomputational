"""Typed trust-layer interfaces (spec Point 20, section 5.1).

Every interface validates its schema on construction, carries an explicit schema
version, and never transports secret key material. Ordinary task payloads and log
annotations receive these structures, never raw private keys.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

SCHEMA_VERSION = "1.0.0"


def utc_timestamp(value: float) -> str:
    """Render a POSIX timestamp as an ISO-8601 UTC instant."""
    return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat(timespec="milliseconds")


class PrincipalKind(str, Enum):
    """Kinds of identity that can hold ORE credentials."""

    NODE = "NODE"
    SERVICE = "SERVICE"
    MODULE = "MODULE"
    USER = "USER"
    AGENT = "AGENT"


class KeyPurpose(str, Enum):
    """Separation of duties for key material (spec step 13)."""

    SIGNING = "SIGNING"
    ENCRYPTION = "ENCRYPTION"
    KEY_ENCRYPTION = "KEY_ENCRYPTION"
    KEY_AGREEMENT = "KEY_AGREEMENT"
    AUTHENTICATION = "AUTHENTICATION"
    CREDENTIAL = "CREDENTIAL"


class KeyStatus(str, Enum):
    """Lifecycle states that govern what a key may be used for."""

    ACTIVE = "ACTIVE"
    ROTATED = "ROTATED"
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    COMPROMISED = "COMPROMISED"


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_positive(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or not float(value) > 0.0:
        raise ValueError(f"{name} must be a positive number")


@dataclass(frozen=True, slots=True)
class Principal:
    """Authenticated identity with issuer, credential reference, scopes and expiry."""

    principal_id: str
    kind: PrincipalKind | str
    issuer: str
    credential_id: str
    scopes: frozenset[str] = field(default_factory=frozenset)
    roles: frozenset[str] = field(default_factory=frozenset)
    expires_at: float = 0.0
    authenticated: bool = False

    def __post_init__(self) -> None:
        _require_nonempty("principal_id", self.principal_id)
        _require_nonempty("issuer", self.issuer)
        _require_nonempty("credential_id", self.credential_id)
        if isinstance(self.kind, str):
            PrincipalKind(self.kind)
        elif not isinstance(self.kind, PrincipalKind):
            raise TypeError("kind must be a PrincipalKind")
        if not isinstance(self.scopes, frozenset):
            raise TypeError("scopes must be a frozenset of scope strings")

    def has_scope(self, scope: str) -> bool:
        """Check one required scope on the authenticated credential."""
        return scope in self.scopes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "principal_id": self.principal_id,
            "kind": PrincipalKind(self.kind).value if isinstance(self.kind, PrincipalKind) else str(self.kind),
            "issuer": self.issuer,
            "credential_id": self.credential_id,
            "scopes": sorted(self.scopes),
            "roles": sorted(self.roles),
            "expires_at": self.expires_at,
            "authenticated": self.authenticated,
        }


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    """One protected-operation request: principal, action, resource, purpose, context."""

    principal_id: str
    action: str
    resource: str
    purpose: str
    context: Mapping[str, Any] = field(default_factory=dict)
    data_classification: str = "INTERNAL"

    def __post_init__(self) -> None:
        for name in ("principal_id", "action", "resource", "purpose"):
            _require_nonempty(name, getattr(self, name))
        _require_nonempty("data_classification", self.data_classification)
        if not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping")


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Default-deny authorization outcome with reason code and evaluation breakdown."""

    allowed: bool
    policy_version: str
    reason_code: str
    decision_expires_at: float = 0.0
    identity_ok: bool = False
    policy_ok: bool = False
    classification_ok: bool = False
    context_ok: bool = False
    matched_rule_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("policy_version", self.policy_version)
        _require_nonempty("reason_code", self.reason_code)
        if not isinstance(self.allowed, bool):
            raise TypeError("allowed must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "allowed": self.allowed,
            "policy_version": self.policy_version,
            "reason_code": self.reason_code,
            "decision_expires_at": self.decision_expires_at,
            "breakdown": {
                "identity": self.identity_ok,
                "policy": self.policy_ok,
                "classification": self.classification_ok,
                "context": self.context_ok,
            },
            "matched_rule_ids": list(self.matched_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class KeyDescriptor:
    """Public key metadata: identity, purpose, algorithm, status, validity, KMS reference.

    This structure intentionally has no field for key material; private bytes never
    leave the KMS through descriptors, task payloads, or logs.
    """

    key_id: str
    purpose: KeyPurpose | str
    algorithm: str
    status: KeyStatus | str
    not_before: float
    not_after: float
    kms_ref: str = "local-software-kms"
    public_commitment: str = ""
    successor_key_id: str | None = None
    predecessor_key_id: str | None = None

    FORBIDDEN_FIELDS = ("private_key", "material", "secret", "seed", "private_bytes")

    def __post_init__(self) -> None:
        _require_nonempty("key_id", self.key_id)
        _require_nonempty("algorithm", self.algorithm)
        if isinstance(self.purpose, str):
            KeyPurpose(self.purpose)
        if isinstance(self.status, str):
            KeyStatus(self.status)
        if self.not_after <= self.not_before:
            raise ValueError("not_after must be strictly after not_before")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "key_id": self.key_id,
            "purpose": KeyPurpose(self.purpose).value if isinstance(self.purpose, KeyPurpose) else str(self.purpose),
            "algorithm": self.algorithm,
            "status": KeyStatus(self.status).value if isinstance(self.status, KeyStatus) else str(self.status),
            "not_before": self.not_before,
            "not_after": self.not_after,
            "kms_ref": self.kms_ref,
            "public_commitment": self.public_commitment,
            "successor_key_id": self.successor_key_id,
            "predecessor_key_id": self.predecessor_key_id,
        }


@dataclass(frozen=True, slots=True)
class SignedArtifactManifest:
    """Canonical manifest binding a signature to digest, version, signer and intended use."""

    artifact_id: str
    artifact_version: str
    digest: str
    hash_algorithm: str
    canonicalization: str
    intended_use: str
    signer_key_id: str
    signature_envelope: Mapping[str, str]
    signed_at: float
    build_provenance: str = ""

    def __post_init__(self) -> None:
        for name in ("artifact_id", "artifact_version", "digest", "hash_algorithm",
                     "canonicalization", "intended_use", "signer_key_id"):
            _require_nonempty(name, getattr(self, name))
        if "algorithm" not in self.signature_envelope or "value" not in self.signature_envelope:
            raise ValueError("signature_envelope requires 'algorithm' and 'value' entries")

    def signed_payload(self) -> dict[str, Any]:
        """Canonical structure covered by the signature (spec step 20)."""
        return {
            "artifact_id": self.artifact_id,
            "artifact_version": self.artifact_version,
            "digest": self.digest,
            "hash_algorithm": self.hash_algorithm,
            "canonicalization": self.canonicalization,
            "intended_use": self.intended_use,
            "signer_key_id": self.signer_key_id,
            "build_provenance": self.build_provenance,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            **self.signed_payload(),
            "signature_envelope": dict(self.signature_envelope),
            "signed_at": self.signed_at,
        }


@dataclass(frozen=True, slots=True)
class IntegrityCheckpoint:
    """Signed commitment over an event range: chain head and Merkle root of the batch."""

    checkpoint_id: str
    range_start: int
    range_end: int
    chain_head: str
    merkle_root: str
    tree_version: str
    signature: str
    signer_key_id: str
    timestamp: float
    anchor_domain: str = "external-anchor-vault"

    def __post_init__(self) -> None:
        _require_nonempty("checkpoint_id", self.checkpoint_id)
        _require_nonempty("chain_head", self.chain_head)
        _require_nonempty("merkle_root", self.merkle_root)
        _require_nonempty("signer_key_id", self.signer_key_id)
        if self.range_start < 1 or self.range_end < self.range_start:
            raise ValueError("range must satisfy 1 <= range_start <= range_end")

    def signed_payload(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "range_start": self.range_start,
            "range_end": self.range_end,
            "chain_head": self.chain_head,
            "merkle_root": self.merkle_root,
            "tree_version": self.tree_version,
            "signer_key_id": self.signer_key_id,
            "anchor_domain": self.anchor_domain,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            **self.signed_payload(),
            "signature": self.signature,
            "timestamp": self.timestamp,
        }


@dataclass(frozen=True, slots=True)
class ProofStatement:
    """Public statement: relation id/version, public inputs, assumptions, task binding."""

    statement_id: str
    relation_id: str
    relation_version: str
    public_inputs: Mapping[str, Any]
    assumptions: str
    task_id: str
    task_nonce: str
    expires_at: float
    verifier_context: str = "ore-security-default-verifier"

    def __post_init__(self) -> None:
        _require_nonempty("statement_id", self.statement_id)
        _require_nonempty("relation_id", self.relation_id)
        _require_nonempty("relation_version", self.relation_version)
        _require_nonempty("task_id", self.task_id)
        _require_nonempty("task_nonce", self.task_nonce)
        _require_nonempty("assumptions", self.assumptions)
        if not isinstance(self.public_inputs, Mapping):
            raise TypeError("public_inputs must be a mapping")

    def binding_payload(self) -> dict[str, Any]:
        """Task-context binding included in the Fiat-Shamir transcript (spec step 29)."""
        return {
            "statement_id": self.statement_id,
            "relation_id": self.relation_id,
            "relation_version": self.relation_version,
            "task_id": self.task_id,
            "task_nonce": self.task_nonce,
            "expires_at": self.expires_at,
            "verifier_context": self.verifier_context,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            **self.binding_payload(),
            "public_inputs": dict(self.public_inputs),
            "assumptions": self.assumptions,
        }


@dataclass(frozen=True, slots=True)
class ProofVerificationResult:
    """Outcome of independent proof verification with recorded statement/verifier versions."""

    proof_ref: str
    statement_id: str
    relation_id: str
    relation_version: str
    verifier_version: str
    verified: bool
    reason: str
    checked_at: float = 0.0

    def __post_init__(self) -> None:
        for name in ("proof_ref", "statement_id", "relation_id", "relation_version", "verifier_version"):
            _require_nonempty(name, getattr(self, name))
        if not isinstance(self.verified, bool):
            raise TypeError("verified must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "proof_ref": self.proof_ref,
            "statement_id": self.statement_id,
            "relation_id": self.relation_id,
            "relation_version": self.relation_version,
            "verifier_version": self.verifier_version,
            "verified": self.verified,
            "reason": self.reason,
            "checked_at": self.checked_at,
        }


@dataclass(frozen=True, slots=True)
class PrivacyReleaseRequest:
    """DP release request: privacy unit, adjacency, query, sensitivity, mechanism, budget."""

    release_id: str
    privacy_unit: str
    adjacency_relation: str
    query: str
    sensitivity: float
    mechanism: str
    epsilon: float
    delta: float
    purpose: str
    dataset_version: str
    contribution_limit: float = 0.0
    calibration_reviewed: bool = False

    def __post_init__(self) -> None:
        _require_nonempty("release_id", self.release_id)
        _require_nonempty("privacy_unit", self.privacy_unit)
        _require_nonempty("adjacency_relation", self.adjacency_relation)
        _require_nonempty("query", self.query)
        _require_nonempty("mechanism", self.mechanism)
        _require_nonempty("purpose", self.purpose)
        _require_nonempty("dataset_version", self.dataset_version)
        _require_positive("sensitivity", self.sensitivity)
        _require_positive("epsilon", self.epsilon)
        if not 0.0 <= self.delta < 1.0:
            raise ValueError("delta must be in [0, 1)")
        if not self.calibration_reviewed:
            raise ValueError("calibration_reviewed must be true: noise is never chosen by intuition")


@dataclass(frozen=True, slots=True)
class PrivacyAccountantRecord:
    """Ledger entry for one approved release and the composed privacy loss so far."""

    release_id: str
    epsilon: float
    delta: float
    composed_epsilon: float
    composed_delta: float
    composition_method: str
    approved: bool
    purpose: str

    def __post_init__(self) -> None:
        _require_nonempty("release_id", self.release_id)
        _require_nonempty("composition_method", self.composition_method)
        _require_positive("epsilon", self.epsilon)
        if self.composed_epsilon < self.epsilon:
            raise ValueError("composed_epsilon cannot be smaller than this release's epsilon")
        if not isinstance(self.approved, bool):
            raise TypeError("approved must be a boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "release_id": self.release_id,
            "epsilon": self.epsilon,
            "delta": self.delta,
            "composed_epsilon": self.composed_epsilon,
            "composed_delta": self.composed_delta,
            "composition_method": self.composition_method,
            "approved": self.approved,
            "purpose": self.purpose,
        }


@dataclass(frozen=True, slots=True)
class MPCSession:
    """Multi-party computation session: participants, protocol, threat model, outputs."""

    session_id: str
    participants: tuple[str, ...]
    protocol: str
    threat_model: str
    collusion_threshold: int
    round_number: int
    input_schema: str
    permitted_outputs: tuple[str, ...]
    dropout_policy: str = "abort"

    def __post_init__(self) -> None:
        _require_nonempty("session_id", self.session_id)
        _require_nonempty("protocol", self.protocol)
        _require_nonempty("threat_model", self.threat_model)
        _require_nonempty("input_schema", self.input_schema)
        if len(self.participants) < 2:
            raise ValueError("an MPC session needs at least two participants")
        if len(set(self.participants)) != len(self.participants):
            raise ValueError("participants must be unique")
        if self.collusion_threshold < 0 or self.collusion_threshold >= len(self.participants):
            raise ValueError("collusion_threshold must be in [0, n-1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "participants": list(self.participants),
            "protocol": self.protocol,
            "threat_model": self.threat_model,
            "collusion_threshold": self.collusion_threshold,
            "round_number": self.round_number,
            "input_schema": self.input_schema,
            "permitted_outputs": list(self.permitted_outputs),
            "dropout_policy": self.dropout_policy,
        }


@dataclass(frozen=True, slots=True)
class AggregateResult:
    """Final aggregate with protocol reference, cohort size, validity and suppression state."""

    protocol_ref: str
    round_id: str
    cohort_size: int
    output: Any
    output_ref: str
    validity_report: Mapping[str, Any]
    suppressed: bool
    suppression_reason: str = ""

    def __post_init__(self) -> None:
        _require_nonempty("protocol_ref", self.protocol_ref)
        _require_nonempty("round_id", self.round_id)
        _require_nonempty("output_ref", self.output_ref)
        if self.cohort_size < 0:
            raise ValueError("cohort_size must be non-negative")
        if self.suppressed and not self.suppression_reason:
            raise ValueError("suppressed aggregates must record a suppression reason")
        if not isinstance(self.validity_report, Mapping):
            raise TypeError("validity_report must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol_ref": self.protocol_ref,
            "round_id": self.round_id,
            "cohort_size": self.cohort_size,
            "output": self.output,
            "output_ref": self.output_ref,
            "validity_report": dict(self.validity_report),
            "suppressed": self.suppressed,
            "suppression_reason": self.suppression_reason,
        }
