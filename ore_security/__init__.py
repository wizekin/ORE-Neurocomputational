"""ORE Cryptographic Trust and Privacy Layer (Point 20).

A cohesive security layer for the ORE neurocomputational architecture implementing
the integrated algorithm "Cryptography, Privacy and Verifiable Computation":
identity, authorization, key management, secure transport, artifact signing,
integrity history, zero-knowledge proofs, verifiable execution, differential
privacy, secure aggregation, MPC, VDAF and post-quantum readiness, all bound
together by a fail-closed security decision pipeline and a tamper-evident audit
ledger. See docs/CRYPTOGRAPHY_PRIVACY_VERIFIABLE_COMPUTATION_POINT20.md.
"""
from __future__ import annotations

from . import audit, fhe, identity, integrity, keys, multiparty, policy, privacy, proofs, pqc, signing, transport, zkvm
from .annotations import record_pipeline_run_to_sqlite, register_annotations
from .canonical import CANONICALIZATION_ID, DEFAULT_HASH_ALGORITHM, canonical_bytes, canonical_json, content_digest, digest_bytes, digest_structure
from .contracts import (
    SCHEMA_VERSION,
    AggregateResult,
    AuthorizationRequest,
    IntegrityCheckpoint,
    KeyDescriptor,
    KeyPurpose,
    KeyStatus,
    MPCSession,
    PolicyDecision,
    Principal,
    PrincipalKind,
    PrivacyAccountantRecord,
    PrivacyReleaseRequest,
    ProofStatement,
    ProofVerificationResult,
    SignedArtifactManifest,
)
from .crypto import provider_report
from .exceptions import SecurityError
from .pipeline import (
    CoordinatorGuard,
    ProtectedOperationResult,
    SecuredRecordStore,
    SecurityDecisionPipeline,
    build_default_trust_stack,
)
from .registry import OPERATION_REGISTRY, OperationSpec, control_review_report, registry_as_dicts

__version__ = "1.0.0"

__all__ = [
    "SCHEMA_VERSION",
    "__version__",
    # subpackages
    "audit", "fhe", "identity", "integrity", "keys", "multiparty", "policy",
    "privacy", "proofs", "pqc", "signing", "transport", "zkvm",
    # contracts
    "AggregateResult", "AuthorizationRequest", "IntegrityCheckpoint", "KeyDescriptor",
    "KeyPurpose", "KeyStatus", "MPCSession", "PolicyDecision", "Principal",
    "PrincipalKind", "PrivacyAccountantRecord", "PrivacyReleaseRequest",
    "ProofStatement", "ProofVerificationResult", "SignedArtifactManifest",
    # canonical + crypto
    "CANONICALIZATION_ID", "DEFAULT_HASH_ALGORITHM", "canonical_bytes", "canonical_json",
    "content_digest", "digest_bytes", "digest_structure", "provider_report",
    # pipeline
    "CoordinatorGuard", "ProtectedOperationResult", "SecuredRecordStore",
    "SecurityDecisionPipeline", "build_default_trust_stack",
    # registry + annotations
    "OPERATION_REGISTRY", "OperationSpec", "control_review_report", "registry_as_dicts",
    "record_pipeline_run_to_sqlite", "register_annotations",
    "SecurityError",
]
