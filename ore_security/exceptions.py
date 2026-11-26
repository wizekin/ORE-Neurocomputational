"""Security exception hierarchy with auditable error categories.

Every exception carries a stable ``error_category`` string so audit events can record
failure classifications (spec Point 20, step 58) without embedding protected content.
All components fail closed: a raised security error never silently downgrades access.
"""
from __future__ import annotations


class SecurityError(Exception):
    """Base class for every ORE Cryptographic Trust and Privacy Layer failure."""

    error_category: str = "SECURITY_ERROR"

    def __init__(self, message: str, *, correlation_id: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.correlation_id = correlation_id


class AuthenticationError(SecurityError):
    """Caller could not be authenticated (unauthenticated request rejected)."""

    error_category = "AUTHENTICATION_FAILED"


class CredentialExpiredError(AuthenticationError):
    """Presented credential passed its expiry instant."""

    error_category = "CREDENTIAL_EXPIRED"


class CredentialRevokedError(AuthenticationError):
    """Presented credential or its subject identity was revoked."""

    error_category = "CREDENTIAL_REVOKED"


class ReplayError(AuthenticationError):
    """A request replay reused a nonce or an already-consumed authentication tag."""

    error_category = "REPLAY_DETECTED"


class AuthorizationError(SecurityError):
    """Authenticated identity is not permitted to perform the operation."""

    error_category = "AUTHORIZATION_DENIED"


class PolicyEngineFailure(AuthorizationError):
    """Authorization service failed; fail-closed semantics deny the operation."""

    error_category = "POLICY_ENGINE_FAILURE"


class ClassificationError(AuthorizationError):
    """Resource classification data was missing or the request exceeded it."""

    error_category = "CLASSIFICATION_UNSATISFIABLE"


class KeyManagementError(SecurityError):
    """Key lifecycle operation failed inside the KMS."""

    error_category = "KEY_MANAGEMENT_ERROR"


class KeyNotFoundError(KeyManagementError):
    """Referenced key identifier does not exist in the KMS."""

    error_category = "KEY_NOT_FOUND"


class KeyNotActiveError(KeyManagementError):
    """Key exists but its status forbids the requested operation."""

    error_category = "KEY_NOT_ACTIVE"


class KMSUnavailableError(KeyManagementError):
    """KMS outage; protected operations must fail closed, not bypass the store."""

    error_category = "KMS_UNAVAILABLE"


class TransportError(SecurityError):
    """Secure transport establishment or message protection failed."""

    error_category = "TRANSPORT_ERROR"


class PeerValidationError(TransportError):
    """Authenticated peer is not the intended service identity."""

    error_category = "PEER_VALIDATION_FAILED"


class ConfigurationRejectedError(SecurityError):
    """Proposed protocol or cipher configuration is obsolete or disallowed."""

    error_category = "CONFIGURATION_REJECTED"


class ArtifactVerificationError(SecurityError):
    """Artifact digest, signature, signer authorization, or scope check failed."""

    error_category = "ARTIFACT_VERIFICATION_FAILED"


class IntegrityError(SecurityError):
    """Hash-chain, Merkle, or checkpoint verification detected tampering."""

    error_category = "INTEGRITY_VIOLATION"


class CheckpointError(IntegrityError):
    """Signed checkpoint missing, invalid, or inconsistent with the event history."""

    error_category = "CHECKPOINT_INVALID"


class MerkleError(IntegrityError):
    """Merkle inclusion proof malformed or inconsistent with the committed root."""

    error_category = "MERKLE_INVALID"


class ProofError(SecurityError):
    """Proof statement, binding, or adapter contract violated."""

    error_category = "PROOF_ERROR"


class StatementVersionError(ProofError):
    """Proof references an unsupported relation or circuit version."""

    error_category = "STATEMENT_VERSION_UNSUPPORTED"


class ProofVerificationError(ProofError):
    """Proof failed cryptographic verification; result must not be marked verified."""

    error_category = "PROOF_VERIFICATION_FAILED"


class ExecutionProofError(ProofError):
    """zkVM-style execution proof inconsistent with program or output commitment."""

    error_category = "EXECUTION_PROOF_INVALID"


class HomomorphicEncryptionError(SecurityError):
    """Homomorphic operation outside the supported ciphertext algebra."""

    error_category = "FHE_UNSUPPORTED_OPERATION"


class PrivacyError(SecurityError):
    """Differential-privacy release violated a requirement or formal definition."""

    error_category = "PRIVACY_REQUIREMENT_VIOLATED"


class BudgetExceededError(PrivacyError):
    """Requested release exceeds the remaining differential-privacy budget."""

    error_category = "PRIVACY_BUDGET_EXCEEDED"


class MultipartyError(SecurityError):
    """Secure aggregation, MPC, or VDAF protocol assumption violated."""

    error_category = "MULTIPARTY_PROTOCOL_ERROR"


class AggregationError(MultipartyError):
    """Aggregation round integrity: duplicate, replayed, or malformed contribution."""

    error_category = "CONTRIBUTION_REJECTED"


class MPCError(MultipartyError):
    """MPC session abort: dropout, threshold, or threat-model violation."""

    error_category = "MPC_ABORTED"


class VDAFError(MultipartyError):
    """VDAF measurement share failed structural or cryptographic validity checks."""

    error_category = "VDAF_MEASUREMENT_INVALID"


class PQCError(SecurityError):
    """Post-quantum migration policy or algorithm agility violation."""

    error_category = "PQC_POLICY_VIOLATION"


class AlgorithmRejectedError(PQCError):
    """Algorithm identifier unsupported, ambiguous, or below the required policy."""

    error_category = "ALGORITHM_REJECTED"


class AuditError(SecurityError):
    """Audit recording, redaction, or checkpoint persistence failed."""

    error_category = "AUDIT_ERROR"


class RedactionError(AuditError):
    """Event payload contained registered secret material and was refused."""

    error_category = "SECRET_LEAK_BLOCKED"
