"""Source-aligned operation registry for the Point 20 trust and privacy algorithm.

Each of the 64 steps of the integrated algorithm (phases A through J) maps to one
``OperationSpec`` that records where the step is implemented inside ``ore_security``.
The registry supports traceability audits and ore_logger annotation without claiming
biological scope: this module is a security layer, not a cognitive model.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one spec step, its mechanism, and its implementing module."""

    operation_id: str
    source_step: int
    phase: str
    name: str
    purpose: str
    mechanism: str
    implemented_in: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible representation of this operation."""
        return asdict(self)


_ROWS = (
    # Phase A - Threat modeling and policy (steps 1-5)
    ("A", "Define assets and trust boundaries", "Inventory identities, keys, records, artifacts, traces and messages with classification.", "Asset inventory with sensitivity classes and boundary map", "ore_security.policy.classification"),
    ("A", "Define adversaries and threat scenarios", "Specify attacker capabilities: theft, replay, tampering, collusion, future cryptanalysis.", "Threat scenario catalog consulted by policy and tests", "ore_security.policy.rules"),
    ("A", "Convert threats into security requirements", "Map each threat to confidentiality, integrity, authenticity, availability, accountability, recovery controls.", "Requirement map recorded per control and test", "ore_security.registry"),
    ("A", "Define data classification and handling rules", "Attach classification, owner, purpose, retention, recipients and transforms to data references.", "ResourceDescriptor registry carried through caches, logs and exports", "ore_security.policy.classification"),
    ("A", "Establish default-deny authorization", "Deny unless identity, policy grant, classification and context all pass.", "A(i,o,r,c) = I(i) and P(i,o,r,c) and D(r) and T(c)", "ore_security.policy.rules"),
    # Phase B - Identity, authentication and authorization (steps 6-12)
    ("B", "Give every component a stable identity", "Register nodes, services, modules and agents with controlled credential lifecycles.", "IdentityService registry with status and roles", "ore_security.identity.identities"),
    ("B", "Authenticate service-to-service requests", "Mutual authentication with issuer, audience, expiry, scope and revocation checks.", "Signed short-lived credential tokens with audience binding", "ore_security.identity.identities"),
    ("B", "Separate authentication from authorization", "Evaluate per-resource permissions after identity proof; identity never implies access.", "PolicyEngine evaluated against AuthorizationRequest", "ore_security.policy.rules"),
    ("B", "Use short-lived, scoped credentials", "Explicit audience, operations, expiry and revocation; no long-lived static secrets.", "TTL-bounded credential issuance with scope sets", "ore_security.identity.identities"),
    ("B", "Enforce authorization at the resource boundary", "The resource owner rechecks permission itself for sensitive state transitions.", "SecurityDecisionPipeline step 4 executed at the owning service", "ore_security.pipeline"),
    ("B", "Prevent replay and request substitution", "Nonces, timestamps, bounded skew, signatures bound to method, destination and digest.", "RequestAuthenticator with nonce cache and request-binding signature", "ore_security.identity.authentication"),
    ("B", "Define safe failure behavior", "Authentication and policy failures fail closed; no silent downgrade to unauthenticated access.", "Exception hierarchy with error categories consumed by the audit trail", "ore_security.exceptions"),
    # Phase C - Key management and secure communication (steps 13-19)
    ("C", "Centralize key lifecycle policy", "Purposes, algorithms, owners, storage, rotation, backup separated per key.", "LocalKMS with KeyDescriptor metadata and purpose separation", "ore_security.keys.kms"),
    ("C", "Protect private keys", "Keys held in the KMS, export restricted, every use audited, administration separated.", "Software KMS adapter with use-audit and secret canaries", "ore_security.keys.kms"),
    ("C", "Establish authenticated encryption in transit", "Validated peer identity, modern suites, obsolete versions rejected.", "TransportPolicy plus X25519+HKDF+AES-GCM channel endpoints", "ore_security.transport.channels"),
    ("C", "Encrypt sensitive data at rest", "Platform encryption for stores and backups with separate key management.", "Envelope encryption with DEK/KEK separation", "ore_security.keys.kms"),
    ("C", "Use envelope encryption where useful", "Wrap per-object DEKs with a KMS-held KEK; rotation rewraps without re-encryption.", "DEK generation, KEK wrapping, rewrap-on-rotation", "ore_security.keys.kms"),
    ("C", "Rotate, expire and revoke keys", "Rotation by exposure risk with validity intervals; revocation blocks future trust.", "rotate/revoke with historical verification preserved", "ore_security.keys.kms"),
    ("C", "Design key backup and recovery", "Protected, tested restoration that does not defeat key separation.", "Passphrase-derived encrypted backup bundles", "ore_security.keys.kms"),
    # Phase D - Signatures, hashes and integrity history (steps 20-25)
    ("D", "Sign approved artifacts and releases", "Sign manifests binding digest, version, provenance, signer and intended use.", "SignedArtifactManifest with Ed25519 signature envelope", "ore_security.signing.artifacts"),
    ("D", "Verify before loading or executing artifacts", "Recompute digest, verify signature, check signer authorization and scope.", "ArtifactSigningService.verify_artifact with layered checks", "ore_security.signing.artifacts"),
    ("D", "Use cryptographic hashes for content identity", "h = H(x) with algorithm and canonicalization identifiers stored.", "ContentHasher with approved algorithm list", "ore_security.integrity.hashing"),
    ("D", "Make audit records tamper-evident", "h_i = H(h_{i-1} || canonical(e_i)) with signed periodic checkpoints.", "HashChain plus CheckpointService", "ore_security.integrity.chain"),
    ("D", "Use Merkle trees for batch verification", "Root over hashed leaves with signed or anchored commitment.", "MerkleTree with inclusion proofs and defined leaf ordering", "ore_security.integrity.merkle"),
    ("D", "Separate integrity from semantic validity", "Intact events and signed outputs still require semantic validation.", "Semantic validators separate from cryptographic verification", "ore_security.pipeline"),
    # Phase E - Zero-knowledge proofs (steps 26-31)
    ("E", "Identify statements that benefit from zero knowledge", "Narrow claims: range constraints, aggregate relations, credential predicates.", "Statement registry with documented motivations", "ore_security.proofs.statements"),
    ("E", "Define public statement and private witness", "Public inputs x, private witness w, relation R(x, w) = 1.", "Schnorr discrete-log relation with explicit group parameters", "ore_security.proofs.schnorr"),
    ("E", "Specify proof security properties", "Completeness, soundness, zero knowledge, setup and cryptographic assumptions.", "Adapter documentation of HVZK and random-oracle assumptions", "ore_security.proofs.schnorr"),
    ("E", "Bind proofs to task context", "Task id, statement version, inputs, nonce, expiry and verifier context bound.", "Binding payload included in the Fiat-Shamir transcript", "ore_security.proofs.service"),
    ("E", "Verify proofs independently", "Validate exact relation version and inputs; record verifier version and result.", "ProofService.verify returning ProofVerificationResult", "ore_security.proofs.service"),
    ("E", "Control circuit and statement upgrades", "Versioned relations, constraint schemas and verifier keys with migration.", "ProofStatementRegistry with supported-version enforcement", "ore_security.proofs.statements"),
    # Phase F - zkVM and verifiable program execution (steps 32-37)
    ("F", "Select external computations worth proving", "Deterministic bounded computations where recomputation is expensive.", "Workload selection documented in adapter docs", "ore_security.zkvm.executor"),
    ("F", "Freeze the program and execution statement", "Commit program, runtime, input encoding, outputs and semantics.", "ProgramCommitment digest over MiniVM program and runtime version", "ore_security.zkvm.minivm"),
    ("F", "Generate an execution proof at the worker", "Worker executes and produces output plus proof of the execution relation.", "Trace Merkle root with Fiat-Shamir spot openings", "ore_security.zkvm.executor"),
    ("F", "Verify proof and bind output to it", "Verify against committed program and public input/output commitments.", "ExecutionVerifier full and probabilistic verification modes", "ore_security.zkvm.executor"),
    ("F", "Distinguish verifiable execution from privacy", "Succinct verifiability without confidentiality; separate mechanism needed.", "provides_privacy flag fixed to False with rationale", "ore_security.zkvm.executor"),
    ("F", "Compare proof cost with recomputation", "Measure proving, verification, memory and proof size before adopting.", "benchmark utility returning cost measurements", "ore_security.zkvm.executor"),
    # Phase G - Differential privacy (steps 38-43)
    ("G", "Identify the privacy unit and neighboring datasets", "Define the protected unit and the adjacency relation precisely.", "PrivacyReleaseRequest with unit and adjacency fields", "ore_security.privacy.accountant"),
    ("G", "Define query and sensitivity", "Delta f under the adjacency relation with clipping and contribution limits.", "Sensitivity review with contribution bounds", "ore_security.privacy.mechanisms"),
    ("G", "Calibrate randomness to a formal definition", "Gaussian noise scale from sensitivity and (epsilon, delta) via reviewed theorem.", "sigma = Delta f * sqrt(2 ln(1.25/delta)) / epsilon", "ore_security.privacy.mechanisms"),
    ("G", "Maintain a privacy budget ledger", "Track mechanism, unit, epsilon, delta, composition, version and purpose.", "PrivacyAccountant with sequential and advanced composition", "ore_security.privacy.accountant"),
    ("G", "Apply DP to appropriate aggregates", "Module telemetry and collaborative statistics, never raw traces.", "Aggregate helpers with clipping and cohort rules", "ore_security.privacy.mechanisms"),
    ("G", "Test utility and privacy together", "Measure error, bias and subgroup utility alongside the formal guarantee.", "evaluate_utility bias/RMSE reporting", "ore_security.privacy.mechanisms"),
    # Phase H - Secure aggregation, MPC and VDAF (steps 44-51)
    ("H", "Decide whether central collection is the problem", "Choose aggregation, MPC or plaintext collection to match the dataflow.", "Decision documentation in protocol module", "ore_security.multiparty.aggregation"),
    ("H", "Specify parties, inputs, outputs and collusion assumptions", "Participant count, trust, dropout behavior and adversary model.", "MPCSession with threat model and collusion threshold", "ore_security.multiparty.mpc"),
    ("H", "Define secure aggregation", "Reveal only the aggregate; authenticate participants and bind rounds.", "Pairwise-canceling PRG masks with round binding", "ore_security.multiparty.aggregation"),
    ("H", "Use MPC for joint private computation when justified", "Compute f(x_1..x_n) revealing only permitted outputs.", "Additive sharing with Beaver triples over a prime field", "ore_security.multiparty.mpc"),
    ("H", "Apply secure aggregation to ORE telemetry selectively", "Contribute module statistics without sharing execution histories.", "Vector aggregation with granularity limits", "ore_security.multiparty.aggregation"),
    ("H", "Use VDAF/Prio-style protocols for structured measurements", "Encoding, validity constraints, aggregation and verifier roles.", "Affine AGREE validity over secret-shared binary vectors", "ore_security.vdaf.prio"),
    ("H", "Verify aggregate validity and participant eligibility", "Protocol proofs, rounds, credentials, limits and cohort minimums.", "Validity reports with cohort size checks", "ore_security.vdaf.prio"),
    ("H", "Handle dropout, malicious submissions and aborts", "Thresholds, retries, partial results and incident logging.", "Seed re-sharing dropout recovery and abort policies", "ore_security.multiparty.aggregation"),
    # Phase I - Post-quantum readiness (steps 52-57)
    ("I", "Inventory cryptographic dependencies", "Protocols, certificates, sizes, libraries, retention horizons, dependencies.", "CryptoInventory fed by KMS, signing and transport adapters", "ore_security.pqc.policy"),
    ("I", "Prioritize migration by exposure and lifetime", "Long-lived confidentiality first; harvest-now/decrypt-later risk.", "MigrationManager priority by data lifetime", "ore_security.pqc.policy"),
    ("I", "Prepare standardized PQC support", "Track ML-KEM, ML-DSA and SLH-DSA families with vetted libraries.", "AlgorithmIdentifier registry with parameter sets", "ore_security.pqc.identifiers"),
    ("I", "Build algorithm agility into formats", "Store algorithm ids, parameter sets, key ids and versioned envelopes.", "Hybrid envelopes with unambiguous identifiers", "ore_security.pqc.hybrid"),
    ("I", "Test hybrid migration and interoperability", "Hybrid classical/PQC establishment and staged signature migration.", "HybridSignatureEnvelope and HybridKeyEstablishment", "ore_security.pqc.hybrid"),
    ("I", "Preserve historical verification", "Keep public keys, chains, metadata and timestamps for old artifacts.", "Historical verification preserved through rotation records", "ore_security.keys.kms"),
    # Phase J - Audit, incident response and ORE integration (steps 58-64)
    ("J", "Define security event schemas", "Identity, action, resource, decision, key id, algorithm, result, correlation.", "SecurityEvent schema with redaction on append", "ore_security.audit.events"),
    ("J", "Annotate security-sensitive functions and classes", "Stable operation ids, schema versions, policy refs, validation state.", "Annotations registered with ore_logger SQLite catalog", "ore_security.annotations"),
    ("J", "Protect the audit system", "Access restriction, chained records, signed exports, replicated checkpoints.", "AuditLedger with checkpoints anchored in a second trust domain", "ore_security.audit.ledger"),
    ("J", "Define key-compromise response", "Revoke, rotate, invalidate sessions, assess artifacts, notify, restore.", "IncidentResponse.compromise_principal drill", "ore_security.audit.service"),
    ("J", "Define recovery tests", "KMS outage, expiry, compromise, corrupted audit, dropout, budget exhaustion.", "Recovery drills exercising fail-closed restoration", "ore_security.audit.service"),
    ("J", "Integrate security checks with ORE state transitions", "Protected records require authentication, authorization, integrity and semantics.", "SecurityDecisionPipeline and SecuredRecordStore", "ore_security.pipeline"),
    ("J", "Review controls and retire obsolete mechanisms", "Periodic review of assumptions, standards, keys, budgets and incidents.", "Control review report over registry and inventory", "ore_security.registry"),
)

OPERATION_REGISTRY = tuple(
    OperationSpec(
        operation_id=f"P20-{index:02d}",
        source_step=index,
        phase=row[0],
        name=row[1],
        purpose=row[2],
        mechanism=row[3],
        implemented_in=row[4],
    )
    for index, row in enumerate(_ROWS, start=1)
)


def registry_as_dicts() -> list[dict[str, object]]:
    """Return all sixty-four Point 20 operations in source order."""
    return [operation.to_dict() for operation in OPERATION_REGISTRY]


def control_review_report() -> dict[str, object]:
    """Summarize control coverage and known limitations for periodic review (step 64)."""
    phases: dict[str, int] = {}
    for operation in OPERATION_REGISTRY:
        phases[operation.phase] = phases.get(operation.phase, 0) + 1
    return {
        "operation_count": len(OPERATION_REGISTRY),
        "phases": phases,
        "known_limitations": [
            "The Schnorr adapter is a reference proof system; production should bind a maintained ZKP library.",
            "The MiniVM execution proof is probabilistic spot-check verification, not a succinct zkVM proof.",
            "The Paillier adapter provides additive homomorphism only, as a stand-in for vetted FHE libraries.",
            "The PQC module provides agility and hybrid envelopes; ML-KEM/ML-DSA/SLH-DSA primitives require a vetted provider.",
            "The local software KMS protects keys in memory only; production requires an HSM-backed or OS-keystore adapter.",
        ],
        "retirement_watchlist": [
            "Reference Lamport one-time signatures are envelope-test vehicles and must not guard production data.",
            "Mock ML-KEM leg in hybrid establishment is for interoperability plumbing only.",
        ],
    }
