"""Registration of the trust layer with the shared ORE contract catalog.

Registers the Point 20 typed interfaces as versioned contracts and publishes a
``ModuleManifest`` so other ORE modules can discover and consume security
capabilities through the standard ``ModuleRegistry``. The registration is
self-contained: it never modifies existing repository files.
"""
from __future__ import annotations

from typing import Any

from .contracts import (
    AggregateResult,
    AuthorizationRequest,
    IntegrityCheckpoint,
    KeyDescriptor,
    MPCSession,
    PolicyDecision,
    Principal,
    PrivacyAccountantRecord,
    PrivacyReleaseRequest,
    ProofStatement,
    ProofVerificationResult,
    SignedArtifactManifest,
)


def register_contracts(catalog: Any) -> dict[str, str]:
    """Register the section-5.1 interfaces in a shared ``ContractCatalog``."""
    registrations = {
        "ore.security.principal": Principal,
        "ore.security.authorization_request": AuthorizationRequest,
        "ore.security.policy_decision": PolicyDecision,
        "ore.security.key_descriptor": KeyDescriptor,
        "ore.security.signed_artifact_manifest": SignedArtifactManifest,
        "ore.security.integrity_checkpoint": IntegrityCheckpoint,
        "ore.security.proof_statement": ProofStatement,
        "ore.security.proof_verification_result": ProofVerificationResult,
        "ore.security.privacy_release_request": PrivacyReleaseRequest,
        "ore.security.privacy_accountant_record": PrivacyAccountantRecord,
        "ore.security.mpc_session": MPCSession,
        "ore.security.aggregate_result": AggregateResult,
    }
    registered: dict[str, str] = {}
    for contract_id, model in registrations.items():
        catalog.register_model(
            contract_id, "1.0.0", model,
            required_fields=tuple(model.__dataclass_fields__),
            description=f"Point 20 CTPL interface {model.__name__}",
        )
        registered[contract_id] = "1.0.0"
    return registered


def build_module_manifest() -> Any:
    """Build the ore_security ModuleManifest for the shared module registry."""
    from ore_contracts.module_manifest import (
        InputPort,
        ModuleDependency,
        ModuleIdentity,
        ModuleManifest,
        OutputPort,
    )

    return ModuleManifest(
        identity=ModuleIdentity(
            module_id="ore.security.trust",
            module_version="1.0.0",
            module_name="ORE Cryptographic Trust and Privacy Layer (Point 20)",
            package_path="ore_security",
        ),
        responsibilities=(
            "Authenticate and authorize every protected ORE operation with default-deny policy.",
            "Manage key lifecycle, envelope encryption and recovery for the trust layer.",
            "Sign and verify artifacts, maintain hash-chained audit history with checkpoints.",
            "Provide versioned ZKP and zkVM-style execution-proof interfaces.",
            "Apply differential privacy accounting, secure aggregation, MPC and VDAF.",
            "Track post-quantum algorithm inventory, agility and hybrid migration.",
        ),
        input_ports=(
            InputPort(
                port_id="authorization-request-in",
                contract_id="ore.security.authorization_request",
                accepted_versions=("1.0.0",),
                expected_data_type="object",
                validation_requirements=("authenticated principal", "classified resource"),
            ),
            InputPort(
                port_id="proof-statement-in",
                contract_id="ore.security.proof_statement",
                accepted_versions=("1.0.0",),
                expected_data_type="object",
                validation_requirements=("supported relation version", "task binding present"),
            ),
            InputPort(
                port_id="privacy-release-in",
                contract_id="ore.security.privacy_release_request",
                accepted_versions=("1.0.0",),
                expected_data_type="object",
                validation_requirements=("reviewed calibration", "declared adjacency"),
            ),
        ),
        output_ports=(
            OutputPort(
                port_id="policy-decision-out",
                contract_id="ore.security.policy_decision",
                contract_version="1.0.0",
                output_data_type="object",
                description="Default-deny decision with predicate breakdown and reason code.",
            ),
            OutputPort(
                port_id="proof-verification-out",
                contract_id="ore.security.proof_verification_result",
                contract_version="1.0.0",
                output_data_type="object",
                description="Independent proof verification with recorded versions.",
            ),
            OutputPort(
                port_id="aggregate-out",
                contract_id="ore.security.aggregate_result",
                contract_version="1.0.0",
                output_data_type="object",
                description="Suppression-aware aggregate with validity report and metadata.",
            ),
        ),
        capabilities=(
            "security.authentication",
            "security.authorization",
            "security.key_management",
            "secure.transport",
            "security.artifact_signing",
            "security.integrity_checkpoints",
            "security.zero_knowledge_proofs",
            "security.verifiable_execution",
            "privacy.differential_privacy",
            "privacy.secure_aggregation",
            "privacy.mpc",
            "privacy.vdaf",
            "security.post_quantum_readiness",
            "security.audit",
        ),
        dependencies=(
            ModuleDependency(module_id="ore.logger", required=False,
                             purpose="SQLite annotations and run persistence"),
            ModuleDependency(module_id="ore.contracts", required=False,
                             purpose="shared contract catalog and module registry"),
        ),
        execution_modes=("FUNCTIONAL", "VALIDATE"),
        limitations=(
            "Reference proof systems: production must bind maintained ZKP and zkVM libraries.",
            "Software KMS: production requires an HSM- or OS-keystore-backed adapter.",
            "PQC primitives: ML-KEM/ML-DSA/SLH-DSA require a vetted provider.",
        ),
    )


def register_with_ore(catalog: Any | None = None,
                      module_registry: Any | None = None) -> dict[str, Any]:
    """Register contracts and the module manifest with shared ORE infrastructure."""
    report: dict[str, Any] = {"contracts_registered": {}, "module_registered": False}
    if catalog is not None:
        report["contracts_registered"] = register_contracts(catalog)
    if module_registry is not None:
        manifest = build_module_manifest()
        module_registry.register(manifest)
        report["module_registered"] = True
        report["module_id"] = manifest.identity.module_id
    return report
