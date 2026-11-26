"""Typed artifact contracts and the artifact-to-finding bridge (Step 17).

Specialists return typed ``ModuleResult`` objects whose artifacts are validated
against registered contracts before any shared-state mutation. This module
registers the blackboard's own contracts and converts validated finding-batch
artifacts into provenance-complete workspace ``Finding`` objects.
"""
from __future__ import annotations

from typing import Any, Mapping

from ore_contracts import ContractCatalog
from ore_contracts.module_artifacts import ContractArtifact
from ore_contracts.module_protocol import ModuleResult, ModuleStatus
from ore_workspace.models import Finding, FindingTarget, Provenance

from .contracts import (
    BlackboardLayer,
    FailureClassification,
    LAYER_FIELDS,
    LifecycleState,
    layer_for_field,
)

BLACKBOARD_INPUT_CONTRACT = "ORE.BLACKBOARD.INPUT"
BLACKBOARD_SNAPSHOT_CONTRACT = "ORE.BLACKBOARD.WORKSPACE_SNAPSHOT"
BLACKBOARD_FINDING_BATCH_CONTRACT = "ORE.BLACKBOARD.FINDING_BATCH"
BLACKBOARD_MEMORY_STORE_CONTRACT = "ORE.BLACKBOARD.MEMORY_STORE"
BLACKBOARD_VERIFICATION_RULES_CONTRACT = "ORE.BLACKBOARD.VERIFICATION_RULES"
BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT = "ORE.BLACKBOARD.USG_MODULE_KNOWLEDGE"
BLACKBOARD_CONTRACT_VERSION = "1.0.0"


def _validate_finding_batch(payload: Any) -> bool | None:
    """Validate the structure of a finding-batch artifact payload."""
    if not isinstance(payload, Mapping):
        return False
    if "findings" not in payload:
        return False
    findings = payload.get("findings")
    if not isinstance(findings, list):
        return False
    for envelope in findings:
        if not isinstance(envelope, Mapping):
            return False
        for required in ("field", "payload"):
            if required not in envelope:
                return False
        if envelope.get("field") not in {
            "current_input", "current_goal", "lexical_hypotheses", "syntactic_hypotheses",
            "semantic_hypotheses", "discourse_context", "working_memory",
            "retrieved_memories", "unresolved_questions", "module_status",
            "artifact_references",
        }:
            return False
        confidence = envelope.get("confidence")
        if confidence is not None:
            try:
                value = float(confidence)
            except (TypeError, ValueError):
                return False
            if not 0.0 <= value <= 1.0:
                return False
        span = envelope.get("source_span")
        if span is not None:
            if not isinstance(span, list) or len(span) != 2:
                return False
            if int(span[0]) < 0 or int(span[1]) < int(span[0]):
                return False
    return True


def _validate_snapshot_payload(payload: Any) -> bool | None:
    """Validate workspace snapshot artifact payloads."""
    if not isinstance(payload, Mapping):
        return False
    if not isinstance(payload.get("revision"), int) or payload["revision"] < 0:
        return False
    return isinstance(payload.get("state"), Mapping)


def _validate_input_root_payload(payload: Any) -> bool | None:
    """Validate input-root artifact payloads."""
    if not isinstance(payload, Mapping):
        return False
    digest = payload.get("input_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        return False
    return isinstance(payload.get("tokens"), list)


def _validate_memory_store_payload(payload: Any) -> bool | None:
    """Validate memory-store artifact payloads."""
    if not isinstance(payload, Mapping):
        return False
    return isinstance(payload.get("episodes"), list)


def _validate_verification_rules_payload(payload: Any) -> bool | None:
    """Validate verification-rule artifact payloads."""
    if not isinstance(payload, Mapping):
        return False
    return isinstance(payload.get("rules"), list)


def register_blackboard_contracts(catalog: ContractCatalog) -> tuple[Any, ...]:
    """Register the blackboard's typed contracts idempotently (Step 16 contract checks)."""
    definitions: list[Any] = []
    existing = {
        (definition.contract_id, definition.version) for definition in catalog.list_contracts()
    }
    specs = (
        (BLACKBOARD_INPUT_CONTRACT, _validate_input_root_payload,
         ("task_id", "text", "input_digest", "tokens")),
        (BLACKBOARD_SNAPSHOT_CONTRACT, _validate_snapshot_payload,
         ("workspace_id", "revision", "state")),
        (BLACKBOARD_FINDING_BATCH_CONTRACT, _validate_finding_batch,
         ("findings", "diagnostics")),
        (BLACKBOARD_MEMORY_STORE_CONTRACT, _validate_memory_store_payload,
         ("episodes",)),
        (BLACKBOARD_VERIFICATION_RULES_CONTRACT, _validate_verification_rules_payload,
         ("rules",)),
    )
    for contract_id, validator, required_fields in specs:
        if (contract_id, BLACKBOARD_CONTRACT_VERSION) in existing:
            continue
        definitions.append(catalog.register_model(
            contract_id,
            BLACKBOARD_CONTRACT_VERSION,
            dict,
            required_fields=required_fields,
            description=f"Blackboard coordination contract {contract_id}.",
            validator=validator,
        ))
    if (BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT, BLACKBOARD_CONTRACT_VERSION) not in existing:
        from .usg_integration import USGModuleKnowledge
        definitions.append(catalog.register_model(
            BLACKBOARD_USG_MODULE_KNOWLEDGE_CONTRACT,
            BLACKBOARD_CONTRACT_VERSION,
            USGModuleKnowledge,
            required_fields=(
                "module_id", "domains", "requested_terms", "predicate_terms",
                "bundle_id", "bundle_digest", "resource_reference",
            ),
            description=(
                "Read-only, domain-scoped USG knowledge slice delivered to one "
                "blackboard specialist with feedback and provenance."
            ),
        ))
    return tuple(definitions)


class ArtifactFindingBridge:
    """Convert validated finding-batch artifacts into provenance-complete findings (Step 17)."""

    def __init__(self, workspace, module_id: str, *, allowed_fields: frozenset[str],
                 input_length: int, provenance_checks: bool = True,
                 dependency_deferral: bool = True) -> None:
        """Bind the bridge to one workspace, producer, authority set, and input bounds."""
        self.workspace = workspace
        self.module_id = module_id
        self.allowed_fields = allowed_fields
        self.input_length = input_length
        self.provenance_checks = provenance_checks
        self.dependency_deferral = dependency_deferral

    def convert(
        self,
        artifact: ContractArtifact,
        *,
        base_workspace_revision: int,
    ) -> tuple[Finding, ...]:
        """Convert one validated finding-batch artifact into typed findings."""
        payload = artifact.payload
        if not isinstance(payload, Mapping) or "findings" not in payload:
            raise ValueError("finding-batch artifact payload must contain a findings list")
        findings: list[Finding] = []
        for envelope in payload["findings"]:
            finding = self._convert_envelope(envelope, base_workspace_revision)
            if finding is not None:
                findings.append(finding)
        return tuple(findings)

    def _convert_envelope(
        self, envelope: Mapping[str, Any], base_revision: int,
    ) -> Finding | None:
        """Convert one envelope record, enforcing authority and span invariants."""
        field = str(envelope.get("field", ""))
        if field not in self.allowed_fields:
            raise PermissionError(
                f"module {self.module_id!r} is not authorized to write field {field!r}"
            )
        layer_raw = envelope.get("layer")
        layer = BlackboardLayer(int(layer_raw)) if layer_raw is not None else layer_for_field(field)
        if field not in LAYER_FIELDS[layer]:
            raise ValueError(
                f"field {field!r} does not belong to blackboard layer {layer.name}"
            )
        raw_lifecycle = envelope.get("lifecycle_state", LifecycleState.CANDIDATE)
        lifecycle = (
            raw_lifecycle if isinstance(raw_lifecycle, LifecycleState)
            else LifecycleState(str(raw_lifecycle))
        )
        span_raw = envelope.get("source_span")
        source_span = tuple(span_raw) if span_raw is not None else None
        if source_span is not None and self.provenance_checks:
            start, end = source_span
            if int(start) < 0 or int(end) < int(start) or int(end) > self.input_length:
                raise ValueError(
                    f"source span ({start}, {end}) is invalid for input length {self.input_length}"
                )
        dependencies = tuple(envelope.get("dependencies", ()))
        if not self.dependency_deferral:
            dependencies = ()
        payload = dict(envelope.get("payload", {}))
        usg_references = tuple(envelope.get("usg_references", ()))
        if usg_references:
            payload["usg_references"] = [dict(item) for item in usg_references]
        payload["layer"] = layer.name
        payload["lifecycle_state"] = lifecycle.value
        provenance = Provenance(
            source=str(envelope.get("source", "module_inference")),
            source_span=source_span,
            rules_applied=tuple(envelope.get("rules_applied", ())),
            evidence=tuple(dict(item) for item in envelope.get("evidence", ())),
            notes=envelope.get("notes"),
        )
        confidence = envelope.get("confidence")
        return Finding(
            workspace_id=self.workspace.workspace_id,
            producer_module=self.module_id,
            finding_type=str(envelope.get("finding_type", "module_hypothesis")),
            target=FindingTarget(field=field, key=envelope.get("key")),
            payload=payload,
            provenance=provenance,
            base_workspace_revision=base_revision,
            dependencies=dependencies,
            confidence=None if confidence is None else float(confidence),
        )


def validate_module_result(
    result: ModuleResult,
    *,
    module_id: str,
    module_version: str,
    run_id: str,
    catalog: ContractCatalog,
) -> tuple[tuple[ContractArtifact, ...], tuple[str, ...]]:
    """Validate one specialist result against its manifest identity (Step 16, V_c).

    Returns the accepted artifacts and the list of validation errors. A result may
    proceed to dependent work only when its status is an allowed successful state
    and every artifact passes contract validation.
    """
    if not isinstance(result, ModuleResult):
        raise TypeError("specialist backend must return a ModuleResult")
    if result.module_id != module_id or result.module_version != module_version:
        raise ValueError(
            "module result identity/version does not match the registered manifest"
        )
    errors: list[str] = []
    accepted: list[ContractArtifact] = []
    if result.status not in (ModuleStatus.SUCCESS, ModuleStatus.PARTIAL):
        return tuple(accepted), tuple(errors)
    for artifact in result.artifacts:
        if not isinstance(artifact, ContractArtifact):
            errors.append("output is not a ContractArtifact")
            continue
        envelope = artifact.envelope
        if envelope.producer_id != module_id or envelope.run_id != run_id:
            errors.append("artifact producer_id or run_id does not match this execution")
            continue
        validation = catalog.validate_artifact(artifact)
        if not validation.valid:
            errors.extend(validation.errors)
            continue
        accepted.append(artifact)
    return tuple(accepted), tuple(errors)


def classify_publication_failure(message: str) -> FailureClassification:
    """Classify downstream publication rejections from validation messages (Step 26)."""
    lowered = message.casefold()
    if "not authorized" in lowered:
        return FailureClassification.PERMISSION_DENIED
    if "missing dependencies" in lowered:
        return FailureClassification.DEPENDENCY_MISSING
    if "different workspace" in lowered or "future workspace revision" in lowered:
        return FailureClassification.CONTRACT_MISMATCH
    if "payload" in lowered and "finite" in lowered:
        return FailureClassification.INVALID_INPUT
    if "span" in lowered and "exceeds" in lowered:
        return FailureClassification.INVALID_INPUT
    return FailureClassification.DOWNSTREAM_VALIDATION_REJECTED


__all__ = [
    "ArtifactFindingBridge",
    "BLACKBOARD_CONTRACT_VERSION",
    "BLACKBOARD_FINDING_BATCH_CONTRACT",
    "BLACKBOARD_INPUT_CONTRACT",
    "BLACKBOARD_MEMORY_STORE_CONTRACT",
    "BLACKBOARD_SNAPSHOT_CONTRACT",
    "BLACKBOARD_VERIFICATION_RULES_CONTRACT",
    "classify_publication_failure",
    "register_blackboard_contracts",
    "validate_module_result",
]
