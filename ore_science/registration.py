"""ORE module registration: manifest, contracts, and backend binding.

This module registers ``ore_science`` with the shared ``ore_contracts`` registry
following the repository's canonical registration sequence: register contracts first,
then the module manifest, then bind a protocol backend exposing ``describe()`` and
``execute(inputs, context)``. The backend adapts ``ScientificReasoner`` results into
``ModuleResult`` envelopes with typed artifacts, so a failed scientific run can never
be mistaken for a successful module execution.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ore_contracts import (
    ContractArtifact,
    ContractCatalog,
    InputPort,
    ModuleDependency,
    ModuleIdentity,
    ModuleManifest,
    ModuleRegistry,
    ModuleResult,
    ModuleStatus,
    OutputPort,
    make_envelope,
)
from ore_contracts.contracts_registry import ContractDefinition

from .contracts import OutcomeKind, ScienceRequest
from .orchestration import MODULE_ID
from .pipeline import ScientificReasoner

__all__ = [
    "ORE_SCIENCE_CAPABILITIES",
    "ORE_SCIENCE_CONTRACT",
    "ScientificReasonerBackend",
    "build_science_manifest",
    "register_science_contracts",
    "register_science_module",
]

ORE_SCIENCE_CONTRACT = "ORE.SCIENCE.RESULT"
_CONTRACT_VERSION = "1.0.0"

ORE_SCIENCE_CAPABILITIES: tuple[str, ...] = (
    "science.task_interpretation",
    "science.representation_selection",
    "science.model_routing",
    "science.independent_validation",
    "science.uncertainty_quantification",
    "science.provenance_recording",
    "science.simulation.heat_diffusion",
    "science.forecasting.ensemble",
    "science.localization.factor_graphs",
    "science.materials.discovery",
    "science.molecular.structure_checks",
    "science.geometric.equivariance_tests",
)


def register_science_contracts(catalog: ContractCatalog) -> ContractDefinition:
    """Register the scientific result contract with its required fields."""
    existing = catalog.get_contract(ORE_SCIENCE_CONTRACT, _CONTRACT_VERSION)
    if existing is not None:
        return existing
    return catalog.register_model(
        ORE_SCIENCE_CONTRACT,
        _CONTRACT_VERSION,
        dict,
        required_fields=(
            "outcome", "question", "result_status", "validation_reports",
            "uncertainty_reports", "operation_trace",
        ),
        optional_fields=(
            "predictions", "task", "domain_decision", "provenance_ref",
            "explanation", "knowledge_update", "diagnostics",
        ),
        description=(
            "Structured scientific result from the ORE scientific reasoning layer: "
            "typed outcome, status ladder position, validation and uncertainty "
            "records, and the full 52-step operation trace."
        ),
    )


def build_science_manifest() -> ModuleManifest:
    """Build the module manifest for the scientific reasoning layer."""
    return ModuleManifest(
        identity=ModuleIdentity(
            module_id=MODULE_ID,
            module_version="1.0.0",
            module_name="ORE Scientific Reasoning Layer",
            package_path="ore_science",
        ),
        responsibilities=(
            "Interpret natural-language scientific requests into typed tasks with "
            "variables, units, constraints, and evidence requirements.",
            "Select domain-appropriate representations, models, and solvers through "
            "hard compatibility gates and the Section 5.8 decision score.",
            "Validate predictions independently: dimensions, convergence, "
            "conservation, baselines, calibration, and equivariance.",
            "Quantify uncertainty, record provenance, and publish results to the "
            "shared cognitive workspace without rewriting USG knowledge.",
        ),
        input_ports=(
            InputPort(
                port_id="science_request",
                contract_id=ORE_SCIENCE_CONTRACT,
                accepted_versions=(_CONTRACT_VERSION,),
                expected_data_type="builtins.dict",
                validation_requirements=("question must be a non-empty string",),
            ),
        ),
        output_ports=(
            OutputPort(
                port_id="science_result",
                contract_id=ORE_SCIENCE_CONTRACT,
                contract_version=_CONTRACT_VERSION,
                output_data_type="builtins.dict",
                description=(
                    "Typed scientific result with outcome, status, validation, "
                    "uncertainty, provenance, and the 52-step operation trace."
                ),
            ),
        ),
        capabilities=ORE_SCIENCE_CAPABILITIES,
        dependencies=(
            ModuleDependency(
                module_id="ore_usg_knowledge",
                required=False,
                purpose="Read-only scientific concepts, equations, and domain rules.",
            ),
            ModuleDependency(
                module_id="ore_workspace",
                required=False,
                purpose="Publication of validated findings and artifact references.",
            ),
        ),
        execution_modes=("FUNCTIONAL", "VALIDATE"),
        limitations=(
            "External scientific services (AlphaFold, RFdiffusion, GNoME/MatterGen, "
            "GraphCast/GenCast, GTSAM) are optional integrations and are never "
            "silently substituted by local reference models.",
            "Generated molecular and material candidates remain CANDIDATE status "
            "until independent validation completes.",
            "The natural-language parser is rule-based; natural-language understanding "
            "is not claimed.",
        ),
    )


class ScientificReasonerBackend:
    """Protocol backend adapting ScientificReasoner to the ORE module contract."""

    def __init__(self, reasoner: ScientificReasoner | None = None) -> None:
        """Bind a reasoner and its manifest description."""
        self._reasoner = reasoner or ScientificReasoner()
        self._manifest = build_science_manifest()

    def describe(self) -> ModuleManifest:
        """Return the module manifest for registry compatibility checks."""
        return self._manifest

    def execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Run the scientific reasoning loop over one request and wrap the result."""
        request_payload = inputs.get("science_request")
        if not isinstance(request_payload, Mapping):
            return ModuleResult(
                module_id=MODULE_ID,
                module_version=self._manifest.identity.module_version,
                status=ModuleStatus.ABSTAINED,
                warnings=("input port 'science_request' requires a mapping payload",),
            )
        question = str(request_payload.get("question", "")).strip()
        if not question:
            return ModuleResult(
                module_id=MODULE_ID,
                module_version=self._manifest.identity.module_version,
                status=ModuleStatus.FAILED,
                warnings=("science request requires a non-empty question",),
            )
        facts = request_payload.get("facts", {}) or {}
        request = ScienceRequest(
            question=question,
            facts=facts if isinstance(facts, Mapping) else {},
            source_reference=str(request_payload.get("source_reference", "ore_contracts")),
            request_id=str(
                request_payload.get("request_id", getattr(context, "run_id", "science_request"))
            ),
        )
        result = self._reasoner.process(request)
        payload = result.to_dict()
        run_id = str(getattr(context, "run_id", "run_local"))
        artifact = ContractArtifact(
            envelope=make_envelope(
                ORE_SCIENCE_CONTRACT,
                run_id,
                MODULE_ID,
                config={
                    "collocation_points": self._reasoner.config.collocation_points,
                    "time_steps": self._reasoner.config.time_steps,
                    "random_seed": self._reasoner.config.random_seed,
                },
                model_version="1.0.0",
                data_hash=artifact_data_hash(payload),
            ),
            payload=payload,
            metadata={
                "outcome": result.outcome.value,
                "result_status": result.result_status,
                "valid": bool(result.diagnostics.get("valid")),
            },
        )
        status = (
            ModuleStatus.SUCCESS
            if result.outcome in (OutcomeKind.COMPLETED,)
            else ModuleStatus.PARTIAL
        )
        return ModuleResult(
            module_id=MODULE_ID,
            module_version=self._manifest.identity.module_version,
            status=status,
            artifacts=(artifact,),
            diagnostics=dict(result.diagnostics),
            warnings=tuple(result.diagnostics.get("warnings", ())),
            provenance=dict(result.provenance_ref or {}),
            processing_metadata={
                "predictions": len(result.predictions),
                "operation_count": len(result.operation_trace),
            },
        )


def artifact_data_hash(payload: Mapping[str, Any]) -> str:
    """Stable content hash for the artifact envelope."""
    from ore_contracts import sha256_json

    return sha256_json(dict(payload))


def register_science_module(
    registry: ModuleRegistry, *, reasoner: ScientificReasoner | None = None
) -> ScientificReasonerBackend:
    """Register contracts, manifest, and backend for the scientific layer."""
    register_science_contracts(registry.contract_catalog)
    manifest = build_science_manifest()
    existing = registry.get_manifest(MODULE_ID)
    if existing is None:
        registry.register(manifest)
    elif existing.identity != manifest.identity:
        raise ValueError("an incompatible ore_science manifest is already registered")
    backend = ScientificReasonerBackend(reasoner)
    registry.bind_backend(MODULE_ID, backend)
    return backend
