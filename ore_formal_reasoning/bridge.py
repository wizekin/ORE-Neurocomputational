"""Bridge between FRCE and the ORE blackboard coordination layer (Point 1).

This module exposes the Formal Reasoning and Constraint Engine as a blackboard
specialist so the Point-1 cognitive coordinator can route a formal-reasoning
task class to it, and registers the FRCE artifact contracts with the shared
``ore_contracts`` catalog. The specialist converts a blackboard task frame into
a typed ``FormalizationRequest``, executes the 52-step algorithm, and returns
the resulting envelopes as a validated finding-batch artifact. The bridge is
additive: nothing in the blackboard package imports FRCE by default.
"""
from __future__ import annotations

from typing import Any, Mapping

from ore_contracts import ContractCatalog, ContractEnvelope, EvidenceKind
from ore_contracts.module_artifacts import ContractArtifact
from ore_contracts.module_manifest import (
    InputPort,
    ModuleIdentity,
    ModuleManifest,
    ModuleDependency,
    OutputPort,
)
from ore_contracts.module_protocol import ModuleResult, ModuleStatus

from .coordinator import FormalReasoningCoordinator
from .contracts import FormalizationRequest, TaskClass
from .version import (
    FRCE_CONTRACT_VERSION,
    FRCE_FINDING_BATCH_CONTRACT,
    FRCE_REQUEST_CONTRACT,
    FRCE_RESULT_CONTRACT,
)

FRCE_SPECIALIST_MODULE_ID = "ore.formal_reasoning.specialist"


def register_formal_reasoning_contracts(catalog: ContractCatalog) -> None:
    """Register the FRCE request/result/finding-batch contracts (typed boundary)."""
    from ore_contracts.contracts_registry import ContractDefinition

    definitions = (
        ContractDefinition(
            contract_id=FRCE_REQUEST_CONTRACT,
            version=FRCE_CONTRACT_VERSION,
            model_type=None,
            required_fields=("request_id", "objective", "task_classes"),
            description="Typed FRCE formalization request envelope",
        ),
        ContractDefinition(
            contract_id=FRCE_RESULT_CONTRACT,
            version=FRCE_CONTRACT_VERSION,
            model_type=None,
            required_fields=("result_id", "kind", "status", "statement"),
            description="Typed FRCE result envelope with epistemic kind and status",
        ),
        ContractDefinition(
            contract_id=FRCE_FINDING_BATCH_CONTRACT,
            version=FRCE_CONTRACT_VERSION,
            model_type=None,
            required_fields=("findings",),
            description="FRCE finding batch published into the shared workspace",
        ),
    )
    for definition in definitions:
        try:
            catalog.register_contract(definition)
        except Exception:  # noqa: BLE001 - already registered is acceptable
            pass


class FormalReasoningSpecialist:
    """A blackboard-compatible specialist backed by the FRCE coordinator.

    Follows the reference specialist protocol of ``ore_blackboard.specialists``:
    ``describe()`` returns a validated manifest and ``execute(inputs, context)``
    returns a ``ModuleResult`` carrying a finding-batch artifact of typed FRCE
    results (inferred facts, satisfiability results, optimization proposals,
    proofs, and conflicts as distinct finding types).
    """

    module_id = FRCE_SPECIALIST_MODULE_ID
    package_path = "ore_formal_reasoning.bridge"
    write_fields = {"frce_out": "artifact_references"}
    subscription_fields = ("current_input", "artifact_references")
    sensitivity_weights = {"current_input": 1.0, "artifact_references": 0.5}
    activation_threshold = 0.5

    def __init__(self, coordinator: FormalReasoningCoordinator | None = None) -> None:
        self.coordinator = coordinator or FormalReasoningCoordinator()

    def describe(self) -> ModuleManifest:
        """Return this specialist's manifest (blackboard Step 6 compatibility)."""
        return ModuleManifest(
            identity=ModuleIdentity(
                module_id=self.module_id,
                module_version="1.0.0",
                module_name="Formal reasoning and constraint engine specialist",
                package_path=self.package_path,
            ),
            responsibilities=(
                "Formalize selected knowledge into a typed IR, run rule inference, "
                "consistency and constraint checks, optimization and proofs, and "
                "publish typed result envelopes.",
            ),
            dependencies=(
                ModuleDependency("usg_knowledge", "ORE.USG.KNOWLEDGE", ("0.1.0",)),
            ),
            input_ports=(
                InputPort("task_frame", FRCE_REQUEST_CONTRACT, (FRCE_CONTRACT_VERSION,),
                          required=True, expected_data_type="builtins.dict"),
            ),
            output_ports=(
                OutputPort("frce_out", FRCE_FINDING_BATCH_CONTRACT, FRCE_CONTRACT_VERSION,
                           "builtins.dict", "Typed FRCE result findings."),
            ),
            capabilities=(
                "reasoning.rule_inference",
                "reasoning.consistency",
                "reasoning.optimization",
                "reasoning.proof_checking",
            ),
            limitations=(
                "Read-only against authoritative USG; derived facts are proposals.",
                "UNKNOWN/timeout never imply falsity; results carry distinct statuses.",
            ),
        )

    def execute(self, inputs: Mapping[str, Any], context: Any) -> ModuleResult:
        """Run FRCE over the task frame and emit the typed finding batch."""
        frame = inputs["task_frame"].payload
        request = FormalizationRequest(
            request_id=str(frame.get("request_id", "blackboard-frce")),
            objective=str(frame.get("objective", "")),
            task_classes=tuple(
                TaskClass(value) for value in frame.get("task_classes", ("rule_inference",))
            ),
            required_property=frame.get("required_property"),
            assumptions=tuple(frame.get("assumptions", ()) or ()),
        )
        # The frame may carry a pre-built IR or typed USG records for the run.
        run = self.coordinator.run(
            request,
            ir=frame.get("ir"),
            usg_records=frame.get("usg_records"),
        )
        findings = []
        for result in run.results:
            findings.append({
                "field": "artifact_references",
                "key": result.result_id,
                "finding_type": f"frce_{result.kind.value}",
                "payload": result.to_dict(),
                "source": "formal_reasoning_engine",
                "rules_applied": list(result.statement_ir_refs),
                "confidence": None,
                "dependencies": [],
            })
        artifact = ContractArtifact(
            envelope=ContractEnvelope(
                contract_id=FRCE_FINDING_BATCH_CONTRACT,
                schema_version=FRCE_CONTRACT_VERSION,
                run_id=getattr(context, "run_id", "frce-run") if context else "frce-run",
                producer_id=self.module_id,
                config_hash="0" * 64,
                created_at=__import__("ore_workspace.models", fromlist=["utc_now"]).utc_now(),
                evidence_kind=EvidenceKind.DERIVED,
            ),
            payload={
                "findings": findings,
                "diagnostics": {
                    "status": run.status,
                    "steps_executed": len(run.step_trace),
                    "failures": list(run.failures),
                },
            },
        )
        return ModuleResult(
            module_id=self.module_id,
            module_version="1.0.0",
            status=ModuleStatus.SUCCESS if run.status != "error" else ModuleStatus.FAILED,
            artifacts=(artifact,),
            diagnostics={"status": run.status, "findings": len(findings)},
        )


__all__ = [
    "FRCE_SPECIALIST_MODULE_ID",
    "FRCE_REQUEST_CONTRACT",
    "FRCE_RESULT_CONTRACT",
    "FRCE_FINDING_BATCH_CONTRACT",
    "FRCE_CONTRACT_VERSION",
    "register_formal_reasoning_contracts",
    "FormalReasoningSpecialist",
]
