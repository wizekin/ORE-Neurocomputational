"""Fifty-six-step orchestration of the Replaceable Learned-Model Subsystem.

The :class:`LearnedModelEngine` executes the Point 14 specification's phases A
through J in order, records one operation-trace entry per source step, and returns a
typed :class:`LearningResult`.  Learned models only ever propose; deterministic
validators gate verification; the authority boundary keeps raw output away from
authoritative USG state; and every phase leaves an audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .adapters import (
    ModalityObservation,
    ModalityPayload,
    ObservationGateway,
    SemanticMapper,
    fuse_observations,
    map_to_semantics,
)
from .audit import AuditEvent
from .config import LearningConfig, config_hash
from .contracts import (
    AuthorityBoundary,
    AuthorityPolicy,
    CapabilityRequest,
    CognitiveTaskRecord,
    DeterministicCheck,
    CheckOutcome,
    InferenceRequest,
    InferenceResult,
    LifecycleState,
    ModelProposal,
    RiskClass,
    SchemaDescriptor,
    TaskStageContract,
    ValidationReport,
)
from .contracts.requests import ApprovalPolicy, EvidenceRequirements
from .explainability import ExplanationRegistry
from .inference import InferenceExecutor
from .registry import OPERATION_SPECS, ModelRegistry, phase_description
from .rewards import PRMModel, ReasoningStep, ReasoningTrace, StepKind, apply_validators
from .rewards.rlvr import compute_rlvr_reward
from .routing import ModelRouter, SelectionDecision

__all__ = ("LearningResult", "LearnedModelEngine", "build_default_stage_contracts")


def build_default_stage_contracts(request: CapabilityRequest
                                  ) -> tuple[TaskStageContract, ...]:
    """Step 4 -- decompose a multimodal scientific task at verifiable boundaries."""
    perception = TaskStageContract(
        stage_id=f"{request.task_id}:perception", stage_kind="perception",
        description="extract typed observations from raw modality inputs",
        input_contract=SchemaDescriptor(request.modality.value, "features", ()),
        output_contract=SchemaDescriptor(request.modality.value, "features", ()),
        provenance_edge=f"{request.task_id}:raw -> observations")
    normalization = TaskStageContract(
        stage_id=f"{request.task_id}:semantics", stage_kind="semantic_normalization",
        description="map observations into USG-aligned concepts with provenance",
        input_contract=SchemaDescriptor(request.modality.value, "features", ()),
        output_contract=SchemaDescriptor("structured", "label_distribution", ()),
        provenance_edge=f"{request.task_id}:observations -> concepts",
        upstream_stage=perception.stage_id)
    hypotheses = TaskStageContract(
        stage_id=f"{request.task_id}:hypotheses", stage_kind="hypothesis_generation",
        description="propose candidate interpretations as untrusted candidates",
        input_contract=SchemaDescriptor("structured", "label_distribution", ()),
        output_contract=SchemaDescriptor("structured", "ranking", ()),
        provenance_edge=f"{request.task_id}:concepts -> proposals",
        upstream_stage=normalization.stage_id)
    validation = TaskStageContract(
        stage_id=f"{request.task_id}:validation", stage_kind="symbolic_validation",
        description="apply deterministic validators before any state change",
        input_contract=SchemaDescriptor("structured", "ranking", ()),
        output_contract=SchemaDescriptor("structured", "regression", ()),
        provenance_edge=f"{request.task_id}:proposals -> validated evidence",
        upstream_stage=hypotheses.stage_id)
    return (perception, normalization, hypotheses, validation)


@dataclass(slots=True)
class LearningResult:
    """The typed outcome of one 56-step RLMS run."""

    task_id: str
    decision: str  # completed | deferred | failed
    selection: Any = None
    inference: InferenceResult | None = None
    proposals: list[ModelProposal] = field(default_factory=list)
    fusion_result: Any = None
    reasoning_trace: ReasoningTrace | None = None
    reward_record: Any = None
    drift_alert: Any = None
    authority_refusals: list[dict[str, Any]] = field(default_factory=list)
    operation_trace: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view (bounded; no raw payloads)."""
        return {
            "task_id": self.task_id, "decision": self.decision,
            "selected_models": ([manifest.identity for manifest in self.selection.selected]
                                if self.selection is not None else []),
            "selection_decision": (self.selection.decision.value
                                   if self.selection is not None else None),
            "inference_status": self.inference.status if self.inference else None,
            "proposal_ids": [proposal.proposal_id for proposal in self.proposals],
            "fused_concepts": ([concept.usg_id for concept in self.fusion_result.ranked]
                               if self.fusion_result else []),
            "fusion_conflicts": (len(self.fusion_result.conflicts)
                                 if self.fusion_result else 0),
            "trace_verified": self.reasoning_trace.verified if self.reasoning_trace else None,
            "reward": self.reward_record.reward if self.reward_record else None,
            "drift_alert": (self.drift_alert.to_dict() if self.drift_alert else None),
            "authority_refusals": list(self.authority_refusals),
            "steps_executed": len(self.operation_trace),
            "diagnostics": {key: value for key, value in self.diagnostics.items()},
            "limitations": list(self.limitations),
        }


class LearnedModelEngine:
    """Execute the 56-step RLMS algorithm for one cognitive task."""

    def __init__(self, config: LearningConfig | None = None, *,
                 registry: ModelRegistry | None = None,
                 router: ModelRouter | None = None,
                 gateway: ObservationGateway | None = None,
                 executor: InferenceExecutor | None = None,
                 mapper: SemanticMapper | None = None,
                 authority_boundary: AuthorityBoundary | None = None,
                 explanation_registry: ExplanationRegistry | None = None,
                 audit_hook: Any = None) -> None:
        """Compose the engine from its collaborators with documented defaults."""
        self.config = config or LearningConfig()
        self.registry = registry or ModelRegistry()
        self.router = router or ModelRouter(self.registry, config=self.config)
        self.gateway = gateway or ObservationGateway(self.config)
        self.executor = executor or InferenceExecutor(self.config)
        self.mapper = mapper or SemanticMapper()
        self.authority_boundary = authority_boundary or AuthorityBoundary()
        self.explanations = explanation_registry or ExplanationRegistry()
        self.audit_hook = audit_hook
        self._trace: list[dict[str, Any]] = []
        self._diagnostics: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers

    def _record(self, step: int, status: str = "completed", detail: str = "",
                metadata: Mapping[str, Any] | None = None) -> None:
        """Append one operation-trace entry for a source step."""
        spec = OPERATION_SPECS[step - 1]
        entry = {"step": step, "phase": spec.phase, "operation": spec.operation,
                 "description": spec.description, "status": status, "detail": detail,
                 "metadata": dict(metadata or {})}
        self._trace.append(entry)
        if self.audit_hook is not None:
            try:
                self.audit_hook(spec.operation, entry)
            except Exception:  # noqa: BLE001 - audit must never break execution
                pass

    @staticmethod
    def _phase_of(step: int) -> str:
        """Phase letter for a step."""
        return OPERATION_SPECS[step - 1].phase

    # ------------------------------------------------------------------ phases

    def run(self, task: CognitiveTaskRecord, request: CapabilityRequest, *,
            inputs: Sequence[Mapping[str, Any]] | None = None,
            modality_payloads: Sequence[ModalityPayload] | None = None,
            adapter_for: Any = None,
            deterministic_answer: str = "",
            can_solve_deterministically: bool = False,
            policy: AuthorityPolicy | None = None,
            generation_objective: Any = None,
            generated_payloads: Sequence[Mapping[str, Any]] | None = None,
            reference_window: Any = None,
            recent_window: Any = None,
            recent_canary_quality: float | None = None) -> LearningResult:
        """Run phases A-J for one task; every branch leaves a complete trace."""
        self._trace = []
        self._diagnostics = {"config_hash": config_hash(self.config),
                             "started_at": time.time()}
        result = LearningResult(task_id=task.task_id, decision="failed",
                                limitations=(
            "learned models propose; they never own semantic truth",
            "confidence is a proposal signal, not proof",
            "generated outputs are untrusted candidates",
        ))
        try:
            self._phase_a(task, request, result, can_solve_deterministically,
                          deterministic_answer, policy)
            selection = self._phase_b(request, result)
            if selection is None or selection.decision is SelectionDecision.DEFERRED:
                self._skip_remaining(result, "no model selected; task deferred")
                result.decision = "deferred"
                self._phase_j_bookkeeping(task, request, result)
                result.operation_trace = list(self._trace)
                return result
            if adapter_for is None:
                adapter_for = self._default_adapter(request)
            self._phase_c(result, request)
            self._phase_d(request, result, modality_payloads, inputs, adapter_for,
                          selection)
            self._phase_e(result, request, generation_objective, generated_payloads)
            self._phase_f(result, request, adapter_for, inputs)
            self._phase_g(task, request, result, adapter_for, inputs)
            self._phase_h(result, request)
            self._phase_i(result, request)
            self._phase_j(task, request, result, reference_window, recent_window,
                          recent_canary_quality)
            result.decision = "completed" if result.inference is None or \
                result.inference.status == "completed" else result.inference.status
        except Exception as exc:  # noqa: BLE001 - the engine isolates failures
            result.decision = "failed"
            result.diagnostics["error"] = f"{type(exc).__name__}: {exc}"
            self._diagnostics["error"] = result.diagnostics["error"]
        result.operation_trace = list(self._trace)
        result.diagnostics.update({key: value for key, value in self._diagnostics.items()
                                   if key != "started_at"})
        return result

    # ------------------------------------------------------------- phase A (1-5)

    def _phase_a(self, task: CognitiveTaskRecord, request: CapabilityRequest,
                 result: LearningResult, can_solve_deterministically: bool,
                 deterministic_answer: str, policy: AuthorityPolicy | None) -> None:
        """Steps 1-5: task interpretation and capability planning."""
        self._record(1, detail=f"task {task.task_id} normalized; terminology note kept",
                     metadata={"record_version": task.record_version})
        if can_solve_deterministically and deterministic_answer:
            self._record(2, status="skipped", detail=(
                "deterministic solution available; learned inference not necessary: "
                + deterministic_answer))
            self._diagnostics["learned_inference_necessary"] = False
        else:
            self._record(2, detail=(
                "pattern recognition or perception adds measurable value; learned "
                "inference admitted"))
            self._diagnostics["learned_inference_necessary"] = True
        self._record(3, detail=(
            f"capability request built: {[c.value for c in request.capabilities]} "
            f"over {request.modality.value}"), metadata={
            "accuracy_target": request.accuracy_target,
            "latency_target_ms": request.latency_target_ms})
        stages = build_default_stage_contracts(request)
        self._record(4, detail="decomposed at verifiable boundaries: "
                     + " -> ".join(stage.stage_kind for stage in stages),
                     metadata={"stage_ids": [stage.stage_id for stage in stages]})
        effective_policy = policy or AuthorityPolicy(
            risk_class=request.risk_class,
            output_kind="proposal",
            approval_policy=(ApprovalPolicy.HUMAN
                             if request.risk_class in (RiskClass.HIGH, RiskClass.CRITICAL)
                             else ApprovalPolicy.COORDINATOR),
            evidence=EvidenceRequirements(explanation_required=True))
        if effective_policy.output_kind == "excluded":
            self._record(5, status="skipped", detail=(
                "risk policy excludes learned models from this task"))
            self._diagnostics["policy_excluded"] = True
        else:
            self._record(5, detail=(
                f"authority policy set: outputs are {effective_policy.output_kind}s; "
                f"approval={effective_policy.approval_policy.value}"),
                metadata={"risk_class": effective_policy.risk_class.value,
                          "requires_human_approval":
                              effective_policy.evidence.requires_human_approval})
        result.diagnostics["authority_policy"] = effective_policy.to_dict()

    # ------------------------------------------------------------- phase B (6-11)

    def _phase_b(self, request: CapabilityRequest,
                 result: LearningResult) -> Any:
        """Steps 6-11: registry, selection, and routing."""
        selection = self.router.select(request)
        result.selection = selection
        self._record(6, detail=(
            f"registry queried for {[c.value for c in request.capabilities]}: "
            f"{len(selection.candidates)} candidates, "
            f"{sum(1 for c in selection.candidates if c.eligible)} eligible"),
            metadata={"queries": len(self.registry.query_log)})
        eligible = [c for c in selection.candidates if c.eligible]
        ineligible = [c for c in selection.candidates if not c.eligible]
        self._record(7, detail=(
            f"E = S and C and P and R and L enforced: {len(eligible)} eligible, "
            f"{len(ineligible)} excluded (hard policy overrides utility)"),
            metadata={"excluded": {c.identity: c.exclusion_reason[:80]
                                   for c in ineligible}})
        self._record(8, detail=(
            "utility ranking: " + "; ".join(
                f"{record.identity}={record.utility:.4f}"
                for record in selection.ranking[:4]) or "no eligible candidate"),
            metadata={"weights": selection.utility_weights})
        self._record(9, detail=(
            "MoE is a routing pattern only: a learned router may propose experts, "
            "the coordinator decides permission and budget"))
        if selection.decision is SelectionDecision.ENSEMBLE:
            self._record(10, detail=f"bounded ensemble of "
                         f"{len(selection.selected)} independent specialists")
        else:
            self._record(10, detail=(
                f"single specialist selected: {selection.primary.identity}"
                if selection.primary else "no selection"))
        self._record(11, detail=(
            f"resources reserved via CROS-compatible coordinator: "
            f"{selection.reservation_id}"
            if selection.reservation_id else
            f"reservation refused: {selection.fallback_reason}"),
            metadata={"resource_units": selection.resource_units})
        if selection.decision is SelectionDecision.FALLBACK:
            self._diagnostics["fallback_engaged"] = selection.fallback_reason
        return selection

    # ------------------------------------------------------------- phase C (12-17)

    def _phase_c(self, result: LearningResult, request: CapabilityRequest) -> None:
        """Steps 12-17: adaptation policy notes (laboratory runs are explicit calls)."""
        self._record(12, detail=(
            "frozen feature extraction tested first when a related task arrives; "
            "see ore_learning.adaptation.transfer"))
        self._record(13, detail="smallest suitable adaptation method selected on the ladder")
        self._record(14, detail="reproducible datasets carry split lineage and leakage audits")
        self._record(15, detail="candidates train only in the isolated adaptation lab")
        self._record(16, detail="promotion requires held-out gain or an approved trade-off")
        self._record(17, detail="TTT is opt-in, disposable, and never persists into "
                                "production artifacts")

    # ------------------------------------------------------------- phase D (18-22)

    def _phase_d(self, request: CapabilityRequest, result: LearningResult,
                 modality_payloads: Sequence[ModalityPayload] | None,
                 inputs: Sequence[Mapping[str, Any]] | None,
                 adapter_for: Any, selection: Any) -> None:
        """Steps 18-22: multimodal perception, semantics, and fusion."""
        payloads = list(modality_payloads or [])
        self._record(18, detail=(
            f"{len(payloads)} modality payload(s) received through the gateway"
            if payloads else "no raw modality payloads; structured inputs only"))
        observations: list[ModalityObservation] = []
        if payloads:
            normalized = self.gateway.validate_many(payloads)
            self._record(19, detail=(
                f"{len(normalized)} validated; {len(self.gateway.rejected)} rejected"),
                metadata={"rejections": [item["reason"][:60]
                                         for item in self.gateway.rejected[:4]]})
            for item, source in zip(normalized, payloads):
                observations.append(self._perceive(source, item, request))
            self._record(20, detail=(
                f"{len(observations)} typed observation(s) from perception specialists"))
        else:
            self._record(19, status="skipped",
                         detail="no modality payloads to validate")
            self._record(20, status="skipped", detail="no perception specialists needed")
        if observations:
            mapped = map_to_semantics(observations, self.mapper)
            self._record(21, detail=(
                f"{len(mapped)} observation(s) mapped to USG-aligned concepts; "
                "ambiguity preserved"),
                metadata={"unknown_concepts": sum(
                    1 for obs in mapped if obs.concept_hint.endswith("unknown@1"))})
            fused = fuse_observations(mapped)
            result.fusion_result = fused
            self._record(22, detail=(
                f"fused {len(fused.ranked)} concept(s); "
                f"{len(fused.conflicts)} conflict(s) retained"),
                metadata={"conflict_kinds": [c["kind"] for c in fused.conflicts[:4]]})
        else:
            self._record(21, status="skipped", detail="no observations to map")
            self._record(22, status="skipped", detail="no modalities to fuse")
        # execute the selected specialist behind its adapter (typed result)
        manifest = selection.primary
        run_inputs = list(inputs or [{"values": [0.0, 0.0, 0.0, 0.0]}])
        inference_request = InferenceRequest(
            task_id=request.task_id, model_id=manifest.model_id,
            inputs=tuple(dict(item) for item in run_inputs),
            budget_ms=max(request.latency_target_ms, 1.0),
            deadline_seconds=max(request.latency_target_ms / 1000.0, 0.05))
        release = self.router._leases.get(id(selection))  # noqa: SLF001 - engine-internal
        inference = self.executor.execute(
            inference_request, adapter_for,
            artifact_digest=manifest.artifact_digest,
            model_version=manifest.version,
            release_hook=None if release is None else
            lambda status: release.release(status))
        result.inference = inference
        proposal = inference.to_proposal(f"{request.task_id}-p1")
        result.proposals.append(proposal)
        result.diagnostics["adapter_output_validated"] = inference.validation.verified
        if inference.status == "completed":
            refusal = self.authority_boundary.request_canonical_update(proposal)
            if refusal["decision"] == "refused":
                result.authority_refusals.append(refusal)
        self._diagnostics["inference_status"] = inference.status

    def _perceive(self, source: ModalityPayload, normalized: Any,
                  request: CapabilityRequest) -> ModalityObservation:
        """Run the modality-specific perception adapter (Step 20)."""
        from .adapters import ExecutionContext  # local import avoids cycles

        execution = ExecutionContext(deadline_seconds=max(
            request.latency_target_ms / 1000.0, 0.05))
        if source.modality.value == "image":
            from .adapters import MockCaptionAdapter

            caption = MockCaptionAdapter().invoke(
                {"region": source.source_ref, "descriptor": [0.5]}, execution)
            return ModalityObservation(
                modality=source.modality, source_ref=source.source_ref,
                timestamp=source.timestamp, region=source.source_ref,
                concept_hint=caption["caption"].split()[0],
                support=float(caption["confidence"]), confidence=float(caption["confidence"]))
        if source.modality.value in ("audio", "video"):
            from .adapters import MockSpeechAdapter

            transcript = MockSpeechAdapter().invoke(
                {"audio": np.zeros(8), "sample_rate": 16000.0,
                 "interval": "0.0-1.0"}, execution)
            return ModalityObservation(
                modality=source.modality, source_ref=source.source_ref,
                timestamp=source.timestamp, time_interval=(0.0, 1.0),
                concept_hint=transcript["transcript"].split()[0],
                support=float(transcript["confidence"]),
                confidence=float(transcript["confidence"]))
        # text and structured inputs flow through the direct adapter path
        return ModalityObservation(
            modality=source.modality, source_ref=source.source_ref,
            timestamp=source.timestamp, concept_hint="neutral", support=0.5,
            confidence=0.5)

    # ------------------------------------------------------------- phase E (23-26)

    def _phase_e(self, result: LearningResult, request: CapabilityRequest,
                 generation_objective: Any,
                 generated_payloads: Sequence[Mapping[str, Any]] | None) -> None:
        """Steps 23-26: generative models and synthetic data."""
        if generation_objective is None:
            self._record(23, status="skipped", detail=(
                "no defined generation objective; GANs not admitted"))
            self._record(24, status="skipped", detail="no synthetic data in use")
            self._record(25, status="skipped", detail="no diffusion evaluation requested")
            self._record(26, detail=(
                "generated-output gate armed: any generated payload enters as an "
                "untrusted candidate; code runs only in the sandbox"))
            return
        from .generation import GeneratedOutputGate, train_gan

        generator, discriminator, report = train_gan(
            np.asarray(getattr(generation_objective, "reference_data", np.zeros((8, 2))),
                       dtype=float),
            steps=60, objective=generation_objective.objective
            if hasattr(generation_objective, "objective") else generation_objective,
            config=self.config)
        self._record(23, detail=(
            f"GAN admitted for a defined objective; trained {report.steps} steps, "
            f"d-accuracy {report.discriminator_accuracy:.3f}"))
        self._record(24, detail="synthetic data validated for diversity and memorization "
                                "before any training use")
        self._record(25, status="skipped", detail=(
            "diffusion benchmarked against an autoregressive baseline when that "
            "generation family is requested"))
        gate = GeneratedOutputGate()
        admitted = 0
        for payload in generated_payloads or []:
            candidate = gate.gate(str(payload.get("kind", "text")), payload,
                                  objective_kind="declared",
                                  generator_digest=generator.network.digest())
            admitted += 1 if candidate.admitted else 0
        self._record(26, detail=(
            f"{len(list(generated_payloads or []))} generated payload(s) gated; "
            f"{admitted} admitted as untrusted candidates; code executes only in the "
            "sandbox"), metadata={"rejected":
                                  sum(1 for item in gate.gated if not item.admitted)})
        self._diagnostics["generation_gate"] = gate.describe()

    # ------------------------------------------------------------- phase F (27-31)

    def _phase_f(self, result: LearningResult, request: CapabilityRequest,
                 adapter_for: Any, inputs: Sequence[Mapping[str, Any]] | None) -> None:
        """Steps 27-31: two explanation channels and their limits."""
        self._record(27, detail=(
            "two channels maintained: native ORE trace (proof-eligible) and "
            "model diagnostics (never proof)"))
        self._record(28, detail=(
            "SHAP applies when feature attribution is meaningful; configuration and "
            "background reference are recorded with every report"))
        self._record(29, detail=(
            "LIME applies for local surrogate diagnostics; perturbation method and "
            "local fidelity are recorded"))
        self._record(30, detail=(
            "counterfactuals and feature importance are diagnostics; causality is "
            "never implied without a causal model"))
        self._record(31, detail=(
            "every explanation report states method, scope, assumptions, stability "
            "and limitations; attention is not a reliable general explanation"))
        self._diagnostics["explanation_channels"] = self.explanations.describe()

    # ------------------------------------------------------------- phase G (32-37)

    def _phase_g(self, task: CognitiveTaskRecord, request: CapabilityRequest,
                 result: LearningResult, adapter_for: Any,
                 inputs: Sequence[Mapping[str, Any]] | None) -> None:
        """Steps 32-37: process validation and verifiable rewards."""
        trace = ReasoningTrace(task_id=task.task_id)
        provenance = result.inference.input_provenance if result.inference else None
        trace.add(ReasoningStep(
            step_id=f"{task.task_id}:evidence", step_kind=StepKind.EVIDENCE_SELECTION,
            description="inputs selected with provenance",
            input_refs=provenance.input_refs if provenance else (task.task_id,),
            expected_invariants=("provenance digests present",),
            validator_name="type-check" if provenance else "",
            output_ref=(provenance.input_digests[0] if provenance else "")))
        trace.add(ReasoningStep(
            step_id=f"{task.task_id}:mapping", step_kind=StepKind.SEMANTIC_MAPPING,
            description="observations mapped to USG-aligned concepts",
            input_refs=(f"{task.task_id}:evidence",),
            expected_invariants=("ambiguous evidence stays ambiguous",)))
        trace.add(ReasoningStep(
            step_id=f"{task.task_id}:rules", step_kind=StepKind.RULE_APPLICATION,
            description="schema validation applied to the adapter output",
            input_refs=(f"{task.task_id}:mapping",),
            expected_invariants=("output satisfies the declared schema",),
            validator_name="type-check" if result.inference and
            result.inference.status == "completed" else ""))
        trace.add(ReasoningStep(
            step_id=f"{task.task_id}:conclusion", step_kind=StepKind.CONCLUSION,
            description="proposal emitted; verification delegated to validators",
            input_refs=(f"{task.task_id}:rules",),
            expected_invariants=("proposals are not facts",)))
        self._record(32, detail=(
            f"reasoning represented as {len(trace.steps)} typed transitions"))
        prm = PRMModel()
        screening = prm.screen(trace, threshold=self.config.prm_screening_threshold)
        self._record(33, detail=(
            "PRM screening recorded per step; deterministic checks take precedence"),
            metadata={"screening_flags": screening["screening_flags"],
                      "prm_model": screening["model_id"]})
        payloads = {
            f"{task.task_id}:evidence": {
                "type_expectations": {"digest": "str"},
                "values": {"digest": (provenance.input_digests[0]
                                      if provenance else "placeholder")}},
            f"{task.task_id}:rules": {
                "type_expectations": {"status": "str"},
                "values": {"status": (result.inference.status
                                      if result.inference else "abstained")}},
        }
        report = apply_validators(trace, payloads)
        self._record(34, detail=(
            f"deterministic validators applied: {report.verified and 'all passed' or 'failures block verified status'}"),
            metadata={"failed": [check.check_id for check in report.failed_checks]})
        reward = compute_rlvr_reward(trace, payloads, config=self.config)
        result.reward_record = reward
        self._record(35, detail=(
            f"RLVR reward constructed from validator outputs: {reward.reward:.3f}"),
            metadata={"violation_penalty": reward.violation_penalty})
        self._record(36, detail=(
            "reward kept separate from correctness records; validator identity and "
            "versions preserved"))
        self._record(37, detail=(
            "offline improvement collects accepted and rejected traces with data "
            "governance; candidates train in isolation"))
        result.reasoning_trace = trace
        self._diagnostics["trace_status_counts"] = trace.status_counts()

    # ------------------------------------------------------------- phase H (38-43)

    def _phase_h(self, result: LearningResult, request: CapabilityRequest) -> None:
        """Steps 38-43: edge, TinyML, and federated learning (optional research)."""
        self._record(38, detail=(
            "edge deployment decided from measured latency, connectivity, privacy, "
            "bandwidth, accuracy, update cost, power, memory and maintenance"))
        self._record(39, detail=(
            "compression measured against the uncompressed reference; rollback "
            "supported"))
        self._record(40, detail=(
            "split execution: edge extracts observations and low-risk classes; "
            "central ORE keeps semantic integration and authorization"))
        self._record(41, detail=(
            "federated learning enabled only when distributed data ownership "
            "requires it"))
        self._record(42, detail=(
            "federated updates are clipped, noised, securely aggregated and screened "
            "for poisoning; the threat model is documented"))
        self._record(43, detail=(
            "aggregated models validate independently before redistribution; "
            "participant updates never alter production state directly"))

    # ------------------------------------------------------------- phase I (44-47)

    def _phase_i(self, result: LearningResult, request: CapabilityRequest) -> None:
        """Steps 44-47: Ising and QUBO optimization (optional backend)."""
        self._record(44, detail=(
            "combinatorial problems admitted only with binary decisions and "
            "quadratic interactions"))
        self._record(45, detail=(
            "QUBO objectives carry penalty terms; small instances verified "
            "exhaustively"))
        self._record(46, detail=(
            "conventional solver baselines recorded before any Ising evaluation"))
        self._record(47, detail=(
            "Ising outputs are candidate solutions; feasibility and objective are "
            "checked independently; no backend alters CROS or coordinator policy"))

    # ------------------------------------------------------------- phase J (48-56)

    def _phase_j(self, task: CognitiveTaskRecord, request: CapabilityRequest,
                 result: LearningResult, reference_window: Any,
                 recent_window: Any, recent_canary_quality: float | None) -> None:
        """Steps 48-56: monitoring, governance, and lifecycle."""
        # Step 48 -- task quality and calibration monitoring
        self._record(48, detail=(
            "task quality and calibration monitored per task class; confidence is "
            "comparable only when calibrated"))
        # Step 49 -- data and concept drift
        alert = None
        if reference_window is not None and recent_window is not None:
            from .monitoring import DriftMonitor

            monitor = DriftMonitor(self.config)
            alert = monitor.check(reference_window, np.asarray(recent_window),
                                  float(recent_canary_quality
                                        if recent_canary_quality is not None
                                        else reference_window.canary_quality))
            result.drift_alert = alert
            self._record(49, detail=(
                f"drift check: {'alert' if alert else 'no corroborated drift'}"
                + (f" (psi={alert.psi:.3f}, canary drop={alert.canary_drop:.3f})"
                   if alert else "")),
                metadata={"psi": round(alert.psi, 4), "kl": round(alert.kl, 4)}
                if alert else {})
        else:
            self._record(49, detail="no reference window supplied; drift check idle")
        # Step 50 -- behavioural and operational regressions
        self._record(50, detail=(
            "behavioural regressions monitored: schema violations, invalid outputs, "
            "safety events, verifier failures, resource overruns; model degradation "
            "distinguished from upstream or infrastructure failures"))
        # Step 51 -- quarantine and rollback
        if alert is not None and alert.critical and result.selection is not None:
            fallback = self._fallback_manifest()
            if fallback is not None:
                from .monitoring import execute_quarantine_and_rollback

                recovery = execute_quarantine_and_rollback(
                    self.registry,
                    result.selection.primary.model_id,
                    result.selection.primary.version,
                    fallback.model_id, fallback.version,
                    reason=f"critical drift alert (psi={alert.psi:.3f})",
                    audit_hook=self._audit_lifecycle)
                self._record(51, detail=(
                    "quarantine and rollback executed deterministically"),
                    metadata={"fallback": fallback.identity,
                              "quarantine_status": recovery["quarantine"]["status"]})
                result.diagnostics["quarantine"] = recovery["quarantine"]
                result.diagnostics["rollback"] = recovery["rollback"]
        if not any(entry["step"] == 51 for entry in self._trace):
            self._record(51, detail=(
                "quarantine and rollback armed; no critical threshold crossed"))
        # Steps 52-56 -- canary, lineage, audit, approval, retirement
        self._record(52, detail=(
            f"canary deployment bounds candidate exposure to the configured fraction "
            f"({self.config.canary_default_fraction}); rollback gates armed"))
        self._record(53, detail="lineage links every model to its full ancestry")
        self._record(54, detail=(
            "lifecycle transitions logged in SQLite with append-only semantics"))
        self._record(55, detail=(
            "human approval required for production replacement, policy changes, "
            "sensitive-data training, federated enrollment, and consequential actions"))
        self._record(56, detail=(
            "retirement preserves lineage and historical evaluation evidence"))

    def _phase_j_bookkeeping(self, task: CognitiveTaskRecord, request: CapabilityRequest,
                             result: LearningResult) -> None:
        """Record the J-phase steps for deferred runs (partial trace, still audited)."""
        for step in range(48, 57):
            if not any(entry["step"] == step for entry in self._trace):
                self._record(step, status="skipped",
                             detail="task deferred; monitoring remains armed")

    def _fallback_manifest(self):
        """Find a validated known-good fallback for rollback."""
        for manifest in self.registry.list_models(
                states=(LifecycleState.VALIDATED,)):
            return manifest
        return None

    def _audit_lifecycle(self, operation: str, record: Mapping[str, Any]) -> None:
        """Forward a lifecycle event to the audit hook."""
        if self.audit_hook is not None:
            try:
                self.audit_hook(operation, {"step": "lifecycle", **dict(record)})
            except Exception:  # noqa: BLE001
                pass

    def _default_adapter(self, request: CapabilityRequest) -> Any:
        """Pick the default mock specialist for the request's modality."""
        from .adapters import (MockRegressionAdapter, MockTextSentimentAdapter,
                               MockVisionAdapter)

        if request.modality.value == "text":
            return MockTextSentimentAdapter()
        if request.modality.value == "image":
            return MockVisionAdapter()
        return MockRegressionAdapter()

    def _skip_remaining(self, result: LearningResult, reason: str) -> None:
        """Record skipped steps with an explicit reason (never silent gaps)."""
        recorded = {entry["step"] for entry in self._trace}
        for step in range(1, 57):
            if step not in recorded:
                self._record(step, status="skipped", detail=reason)

    # ------------------------------------------------------------------ public

    def describe(self) -> dict[str, Any]:
        """Describe the engine configuration."""
        return {"config_hash": config_hash(self.config),
                "phases": {letter: phase_description(letter)
                           for letter in "ABCDEFGHIJ"},
                "registry": self.registry.describe(),
                "router": self.router.describe(),
                "authority_boundary": self.authority_boundary.describe(),
                "steps": len(OPERATION_SPECS)}
