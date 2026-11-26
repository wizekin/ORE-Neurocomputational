"""The 52-step HIC compilation coordinator with gates and human approval.

Implements the integrated pipeline of spec Section 5 and the data flow and
approval gates of Section 7:

    input -> semantic interpretation -> ambiguity analysis -> requirements ->
    math model -> AlgoIR -> IR validation -> candidate discovery ->
    independent verification -> code generation (approved candidates only) ->
    sandboxed execution -> human approval -> controlled execution -> learning.

No stage may silently convert uncertainty into fact: blocking ambiguities and
unconfirmed high-impact assumptions stop the pipeline before code generation;
a changed artifact invalidates its approval and requires renewal.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from .algoir import AlgoIRBuilder, AlgoIRGraph
from .ambiguity import AmbiguityAnalyzer, AmbiguitySet
from .codegen import CodeArtifact, PythonGenerator
from .contracts import (
    ArtifactKind,
    CanonicalArtifact,
    GateViolation,
    NodeType,
)
from .cros_adapter import CROSAdapter
from .discovery import AlgorithmDiscovery, CandidateSet
from .input import Normalizer, TextInputAdapter
from .learning import LearningJournal, LearningRecord
from .mathmodel import MathFormalizer, MathModel
from .requirements import RequirementsFormalizer, RequirementSpec
from .sandbox import ExecutionReport, SandboxExecutor, SandboxPolicy
from .segmentation import IdeaSegmenter
from .semantic import SemanticGraph, SemanticInterpreter
from .session import CompilationSession, InputRecord, SessionSettings
from .usg_adapter import HICUSGAdapter
from .verification import VerificationReport, Verifier

# ---------------------------------------------------------------------------
# Approval records (spec 2.2, Section 7)
# ---------------------------------------------------------------------------

class ApprovalRecord(CanonicalArtifact):
    """Human approval bound to an exact artifact digest and scope."""

    kind = ArtifactKind.APPROVAL_RECORD

    def __init__(
        self,
        *,
        artifact_id: str,
        approved_digest: str,
        scope: str,
        permissions: tuple[str, ...],
        approved_by: str,
        expires_at: str,
        auto: bool = False,
        version: int = 1,
    ) -> None:
        if not approved_by.strip():
            raise GateViolation("approval requires a human approver identity")
        self.artifact_id = artifact_id
        self.approved_digest = approved_digest
        self.scope = scope
        self.permissions = tuple(permissions)
        self.approved_by = approved_by
        self.expires_at = expires_at
        self.auto = auto
        self.version = version

    def is_valid_at(self, moment: str) -> bool:
        return moment <= self.expires_at

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "approved_digest": self.approved_digest,
            "scope": self.scope,
            "permissions": list(self.permissions),
            "approved_by": self.approved_by,
            "expires_at": self.expires_at,
            "auto": self.auto,
            "version": self.version,
        }

    @property
    def semantic_digest(self) -> str:
        """Digest over the approval decision; issue-time metadata excluded.

        ``expires_at`` is wall-clock data, not part of the approved content:
        the same decision (digest, scope, permissions, approver) must yield
        the same semantic digest across deterministic recompilations.
        """
        from .contracts import digest_payload

        payload = self.content_payload()
        payload.pop("artifact_id", None)
        payload.pop("expires_at", None)
        return digest_payload(payload)


# ---------------------------------------------------------------------------
# Step log
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class StepRecord:
    """One of the 52 pipeline steps and its outcome."""

    step_id: str
    phase: str
    title: str
    status: str  # EXECUTED | SKIPPED | BLOCKED
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "phase": self.phase,
            "title": self.title,
            "status": self.status,
            "detail": self.detail,
        }


# ---------------------------------------------------------------------------
# Compilation result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class CompilationResult:
    """Everything one compilation run produced, with gates and lineage."""

    session: CompilationSession
    status: str  # COMPLETED | AWAITING_CLARIFICATION | AWAITING_APPROVAL | IR_REJECTED | VERIFICATION_FAILED | NO_CANDIDATE
    input_record: InputRecord
    semantic_graph: SemanticGraph | None
    ambiguity_set: AmbiguitySet | None
    requirement_spec: RequirementSpec | None
    math_model: MathModel | None
    algo_ir: AlgoIRGraph | None
    ir_violations: tuple[Any, ...]
    candidate_set: CandidateSet | None
    verification_report: VerificationReport | None
    code_artifact: CodeArtifact | None
    execution_report: ExecutionReport | None
    approval: ApprovalRecord | None
    learning_record: LearningRecord | None
    step_log: tuple[StepRecord, ...]
    explanation: str

    @property
    def clarification_questions(self) -> tuple[Any, ...]:
        return self.ambiguity_set.questions if self.ambiguity_set else ()

    def lineage(self) -> list[Any]:
        digest = self._final_digest()
        return self.session.ledger.reconstruct_lineage(digest) if digest else []

    def _final_digest(self) -> str | None:
        for artifact in (self.learning_record, self.execution_report, self.code_artifact,
                         self.verification_report, self.candidate_set, self.algo_ir,
                         self.math_model, self.requirement_spec, self.ambiguity_set,
                         self.semantic_graph, self.input_record):
            if artifact is not None:
                return artifact.content_digest
        return None

    def step_log_dicts(self) -> list[dict[str, Any]]:
        return [s.to_dict() for s in self.step_log]


# ---------------------------------------------------------------------------
# The compiler
# ---------------------------------------------------------------------------

class HICCompiler:
    """The 52-step Human Idea-to-Algorithm Compiler (spec Section 5)."""

    PHASES = (
        "A: Capture and normalize human intent (Steps 1-7)",
        "B: Semantic interpretation and USG grounding (Steps 8-16)",
        "C: Formalize requirements and behavior (Steps 17-26)",
        "D: Mathematical formalization (Steps 27-33)",
        "E: Generate and validate AlgoIR (Steps 34-39)",
        "F: Algorithm discovery and program synthesis (Steps 40-44)",
        "G: Verification and independent validation (Steps 45-49)",
        "H: Code generation, sandboxing and learning (Steps 50-52)",
    )

    def __init__(
        self,
        session: CompilationSession,
        *,
        usg: HICUSGAdapter | None = None,
        cros: CROSAdapter | None = None,
        verifier: Verifier | None = None,
    ) -> None:
        self.session = session
        self.usg = usg or HICUSGAdapter()
        self.cros = cros or CROSAdapter()
        self.verifier = verifier or Verifier(
            cases=session.settings.property_test_cases,
            seed=session.settings.property_test_seed,
        )
        self.segmenter = IdeaSegmenter()
        self.interpreter = SemanticInterpreter(self.usg)
        self.ambiguity_analyzer = AmbiguityAnalyzer()
        self.formalizer = RequirementsFormalizer()
        self.math_formalizer = MathFormalizer()
        self.ir_builder = AlgoIRBuilder()
        self.discovery = AlgorithmDiscovery()
        self.generator = PythonGenerator()
        self.sandbox = SandboxExecutor(
            SandboxPolicy.default(
                cpu_seconds=session.settings.sandbox_cpu_seconds,
                memory_mb=session.settings.sandbox_memory_mb,
                wall_seconds=session.settings.sandbox_wall_seconds,
                output_byte_limit=session.settings.sandbox_output_byte_limit,
            )
        )
        self.journal = LearningJournal()
        self._steps: list[StepRecord] = []
        self._approvals: dict[str, ApprovalRecord] = {}

    # -- step log ------------------------------------------------------------

    def _log(self, step_id: str, phase: str, title: str, status: str = "EXECUTED", detail: str = "") -> None:
        self._steps.append(StepRecord(step_id, phase, title, status, detail))

    # -- approvals -------------------------------------------------------------

    def approve(
        self,
        artifact: CanonicalArtifact,
        *,
        approved_by: str,
        scope: str = "sandbox-execution",
        permissions: Iterable[str] = ("sandbox:execute",),
        ttl_hours: float = 24.0,
    ) -> ApprovalRecord:
        """Bind a human approval to the exact semantic digest and scope."""
        expires = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()
        record = ApprovalRecord(
            artifact_id=self.session.next_artifact_id(ArtifactKind.APPROVAL_RECORD),
            approved_digest=artifact.semantic_digest,
            scope=scope,
            permissions=tuple(permissions),
            approved_by=approved_by,
            expires_at=expires,
        )
        self.session.register(record)
        self._approvals[record.approved_digest] = record
        self.session.ledger.record(
            stage="H",
            step_id="APPROVAL",
            description=f"Human approval by {approved_by} for scope '{scope}'",
            inputs=[artifact],
            outputs=[record],
            rule_references={"HIC-APPROVAL-1"},
        )
        self.session.ledger.audit(
            "APPROVAL_GRANTED", record.content_digest,
            f"digest={record.approved_digest}", f"scope={scope}", f"by={approved_by}",
        )
        return record

    def _require_approval(self, artifact: CanonicalArtifact) -> ApprovalRecord:
        record = self._approvals.get(artifact.semantic_digest)
        if record is None:
            if self.session.settings.allow_codegen_without_approval:
                # Explicit, audited auto-approval: never silent.
                record = self.approve(
                    artifact,
                    approved_by="session-auto-policy",
                    scope="sandbox-execution",
                    permissions=("sandbox:execute",),
                    ttl_hours=1.0,
                )
                record = ApprovalRecord(
                    artifact_id=record.artifact_id,
                    approved_digest=record.approved_digest,
                    scope=record.scope,
                    permissions=record.permissions,
                    approved_by=record.approved_by,
                    expires_at=record.expires_at,
                    auto=True,
                    version=record.version,
                )
                self._approvals[record.approved_digest] = record
                self.session.ledger.audit(
                    "AUTO_APPROVAL", record.content_digest,
                    "generated under allow_codegen_without_approval=true; scope limited to sandbox",
                )
            else:
                raise GateViolation(
                    "code generation requires human approval bound to the candidate-set digest "
                    "(spec Section 7); call compiler.approve(candidate_set, approved_by=...)"
                )
        now = datetime.now(timezone.utc).isoformat()
        if not record.is_valid_at(now):
            raise GateViolation("approval expired; a renewed approval is required")
        return record

    # -- the pipeline ------------------------------------------------------------

    def compile(
        self,
        idea: str,
        *,
        source_identity: str = "user",
        confirm_assumptions: Sequence[str] = (),
    ) -> CompilationResult:
        """Run all 52 steps with gates; pause before codegen when blocked.

        ``confirm_assumptions`` lists assumption ids the user confirms in this
        call (e.g. ``("as-model-linearity",)``); previously answered
        clarification questions stored via ``answer_clarifications`` also
        resolve their blocking ambiguities.
        """
        s = self.session

        # Phase A ---------------------------------------------------------------
        self.cros.allocate("A")
        self._log("HIC01", "A", "Create a compilation session", detail=s.session_id)
        adapter = TextInputAdapter()
        record = adapter.capture(s, idea, source_identity=source_identity)
        self._log("HIC02", "A", "Preserve the raw input", detail=f"digest={record.integrity_digest[:12]}")
        self._log("HIC03", "A", "Identify input modality and quality",
                  detail=f"modality={record.modality.value} flags={[f.value for f in record.quality_flags]}")
        normalizer = Normalizer()
        view = normalizer.normalize(record)
        self._log("HIC04", "A", "Normalize without semantic loss",
                  detail=f"corrections={list(view.corrections)}")
        self.cros.allocate("B")
        segmentation = self.segmenter.run(s, record, view)
        self._log("HIC05", "A", "Segment the idea into claims and intentions",
                  detail=f"{len(segmentation.segments)} segments")
        self._log("HIC06", "A", "Identify the requested deliverable",
                  detail=segmentation.deliverable.value)
        self._log("HIC07", "A", "Resolve project context",
                  detail=f"{sum(1 for f in segmentation.context_facts if f.status == 'RESOLVED')} resolved, "
                         f"{sum(1 for f in segmentation.context_facts if f.status == 'UNKNOWN')} unknown")

        # Phase B ---------------------------------------------------------------
        graph = self.interpreter.interpret(s, record, segmentation)
        self._log("HIC08", "B", "Extract candidate concepts and entities",
                  detail=f"{len(graph.concepts)} concepts, {len(graph.dangling_references)} unresolved")
        self._log("HIC09", "B", "Extract actions and relations", detail=f"{len(graph.relations)} relations")
        self._log("HIC10", "B", "Extract attributes, units and modalities", detail=f"{len(graph.attributes)} attributes")
        self._log("HIC11", "B", "Resolve semantic classes and inheritance", detail=self.usg.backend)
        self._log("HIC12", "B", "Build the preliminary semantic graph",
                  detail=f"{len(graph.concepts)} nodes")
        self._log("HIC13", "B", "Preserve competing interpretations",
                  detail=f"{len(graph.interpretations)} ambiguous phrases")
        resolved_from_answers = {
            key[len("answer:q-"):]
            for key in s.settings.context_facts
            if key.startswith("answer:q-")
        }
        confirmed_from_answers = {
            key[len("answer:"):]
            for key in s.settings.context_facts
            if key.startswith("answer:as-")
        }
        confirmed = set(confirm_assumptions) | confirmed_from_answers
        ambiguity = self.ambiguity_analyzer.analyze(
            s, graph, resolved=resolved_from_answers, confirmed=confirmed
        )
        self._log("HIC14", "B", "Detect ambiguity and missing information",
                  detail=f"{len(ambiguity.ambiguities)} ambiguities ({len(ambiguity.blocking)} blocking)")
        self._log("HIC15", "B", "Rank clarification questions",
                  detail=f"top IG={ambiguity.questions[0].information_gain_bits:.3f} bits"
                  if ambiguity.questions else "no questions")
        self._log("HIC16", "B", "Record assumptions explicitly",
                  detail=f"{len(ambiguity.assumptions)} assumptions "
                         f"({len(ambiguity.unconfirmed_high_impact_assumptions)} high-impact unconfirmed)")

        blocking = ambiguity.blocking or ambiguity.unconfirmed_high_impact_assumptions
        if blocking and not s.settings.allow_codegen_without_approval:
            # Gate: blocking uncertainty returns to the user; no code generation.
            self.session.ledger.audit(
                "GATE_BLOCKED", ambiguity.content_digest,
                "blocking ambiguity; pipeline paused before requirements formalization of code artifacts",
            )
            self._log("HIC17", "C", "Define the objective", "BLOCKED", "awaiting clarification")
            return self._result(
                "AWAITING_CLARIFICATION", record, graph, ambiguity, None, None, None, (),
                None, None, None, None, None, None,
            )

        # Phase C ---------------------------------------------------------------
        self.cros.allocate("C")
        spec = self.formalizer.formalize(s, graph, segmentation, ambiguity)
        self._log("HIC17", "C", "Define the objective", detail=spec.objectives[0].title)
        self._log("HIC18", "C", "Define inputs and outputs", detail=f"{len(spec.interfaces)} interfaces")
        self._log("HIC19", "C", "Define actors and responsibilities", detail=f"{len(spec.actors)} actors")
        self._log("HIC20", "C", "Define preconditions and postconditions",
                  detail=f"{len(spec.preconditions)} pre / {len(spec.postconditions)} post")
        self._log("HIC21", "C", "Define invariants", detail=f"{len(spec.invariants)} invariants")
        self._log("HIC22", "C", "Define constraints and priorities",
                  detail=f"{len(spec.hard_constraints())} hard / {len(spec.soft_constraints())} soft")
        self._log("HIC23", "C", "Define state transitions",
                  detail=f"{len(spec.state_machine.transitions)} legal transitions")
        self._log("HIC24", "C", "Define failure and recovery behavior",
                  detail=f"{len(spec.failure_cases)} failure cases")
        self._log("HIC25", "C", "Define termination and completion",
                  detail=spec.termination.termination_argument[:60])
        self._log("HIC26", "C", "Define measurable acceptance criteria",
                  detail=f"{len(spec.acceptance_criteria)} criteria")

        # Phase D ---------------------------------------------------------------
        self.cros.allocate("D")
        model = self.math_formalizer.formalize(s, graph, spec)
        structures = ", ".join(x.value for x in model.structures)
        self._log("HIC27", "D", "Identify mathematical structure", detail=structures)
        self._log("HIC28", "D", "Declare variables and domains", detail=f"{len(model.variables)} variables")
        self._log("HIC29", "D", "Formulate equations and constraints", detail=f"{len(model.equations)} equations")
        self._log("HIC30", "D", "Select objective functions", detail=f"{len(model.objectives)} objectives")
        invalid = model.invalid_assumptions
        self._log("HIC31", "D", "Validate model assumptions",
                  detail=f"{len(invalid)} unsupported" if invalid else "all assumptions supported",
                  status="EXECUTED" if not invalid else "BLOCKED")
        self._log("HIC32", "D", "Choose mathematical tools",
                  detail=", ".join(t.tool.value for t in model.tools if t.fits_structure))
        self._log("HIC33", "D", "Estimate computational complexity",
                  detail=f"{len(model.complexity)} estimates")

        # Phase E ---------------------------------------------------------------
        self.cros.allocate("E")
        ir = self.ir_builder.build(s, graph, spec, model)
        self._log("HIC34", "E", "Instantiate the AlgoIR schema", detail=f"{len(ir)} nodes")
        self._log("HIC35", "E", "Map semantic concepts to executable types",
                  detail=f"{len(ir.nodes_of(NodeType.TYPE))} types")
        self._log("HIC36", "E", "Generate operations and control flow",
                  detail=f"{len(ir.nodes_of(NodeType.OPERATION))} operations")
        self._log("HIC37", "E", "Attach constraints and invariants",
                  detail=f"{len(ir.nodes_of(NodeType.CONSTRAINT))} constraints")
        ir, violations, explanation = self.ir_builder.validate_and_explain(s, ir)
        self._log("HIC38", "E", "Validate AlgoIR well-formedness",
                  detail=f"{len(violations)} violations",
                  status="EXECUTED" if not violations else "BLOCKED")
        self._log("HIC39", "E", "Produce a human-readable explanation",
                  detail=f"{len(explanation.splitlines())} lines")
        if violations:
            self.session.ledger.audit(
                "IR_REJECTED", ir.content_digest,
                *(f"{v.code}:{v.node_id}" for v in violations),
            )
            return self._result(
                "IR_REJECTED", record, graph, ambiguity, spec, model, ir, violations,
                None, None, None, None, None, None,
            )

        # Phase F ---------------------------------------------------------------
        self.cros.allocate("F")
        candidate_set = self.discovery.discover(s, model)
        self._log("HIC40", "F", "Generate candidate algorithm families",
                  detail=f"{len(candidate_set.candidates)} candidates")
        self._log("HIC41", "F", "Build candidate-specific plans",
                  detail=f"{len({c.plan.external_specification for c in candidate_set.candidates})} shared spec")
        self._log("HIC42", "F", "Synthesize candidate procedures",
                  detail="template-based synthesis")
        self._log("HIC43", "F", "Compare candidates systematically",
                  detail=f"{len(candidate_set.comparisons)} comparisons")
        self._log("HIC44", "F", "Select or defer",
                  detail=f"selected={candidate_set.selected_id} deferred={candidate_set.deferred}")

        # Phase G ---------------------------------------------------------------
        self.cros.allocate("G")
        verification = self.verifier.verify(s, candidate_set, spec, ir)
        static_errors = verification.blocking_errors
        self._log("HIC45", "G", "Perform static checks",
                  detail=f"{len(static_errors)} errors",
                  status="EXECUTED" if not static_errors else "BLOCKED")
        self._log("HIC46", "G", "Discharge formal proof obligations where possible",
                  detail=f"{sum(1 for o in verification.obligations if o.status == 'DISCHARGED')} discharged / "
                         f"{len(verification.obligations)}")
        self._log("HIC47", "G", "Generate property-based and example tests",
                  detail=f"{len(verification.property_results)} properties")
        self._log("HIC48", "G", "Perform differential and metamorphic testing",
                  detail=f"{len(verification.differential_results)} differential results")
        self._log("HIC49", "G", "Review traceability and verification gaps",
                  detail=f"{len(verification.traceability.gaps)} gaps")
        hard_ok = {
            comp.candidate_id: comp.hard_constraints_satisfied for comp in candidate_set.comparisons
        }
        if verification.verdict != "PASSED" or not any(hard_ok.values()):
            self.session.ledger.audit(
                "VERIFICATION_GATE", verification.content_digest, "no candidate passed verification"
            )
            return self._result(
                "VERIFICATION_FAILED", record, graph, ambiguity, spec, model, ir, violations,
                candidate_set, verification, None, None, None, None,
            )
        # Re-select among verified candidates only (Step 43/44 gate).
        verified_candidates = [
            c for c, comp in zip(candidate_set.candidates, candidate_set.comparisons)
            if comp.hard_constraints_satisfied
        ]
        selected_id, shortlist, deferred = self.discovery.select(
            verified_candidates,
            [c for c in candidate_set.comparisons if c.candidate_id in {v.candidate_id for v in verified_candidates}],
        )
        if selected_id is None:
            return self._result(
                "NO_CANDIDATE", record, graph, ambiguity, spec, model, ir, violations,
                candidate_set, verification, None, None, None, None,
            )
        if selected_id != candidate_set.selected_id:
            candidate_set = CandidateSet(
                artifact_id=candidate_set.artifact_id,
                model_id=candidate_set.model_id,
                candidates=candidate_set.candidates,
                comparisons=candidate_set.comparisons,
                selected_id=selected_id,
                shortlist=shortlist,
                deferred=deferred,
                version=candidate_set.version + 1,
            )
            self.session.register(candidate_set)
            self._log("HIC44", "F", "Select or defer (post-verification re-selection)",
                      detail=f"selected={selected_id}")

        # Phase H ---------------------------------------------------------------
        self.cros.allocate("H")
        try:
            approval = self._require_approval(candidate_set)
        except GateViolation as exc:
            # Gate: code generation needs a human approval bound to this digest.
            self.session.ledger.audit(
                "APPROVAL_GATE", candidate_set.content_digest, str(exc)
            )
            self._log("HIC50", "H", "Generate target-language code", "BLOCKED",
                      "awaiting human approval of the candidate set digest")
            return self._result(
                "AWAITING_APPROVAL", record, graph, ambiguity, spec, model, ir, violations,
                candidate_set, verification, None, None, None, None, explanation,
            )
        code = self.generator.generate(s, ir, candidate_set, approved_digest=approval.approved_digest)
        self._log("HIC50", "H", "Generate target-language code",
                  detail=f"language={code.language} lines={len(code.source.splitlines())}")
        fixtures = (
            ("example", "generated.run_examples()"),
            ("empty-input-rejected", "generated.validate_inputs([], [])", True),
        )
        execution = self.sandbox.execute(s, code, fixtures)
        self._log("HIC51", "H", "Execute in a restricted environment",
                  detail=f"exit={execution.exit_code} outcomes={[o.status for o in execution.outcomes]}")
        lineage_digests = {
            "input_record": record.semantic_digest,
            "semantic_graph": graph.semantic_digest,
            "ambiguity_set": ambiguity.semantic_digest,
            "requirement_spec": spec.semantic_digest,
            "math_model": model.semantic_digest,
            "algo_ir": ir.semantic_digest,
            "candidate_set": candidate_set.semantic_digest,
            "verification_report": verification.semantic_digest,
            "code_artifact": code.semantic_digest,
            "execution_report": execution.semantic_digest,
            "approval_record": approval.semantic_digest,
        }
        learning = self.journal.evaluate(s, execution, spec, lineage_digests)
        self._log("HIC52", "H", "Evaluate, version and learn safely",
                  detail=f"{len(learning.proposals)} proposals")
        return self._result(
            "COMPLETED", record, graph, ambiguity, spec, model, ir, violations,
            candidate_set, verification, code, execution, approval, learning, explanation,
        )

    # -- helpers ---------------------------------------------------------------

    def _result(
        self,
        status: str,
        record: InputRecord,
        graph: SemanticGraph | None,
        ambiguity: AmbiguitySet | None,
        spec: RequirementSpec | None,
        model: MathModel | None,
        ir: AlgoIRGraph | None,
        violations: tuple[Any, ...],
        candidate_set: CandidateSet | None,
        verification: VerificationReport | None,
        code: CodeArtifact | None,
        execution: ExecutionReport | None,
        approval: ApprovalRecord | None,
        learning: LearningRecord | None,
        explanation: str = "",
    ) -> CompilationResult:
        return CompilationResult(
            session=self.session,
            status=status,
            input_record=record,
            semantic_graph=graph,
            ambiguity_set=ambiguity,
            requirement_spec=spec,
            math_model=model,
            algo_ir=ir,
            ir_violations=violations,
            candidate_set=candidate_set,
            verification_report=verification,
            code_artifact=code,
            execution_report=execution,
            approval=approval,
            learning_record=learning,
            step_log=tuple(self._steps),
            explanation=explanation,
        )

    def answer_clarifications(self, answers: Mapping[str, str]) -> None:
        """Record user answers to clarification questions (spec Step 15).

        Answers are stored as user-sourced facts on the session; a subsequent
        ``compile`` call will then pass the blocking gate.
        """
        for question_id, answer in answers.items():
            self.session.settings.context_facts[f"answer:{question_id}"] = str(answer)


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------

def compile_idea(
    idea: str,
    *,
    user: str = "user",
    settings: SessionSettings | None = None,
) -> CompilationResult:
    """Run the full HIC pipeline over one idea in a fresh session."""
    session = CompilationSession(user=user, settings=settings)
    compiler = HICCompiler(session)
    return compiler.compile(idea)
