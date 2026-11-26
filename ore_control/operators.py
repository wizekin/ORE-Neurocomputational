"""Cognitive operator registry and reference operators (Point 02 Phase B).

Operators are decision-level cognitive actions (Point 02 §3.2): each declares
a spec, and its handler receives detached context state plus services and
returns a typed outcome with findings to publish through the authoritative
workspace path. The handlers below are deterministic reference operators that
validate the control algorithm; they are explicitly not claims of
language-understanding quality (see docs/COGNITIVE_CONTROL_IMPLEMENTATION.md).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Mapping, Sequence

from ore_blackboard.specialists import FALLBACK_LEXICON

from .contracts import (
    CognitiveContext,
    OperatorSpec,
    RiskClass,
    RetryPolicy,
    sha256_json,
)
from .routing import (
    LEARNED_PROPOSAL_NOTE,
    LearnedComponentRegistry,
    default_learned_components,
)

CONTROL_INPUT_SCHEMA = "control.operator.input/1.0"
CONTROL_OUTPUT_SCHEMA = "control.operator.output/1.0"


# ------------------------------------------------------------------- services


@dataclass
class ControlServices:
    """Read-only service handles passed to operator handlers (Step 31).

    Handlers receive detached state and services; they never receive the
    workspace object itself, so they cannot bypass controlled publication.
    """

    task_text: str = ""
    usg_index: Any | None = None
    usg_reference: Mapping[str, Any] | None = None
    retrieved_knowledge: tuple[Mapping[str, Any], ...] = ()
    memory_records: tuple[Mapping[str, Any], ...] = ()
    learned_components: LearnedComponentRegistry = field(
        default_factory=default_learned_components,
    )
    blackboard: Any | None = None
    run_id: str = "control_run"
    usage_history: Mapping[str, tuple[float, ...]] = field(default_factory=dict)


# -------------------------------------------------------------------- outcome


@dataclass(frozen=True, slots=True)
class OperatorFinding:
    """One typed finding proposed by an operator handler (Step 33 input)."""

    field: str
    payload: Mapping[str, Any]
    key: str | None = None
    finding_type: str = "control_operator_result"
    source: str = "control_operator"
    rules_applied: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    dependencies: tuple[str, ...] = ()
    confidence: float | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate the proposed finding envelope."""
        if not isinstance(self.payload, Mapping):
            raise TypeError("finding payload must be a mapping")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("finding confidence must lie in [0, 1]")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "rules_applied", tuple(self.rules_applied))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))


@dataclass(frozen=True, slots=True)
class ChildGoalSpec:
    """One child goal description produced by a decomposition (Step 23)."""

    goal_text: str
    hypothesis_id: str | None = None
    target_term: str = ""
    required_evidence: tuple[str, ...] = ()
    allowed_operator_ids: tuple[str, ...] = ()
    success_predicate: str = "hypothesis_resolved"

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable child-goal spec."""
        return {
            "goal_text": self.goal_text,
            "hypothesis_id": self.hypothesis_id,
            "target_term": self.target_term,
            "required_evidence": list(self.required_evidence),
            "allowed_operator_ids": list(self.allowed_operator_ids),
            "success_predicate": self.success_predicate,
        }


@dataclass(frozen=True, slots=True)
class OperatorOutcome:
    """Typed result of one operator execution (Step 32 input)."""

    status: str  # completed | failed | deferred | waiting
    payload: Mapping[str, Any] = field(default_factory=dict)
    findings: tuple[OperatorFinding, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    decompose: bool = False
    child_goals: tuple[ChildGoalSpec, ...] = ()
    wait_reason: str | None = None
    checkpoint_events: tuple[Mapping[str, Any], ...] = ()
    elapsed_ms: float = 0.0
    failure_classification: str | None = None

    def __post_init__(self) -> None:
        """Validate the outcome structure."""
        allowed = {"completed", "failed", "deferred", "waiting"}
        if self.status not in allowed:
            raise ValueError(f"operator outcome status must be one of {sorted(allowed)}")
        object.__setattr__(self, "payload", dict(self.payload))
        object.__setattr__(self, "findings", tuple(self.findings))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))
        object.__setattr__(self, "child_goals", tuple(self.child_goals))
        object.__setattr__(
            self, "checkpoint_events", tuple(dict(item) for item in self.checkpoint_events),
        )


@dataclass(frozen=True, slots=True)
class OperatorAction:
    """One concrete, deduplicated operator invocation (Step 8-9)."""

    operator_id: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    arguments_fingerprint: str = ""
    child_context_id: str | None = None

    def __post_init__(self) -> None:
        """Validate and freeze the action arguments."""
        object.__setattr__(self, "arguments", dict(self.arguments))
        if not self.arguments_fingerprint:
            object.__setattr__(
                self, "arguments_fingerprint",
                sha256_json({"operator_id": self.operator_id, "arguments": self.arguments}),
            )


# ----------------------------------------------------------- precondition(s)


PRECONDITION_LIBRARY: dict[str, Callable[[Mapping[str, Any], ControlServices], bool]] = {
    "task_text_available": lambda state, services: bool(services.task_text.strip()),
    "open_questions_present": lambda state, services: any(
        not question.get("resolved", False)
        for question in state.get("unresolved_questions", [])
    ),
    "hypotheses_present": lambda state, services: (
        bool(state.get("lexical_hypotheses")) or bool(state.get("semantic_hypotheses"))
    ),
    "competing_hypotheses_present": lambda state, services: _competing(state),
    "unresolved_conflicts_present": lambda state, services: _conflicts_open(state),
    "sufficient_independent_evidence": lambda state, services: _independent_evidence(state),
    "retrieval_domains_available": lambda state, services: (
        services.usg_index is not None or bool(services.memory_records)
    ),
    "blackboard_backend_available": lambda state, services: services.blackboard is not None,
    "unresolvable_questions_present": lambda state, services: any(
        not question.get("resolved", False)
        and question.get("awaiting_external_evidence", False)
        for question in state.get("unresolved_questions", [])
    ),
}


def _records_for_field(
    state: Mapping[str, Any], field: str,
) -> tuple[Mapping[str, Any], ...]:
    """Flatten one workspace hypothesis field into payload records.

    The workspace stores keyed fields as ``{key: [payload, ...]}`` and
    append-only fields as ``[payload, ...]``; both shapes flatten to the same
    tuple of payload mappings.
    """
    value = state.get(field)
    records: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        for entries in value.values():
            if isinstance(entries, Sequence) and not isinstance(entries, (str, bytes)):
                records.extend(item for item in entries if isinstance(item, Mapping))
            elif isinstance(entries, Mapping):
                records.append(entries)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        records.extend(item for item in value if isinstance(item, Mapping))
    return tuple(records)


def _hypothesis_records(state: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return every lexical and semantic hypothesis payload in the state."""
    return (
        *_records_for_field(state, "lexical_hypotheses"),
        *_records_for_field(state, "semantic_hypotheses"),
    )


def _competing(state: Mapping[str, Any]) -> bool:
    """Return whether any term currently carries more than one hypothesis."""
    return _max_hypotheses_per_term(state) > 1


def _conflicts_open(state: Mapping[str, Any]) -> bool:
    """Return whether unresolved conflict records remain in the state view."""
    return bool(int(state.get("unresolved_conflicts", 0) or 0))


def _independent_evidence(state: Mapping[str, Any]) -> bool:
    """Check whether any hypothesis has two distinct supporting source groups."""
    for record in _hypothesis_records(state):
        groups = {
            str(item.get("source_group", "unknown"))
            for item in (
                value for value in record.get("evidence", [])
                if isinstance(value, Mapping)
            )
        }
        if len(groups) >= 2:
            return True
    return False


def _max_hypotheses_per_term(state: Mapping[str, Any]) -> int:
    """Count the maximum number of hypotheses currently held for one term."""
    best = 0
    for record in _hypothesis_records(state):
        if record.get("hypothesis_id") or record.get("term"):
            best = max(best, 1)
            alternates = record.get("competing_hypotheses", [])
            if isinstance(alternates, Sequence) and not isinstance(alternates, (str, bytes)):
                best = max(best, len(alternates))
    return best


def evaluate_preconditions(
    spec: OperatorSpec,
    state_summary: Mapping[str, Any],
    services: ControlServices,
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate one operator's declared preconditions (Step 7).

    Preconditions are deterministic predicates over the detached state summary
    and services; failures are returned as machine-readable precondition
    names so the trace can explain ineligibility.
    """
    failures: list[str] = []
    for name in spec.preconditions:
        predicate = PRECONDITION_LIBRARY.get(name)
        if predicate is None:
            failures.append(f"unknown_precondition:{name}")
            continue
        try:
            if not predicate(state_summary, services):
                failures.append(name)
        except Exception as error:  # pragma: no cover - defensive
            failures.append(f"precondition_error:{name}:{type(error).__name__}")
    return (not failures, tuple(failures))


# ------------------------------------------------------------- registry


@dataclass(frozen=True, slots=True)
class OperatorEligibility:
    """One operator's registry-query verdict with reasons (Step 6)."""

    spec: OperatorSpec
    eligible: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the serializable eligibility record."""
        return {
            "operator_id": self.spec.operator_id,
            "version": self.spec.version,
            "eligible": self.eligible,
            "reasons": list(self.reasons),
        }


class OperatorRegistry:
    """Registry of cognitive operators with stable identities (Steps 6, 40)."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._operators: dict[str, tuple[OperatorSpec, Callable]] = {}

    def register(self, spec: OperatorSpec, handler: Callable) -> None:
        """Register one operator spec with its execution handler."""
        if not callable(handler):
            raise TypeError("operator handler must be callable")
        if spec.operator_id in self._operators:
            raise ValueError(f"operator already registered: {spec.operator_id}")
        self._operators[spec.operator_id] = (spec, handler)

    def unregister(self, operator_id: str) -> bool:
        """Remove one operator; withdrawn operators are no longer eligible."""
        return self._operators.pop(operator_id, None) is not None

    def replace(self, spec: OperatorSpec, handler: Callable) -> None:
        """Replace a registration (used by procedure version bumps, Step 39)."""
        if not callable(handler):
            raise TypeError("operator handler must be callable")
        if spec.operator_id not in self._operators:
            raise ValueError(f"cannot replace unregistered operator: {spec.operator_id}")
        self._operators[spec.operator_id] = (spec, handler)

    def get_spec(self, operator_id: str) -> OperatorSpec | None:
        """Return the spec for one operator ID."""
        entry = self._operators.get(operator_id)
        return entry[0] if entry else None

    def handler(self, operator_id: str) -> Callable | None:
        """Return the handler for one operator ID."""
        entry = self._operators.get(operator_id)
        return entry[1] if entry else None

    def specs(self) -> tuple[OperatorSpec, ...]:
        """Return all registered operator specs in stable ID order."""
        return tuple(
            entry[0] for _, entry in sorted(self._operators.items())
        )

    def query(
        self,
        *,
        task_type: str = "sentence_understanding",
        required_capabilities: Iterable[str] = (),
    ) -> tuple[OperatorEligibility, ...]:
        """Query operators compatible with the task and capabilities (Step 6).

        An operator is ineligible when it is disabled, its schema version is
        unsupported, its handler is unavailable, or a required capability is
        missing from its declaration. Ineligibility reasons are
        machine-readable and flow into the decision trace.
        """
        required = tuple(required_capabilities)
        verdicts: list[OperatorEligibility] = []
        for operator_id, (spec, handler) in sorted(self._operators.items()):
            reasons: list[str] = []
            if not spec.enabled:
                reasons.append("operator_disabled")
            if not callable(handler):
                reasons.append("handler_unavailable")
            if spec.input_schema_id != CONTROL_INPUT_SCHEMA:
                reasons.append(
                    f"unsupported_input_schema:{spec.input_schema_id}"
                )
            if spec.output_schema_id != CONTROL_OUTPUT_SCHEMA:
                reasons.append(
                    f"unsupported_output_schema:{spec.output_schema_id}"
                )
            if required and not set(spec.required_capabilities) & set(required):
                reasons.append("capability_mismatch")
            verdicts.append(OperatorEligibility(
                spec=spec, eligible=not reasons, reasons=tuple(reasons),
            ))
        return tuple(verdicts)

    def apply(self, action: OperatorAction, *args, **kwargs) -> OperatorOutcome:
        """Invoke one registered handler; unknown operators are rejected (Step 31)."""
        handler = self.handler(action.operator_id)
        if handler is None:
            raise KeyError(f"operator not registered: {action.operator_id}")
        return handler(*args, **kwargs)


# ---------------------------------------------------------- reference operators


def _sense_inventory(term: str, services: ControlServices) -> tuple[tuple[str, str], ...]:
    """Return the sense inventory for one term from USG or the fallback lexicon.

    A USG lexical entry's ``details`` list carries one ``meaning:<id>:<gloss>``
    entry per sense, so distinct senses are preserved separately rather than
    being collapsed into a single record (Point 02 Step 12).
    """
    if services.usg_index is not None:
        result = services.usg_index.lookup(term)
        if result.exact_match:
            senses: list[tuple[str, str]] = []
            for item in result.items:
                if item.kind != "LEXICAL_ENTRY":
                    continue
                for detail in item.details:
                    parts = str(detail).split(":", 2)
                    if len(parts) == 3 and parts[0] == "meaning":
                        senses.append((parts[1], parts[2]))
            if senses:
                return tuple(senses)
    return FALLBACK_LEXICON.get(term, ())


def _terms(text: str) -> tuple[str, ...]:
    """Tokenize the task text deterministically (reference tokenization)."""
    return tuple(
        token for token in (
            part.strip(".,!?;:").casefold() for part in text.split()
        ) if token
    )


def _ambiguous_terms(services: ControlServices) -> tuple[str, ...]:
    """Return task-text terms whose sense inventory holds more than one sense."""
    return tuple(
        term for term in _terms(services.task_text)
        if len(_sense_inventory(term, services)) > 1
    )


def _retrieve_knowledge_handler(
    context: CognitiveContext,
    state: Mapping[str, Any],
    action: OperatorAction,
    services: ControlServices,
) -> OperatorOutcome:
    """Retrieve lexical/semantic knowledge for cue terms (Step 8 example)."""
    terms = tuple(action.arguments.get("cue_terms", ()) ) or _terms(services.task_text)
    retrieved: list[OperatorFinding] = []
    diagnostics: dict[str, Any] = {"hits": 0, "misses": []}
    for term in terms:
        if services.usg_index is not None:
            result = services.usg_index.lookup(term)
            if result.exact_match:
                for item in result.items:
                    retrieved.append(OperatorFinding(
                        field="retrieved_memories",
                        payload={
                            "record_reference": item.record_id,
                            "kind": item.kind,
                            "label": item.label,
                            "language": item.language,
                            "details": list(item.details),
                            "retrieved_by": "usg_exact_lookup",
                            "usg_bundle_id": result.bundle_id,
                            "usg_bundle_digest": result.bundle_digest,
                        },
                        key=f"usg:{item.record_id}",
                        finding_type="retrieved_knowledge_reference",
                        source="usg_knowledge_bundle",
                        rules_applied=("normalized_exact_match",),
                        evidence=({
                            "source_group": f"usg:{result.bundle_id}",
                            "record_reference": item.record_id,
                        },),
                        notes="USG remains read-only; only references are copied.",
                    ))
                diagnostics["hits"] += len(result.items)
            else:
                diagnostics["misses"].append(term)
        else:
            diagnostics["misses"].append(term)
    return OperatorOutcome(
        status="completed",
        payload={"retrieved": len(retrieved), "misses": diagnostics["misses"]},
        findings=tuple(retrieved),
        diagnostics=diagnostics,
        checkpoint_events=({"event": "progress", "detail": f"retrieved={len(retrieved)}"},),
    )


def _hypothesize_handler(
    context: CognitiveContext,
    state: Mapping[str, Any],
    action: OperatorAction,
    services: ControlServices,
) -> OperatorOutcome:
    """Generate competing hypotheses for ambiguous terms (Step 8 example).

    This is the hybrid path: sense enumeration is explicit symbolic work; the
    ranking of senses comes from the registered learned component strictly as
    a proposal with uncalibrated confidence (Point 02 Steps 26-28).
    """
    ambiguous = _ambiguous_terms(services)
    findings: list[OperatorFinding] = []
    proposals: list[Mapping[str, Any]] = []
    component = services.learned_components.get("control.toy.sense_ranker")
    primary_senses: dict[str, str] = {}
    for term in ambiguous:
        senses = _sense_inventory(term, services)
        if len(senses) < 2:
            continue
        ranked_senses: tuple[tuple[int, float, str], ...]
        if component is not None and component.ranker is not None:
            check = services.learned_components.check("control.toy.sense_ranker")
            if check.suitable:
                ranked_senses = component.ranker(term, senses)
                proposals.append({
                    "component": "control.toy.sense_ranker",
                    "term": term,
                    "ranked_senses": [
                        {"index": index, "score": score, "sense_id": sense_id}
                        for index, score, sense_id in ranked_senses
                    ],
                    "confidence": None,
                    "note": LEARNED_PROPOSAL_NOTE,
                })
            else:
                ranked_senses = tuple(
                    (index, 0.0, sense_id) for index, (sense_id, _gloss) in enumerate(senses)
                )
        else:
            ranked_senses = tuple(
                (index, 0.0, sense_id) for index, (sense_id, _gloss) in enumerate(senses)
            )
        primary = ranked_senses[0] if ranked_senses else None
        if primary is not None:
            primary_senses[term] = primary[2]
        hypothesis_id = f"hyp:{term}:{context.cycle_fingerprint[:8] or 'root'}"
        evidence: list[Mapping[str, Any]] = []
        if services.usg_reference:
            evidence.append({
                "source_group": f"usg:{services.usg_reference.get('bundle_id', 'unknown')}",
                "record_reference": "usg_lexicon",
            })
        findings.append(OperatorFinding(
            field="lexical_hypotheses",
            payload={
                "hypothesis_id": hypothesis_id,
                "term": term,
                "sense": primary[2] if primary else None,
                "sense_inventory": [sense_id for _, sense_id in senses],
                "competing_hypotheses": [
                    {"sense_id": sense_id, "rank_score": score}
                    for _, score, sense_id in ranked_senses
                ],
                "ranked_by": (
                    "learned_proposal:control.toy.sense_ranker"
                    if proposals else "symbolic_declaration_order"
                ),
                "evidence": evidence,
            },
            key=f"hypothesis:{term}",
            finding_type="lexical_hypothesis",
            source="control_operator",
            rules_applied=("sense_enumeration", "learned_ranking_proposal"),
            evidence=tuple(evidence),
            confidence=None,
            notes=(
                "Competing senses retained; ranking is an uncalibrated learned "
                "proposal, not a verified fact"
            ),
        ))
    for term in _terms(services.task_text):
        if term in primary_senses:
            continue
        senses = _sense_inventory(term, services)
        if not senses:
            continue
        primary_senses[term] = senses[0][1]
        findings.append(OperatorFinding(
            field="lexical_hypotheses",
            payload={
                "hypothesis_id": f"hyp:{term}:{context.cycle_fingerprint[:8] or 'root'}",
                "term": term,
                "sense": senses[0][1],
                "sense_inventory": [sense_id for sense_id, _ in senses],
                "competing_hypotheses": [],
                "ranked_by": "symbolic_declaration_order",
                "evidence": [],
            },
            key=f"hypothesis:{term}",
            finding_type="lexical_hypothesis",
            source="control_operator",
            rules_applied=("sense_enumeration",),
            notes="Unambiguous term: one declared reading.",
        ))
    # Structural segmentation (deterministic reference syntax record).
    tokens = _terms(services.task_text)
    findings.append(OperatorFinding(
        field="syntactic_hypotheses",
        payload={
            "structure_id": (
                f"struct:{context.cycle_fingerprint[:8] or 'root'}"
            ),
            "tokens": list(tokens),
            "segmentation": "linear_token_order",
            "entity_order": [index for index, _ in enumerate(tokens)],
        },
        key="structure:segmentation",
        finding_type="structural_segmentation",
        source="control_operator",
        rules_applied=("deterministic_token_segmentation",),
        notes="Reference structural record; not a parser-quality claim.",
    ))
    # Composed semantic reading from the current primary senses.
    findings.append(OperatorFinding(
        field="semantic_hypotheses",
        payload={
            "hypothesis_id": (
                f"sem:{context.cycle_fingerprint[:8] or 'root'}"
            ),
            "reading": " ".join(
                primary_senses.get(term, term) for term in tokens
            ) or " ".join(tokens),
            "term_senses": dict(primary_senses),
            "competing_hypotheses": [
                {"term": term, "note": "competing senses retained"}
                for term in ambiguous
            ],
            "ranked_by": (
                "learned_proposal:control.toy.sense_ranker"
                if proposals else "symbolic_declaration_order"
            ),
            "evidence": [],
        },
        key="semantic:reading",
        finding_type="semantic_hypothesis",
        source="control_operator",
        rules_applied=("sense_composition", "learned_ranking_proposal"),
        notes=(
            "Composed reference reading; competing senses stay retained and "
            "the learned ranking is an uncalibrated proposal"
        ),
    ))
    # Single-sentence discourse frame (deterministic reference record).
    findings.append(OperatorFinding(
        field="discourse_context",
        payload={
            "frame_id": (
                f"discourse:{context.cycle_fingerprint[:8] or 'root'}"
            ),
            "sentence_count": 1,
            "token_count": len(tokens),
            "anchoring": "single_sentence_segmentation",
        },
        key="discourse:frame",
        finding_type="discourse_frame",
        source="control_operator",
        rules_applied=("single_sentence_frame",),
        notes=(
            "Reference discourse frame for single-sentence tasks; not a "
            "multi-sentence discourse model."
        ),
    ))
    return OperatorOutcome(
        status="completed",
        payload={"hypotheses": len(findings), "learned_proposals": len(proposals)},
        findings=tuple(findings),
        diagnostics={"ambiguous_terms": list(ambiguous), "proposals": proposals},
        decompose=bool(ambiguous),
        child_goals=tuple(
            ChildGoalSpec(
                goal_text=f"Gather discriminating evidence for the readings of '{term}'.",
                hypothesis_id=f"hyp:{term}:{context.cycle_fingerprint[:8] or 'root'}",
                target_term=term,
                required_evidence=("independent_source_groups",),
                allowed_operator_ids=(
                    "control.retrieve_knowledge",
                    "control.verify_hypotheses",
                    "control.request_missing_evidence",
                ),
                success_predicate="hypothesis_resolved",
            )
            for term in ambiguous
        ),
    )


def _verify_handler(
    context: CognitiveContext,
    state: Mapping[str, Any],
    action: OperatorAction,
    services: ControlServices,
) -> OperatorOutcome:
    """Verify competing hypotheses against accumulated evidence (Step 8 example).

    Deterministic verification rule: a reading is verified when at least two
    distinct source groups support it, otherwise the question stays open.
    A successful process exit is not treated as a correct result (Step 32).
    """
    term = str(action.arguments.get("target_term", ""))
    hypotheses = _hypothesis_records(state)
    findings: list[OperatorFinding] = []
    verified = 0
    for record in hypotheses:
        if term and record.get("term") != term:
            continue
        evidence = [
            item for item in record.get("evidence", [])
            if isinstance(item, Mapping)
        ]
        groups = {str(item.get("source_group", "unknown")) for item in evidence}
        resolved = len(groups) >= 2
        if resolved:
            verified += 1
        findings.append(OperatorFinding(
            field="unresolved_questions",
            payload={
                "question_id": f"q:{record.get('term', record.get('hypothesis_id', 'unknown'))}",
                "text": (
                    f"Which reading of '{record.get('term', 'unknown')}' is supported?"
                ),
                "priority": 0.5,
                "resolved": resolved,
                "resolution": (
                    "verified_two_independent_source_groups" if resolved
                    else "insufficient_independent_evidence"
                ),
                "hypothesis_id": record.get("hypothesis_id"),
                "target_term": record.get("term"),
                "supporting_groups": sorted(groups),
            },
            key=f"question:{record.get('term', record.get('hypothesis_id', 'unknown'))}",
            finding_type="hypothesis_verification",
            source="control_operator",
            rules_applied=("two_independent_source_groups_rule",),
            evidence=tuple(evidence),
            notes=(
                "Deterministic verification rule; unresolved questions stay open "
                "rather than being forced to a winner."
            ),
        ))
    status = "completed" if findings else "deferred"
    return OperatorOutcome(
        status=status,
        payload={"verified": verified, "checked": len(findings)},
        findings=tuple(findings),
        diagnostics={"target_term": term},
        wait_reason=None if findings else "no hypotheses available to verify yet",
    )


def _resolve_conflict_handler(
    context: CognitiveContext,
    state: Mapping[str, Any],
    action: OperatorAction,
    services: ControlServices,
) -> OperatorOutcome:
    """Apply the explicit conflict-resolution rule (Step 21 of Point 01 kin).

    Resolution is only permitted when one alternative is supported by two
    independent source groups; otherwise the conflict is preserved untouched.
    """
    findings: list[OperatorFinding] = []
    resolutions = 0
    for record in _hypothesis_records(state):
        competing = [
            item for item in record.get("competing_hypotheses", [])
            if isinstance(item, Mapping)
        ]
        if not competing:
            continue
        evidence = [
            item for item in record.get("evidence", [])
            if isinstance(item, Mapping)
        ]
        groups = {str(item.get("source_group", "unknown")) for item in evidence}
        record_id = str(record.get("hypothesis_id") or record.get("term") or "unknown")
        if len(groups) >= 2:
            resolutions += 1
            findings.append(OperatorFinding(
                field="semantic_hypotheses",
                payload={
                    **dict(record),
                    "resolution": "selected_by_two_independent_source_groups",
                    "resolved_conflict": True,
                },
                key=f"resolved:{record_id}",
                finding_type="resolved_hypothesis",
                source="control_operator",
                rules_applied=("explicit_resolution_rule",),
                evidence=tuple(evidence),
                notes="Explicit, evidence-backed resolution action (Step 21).",
            ))
        else:
            findings.append(OperatorFinding(
                field="unresolved_questions",
                payload={
                    "question_id": f"q:conflict:{record_id}",
                    "text": "Conflict retained pending independent evidence.",
                    "priority": 0.7,
                    "resolved": False,
                    "resolution": "conflict_preserved_insufficient_evidence",
                    "hypothesis_id": record.get("hypothesis_id"),
                },
                key=f"conflict:{record_id}",
                finding_type="conflict_preservation",
                source="control_operator",
                rules_applied=("conflict_preservation_rule",),
                notes="Contradictory hypotheses are kept until resolution is justified.",
            ))
    return OperatorOutcome(
        status="completed",
        payload={"resolutions": resolutions},
        findings=tuple(findings),
        diagnostics={"resolutions": resolutions},
    )


def _request_evidence_handler(
    context: CognitiveContext,
    state: Mapping[str, Any],
    action: OperatorAction,
    services: ControlServices,
) -> OperatorOutcome:
    """Mark an unanswerable question as awaiting external evidence (Step 36)."""
    question = str(action.arguments.get("question_id", ""))
    findings: list[OperatorFinding] = []
    for record in state.get("unresolved_questions", []):
        if not isinstance(record, Mapping):
            continue
        if question and record.get("question_id") != question:
            continue
        if record.get("resolved", False):
            continue
        findings.append(OperatorFinding(
            field="unresolved_questions",
            payload={
                **dict(record),
                "awaiting_external_evidence": True,
                "escalation": "targeted_retrieval_or_human_review",
            },
            key=str(record.get("question_id", "question")),
            finding_type="evidence_request",
            source="control_operator",
            rules_applied=("external_evidence_request_rule",),
            notes="Waiting for an external dependency; no forced completion.",
        ))
    return OperatorOutcome(
        status="waiting",
        payload={"requested": len(findings)},
        findings=tuple(findings),
        diagnostics={"question_id": question},
        wait_reason="external evidence requested for an open question",
    )


def _run_subtask_handler(
    context: CognitiveContext,
    state: Mapping[str, Any],
    action: OperatorAction,
    services: ControlServices,
) -> OperatorOutcome:
    """Delegate one bounded subtask to the existing blackboard coordinator.

    This is the Point 02 integration path: the control layer stays
    executive-only and delegates execution to the existing coordination layer
    (which itself plans through the repository's CognitiveTaskCoordinator and
    ModuleScheduler).
    """
    if services.blackboard is None:
        return OperatorOutcome(
            status="failed",
            payload={},
            diagnostics={"reason": "blackboard backend unavailable"},
            failure_classification="backend_unavailable",
        )
    from ore_blackboard.contracts import BlackboardTaskRequest

    subtask_text = str(
        action.arguments.get("subtask_text", services.task_text),
    )
    request = BlackboardTaskRequest(
        input=subtask_text,
        goal=str(action.arguments.get("goal", "understand_sentence")),
        task_id=f"control_subtask_{action.arguments_fingerprint[:12]}",
        task_family="sentence_understanding",
    )
    result = services.blackboard.run(request)
    findings: list[OperatorFinding] = [
        OperatorFinding(
            field="artifact_references",
            payload={
                "reference_type": "blackboard_subtask_result",
                "task_id": result.task_id,
                "run_id": result.run_id,
                "status": result.status,
                "goal_score": result.goal_score,
                "accepted_finding_count": result.accepted_finding_count,
                "conflict_count": result.conflict_count,
                "hypothesis_labels": [
                    item.label for item in result.supported_hypotheses
                ],
                "trace_completeness": result.trace_completeness,
            },
            key=f"blackboard:{result.task_id}",
            finding_type="delegated_subtask_reference",
            source="ore_blackboard.coordinator",
            rules_applied=("delegation_to_existing_coordinator",),
            evidence=({
                "source_group": "ore_blackboard",
                "record_reference": result.run_id,
            },),
            notes="Executive layer delegates execution; state authority stays here.",
        ),
    ]
    for hypothesis in result.supported_hypotheses:
        findings.append(OperatorFinding(
            field="semantic_hypotheses",
            payload={
                "hypothesis_id": hypothesis.hypothesis_id,
                "term": hypothesis.label,
                "sense": hypothesis.label,
                "sense_inventory": [],
                "competing_hypotheses": [],
                "ranked_by": "blackboard_evidence_score",
                "evidence": [
                    dict(item) for item in hypothesis.used_evidence
                ],
            },
            key=f"blackboard_hyp:{hypothesis.hypothesis_id}",
            finding_type="semantic_hypothesis",
            source="ore_blackboard.coordinator",
            rules_applied=("evidence_score_aggregation",),
            evidence=tuple(dict(item) for item in hypothesis.used_evidence),
            confidence=None,
            notes="Imported hypothesis retains its ORE evidence-score semantics.",
        ))
    status = "completed" if result.status in {
        "completed", "completed_with_uncertainty", "needs_more_processing",
    } else "deferred"
    return OperatorOutcome(
        status=status,
        payload={
            "delegated_status": result.status,
            "workspace_id": result.workspace_id,
        },
        findings=tuple(findings),
        diagnostics={"delegated_status": result.status},
        checkpoint_events=({"event": "progress", "detail": f"delegated={result.status}"},),
    )


def default_operators() -> tuple[tuple[OperatorSpec, Callable], ...]:
    """Return the deterministic reference operator set (Step 8 vocabulary).

    This list is a reference vocabulary, not the complete future operator
    vocabulary: domain-specific operators can register through the same
    registry without changes to the cycle.
    """
    return (
        (
            OperatorSpec(
                operator_id="control.retrieve_knowledge",
                version="1.0.0",
                purpose="Retrieve lexical/semantic knowledge for cue terms from attached stores.",
                preconditions=("task_text_available", "retrieval_domains_available"),
                required_capabilities=("knowledge.retrieval",),
                estimated_time_seconds=0.01,
                estimated_memory_mb=1.0,
                expected_progress=0.3,
                expected_information_gain=0.6,
                writes_workspace=True,
                write_fields=("retrieved_memories",),
                read_fields=("current_input", "unresolved_questions"),
                handler_reference="ore_control.operators:_retrieve_knowledge_handler",
                processing_path="EXPLICIT_SYMBOLIC",
            ),
            _retrieve_knowledge_handler,
        ),
        (
            OperatorSpec(
                operator_id="control.hypothesize_interpretations",
                version="1.0.0",
                purpose="Generate competing interpretations for ambiguous terms.",
                preconditions=("task_text_available",),
                required_capabilities=("language.semantic_composition",),
                estimated_time_seconds=0.02,
                estimated_memory_mb=2.0,
                expected_progress=0.5,
                expected_information_gain=0.7,
                writes_workspace=True,
                write_fields=(
                    "lexical_hypotheses", "syntactic_hypotheses",
                    "semantic_hypotheses", "discourse_context",
                ),
                read_fields=("current_input", "retrieved_memories"),
                handler_reference="ore_control.operators:_hypothesize_handler",
                processing_path="HYBRID",
            ),
            _hypothesize_handler,
        ),
        (
            OperatorSpec(
                operator_id="control.verify_hypotheses",
                version="1.0.0",
                purpose="Verify competing hypotheses against accumulated evidence.",
                preconditions=("hypotheses_present",),
                required_capabilities=("cognition.verification",),
                estimated_time_seconds=0.01,
                estimated_memory_mb=1.0,
                expected_progress=0.6,
                expected_information_gain=0.5,
                writes_workspace=True,
                write_fields=("unresolved_questions",),
                read_fields=("lexical_hypotheses", "semantic_hypotheses"),
                handler_reference="ore_control.operators:_verify_handler",
                processing_path="EXPLICIT_SYMBOLIC",
            ),
            _verify_handler,
        ),
        (
            OperatorSpec(
                operator_id="control.resolve_conflict",
                version="1.0.0",
                purpose="Apply the explicit resolution rule to retained conflicts.",
                preconditions=("unresolved_conflicts_present", "sufficient_independent_evidence"),
                required_capabilities=("cognition.verification",),
                estimated_time_seconds=0.01,
                estimated_memory_mb=1.0,
                expected_progress=0.8,
                expected_information_gain=0.4,
                risk_class=RiskClass.MEDIUM,
                reversible=True,
                writes_workspace=True,
                write_fields=("semantic_hypotheses", "unresolved_questions"),
                read_fields=("lexical_hypotheses", "semantic_hypotheses"),
                handler_reference="ore_control.operators:_resolve_conflict_handler",
                processing_path="EXPLICIT_SYMBOLIC",
            ),
            _resolve_conflict_handler,
        ),
        (
            OperatorSpec(
                operator_id="control.request_missing_evidence",
                version="1.0.0",
                purpose="Mark unanswerable questions as awaiting external evidence.",
                preconditions=("open_questions_present",),
                required_capabilities=("cognition.meta",),
                estimated_time_seconds=0.005,
                estimated_memory_mb=0.5,
                expected_progress=0.1,
                expected_information_gain=0.1,
                writes_workspace=True,
                write_fields=("unresolved_questions",),
                read_fields=("unresolved_questions",),
                handler_reference="ore_control.operators:_request_evidence_handler",
                processing_path="EXPLICIT_SYMBOLIC",
            ),
            _request_evidence_handler,
        ),
        (
            OperatorSpec(
                operator_id="control.run_coordinated_subtask",
                version="1.0.0",
                purpose="Delegate a bounded subtask to the existing blackboard coordinator.",
                preconditions=("task_text_available", "blackboard_backend_available"),
                required_capabilities=("coordination.delegation",),
                estimated_time_seconds=0.5,
                estimated_memory_mb=16.0,
                expected_progress=0.7,
                expected_information_gain=0.6,
                risk_class=RiskClass.LOW,
                retry_policy=RetryPolicy(timeout_seconds=30.0, max_retries=1),
                writes_workspace=True,
                write_fields=("artifact_references", "semantic_hypotheses"),
                read_fields=("current_input",),
                handler_reference="ore_control.operators:_run_subtask_handler",
                processing_path="EXPLICIT_SYMBOLIC",
            ),
            _run_subtask_handler,
        ),
    )


def register_default_operators(registry: OperatorRegistry) -> None:
    """Register the reference operator set (Step 6)."""
    for spec, handler in default_operators():
        registry.register(spec, handler)


def candidate_actions_for_state(
    specs: Iterable[OperatorSpec],
    *,
    state_summary: Mapping[str, Any],
    services: ControlServices,
    context: CognitiveContext,
) -> tuple[OperatorAction, ...]:
    """Generate typed candidate actions with normalized arguments (Step 8)."""
    actions: list[OperatorAction] = []
    ambiguous = _ambiguous_terms(services)
    open_questions = [
        item for item in state_summary.get("unresolved_questions", [])
        if isinstance(item, Mapping) and not item.get("resolved", False)
    ]
    unresolvable = [
        item for item in open_questions
        if item.get("awaiting_external_evidence", False)
    ]
    for spec in specs:
        arguments: dict[str, Any] = {}
        if spec.operator_id == "control.retrieve_knowledge":
            arguments = {
                "cue_terms": tuple(
                    dict.fromkeys(
                        [item.get("target_term") for item in open_questions
                         if item.get("target_term")] + list(ambiguous)
                    )
                ),
            }
        elif spec.operator_id == "control.hypothesize_interpretations":
            arguments = {"ambiguous_terms": ambiguous}
        elif spec.operator_id == "control.verify_hypotheses":
            arguments = {
                "target_term": (
                    open_questions[0].get("target_term", "")
                    if open_questions else (ambiguous[0] if ambiguous else "")
                ),
            }
        elif spec.operator_id == "control.request_missing_evidence":
            arguments = {
                "question_id": (
                    unresolvable[0].get("question_id", "")
                    if unresolvable else ""
                ),
            }
        elif spec.operator_id == "control.run_coordinated_subtask":
            arguments = {"subtask_text": services.task_text, "goal": "understand_sentence"}
        elif spec.operator_id.startswith("procedure:"):
            arguments = {"goal_id": context.goal_id}
        actions.append(OperatorAction(
            operator_id=spec.operator_id,
            arguments=arguments,
            arguments_fingerprint=sha256_json({
                "operator_id": spec.operator_id,
                "operator_version": spec.version,
                "arguments": arguments,
                "context_status": context.status.value,
            }),
        ))
    return tuple(actions)


def deduplicate_actions(
    actions: Iterable[OperatorAction],
    *,
    executed_fingerprints: Iterable[str] = (),
) -> tuple[OperatorAction, ...]:
    """Deduplicate candidate actions (Step 9).

    Deduplication keys on operator identity plus normalized arguments; a
    fingerprint already executed successfully in this run is removed as
    redundant work rather than re-run.
    """
    seen: set[str] = set()
    executed = set(executed_fingerprints)
    unique: list[OperatorAction] = []
    for action in actions:
        identity = sha256_json({
            "operator_id": action.operator_id,
            "arguments": action.arguments,
        })
        if identity in seen or action.arguments_fingerprint in executed:
            continue
        seen.add(identity)
        unique.append(action)
    return tuple(unique)


__all__ = [
    "CONTROL_INPUT_SCHEMA",
    "CONTROL_OUTPUT_SCHEMA",
    "ChildGoalSpec",
    "ControlServices",
    "OperatorAction",
    "OperatorEligibility",
    "OperatorFinding",
    "OperatorOutcome",
    "OperatorRegistry",
    "PRECONDITION_LIBRARY",
    "candidate_actions_for_state",
    "deduplicate_actions",
    "default_operators",
    "evaluate_preconditions",
    "register_default_operators",
]
