"""Ambiguity analysis, clarification ranking, and explicit assumptions (Steps 14-16).

Uncertainty over candidate interpretations H is estimated by entropy
H(H) = -sum_h P(h) log P(h); a clarification question q is valued by expected
information gain IG(q) = H(H) - E_a[H(H | answer=a)] and prioritized together
with decision impact and user effort (spec 3.2). Blocking ambiguity stops
unsafe downstream generation; harmless optional detail becomes a configurable
parameter instead of an interruption. Every assumption records rationale,
source, confidence, impact and whether human confirmation is required.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import (
    ArtifactKind,
    AssumptionSource,
    CanonicalArtifact,
    Impact,
)
from .semantic import SemanticGraph
from .session import CompilationSession


def entropy(probabilities: Sequence[float], *, base: float = 2.0) -> float:
    """Shannon entropy in bits of a non-negative weight vector (spec 3.2)."""
    total = sum(probabilities)
    if total <= 0:
        return 0.0
    result = 0.0
    for weight in probabilities:
        p = weight / total
        if p > 0:
            result -= p * math.log(p, base)
    return result


def normalized(weights: Sequence[float]) -> list[float]:
    total = sum(weights)
    if total <= 0:
        return [0.0] * len(weights)
    return [w / total for w in weights]


@dataclass(frozen=True, slots=True)
class Ambiguity:
    """One unresolved interpretation, missing parameter, or conflict (Step 14)."""

    ambiguity_id: str
    subject: str
    kind: str  # MISSING_PARAMETER | INTERPRETATION | CONFLICT | MISSING_DEFINITION
    description: str
    blocking: bool
    candidate_answers: tuple[str, ...]
    span_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ambiguity_id": self.ambiguity_id,
            "subject": self.subject,
            "kind": self.kind,
            "description": self.description,
            "blocking": self.blocking,
            "candidate_answers": list(self.candidate_answers),
            "span_text": self.span_text,
        }


@dataclass(frozen=True, slots=True)
class ClarificationQuestion:
    """A ranked clarification question (spec Step 15)."""

    question_id: str
    ambiguity_id: str
    question: str
    options: tuple[str, ...]
    information_gain_bits: float
    decision_impact: Impact
    user_effort: Impact  # LOW effort is good
    rank: int = 0

    def priority_score(self) -> float:
        """High information gain, high impact, low effort first."""
        impact_score = {"HIGH": 3.0, "MEDIUM": 2.0, "LOW": 1.0}[self.decision_impact.value]
        effort_penalty = {"LOW": 0.0, "MEDIUM": 0.5, "HIGH": 1.0}[self.user_effort.value]
        return self.information_gain_bits * impact_score - effort_penalty

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "ambiguity_id": self.ambiguity_id,
            "question": self.question,
            "options": list(self.options),
            "information_gain_bits": self.information_gain_bits,
            "decision_impact": self.decision_impact.value,
            "user_effort": self.user_effort.value,
            "priority_score": self.priority_score(),
            "rank": self.rank,
        }


@dataclass(frozen=True, slots=True)
class Assumption:
    """One explicit assumption with source, confidence and impact (Step 16)."""

    assumption_id: str
    statement: str
    rationale: str
    source: AssumptionSource
    confidence: float
    impact: Impact
    requires_confirmation: bool
    configurable_parameter: str | None = None
    span_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assumption_id": self.assumption_id,
            "statement": self.statement,
            "rationale": self.rationale,
            "source": self.source.value,
            "confidence": self.confidence,
            "impact": self.impact.value,
            "requires_confirmation": self.requires_confirmation,
            "configurable_parameter": self.configurable_parameter,
            "span_text": self.span_text,
        }


class AmbiguitySet(CanonicalArtifact):
    """Canonical artifact: ambiguities, questions, and assumptions."""

    kind = ArtifactKind.AMBIGUITY_SET

    def __init__(
        self,
        *,
        artifact_id: str,
        graph_id: str,
        ambiguities: Iterable[Ambiguity],
        questions: Iterable[ClarificationQuestion],
        assumptions: Iterable[Assumption],
        entropy_bits: float,
        version: int = 1,
    ) -> None:
        self.artifact_id = artifact_id
        self.graph_id = graph_id
        self.ambiguities = tuple(ambiguities)
        self.questions = tuple(questions)
        self.assumptions = tuple(assumptions)
        self.entropy_bits = entropy_bits
        self.version = version

    @property
    def blocking(self) -> tuple[Ambiguity, ...]:
        return tuple(a for a in self.ambiguities if a.blocking)

    @property
    def unconfirmed_high_impact_assumptions(self) -> tuple[Assumption, ...]:
        return tuple(a for a in self.assumptions if a.requires_confirmation and a.impact is Impact.HIGH)

    def answer_for(self, ambiguity_id: str) -> ClarificationQuestion | None:
        for question in self.questions:
            if question.ambiguity_id == ambiguity_id:
                return question
        return None

    def content_payload(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "graph_id": self.graph_id,
            "ambiguities": [a.to_dict() for a in self.ambiguities],
            "questions": [q.to_dict() for q in self.questions],
            "assumptions": [a.to_dict() for a in self.assumptions],
            "entropy_bits": self.entropy_bits,
            "version": self.version,
        }


class AmbiguityAnalyzer:
    """Detect ambiguity, rank questions by information gain (Steps 14-16)."""

    REQUIRED_OPERATION_FIELDS = (
        "actors", "inputs", "outputs", "measurement_definitions", "limits",
        "failure_behavior", "concurrency", "authorization", "success_criteria",
    )

    def detect(self, graph: SemanticGraph) -> tuple[Ambiguity, ...]:
        """Step 14: distinguish blocking ambiguity from harmless optional detail."""
        ambiguities: list[Ambiguity] = []

        for index, interpretation in enumerate(graph.interpretations, start=1):
            hypotheses = interpretation.hypotheses
            weights = normalized([h.posterior_weight for h in hypotheses])
            gap = abs(weights[0] - weights[1]) if len(weights) >= 2 else 1.0
            blocking = gap < 0.15  # near-tie readings change the algorithm materially
            ambiguities.append(
                Ambiguity(
                    ambiguity_id=f"amb-{index:03d}",
                    subject=interpretation.phrase,
                    kind="INTERPRETATION",
                    description=(
                        f"phrase '{interpretation.phrase}' has {len(hypotheses)} plausible readings: "
                        + "; ".join(h.reading for h in hypotheses)
                    ),
                    blocking=blocking,
                    candidate_answers=tuple(h.reading for h in hypotheses),
                    span_text=interpretation.span.text,
                )
            )

        for attribute in graph.attributes:
            if attribute.value == "unspecified" and attribute.unit is None:
                ambiguities.append(
                    Ambiguity(
                        ambiguity_id=f"amb-{len(ambiguities) + 1:03d}",
                        subject=attribute.name,
                        kind="MISSING_PARAMETER",
                        description=(
                            f"attribute '{attribute.name}' has no value or unit; "
                            "it can become a configurable parameter"
                        ),
                        blocking=False,
                        candidate_answers=("configurable parameter", "fixed default"),
                        span_text=attribute.span.text,
                    )
                )

        required_but_missing = [
            field for field in self.REQUIRED_OPERATION_FIELDS
            if field in {"measurement_definitions"} and not graph.attributes
        ]
        for field in required_but_missing:
            ambiguities.append(
                Ambiguity(
                    ambiguity_id=f"amb-{len(ambiguities) + 1:03d}",
                    subject=field,
                    kind="MISSING_DEFINITION",
                    description=f"required operation field '{field}' is missing",
                    blocking=True,
                    candidate_answers=(),
                )
            )
        return tuple(ambiguities)

    def expected_information_gain(
        self,
        ambiguity: Ambiguity,
        hypothesis_weights: Sequence[float],
    ) -> float:
        """IG(q) = H(H) - E_a[H(H | a)] for a binary split of the hypotheses."""
        prior = entropy(hypothesis_weights)
        if len(hypothesis_weights) < 2:
            return 0.0
        probs = normalized(hypothesis_weights)
        p_yes = probs[0]
        # A perfectly separating answer leaves zero residual entropy.
        residual = p_yes * entropy([p_yes]) + (1 - p_yes) * entropy([1 - p_yes])
        return max(prior - residual, 0.0)

    def rank_questions(
        self,
        ambiguities: Sequence[Ambiguity],
        graph: SemanticGraph,
    ) -> tuple[ClarificationQuestion, ...]:
        """Step 15: ask the fewest high-value questions first."""
        questions: list[ClarificationQuestion] = []
        for ambiguity in ambiguities:
            if not ambiguity.candidate_answers:
                continue
            weights = self._weights_for(ambiguity, graph)
            ig = self.expected_information_gain(ambiguity, weights)
            impact = Impact.HIGH if ambiguity.blocking else Impact.LOW
            effort = Impact.LOW if len(ambiguity.candidate_answers) <= 3 else Impact.MEDIUM
            questions.append(
                ClarificationQuestion(
                    question_id=f"q-{ambiguity.ambiguity_id}",
                    ambiguity_id=ambiguity.ambiguity_id,
                    question=f"Which reading of '{ambiguity.subject}' do you intend?",
                    options=ambiguity.candidate_answers,
                    information_gain_bits=ig,
                    decision_impact=impact,
                    user_effort=effort,
                )
            )
        questions.sort(key=lambda q: (-q.priority_score(), q.question_id))
        return tuple(
            ClarificationQuestion(
                question_id=q.question_id,
                ambiguity_id=q.ambiguity_id,
                question=q.question,
                options=q.options,
                information_gain_bits=q.information_gain_bits,
                decision_impact=q.decision_impact,
                user_effort=q.user_effort,
                rank=rank,
            )
            for rank, q in enumerate(questions, start=1)
        )

    @staticmethod
    def _weights_for(ambiguity: Ambiguity, graph: SemanticGraph) -> list[float]:
        for interpretation in graph.interpretations:
            if interpretation.phrase == ambiguity.subject:
                return [h.posterior_weight for h in interpretation.hypotheses]
        return [1.0] * max(len(ambiguity.candidate_answers), 2)

    def record_assumptions(
        self,
        ambiguities: Sequence[Ambiguity],
        graph: SemanticGraph,
        *,
        auto_assume_nonblocking: bool = True,
        confirmed: frozenset[str] | set[str] = frozenset(),
    ) -> tuple[Assumption, ...]:
        """Step 16: store rationale, source, confidence, impact per assumption."""
        assumptions: list[Assumption] = []
        for ambiguity in ambiguities:
            if ambiguity.blocking:
                assumptions.append(
                    Assumption(
                        assumption_id=f"as-{ambiguity.ambiguity_id}",
                        statement=f"'{ambiguity.subject}' resolves to '{ambiguity.candidate_answers[0]}'",
                        rationale="blocking ambiguity: no assumption may silently decide it",
                        source=AssumptionSource.COMPILER,
                        confidence=0.0,
                        impact=Impact.HIGH,
                        requires_confirmation=True,
                        span_text=ambiguity.span_text,
                    )
                )
            elif auto_assume_nonblocking:
                param = ambiguity.subject.replace(" ", "_")
                assumptions.append(
                    Assumption(
                        assumption_id=f"as-{ambiguity.ambiguity_id}",
                        statement=(
                            f"'{ambiguity.subject}' is represented as configurable parameter "
                            f"'{param}' instead of interrupting the workflow"
                        ),
                        rationale="noncritical detail that does not change algorithm safety",
                        source=AssumptionSource.COMPILER,
                        confidence=0.8,
                        impact=Impact.LOW,
                        requires_confirmation=False,
                        configurable_parameter=param,
                        span_text=ambiguity.span_text,
                    )
                )
        linearity_confirmed = "as-model-linearity" in confirmed
        assumptions.append(
            Assumption(
                assumption_id="as-model-linearity",
                statement="assignment costs and demands are additive across tasks",
                rationale="required by the task-to-module optimization model of spec 3.3",
                source=AssumptionSource.USER if linearity_confirmed else AssumptionSource.COMPILER,
                confidence=0.95 if linearity_confirmed else 0.7,
                impact=Impact.HIGH,
                requires_confirmation=not linearity_confirmed,
            )
        )
        return tuple(assumptions)

    def analyze(
        self,
        session: CompilationSession,
        graph: SemanticGraph,
        *,
        auto_assume_nonblocking: bool = True,
        resolved: Iterable[str] = (),
        confirmed: Iterable[str] = (),
    ) -> AmbiguitySet:
        """Run Steps 14-16 and register the AmbiguitySet artifact.

        ``resolved`` are ambiguity ids the user has already answered;
        ``confirmed`` are assumption ids the user has explicitly confirmed
        (e.g. the additive-cost modeling assumption of spec 3.3).
        """
        resolved_ids = frozenset(resolved)
        confirmed_ids = frozenset(confirmed)
        ambiguities = tuple(
            Ambiguity(
                ambiguity_id=a.ambiguity_id,
                subject=a.subject,
                kind=a.kind,
                description=a.description,
                blocking=False if a.ambiguity_id in resolved_ids else a.blocking,
                candidate_answers=a.candidate_answers,
                span_text=a.span_text,
            )
            for a in self.detect(graph)
        )
        questions = self.rank_questions(
            tuple(a for a in ambiguities if a.ambiguity_id not in resolved_ids), graph
        )
        assumptions = self.record_assumptions(
            ambiguities, graph, auto_assume_nonblocking=auto_assume_nonblocking,
            confirmed=confirmed_ids,
        )
        weights: list[float] = []
        for interpretation in graph.interpretations:
            weights.extend(h.posterior_weight for h in interpretation.hypotheses)
        overall_entropy = entropy(weights) if weights else 0.0
        result = AmbiguitySet(
            artifact_id=session.next_artifact_id(ArtifactKind.AMBIGUITY_SET),
            graph_id=graph.artifact_id,
            ambiguities=ambiguities,
            questions=questions,
            assumptions=assumptions,
            entropy_bits=overall_entropy,
        )
        session.register(result)
        session.ledger.record(
            stage="B",
            step_id="HIC14-16",
            description="Ambiguity detection, clarification ranking, explicit assumptions",
            inputs=[graph],
            outputs=[result],
            rule_references={"HIC-AMBIG-1", "HIC-AMBIG-2"},
        )
        return result
