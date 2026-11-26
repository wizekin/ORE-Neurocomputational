"""Replaceable learned-candidate adapter with validated proposals (Phase F, Steps F1-F5).

The registry keeps deterministic symbolic reasoning fully operational when no
model is registered, when a model fails, or when a model is removed (Stage 6
gate). Model responses must validate: schema, version, input hash, score
range/meaning, calibration, latency and policy. Learned outputs become
hypotheses (``EpistemicStatus.MODEL_PROPOSAL``), never authoritative facts, and
they cannot mutate USG (Step F3). Symbolic filtering retains bounded rejection
reasons (Step F4). Routing/ranking parameters update only from *verified*
feedback with bounded, versioned deltas (Step F5; Stage 7 gate).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from .contracts import (
    CandidateInterpretation,
    CandidateStatus,
    EpistemicStatus,
    ModelProposal,
    make_id,
    sha256_json,
)


class ModelInvocationError(RuntimeError):
    """Raised when a model adapter fails, returns malformed output, or is unavailable."""


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    """One registered model's capability, quality, cost and policy profile."""

    model_id: str
    model_version: str
    capability: str  # entity_linking | relation_similarity | rule_retrieval | ranking
    output_schema: str
    score_semantics: str
    calibration: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    quality: float = 0.5
    cost_per_call: float = 1.0
    allowed_by_policy: bool = True

    def __post_init__(self) -> None:
        """Validate identity, capability and bounded quality."""
        for name in ("model_id", "model_version", "capability", "output_schema", "score_semantics"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"ModelDescriptor.{name} must be a non-empty string")
        allowed = {
            "entity_linking", "relation_similarity", "rule_retrieval", "ranking",
        }
        if self.capability not in allowed:
            raise ValueError(f"ModelDescriptor.capability must be one of {sorted(allowed)}")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("ModelDescriptor.quality must be in [0, 1]")
        if self.cost_per_call < 0:
            raise ValueError("ModelDescriptor.cost_per_call cannot be negative")


class ModelAdapter(Protocol):
    """The minimal contract one replaceable model backend must satisfy."""

    def invoke(
        self,
        capability: str,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Return one raw model response for the requested capability."""
        ...


def _default_score_validator(proposal: ModelProposal) -> tuple[bool, str | None]:
    """Reject proposals whose candidate scores fall outside [0, 1]."""
    for candidate in proposal.candidates:
        for name, value in candidate.neural_scores.items():
            if not 0.0 <= float(value) <= 1.0:
                return False, f"score {name!r}={value!r} outside [0, 1]"
    return True, None


@dataclass(slots=True)
class RejectionRecord:
    """One bounded symbolic-filter rejection reason (Step F4)."""

    candidate_id: str
    reason_code: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "candidate_id": self.candidate_id,
            "reason_code": self.reason_code,
            "reason": self.reason,
        }


@dataclass(slots=True)
class AdaptationLedger:
    """Versioned record of verified-feedback parameter updates (Step F5)."""

    version: int = 1
    updates: tuple[Mapping[str, Any], ...] = ()
    rejected_updates: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "version": self.version,
            "updates": [dict(item) for item in self.updates],
            "rejected_updates": [dict(item) for item in self.rejected_updates],
        }


class LearnedCandidateAdapter:
    """Register, invoke, validate and filter learned components (Phase F)."""

    def __init__(
        self,
        *,
        max_proposals_per_run: int = 64,
        max_latency_ms: float = 5000.0,
        max_weight_delta: float = 0.1,
        score_validator: Callable[[ModelProposal], tuple[bool, str | None]] | None = None,
    ) -> None:
        """Bind invocation budgets and the bounded adaptation delta."""
        if max_proposals_per_run <= 0:
            raise ValueError("max_proposals_per_run must be positive")
        if max_latency_ms <= 0:
            raise ValueError("max_latency_ms must be positive")
        if not 0.0 < max_weight_delta <= 1.0:
            raise ValueError("max_weight_delta must be in (0, 1]")
        self.max_proposals_per_run = max_proposals_per_run
        self.max_latency_ms = max_latency_ms
        self.max_weight_delta = max_weight_delta
        self._descriptors: dict[str, ModelDescriptor] = {}
        self._backends: dict[str, ModelAdapter] = {}
        self._score_validator = score_validator or _default_score_validator
        self.ledger = AdaptationLedger()
        self.calls_made: int = 0
        self.failures: int = 0

    # ---------------------------------------------------------------- registry

    def register(
        self, descriptor: ModelDescriptor, backend: ModelAdapter,
    ) -> None:
        """Register one replaceable model backend behind its descriptor."""
        self._descriptors[descriptor.capability] = descriptor
        self._backends[descriptor.capability] = backend

    def remove(self, capability: str) -> bool:
        """Remove one capability's model; symbolic reasoning must keep working."""
        if capability in self._descriptors:
            del self._descriptors[capability]
            del self._backends[capability]
            return True
        return False

    def available_capabilities(self) -> tuple[str, ...]:
        """Return the currently registered capabilities."""
        return tuple(sorted(self._descriptors))

    # ------------------------------------------------------------------ Step F1

    def invoke_when_needed(
        self,
        capability: str,
        payload: Mapping[str, Any],
        *,
        deterministic_indexes_sufficient: bool,
        invocation_count: int = 0,
    ) -> ModelProposal | None:
        """Invoke a learned model only when deterministic indexes are insufficient (F1).

        Selection weighs capability fit first, then quality, cost and policy.
        Returns ``None`` (never an error) when the model is absent, disabled,
        over budget or fails; symbolic reasoning continues without it.
        """
        if deterministic_indexes_sufficient:
            return None
        if capability not in self._descriptors:
            return None
        descriptor = self._descriptors[capability]
        if not descriptor.allowed_by_policy:
            return None
        if invocation_count >= self.max_proposals_per_run:
            return None
        try:
            raw = self._backends[capability].invoke(capability, payload)
            proposal = self._build_proposal(descriptor, payload, raw)
        except (ModelInvocationError, KeyError, TypeError, ValueError):
            self.failures += 1
            return None
        self.calls_made += 1
        return proposal

    def _build_proposal(
        self,
        descriptor: ModelDescriptor,
        payload: Mapping[str, Any],
        raw: Mapping[str, Any],
    ) -> ModelProposal:
        """Validate the raw response and build the typed proposal (Step F2)."""
        if not isinstance(raw, Mapping):
            raise ModelInvocationError("model response must be a mapping")
        schema = str(raw.get("output_schema", descriptor.output_schema))
        if schema != descriptor.output_schema:
            raise ModelInvocationError(
                f"model responded with schema {schema!r}, expected {descriptor.output_schema!r}",
            )
        version = str(raw.get("model_version", descriptor.model_version))
        if version != descriptor.model_version:
            raise ModelInvocationError(
                f"model responded with version {version!r}, expected "
                f"{descriptor.model_version!r}",
            )
        input_hash = str(raw.get("input_hash", "")) or sha256_json(dict(payload))
        candidates = tuple(
            CandidateInterpretation(
                candidate_id=item.get("candidate_id") or make_id("proposal_candidate"),
                semantic_structure=dict(item.get("semantic_structure", {})),
                evidence_refs=tuple(item.get("evidence_refs", ())),
                neural_scores={
                    key: float(value)
                    for key, value in item.get("neural_scores", {}).items()
                },
                constraint_status={},
                ranking_features={},
                status=CandidateStatus.GENERATED,
            )
            for item in raw.get("candidates", ())
        )
        proposal = ModelProposal(
            model_id=descriptor.model_id,
            model_version=descriptor.model_version,
            input_hash=input_hash,
            output_schema=schema,
            candidates=candidates,
            score_semantics=str(raw.get("score_semantics", descriptor.score_semantics)),
            calibration=dict(raw.get("calibration", descriptor.calibration)),
            limitations=descriptor.limitations,
            latency_ms=float(raw.get("latency_ms", 0.0)),
        )
        ok, reason = self._score_validator(proposal)
        if not ok:
            raise ModelInvocationError(f"score validation failed: {reason}")
        if proposal.latency_ms > self.max_latency_ms:
            raise ModelInvocationError(
                f"model latency {proposal.latency_ms}ms exceeds policy {self.max_latency_ms}ms",
            )
        return proposal

    # ------------------------------------------------------------------ Step F3

    def to_hypotheses(self, proposal: ModelProposal) -> tuple[CandidateInterpretation, ...]:
        """Convert validated model outputs into hypothesis candidates (Step F3).

        Candidates carry ``EpistemicStatus.MODEL_PROPOSAL`` semantics in their
        structure and never receive fact status; USG is untouched.
        """
        hypotheses: list[CandidateInterpretation] = []
        for candidate in proposal.candidates:
            structure = {
                **dict(candidate.semantic_structure),
                "epistemic_status": EpistemicStatus.MODEL_PROPOSAL.value,
                "model_id": proposal.model_id,
                "model_version": proposal.model_version,
                "score_semantics": proposal.score_semantics,
                "limitations": list(proposal.limitations),
            }
            hypotheses.append(CandidateInterpretation(
                candidate_id=make_id("hypothesis_candidate"),
                semantic_structure=structure,
                evidence_refs=(),
                neural_scores=dict(candidate.neural_scores),
                constraint_status={},
                ranking_features=dict(candidate.ranking_features),
                status=CandidateStatus.GENERATED,
            ))
        return tuple(hypotheses)

    # ------------------------------------------------------------------ Step F4

    def filter_symbolically(
        self,
        proposals: tuple[CandidateInterpretation, ...],
        valid_predicates: Mapping[str, tuple[int, ...]] | None = None,
        required_references: tuple[str, ...] = (),
    ) -> tuple[tuple[CandidateInterpretation, ...], tuple[RejectionRecord, ...]]:
        """Apply type/rule/constraint/evidence checks with bounded reasons (F4)."""
        kept: list[CandidateInterpretation] = []
        rejections: list[RejectionRecord] = []
        for candidate in proposals:
            structure = candidate.semantic_structure
            predicate = structure.get("predicate")
            arity = structure.get("arity")
            if valid_predicates is not None:
                arities = valid_predicates.get(str(predicate)) if predicate else None
                if predicate is None or arities is None:
                    rejections.append(RejectionRecord(
                        candidate_id=candidate.candidate_id,
                        reason_code="unknown_predicate",
                        reason=f"predicate {predicate!r} is not in the validated ruleset",
                    ))
                    continue
                if arity is not None and int(arity) not in arities:
                    rejections.append(RejectionRecord(
                        candidate_id=candidate.candidate_id,
                        reason_code="arity_mismatch",
                        reason=f"arity {arity} not declared for {predicate!r}",
                    ))
                    continue
            if required_references:
                refs = set(candidate.evidence_refs)
                if not refs.intersection(required_references):
                    rejections.append(RejectionRecord(
                        candidate_id=candidate.candidate_id,
                        reason_code="missing_evidence_reference",
                        reason="no reference to a known evidence record",
                    ))
                    continue
            kept.append(candidate)
        return tuple(kept), tuple(rejections)

    # ------------------------------------------------------------------ Step F5

    def record_verified_feedback(
        self,
        proposal_keys: tuple[Mapping[str, Any], ...],
        verified: bool,
        *,
        verification_source: str,
    ) -> AdaptationLedger:
        """Update routing/ranking parameters only from verified outcomes (Step F5).

        ``verified=True`` means an independent verifier or genuine ground truth
        confirmed the outcome; unverified inferences can never label themselves
        as training data (Stage 7 gate). Each accepted update is bounded by
        ``max_weight_delta`` and versioned in the adaptation ledger.
        """
        if not verification_source:
            raise ValueError("verified feedback requires a verification_source")
        accepted: list[Mapping[str, Any]] = []
        rejected: list[Mapping[str, Any]] = []
        if verified:
            for key in proposal_keys:
                proposed_delta = float(key.get("weight_delta", 0.0))
                if abs(proposed_delta) > self.max_weight_delta:
                    rejected.append({
                        **dict(key),
                        "reason": "delta_exceeds_bounded_limit",
                        "limit": self.max_weight_delta,
                    })
                    continue
                accepted.append({
                    "weight_delta": proposed_delta,
                    "verification_source": verification_source,
                    "target": str(key.get("target", "ranking")),
                })
        else:
            for key in proposal_keys:
                rejected.append({
                    **dict(key),
                    "reason": "unverified_outcome_is_not_training_data",
                })
        self.ledger = AdaptationLedger(
            version=self.ledger.version + 1,
            updates=(*self.ledger.updates, *accepted),
            rejected_updates=(*self.ledger.rejected_updates, *rejected),
        )
        return self.ledger


__all__ = [
    "AdaptationLedger",
    "LearnedCandidateAdapter",
    "ModelAdapter",
    "ModelDescriptor",
    "ModelInvocationError",
    "RejectionRecord",
]
