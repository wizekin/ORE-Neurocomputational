"""Typed request, evidence, belief, configuration, and result contracts for Point 21."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _unit(name: str, value: float) -> float:
    """Validate a finite normalized parameter and return its float representation."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return number


@dataclass(frozen=True)
class Hypothesis:
    """A candidate explanation with a prior and optional feature-level predictions."""

    hypothesis_id: str
    statement: str
    prior_probability: float
    predicted_features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity, prior probability, and predicted feature probabilities."""
        if not self.hypothesis_id.strip() or not self.statement.strip():
            raise ValueError("hypothesis_id and statement must be non-empty")
        _unit("prior_probability", self.prior_probability)
        for feature, probability in self.predicted_features.items():
            if not str(feature).strip():
                raise ValueError("predicted feature names must be non-empty")
            _unit(f"predicted_features[{feature}]", probability)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-data hypothesis record."""
        return asdict(self)


@dataclass(frozen=True)
class EvidenceSource:
    """Describe a source independently of whether its information agrees with a belief."""

    source_id: str
    source_type: str
    reliability: float = 0.7
    diagnostic_value: float = 0.5
    accessibility: float = 0.5
    social_trust: float = 0.5
    challenge_cost: float = 0.0
    independence_group: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate source identifiers and bounded quality/accessibility estimates."""
        if not self.source_id.strip() or not self.source_type.strip():
            raise ValueError("source_id and source_type must be non-empty")
        for name in ("reliability", "diagnostic_value", "accessibility", "social_trust", "challenge_cost"):
            _unit(name, getattr(self, name))
        if self.independence_group is not None and not self.independence_group.strip():
            raise ValueError("independence_group cannot be an empty string")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached source record."""
        return asdict(self)


@dataclass(frozen=True)
class EvidenceItem:
    """One observation with explicit source provenance and likelihoods under each hypothesis.

    likelihoods are relative P(observation | hypothesis) values supplied by the caller. They are
    not automatically treated as objective probabilities or empirical truth labels.
    """

    evidence_id: str
    source_id: str
    content: str
    feature: str = "unspecified"
    likelihoods: Mapping[str, float] = field(default_factory=dict)
    salience: float = 0.5
    relevance: float = 0.5
    novelty: float = 0.5
    uncertainty_reduction: float = 0.5
    measurement_quality: float = 0.7
    direct_observation: bool = False
    context: Mapping[str, Any] = field(default_factory=dict)
    timestamp: float | None = None

    def __post_init__(self) -> None:
        """Validate identifiers, evidence likelihoods, and normalized processing features."""
        if not self.evidence_id.strip() or not self.source_id.strip() or not self.content.strip():
            raise ValueError("evidence_id, source_id, and content must be non-empty")
        if not self.feature.strip():
            raise ValueError("feature must be non-empty")
        for name in ("salience", "relevance", "novelty", "uncertainty_reduction", "measurement_quality"):
            _unit(name, getattr(self, name))
        if not self.likelihoods:
            raise ValueError("each evidence item must provide likelihoods for competing hypotheses")
        for hypothesis_id, likelihood in self.likelihoods.items():
            number = float(likelihood)
            if not str(hypothesis_id).strip() or not math.isfinite(number) or number <= 0.0:
                raise ValueError("likelihoods must have non-empty hypothesis IDs and finite positive values")
        if self.timestamp is not None and not math.isfinite(float(self.timestamp)):
            raise ValueError("timestamp must be finite when provided")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached evidence record."""
        return asdict(self)


@dataclass(frozen=True)
class Point21Config:
    """Tunable parameters for functional effects; none are asserted to be neural constants."""

    selection_congruence_bias: float = 0.55
    selection_challenge_cost: float = 0.35
    attention_congruence_bias: float = 0.25
    interpretation_prior_bias: float = 1.0
    readout_congruence_bias: float = 0.30
    memory_congruence_bias: float = 0.25
    selection_temperature: float = 0.35
    interpretation_temperature: float = 1.0
    evidence_weight_scale: float = 1.0
    update_rate: float = 0.75
    max_sources: int = 3
    accuracy_first: bool = False
    simulate_biology: bool = True
    population_decay: float = 0.22
    population_recurrent_gain: float = 0.28
    population_inhibition_gain: float = 0.16

    def __post_init__(self) -> None:
        """Reject unbounded bias controls and invalid numeric model parameters."""
        for name in (
            "selection_congruence_bias", "selection_challenge_cost",
            "attention_congruence_bias", "readout_congruence_bias",
            "memory_congruence_bias", "update_rate", "population_decay",
            "population_inhibition_gain",
        ):
            _unit(name, getattr(self, name))
        for name in ("interpretation_prior_bias", "evidence_weight_scale", "population_recurrent_gain"):
            number = float(getattr(self, name))
            if not math.isfinite(number) or number < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("selection_temperature", "interpretation_temperature"):
            number = float(getattr(self, name))
            if not math.isfinite(number) or number <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.max_sources, int) or self.max_sources < 1:
            raise ValueError("max_sources must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        """Return an immutable configuration snapshot."""
        return asdict(self)


@dataclass(frozen=True)
class Point21Request:
    """One claim evaluation with competing hypotheses, candidate sources, and observations."""

    claim_id: str
    claim: str
    hypotheses: Sequence[Hypothesis]
    sources: Sequence[EvidenceSource]
    evidence: Sequence[EvidenceItem]
    focus_hypothesis_id: str | None = None
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate referential integrity and ensure priors form a probability distribution."""
        if not self.claim_id.strip() or not self.claim.strip():
            raise ValueError("claim_id and claim must be non-empty")
        hypotheses = tuple(self.hypotheses)
        sources = tuple(self.sources)
        evidence = tuple(self.evidence)
        if len(hypotheses) < 2:
            raise ValueError("at least two competing hypotheses are required")
        if len({item.hypothesis_id for item in hypotheses}) != len(hypotheses):
            raise ValueError("hypothesis IDs must be unique")
        prior_total = sum(item.prior_probability for item in hypotheses)
        if prior_total <= 0.0 or not math.isclose(prior_total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError("hypothesis prior probabilities must sum to 1")
        hypothesis_ids = {item.hypothesis_id for item in hypotheses}
        source_ids = {item.source_id for item in sources}
        if len(source_ids) != len(sources):
            raise ValueError("source IDs must be unique")
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise ValueError("evidence IDs must be unique")
        for item in evidence:
            if item.source_id not in source_ids:
                raise ValueError(f"evidence {item.evidence_id!r} references an unknown source")
            unknown = set(item.likelihoods) - hypothesis_ids
            if unknown:
                raise ValueError(f"evidence {item.evidence_id!r} references unknown hypotheses: {sorted(unknown)}")
        if self.focus_hypothesis_id is not None and self.focus_hypothesis_id not in hypothesis_ids:
            raise ValueError("focus_hypothesis_id must identify one of the hypotheses")
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "sources", sources)
        object.__setattr__(self, "evidence", evidence)
        if not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached request snapshot."""
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "sources": [item.to_dict() for item in self.sources],
            "evidence": [item.to_dict() for item in self.evidence],
            "focus_hypothesis_id": self.focus_hypothesis_id,
            "context": dict(self.context),
        }


@dataclass
class Point21State:
    """Persist revisable belief distributions and evidence memories across claim evaluations."""

    beliefs_by_claim: dict[str, dict[str, float]] = field(default_factory=dict)
    memory_traces: dict[str, dict[str, Any]] = field(default_factory=dict)
    audit_history: list[dict[str, Any]] = field(default_factory=list)
    run_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize state without returning mutable internal mappings."""
        return {
            "beliefs_by_claim": {key: dict(value) for key, value in self.beliefs_by_claim.items()},
            "memory_traces": {key: dict(value) for key, value in self.memory_traces.items()},
            "audit_history": [dict(item) for item in self.audit_history],
            "run_count": self.run_count,
        }


@dataclass(frozen=True)
class Point21Result:
    """Return beliefs, staged evidence audit, corrective controls, and biological-reference outputs."""

    claim_id: str
    beliefs: Mapping[str, float]
    confidence: float
    evidence_audit: Sequence[Mapping[str, Any]]
    source_selection: Mapping[str, Any]
    attention: Mapping[str, Any]
    interpretations: Mapping[str, Any]
    evidence_quality: Mapping[str, Any]
    evidence_readout: Mapping[str, Any]
    memory_retrieval: Mapping[str, Any]
    feedback_loop: Mapping[str, Any]
    corrective_control: Mapping[str, Any]
    validation: Mapping[str, Any]
    biological: Mapping[str, Any]
    operation_trace: Sequence[Mapping[str, Any]]
    final_state: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the complete output as plain serializable data."""
        return {
            "schema": "ore.confirmation_bias.point21.1.0",
            "claim_id": self.claim_id,
            "beliefs": dict(self.beliefs),
            "confidence": self.confidence,
            "evidence_audit": [dict(item) for item in self.evidence_audit],
            "source_selection": dict(self.source_selection),
            "attention": dict(self.attention),
            "interpretations": dict(self.interpretations),
            "evidence_quality": dict(self.evidence_quality),
            "evidence_readout": dict(self.evidence_readout),
            "memory_retrieval": dict(self.memory_retrieval),
            "feedback_loop": dict(self.feedback_loop),
            "corrective_control": dict(self.corrective_control),
            "validation": dict(self.validation),
            "biological": dict(self.biological),
            "operation_trace": [dict(item) for item in self.operation_trace],
            "final_state": dict(self.final_state),
            "metadata": dict(self.metadata),
        }
