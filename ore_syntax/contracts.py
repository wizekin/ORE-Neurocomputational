"""Typed contracts for the nine-stage ORE syntax-processing system."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
import math


def _validate_weights(name: str, values: Mapping[str, float]) -> None:
    for key, raw in values.items():
        value = float(raw)
        if not isinstance(key, str) or not key.strip() or not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must map non-empty labels to finite values in [0, 1]")


@dataclass(frozen=True, slots=True)
class SyntaxRequest:
    """A sentence plus optional lexical/POS and semantic evidence from upstream ORE stages."""

    sequence_id: str
    text: str | None = None
    tokens: Sequence[str] | None = None
    language: str = "en"
    lexical_evidence: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    semantic_features: Mapping[int, Mapping[str, float]] = field(default_factory=dict)
    source_reference: str | None = None

    def __post_init__(self) -> None:
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if (self.text is None) == (self.tokens is None):
            raise ValueError("provide exactly one of text or tokens")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.tokens is not None and (
            not self.tokens or any(not isinstance(token, str) or not token.strip() for token in self.tokens)
        ):
            raise ValueError("tokens must be a non-empty sequence of non-empty strings")
        for name, evidence in (
            ("lexical_evidence", self.lexical_evidence),
            ("semantic_features", self.semantic_features),
        ):
            for index, weights in evidence.items():
                if not isinstance(index, int) or index < 0:
                    raise ValueError(f"{name} keys must be non-negative token indices")
                _validate_weights(f"{name}[{index}]", weights)


@dataclass(frozen=True, slots=True)
class SyntaxConfig:
    """Numerical controls; default values are engineering hypotheses, not neural constants."""

    pos_temperature: float = 0.24
    syntax_temperature: float = 0.32
    ambiguity_margin: float = 0.10
    confidence_threshold: float = 0.52
    max_parse_alternatives: int = 5
    max_tokens: int = 100
    inference_iterations: int = 4
    syntactic_gain: float = 0.55
    semantic_gain: float = 0.25
    recurrent_damping: float = 0.45
    gap_distance_penalty: float = 0.035
    simulate_biology: bool = True
    population_size: int = 10
    dt_ms: float = 1.0
    steps_per_operation: int = 2
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    membrane_resistance_mohm: float = 100.0
    current_gain_pa: float = 300.0
    recurrent_gain_pa: float = 18.0
    inhibition_gain_pa: float = 20.0

    def __post_init__(self) -> None:
        if self.max_parse_alternatives < 1 or self.max_tokens < 1 or self.inference_iterations < 1:
            raise ValueError("parse/token/iteration limits must be positive")
        if self.population_size < 2 or self.steps_per_operation < 1:
            raise ValueError("population_size must be >= 2 and steps_per_operation positive")
        for name in ("pos_temperature", "syntax_temperature", "dt_ms", "membrane_tau_ms", "membrane_resistance_mohm", "current_gain_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("ambiguity_margin", "confidence_threshold", "recurrent_damping"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        for name in ("syntactic_gain", "semantic_gain", "gap_distance_penalty"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.refractory_ms < 0 or self.recurrent_gain_pa < 0 or self.inhibition_gain_pa < 0:
            raise ValueError("refractory and synaptic-current parameters must be non-negative")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SyntaxResult:
    """Inspectable result containing all functional, structural, and biological tracks."""

    sequence_id: str
    original_text: str
    tokens: list[str]
    token_records: list[dict[str, Any]]
    pos_candidates: list[list[dict[str, Any]]]
    selected_pos: list[dict[str, Any]]
    phrases: list[dict[str, Any]]
    dependency_arcs: list[dict[str, Any]]
    constituency_parses: list[dict[str, Any]]
    constituency_tree: dict[str, Any]
    hierarchy: dict[str, Any]
    predicate_arguments: list[dict[str, Any]]
    thematic_roles: list[dict[str, Any]]
    ambiguity: dict[str, Any]
    long_distance_dependencies: list[dict[str, Any]]
    biological: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        def clean(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): clean(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [clean(item) for item in value]
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)

        return clean({
            "schema": "ore.syntax.9stage.v1",
            "sequence_id": self.sequence_id,
            "original_text": self.original_text,
            "tokens": self.tokens,
            "token_records": self.token_records,
            "pos_candidates": self.pos_candidates,
            "selected_pos": self.selected_pos,
            "phrases": self.phrases,
            "dependency_arcs": self.dependency_arcs,
            "constituency_parses": self.constituency_parses,
            "constituency_tree": self.constituency_tree,
            "hierarchy": self.hierarchy,
            "predicate_arguments": self.predicate_arguments,
            "thematic_roles": self.thematic_roles,
            "ambiguity": self.ambiguity,
            "long_distance_dependencies": self.long_distance_dependencies,
            "biological": self.biological,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
