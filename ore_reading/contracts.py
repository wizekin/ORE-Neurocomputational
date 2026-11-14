"""Public contracts for the Integrated Reading Algorithm."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence
import math


@dataclass(frozen=True, slots=True)
class ReadingLexiconEntry:
    """A provenance-tracked orthographic, phonological and semantic entry."""

    entry_id: str
    orthography: str
    phonemes: tuple[str, ...]
    frequency: float = 1.0
    semantic_features: Mapping[str, float] = field(default_factory=dict)
    lexical_category: str = "unknown"
    stress: tuple[int, ...] | None = None
    verified_by: str = "unspecified"
    source_reference: str = "unspecified"
    language: str = "und"

    def __post_init__(self) -> None:
        if not self.entry_id.strip() or not self.orthography.strip():
            raise ValueError("entry_id and orthography must be non-empty")
        if not self.phonemes or any(not isinstance(p, str) or not p.strip() for p in self.phonemes):
            raise ValueError("phonemes must contain at least one non-empty symbol")
        if not math.isfinite(self.frequency) or self.frequency <= 0:
            raise ValueError("frequency must be finite and positive")
        for key, value in self.semantic_features.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("semantic feature names must be non-empty strings")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("semantic feature values must lie in [0, 1]")
        if self.stress is not None and any(int(x) < 0 for x in self.stress):
            raise ValueError("stress indices must be non-negative")


@dataclass(frozen=True, slots=True)
class G2PRule:
    """One grapheme-to-phoneme rule, ordered by longest grapheme match."""

    graphemes: str
    phonemes: tuple[str, ...]
    confidence: float = 1.0
    source_reference: str = "default-rule-set"

    def __post_init__(self) -> None:
        if not self.graphemes or not self.phonemes:
            raise ValueError("G2P rules require graphemes and at least one phoneme")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("G2P confidence must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class ReadingRequest:
    """A sequence of text or pre-tokenized orthographic word forms."""

    sequence_id: str
    text: str | None = None
    tokens: Sequence[str] | None = None
    language: str = "en"
    reading_direction: str = "ltr"
    context: Mapping[str, float] = field(default_factory=dict)
    source_reference: str | None = None

    def __post_init__(self) -> None:
        if not self.sequence_id.strip():
            raise ValueError("sequence_id must be non-empty")
        if (self.text is None) == (self.tokens is None):
            raise ValueError("provide exactly one of text or tokens")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.tokens is not None and (not self.tokens or any(not str(t).strip() for t in self.tokens)):
            raise ValueError("tokens must be a non-empty sequence of non-empty strings")
        if self.reading_direction not in {"ltr", "rtl"}:
            raise ValueError("reading_direction must be 'ltr' or 'rtl'")
        for key, value in self.context.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("context feature names must be non-empty strings")
            if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
                raise ValueError("context feature values must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class ReadingConfig:
    """Functional inference and optional LIF population-simulation controls."""

    max_candidates: int = 12
    interactive_steps: int = 6
    interactive_feedback_gain: float = 0.30
    lateral_inhibition: float = 0.18
    lexical_route_gain: float = 1.20
    sublexical_route_gain: float = 1.00
    bayesian_temperature: float = 1.0
    phonological_noise: float = 0.12
    confidence_threshold: float = 0.55
    ambiguity_margin: float = 0.08
    context_gain: float = 0.20
    simulate_biology: bool = True
    neural_dt_ms: float = 1.0
    neural_steps: int = 30
    membrane_tau_ms: float = 20.0
    refractory_ms: float = 2.0
    input_current_gain_pa: float = 38.0
    recurrent_gain_pa: float = 2.5
    inhibition_gain_pa: float = 3.0

    def __post_init__(self) -> None:
        if self.max_candidates < 1 or self.interactive_steps < 1 or self.neural_steps < 1:
            raise ValueError("candidate, interactive and neural step counts must be positive")
        for name in ("interactive_feedback_gain", "lateral_inhibition", "phonological_noise",
                     "confidence_threshold", "ambiguity_margin", "context_gain"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("lexical_route_gain", "sublexical_route_gain", "bayesian_temperature",
                     "neural_dt_ms", "membrane_tau_ms", "input_current_gain_pa"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("refractory_ms", "recurrent_gain_pa", "inhibition_gain_pa"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReadingResult:
    """Complete functional and optional biological outputs for one sequence."""

    sequence_id: str
    original_text: str
    normalized_text: str
    tokens: list[dict[str, Any]]
    decoded_text: str
    semantic_state: dict[str, float]
    biological: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "ore.integrated_reading.v1",
            "sequence_id": self.sequence_id,
            "original_text": self.original_text,
            "normalized_text": self.normalized_text,
            "tokens": self.tokens,
            "decoded_text": self.decoded_text,
            "semantic_state": self.semantic_state,
            "biological": self.biological,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        }
