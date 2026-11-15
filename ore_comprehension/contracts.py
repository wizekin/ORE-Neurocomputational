"""Typed requests, configuration, and results for predictive sentence comprehension."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Mapping, Sequence


_TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)?|[^\w\s]", flags=re.UNICODE)


def tokenize_text(text: str) -> list[str]:
    """Split text into word and punctuation tokens without discarding sentence boundaries."""
    return _TOKEN_PATTERN.findall(text)


@dataclass(frozen=True, slots=True)
class ComprehensionRequest:
    """Describe one language-comprehension input and optional caller-supplied linguistic evidence."""

    sequence_id: str
    text: str | None = None
    tokens: Sequence[str] | None = None
    language: str = "en"
    lexical_categories: Mapping[str, str] = field(default_factory=dict)
    semantic_features: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    world_knowledge: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    discourse_context: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Reject ambiguous input forms and malformed optional evidence before processing."""
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        if (self.text is None) == (self.tokens is None):
            raise ValueError("provide exactly one of text or tokens")
        if self.text is not None and not self.text.strip():
            raise ValueError("text cannot be empty")
        if self.tokens is not None and (
            not self.tokens or any(not isinstance(token, str) or not token.strip() for token in self.tokens)
        ):
            raise ValueError("tokens must contain non-empty strings")
        if not self.language.strip():
            raise ValueError("language must be non-empty")
        for collection_name, collection in (
            ("semantic_features", self.semantic_features),
            ("world_knowledge", self.world_knowledge),
        ):
            for key, values in collection.items():
                if not isinstance(key, str) or not key.strip() or not isinstance(values, Mapping):
                    raise ValueError(f"{collection_name} must map non-empty words to feature mappings")
                for feature, raw in values.items():
                    value = float(raw)
                    if not isinstance(feature, str) or not feature.strip() or not math.isfinite(value):
                        raise ValueError(f"{collection_name} values must be finite numeric features")
        for word, category in self.lexical_categories.items():
            if not isinstance(word, str) or not word.strip() or not isinstance(category, str) or not category.strip():
                raise ValueError("lexical_categories must map non-empty words to non-empty category labels")
        if any(not isinstance(item, str) or not item.strip() for item in self.discourse_context):
            raise ValueError("discourse_context must contain non-empty strings")

    def to_tokens(self) -> list[str]:
        """Return the supplied tokens or tokenize the source text using the package's deterministic tokenizer."""
        if self.tokens is not None:
            return list(self.tokens)
        return tokenize_text(self.text or "")


@dataclass(frozen=True, slots=True)
class ComprehensionConfig:
    """Numerical controls for the functional and biological-reference tracks.

    Defaults are transparent engineering parameters, not measured human neural constants.
    """
    prediction_temperature: float = 0.85
    syntax_weight: float = 1.0
    semantic_weight: float = 0.9
    world_weight: float = 0.65
    lexical_weight: float = 0.35
    max_candidates: int = 12
    max_hypotheses: int = 6
    context_decay: float = 0.82
    conflict_threshold: float = 0.22
    completion_threshold: float = 0.65
    n400_gain_uv: float = 4.5
    p600_gain_uv: float = 3.0
    learning_rate: float = 0.08
    simulate_biology: bool = True
    lif_population_size: int = 12
    lif_steps_per_operation: int = 2
    lif_dt_ms: float = 1.0
    lif_membrane_tau_ms: float = 20.0
    lif_refractory_ms: float = 2.0
    lif_rest_mv: float = -70.0
    lif_reset_mv: float = -68.0
    lif_threshold_mv: float = -50.0
    lif_resistance_mohm: float = 100.0
    lif_current_gain_pa: float = 300.0
    lif_recurrent_gain_pa: float = 18.0
    lif_inhibition_gain_pa: float = 20.0

    def __post_init__(self) -> None:
        """Validate bounded weights and positive numerical parameters."""
        if self.max_candidates < 2 or self.max_hypotheses < 2:
            raise ValueError("candidate and hypothesis limits must be at least 2")
        if self.lif_population_size < 2 or self.lif_steps_per_operation < 1:
            raise ValueError("LIF population size must be >= 2 and steps per operation must be positive")
        bounded = ("context_decay", "conflict_threshold", "completion_threshold")
        for name in bounded:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        positive = (
            "prediction_temperature", "n400_gain_uv", "p600_gain_uv",
            "lif_dt_ms", "lif_membrane_tau_ms", "lif_resistance_mohm", "lif_current_gain_pa",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        nonnegative = (
            "syntax_weight", "semantic_weight", "world_weight", "lexical_weight",
            "learning_rate", "lif_refractory_ms", "lif_recurrent_gain_pa", "lif_inhibition_gain_pa",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.lif_rest_mv <= self.lif_reset_mv < self.lif_threshold_mv:
            raise ValueError("LIF voltages must satisfy rest <= reset < threshold")

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary representation suitable for metadata and JSON serialization."""
        return asdict(self)


@dataclass(slots=True)
class ComprehensionResult:
    """Expose per-word predictions, all 38 operation traces, final meaning, and model diagnostics."""

    sequence_id: str
    source_text: str
    tokens: list[str]
    token_records: list[dict[str, Any]]
    sentence_interpretations: list[dict[str, Any]]
    operation_trace: list[dict[str, Any]]
    stage_trace: list[dict[str, Any]]
    next_word_predictions: list[dict[str, Any]]
    biological: dict[str, Any]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert the complete result into JSON-safe primitive Python values."""
        def clean(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {str(key): clean(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [clean(item) for item in value]
            if hasattr(value, "tolist") and callable(value.tolist):
                return clean(value.tolist())
            if isinstance(value, float) and not math.isfinite(value):
                return None
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            return repr(value)
        return clean({
            "schema": "ore.comprehension.tdpi.38.v1",
            "sequence_id": self.sequence_id,
            "source_text": self.source_text,
            "tokens": self.tokens,
            "token_records": self.token_records,
            "sentence_interpretations": self.sentence_interpretations,
            "operation_trace": self.operation_trace,
            "stage_trace": self.stage_trace,
            "next_word_predictions": self.next_word_predictions,
            "biological": self.biological,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
