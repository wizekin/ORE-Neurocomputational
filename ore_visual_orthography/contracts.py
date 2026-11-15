"""Typed contracts for the five-stage visual and orthographic representation algorithm."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import math
import unicodedata
from typing import Any, Mapping, Sequence

import numpy as np


def _finite_array(name: str, value: Any, ndim: int | None = None) -> np.ndarray:
    """Convert an input to finite float64 values and optionally enforce its dimensionality."""
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; received {array.ndim}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and contain only finite values")
    return array


def _jsonable(value: Any) -> Any:
    """Convert dataclasses, NumPy values and nested containers to JSON-compatible values."""
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return repr(value)


def _normalize_sequence(value: str | Sequence[str]) -> tuple[str, ...]:
    """Normalize a word to ordered Unicode letter units without removing internal order."""
    if isinstance(value, str):
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return tuple(character for character in normalized if not character.isspace())
    output = []
    for item in value:
        normalized = unicodedata.normalize("NFKC", str(item)).casefold()
        if normalized and not normalized.isspace():
            output.append(normalized)
    return tuple(output)


@dataclass(frozen=True, slots=True)
class LetterPrototype:
    """A labelled visual-form prototype with diagonal feature variance.

    Multiple prototypes can share a label while representing different fonts, cases, or glyph
    styles. Their feature vectors are engineering templates, not a claim about cortical storage.
    """

    label: str
    mean: Sequence[float]
    variance: Sequence[float] | None = None
    form_id: str = ""

    def __post_init__(self) -> None:
        """Validate the abstract letter label and its finite feature statistics."""
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("letter prototype label must be a non-empty string")
        mean = _finite_array("prototype mean", self.mean, ndim=1)
        variance = np.ones_like(mean) if self.variance is None else _finite_array(
            "prototype variance", self.variance, ndim=1
        )
        if mean.shape != variance.shape or np.any(variance <= 0.0):
            raise ValueError("prototype variance must be positive and match prototype mean")
        object.__setattr__(self, "label", _normalize_sequence(self.label)[0])
        object.__setattr__(self, "mean", tuple(float(item) for item in mean))
        object.__setattr__(self, "variance", tuple(float(item) for item in variance))
        object.__setattr__(self, "form_id", str(self.form_id))


@dataclass(frozen=True, slots=True)
class LexicalCandidate:
    """A caller-supplied orthographic word candidate and its positive exposure-frequency prior."""

    word: str
    frequency: float = 1.0
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Normalize the candidate spelling and reject invalid frequency priors."""
        if not isinstance(self.word, str) or not self.word.strip():
            raise ValueError("lexical candidate word must be a non-empty string")
        if not math.isfinite(float(self.frequency)) or float(self.frequency) <= 0.0:
            raise ValueError("lexical candidate frequency must be finite and positive")
        normalized = "".join(_normalize_sequence(self.word))
        if not normalized:
            raise ValueError("lexical candidate must contain at least one non-space symbol")
        object.__setattr__(self, "word", normalized)
        object.__setattr__(self, "frequency", float(self.frequency))


@dataclass(frozen=True, slots=True)
class VisualOrthographyConfig:
    """Reproducible controls for feature extraction, orthographic computation and the LIF reference.

    These values are software defaults for a research prototype. They are not fitted human
    physiological constants, universal letter-position parameters, or measured neural thresholds.
    """

    gabor_size: int = 15
    orientations: Sequence[float] = (0.0, 0.7853981634, 1.5707963268, 2.3561944902)
    spatial_frequencies: Sequence[float] = (0.08, 0.16)
    gabor_sigma: float = 3.0
    gabor_gamma: float = 0.7
    visual_threshold_quantile: float = 0.72
    max_candidate_regions: int = 64
    prototype_temperature: float = 0.25
    prototype_learning_rate: float = 0.05
    position_grid_size: int = 32
    position_sigma: float = 0.10
    conjunction_distance_sigma: float = 0.12
    bigram_distance_decay: float = 0.35
    recurrent_steps: int = 8
    recurrent_damping: float = 0.35
    lateral_inhibition: float = 0.15
    identity_weight: float = 0.15
    position_weight: float = 0.20
    conjunction_weight: float = 0.20
    bigram_weight: float = 0.35
    frequency_weight: float = 0.10
    lexical_temperature: float = 0.20
    max_settling_steps: int = 40
    settling_damping: float = 0.55
    stability_epsilon: float = 1e-4
    confidence_threshold: float = 0.65
    simulate_biology: bool = True
    neurons_per_population: int = 12
    lif_steps_per_operation: int = 4
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    membrane_resistance_mohm: float = 100.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    refractory_ms: float = 2.0
    input_gain_pa: float = 280.0
    recurrent_gain_pa: float = 24.0
    inhibition_gain_pa: float = 14.0
    random_seed: int = 17

    def __post_init__(self) -> None:
        """Validate dimensions, temperatures, bounded learning rates and stable LIF settings."""
        if self.gabor_size < 5 or self.gabor_size % 2 == 0:
            raise ValueError("gabor_size must be an odd integer >= 5")
        if not self.orientations or not self.spatial_frequencies:
            raise ValueError("at least one orientation and spatial frequency are required")
        if self.max_candidate_regions < 1 or self.position_grid_size < 4:
            raise ValueError("candidate-region and position-grid sizes must be positive")
        if self.recurrent_steps < 1 or self.max_settling_steps < 1:
            raise ValueError("recurrent and settling iteration limits must be positive")
        if self.neurons_per_population < 2 or self.lif_steps_per_operation < 1:
            raise ValueError("LIF population size must be >= 2 and steps_per_operation positive")
        positive = (
            "gabor_sigma", "gabor_gamma", "prototype_temperature", "position_sigma",
            "conjunction_distance_sigma", "lexical_temperature", "dt_ms",
            "membrane_tau_ms", "membrane_resistance_mohm", "input_gain_pa",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        bounded = (
            "visual_threshold_quantile", "prototype_learning_rate", "recurrent_damping",
            "lateral_inhibition", "settling_damping", "confidence_threshold",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        nonnegative = (
            "bigram_distance_decay", "identity_weight", "position_weight",
            "conjunction_weight", "bigram_weight", "frequency_weight",
            "refractory_ms", "recurrent_gain_pa", "inhibition_gain_pa",
            "stability_epsilon",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if sum(float(getattr(self, name)) for name in (
            "identity_weight", "position_weight", "conjunction_weight",
            "bigram_weight", "frequency_weight",
        )) <= 0.0:
            raise ValueError("at least one lexical integration weight must be positive")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")
        object.__setattr__(self, "orientations", tuple(float(x) for x in self.orientations))
        object.__setattr__(self, "spatial_frequencies", tuple(float(x) for x in self.spatial_frequencies))

    def to_dict(self) -> dict[str, Any]:
        """Return all configuration controls as a plain dictionary for experiment reports."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VisualOrthographyRequest:
    """One sample plus optional image, known sequence, labelled visual prototypes and lexicon.

    A visual image is not automatically converted to text. Letter labels must come from a supplied
    prototype bank or from an explicitly supplied orthographic sequence. Labeled variants support
    transparent, supervised prototype updates.
    """

    sample_id: str
    visual_image: Any | None = None
    orthographic_sequence: str | Sequence[str] | None = None
    prototypes: Sequence[LetterPrototype] = ()
    lexicon: Sequence[LexicalCandidate] = ()
    comparison_strings: Sequence[str] = ()
    labeled_variants: Mapping[str, Sequence[Sequence[float]]] = field(default_factory=dict)
    position_uncertainty: float | None = None
    language: str = "und"

    def __post_init__(self) -> None:
        """Validate sample inputs and freeze normalized sequence, prototype and training records."""
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if self.visual_image is None and self.orthographic_sequence is None:
            raise ValueError("supply visual_image, orthographic_sequence, or both")
        if self.visual_image is not None:
            image = _finite_array("visual_image", self.visual_image)
            if image.ndim not in (2, 3):
                raise ValueError("visual_image must be a 2-D grayscale or 3-D RGB/RGBA array")
            if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
                raise ValueError("visual_image channel count must be 1, 3, or 4")
            object.__setattr__(self, "visual_image", image)
        if self.orthographic_sequence is not None:
            sequence = _normalize_sequence(self.orthographic_sequence)
            if not sequence:
                raise ValueError("orthographic_sequence must contain at least one symbol")
            object.__setattr__(self, "orthographic_sequence", sequence)
        prototypes = tuple(self.prototypes)
        if any(not isinstance(item, LetterPrototype) for item in prototypes):
            raise TypeError("prototypes must contain LetterPrototype instances")
        lexicon = tuple(self.lexicon)
        if any(not isinstance(item, LexicalCandidate) for item in lexicon):
            raise TypeError("lexicon must contain LexicalCandidate instances")
        object.__setattr__(self, "prototypes", prototypes)
        object.__setattr__(self, "lexicon", lexicon)
        object.__setattr__(self, "comparison_strings", tuple(
            "".join(_normalize_sequence(item)) for item in self.comparison_strings
        ))
        normalized_variants: dict[str, tuple[tuple[float, ...], ...]] = {}
        for label, vectors in self.labeled_variants.items():
            normalized_label = "".join(_normalize_sequence(str(label)))
            if not normalized_label:
                raise ValueError("labeled_variants labels must be non-empty")
            validated = []
            for vector in vectors:
                values = _finite_array("labeled variant feature vector", vector, ndim=1)
                validated.append(tuple(float(value) for value in values))
            normalized_variants[normalized_label] = tuple(validated)
        object.__setattr__(self, "labeled_variants", normalized_variants)
        if self.position_uncertainty is not None:
            uncertainty = float(self.position_uncertainty)
            if not math.isfinite(uncertainty) or uncertainty <= 0.0:
                raise ValueError("position_uncertainty must be finite and positive")
            object.__setattr__(self, "position_uncertainty", uncertainty)
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty identifier")


@dataclass(slots=True)
class VisualOrthographyResult:
    """Return stage outputs, operation trace and the isolated biological-reference simulation."""

    sample_id: str
    functional: dict[str, Any]
    biological: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result while preserving functional and biological evidence boundaries."""
        return _jsonable({
            "schema": "ore.visual_orthography.1.0",
            "sample_id": self.sample_id,
            "functional": self.functional,
            "biological": self.biological,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
