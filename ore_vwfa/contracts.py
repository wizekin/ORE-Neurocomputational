"""Typed, validated inputs and outputs for the six-stage VWFA reference implementation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


def _finite_array(name: str, value: Any, ndim: int | None = None) -> np.ndarray:
    """Convert a numerical value to float64 and reject empty, non-finite, or wrongly shaped data."""
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; received {array.ndim}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and contain only finite values")
    return array


def _jsonable(value: Any) -> Any:
    """Convert dataclasses and NumPy values into JSON-compatible built-in values."""
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
    return repr(value)


@dataclass(frozen=True, slots=True)
class GraphemePrototype:
    """A labelled visual-population prototype with per-feature variance and provenance.

    The mean and variance must follow the feature-vector contract emitted by Stage B. They are
    engineering reference statistics, not a claim that cortical neurons store literal vectors.
    """

    label: str
    mean: Sequence[float]
    variance: Sequence[float] | None = None
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Validate the label and convert compatible positive-variance vectors to immutable tuples."""
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("prototype label must be a non-empty string")
        mean = _finite_array("prototype mean", self.mean, ndim=1)
        variance = np.ones_like(mean) if self.variance is None else _finite_array(
            "prototype variance", self.variance, ndim=1
        )
        if variance.shape != mean.shape or np.any(variance <= 0.0):
            raise ValueError("prototype variance must be positive and match prototype mean")
        object.__setattr__(self, "mean", tuple(float(item) for item in mean))
        object.__setattr__(self, "variance", tuple(float(item) for item in variance))


@dataclass(frozen=True, slots=True)
class LexicalEntry:
    """A caller-supplied whole-word form, exposure frequency, semantic evidence, and provenance."""

    word: str
    frequency: float = 1.0
    semantic_features: Mapping[str, float] = field(default_factory=dict)
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Reject empty words and invalid exposure counts, then freeze semantic feature values."""
        if not isinstance(self.word, str) or not self.word.strip():
            raise ValueError("lexical entry word must be non-empty")
        if not math.isfinite(float(self.frequency)) or float(self.frequency) <= 0.0:
            raise ValueError("lexical frequency must be finite and positive")
        features = {str(key): float(value) for key, value in self.semantic_features.items()}
        if any(not key.strip() or not math.isfinite(value) for key, value in features.items()):
            raise ValueError("semantic feature keys and values must be non-empty and finite")
        object.__setattr__(self, "semantic_features", features)


@dataclass(frozen=True, slots=True)
class MorphemeEntry:
    """A known written morpheme and optional caller-supplied semantic/grammatical attributes.

    Roles are orthographic hypotheses: prefix, stem, suffix, inflection, or unspecified. Meaning
    is never inferred from spelling when the caller has not supplied feature evidence.
    """

    form: str
    role: str = "unspecified"
    frequency: float = 1.0
    semantic_features: Mapping[str, float] = field(default_factory=dict)
    grammatical_features: Mapping[str, float] = field(default_factory=dict)
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Validate the written morpheme and its supplied frequency and feature maps."""
        if not isinstance(self.form, str) or not self.form.strip():
            raise ValueError("morpheme form must be non-empty")
        if self.role not in {"prefix", "stem", "suffix", "inflection", "unspecified"}:
            raise ValueError("morpheme role must be prefix, stem, suffix, inflection, or unspecified")
        if not math.isfinite(float(self.frequency)) or float(self.frequency) <= 0.0:
            raise ValueError("morpheme frequency must be finite and positive")
        for name in ("semantic_features", "grammatical_features"):
            features = {str(key): float(value) for key, value in getattr(self, name).items()}
            if any(not key.strip() or not math.isfinite(value) for key, value in features.items()):
                raise ValueError(f"{name} keys and values must be non-empty and finite")
            object.__setattr__(self, name, features)


@dataclass(frozen=True, slots=True)
class VWFAExample:
    """An explicitly labelled literacy-learning example pairing written form with language features."""

    orthographic_form: str
    phonological_features: Sequence[float]
    category: str = "word"

    def __post_init__(self) -> None:
        """Validate an orthographic form and its finite, non-empty phonological target vector."""
        if not isinstance(self.orthographic_form, str) or not self.orthographic_form.strip():
            raise ValueError("learning example orthographic_form must be non-empty")
        if self.category not in {"word", "control"}:
            raise ValueError("learning example category must be word or control")
        features = _finite_array("phonological_features", self.phonological_features, ndim=1)
        object.__setattr__(self, "phonological_features", tuple(float(item) for item in features))


@dataclass(frozen=True, slots=True)
class VWFAConfig:
    """Reproducible numerical settings for the functional pipeline and LIF reference track.

    Values are software defaults for experiments, not measured physiological constants or fitted
    properties of a particular participant. Use held-out data for model selection and validation.
    """

    gabor_size: int = 15
    orientations: Sequence[float] = (0.0, 0.7853981634, 1.5707963268, 2.3561944902)
    spatial_frequencies: Sequence[float] = (0.08, 0.16)
    gabor_sigma: float = 3.0
    gabor_gamma: float = 0.7
    local_contrast_sigma: float = 3.0
    epsilon: float = 1e-8
    population_temperature: float = 0.25
    invariance_weight: float = 0.2
    max_morphology_depth: int = 4
    recurrent_steps: int = 12
    recurrent_damping: float = 0.35
    convergence_tolerance: float = 1e-5
    lif_population_size: int = 12
    lif_steps_per_operation: int = 4
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    membrane_resistance_mohm: float = 100.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    refractory_ms: float = 2.0
    input_current_pa: float = 350.0
    recurrent_gain: float = 0.16
    inhibition_gain: float = 0.06
    learning_rate: float = 0.03
    consolidation_rate: float = 0.005
    max_synaptic_weight: float = 0.5
    random_seed: int = 23
    simulate_biology: bool = True

    def __post_init__(self) -> None:
        """Validate image-filter geometry, dynamics, learning rates, and bounded iteration counts."""
        if self.gabor_size < 5 or self.gabor_size % 2 == 0:
            raise ValueError("gabor_size must be an odd integer >= 5")
        if not self.orientations or not self.spatial_frequencies:
            raise ValueError("at least one orientation and spatial frequency are required")
        if self.max_morphology_depth < 1 or self.recurrent_steps < 1:
            raise ValueError("morphology depth and recurrent steps must be positive")
        if self.lif_population_size < 2 or self.lif_steps_per_operation < 1:
            raise ValueError("LIF population size must be >= 2 and steps_per_operation positive")
        positive = (
            "gabor_sigma", "gabor_gamma", "local_contrast_sigma", "epsilon",
            "population_temperature", "dt_ms", "membrane_tau_ms", "membrane_resistance_mohm",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        bounded = (
            "invariance_weight", "recurrent_damping", "learning_rate", "consolidation_rate",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        nonnegative = (
            "refractory_ms", "input_current_pa", "recurrent_gain", "inhibition_gain", "max_synaptic_weight",
        )
        for name in nonnegative:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")
        object.__setattr__(self, "orientations", tuple(float(item) for item in self.orientations))
        object.__setattr__(self, "spatial_frequencies", tuple(float(item) for item in self.spatial_frequencies))

    def to_dict(self) -> dict[str, Any]:
        """Return all numerical settings as a plain dictionary for reproducible experiment metadata."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class VWFARequest:
    """One printed-word sample plus optional upstream sequence, lexical data, context, and learning labels.

    A grayscale/RGB image is preferred. An ``orthographic_sequence`` can be supplied when an upstream
    recognizer has already decoded letters; that sequence is kept explicitly separate from model-derived
    visual evidence. Prototypes, lexical entries, morphemes, and learning labels are never fabricated.
    """

    sample_id: str
    visual_image: Any | None = None
    orthographic_sequence: Sequence[str] = ()
    visual_views: Sequence[Any] = ()
    grapheme_prototypes: Sequence[GraphemePrototype] = ()
    lexicon: Sequence[LexicalEntry] = ()
    morphology_inventory: Sequence[MorphemeEntry] = ()
    context_features: Mapping[str, float] = field(default_factory=dict)
    learning_examples: Sequence[VWFAExample] = ()
    learning_authorized: bool = False
    language: str = "und"
    subject_id: str | None = None

    def __post_init__(self) -> None:
        """Validate sample identity, sensory inputs, inventories, and explicitly authorized learning data."""
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if self.visual_image is None and not self.orthographic_sequence:
            raise ValueError("visual_image or an upstream orthographic_sequence is required")
        if self.visual_image is not None:
            image = _finite_array("visual_image", self.visual_image)
            if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
                raise ValueError("visual_image must be 2-D grayscale or 3-D RGB/RGBA")
            if np.issubdtype(np.asarray(self.visual_image).dtype, np.integer):
                maximum = float(np.iinfo(np.asarray(self.visual_image).dtype).max)
                image = image / maximum
            if np.min(image) < 0.0 or np.max(image) > 1.0:
                raise ValueError("visual_image values must lie in [0, 1] after integer scaling")
            object.__setattr__(self, "visual_image", image)
        sequence = tuple(str(item) for item in self.orthographic_sequence)
        if any(not item for item in sequence):
            raise ValueError("orthographic_sequence items must be non-empty strings")
        object.__setattr__(self, "orthographic_sequence", sequence)
        views = []
        for index, raw in enumerate(self.visual_views):
            view = _finite_array(f"visual_views[{index}]", raw)
            if view.ndim not in (2, 3):
                raise ValueError("each visual view must be a 2-D grayscale or 3-D color array")
            if np.issubdtype(np.asarray(raw).dtype, np.integer):
                view = view / float(np.iinfo(np.asarray(raw).dtype).max)
            if np.min(view) < 0.0 or np.max(view) > 1.0:
                raise ValueError("visual view values must lie in [0, 1]")
            views.append(view)
        object.__setattr__(self, "visual_views", tuple(views))
        for name in ("grapheme_prototypes", "lexicon", "morphology_inventory", "learning_examples"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        context = {str(key): float(value) for key, value in self.context_features.items()}
        if any(not key.strip() or not math.isfinite(value) for key, value in context.items()):
            raise ValueError("context feature keys and values must be non-empty and finite")
        object.__setattr__(self, "context_features", context)
        if self.learning_examples and not self.learning_authorized:
            # Keep data available for an explicit preview, but the processor must not update state.
            pass
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty identifier")


@dataclass(slots=True)
class VWFAResult:
    """Complete six-stage result with isolated functional, biological-reference, and diagnostic tracks."""

    sample_id: str
    functional: dict[str, Any]
    biological: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize arrays and nested records while preserving functional/biological track boundaries."""
        return _jsonable({
            "schema": "ore.vwfa.six_stage.1.0",
            "sample_id": self.sample_id,
            "functional": self.functional,
            "biological": self.biological,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
