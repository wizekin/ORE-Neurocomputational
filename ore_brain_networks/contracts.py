"""Typed contracts for the Brain Networks and Biological Evidence Algorithm (BNBEA).

Functional representations, simulated biological states, and experimentally observed data are
kept in separate fields so a successful software prediction is never mistaken for neural evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


def _finite_array(name: str, value: Any, ndim: int | None = None) -> np.ndarray:
    """Convert numerical input to a finite float array and validate its dimensionality."""
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions; received {array.ndim}")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be non-empty and contain only finite values")
    return array


def _jsonable(value: Any) -> Any:
    """Convert nested contracts and NumPy values into JSON-compatible Python values."""
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


@dataclass(frozen=True, slots=True)
class GraphemePrototype:
    """A labelled population prototype for one visual grapheme realization.

    ``mean`` and ``variance`` must use the same feature schema emitted by the grapheme decoder.
    They are engineering reference statistics, not a claim about a literal cortical memory format.
    """

    label: str
    mean: Sequence[float]
    variance: Sequence[float] | None = None
    prototype_id: str = ""

    def __post_init__(self) -> None:
        """Validate a non-empty label and finite, dimensionally consistent prototype statistics."""
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("grapheme prototype label must be a non-empty string")
        mean = _finite_array("prototype mean", self.mean, ndim=1)
        variance = np.ones_like(mean) if self.variance is None else _finite_array(
            "prototype variance", self.variance, ndim=1
        )
        if variance.shape != mean.shape or np.any(variance <= 0.0):
            raise ValueError("prototype variance must be positive and match the mean dimensions")
        object.__setattr__(self, "mean", tuple(float(x) for x in mean))
        object.__setattr__(self, "variance", tuple(float(x) for x in variance))


@dataclass(frozen=True, slots=True)
class LexicalEntry:
    """An explicitly supplied orthographic lexical item with optional metadata and provenance."""

    word: str
    frequency: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_reference: str | None = None

    def __post_init__(self) -> None:
        """Reject empty lexical items and non-positive or non-finite frequency values."""
        if not isinstance(self.word, str) or not self.word.strip():
            raise ValueError("lexical entry word must be a non-empty string")
        if not math.isfinite(float(self.frequency)) or float(self.frequency) <= 0.0:
            raise ValueError("lexical entry frequency must be finite and positive")
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True, slots=True)
class BNBEAConfig:
    """Transparent numerical settings for the functional, LIF and measurement-reference tracks.

    Parameter values are reproducible software defaults, not universal physiological constants or
    participant-specific fits. Users must provide head models and empirical data for real comparisons.
    """

    gabor_size: int = 15
    orientations: Sequence[float] = (0.0, 0.7853981634, 1.5707963268, 2.3561944902)
    spatial_frequencies: Sequence[float] = (0.08, 0.16)
    gabor_sigma: float = 3.0
    gabor_gamma: float = 0.7
    visual_threshold_quantile: float = 0.72
    max_candidate_regions: int = 128
    population_temperature: float = 0.25
    prototype_learning_rate: float = 0.05
    recurrent_steps: int = 6
    recurrent_damping: float = 0.45
    lif_population_size: int = 12
    lif_steps_per_stage: int = 8
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    membrane_resistance_mohm: float = 100.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    refractory_ms: float = 2.0
    current_gain_pa: float = 300.0
    recurrent_gain_pa: float = 18.0
    inhibition_gain_pa: float = 20.0
    eeg_noise_sd: float = 0.0
    meg_noise_sd: float = 0.0
    inverse_regularization: float = 0.05
    hrf_duration_seconds: float = 32.0
    random_seed: int = 17
    simulate_biology: bool = True

    def __post_init__(self) -> None:
        """Validate positive numerical controls, normalized thresholds and stable LIF parameters."""
        if self.gabor_size < 5 or self.gabor_size % 2 == 0:
            raise ValueError("gabor_size must be an odd integer >= 5")
        if not self.orientations or not self.spatial_frequencies:
            raise ValueError("at least one orientation and spatial frequency are required")
        if self.max_candidate_regions < 1 or self.recurrent_steps < 1:
            raise ValueError("candidate and recurrent-step limits must be positive")
        if self.lif_population_size < 2 or self.lif_steps_per_stage < 1:
            raise ValueError("LIF population size must be >= 2 and steps_per_stage positive")
        for name in (
            "gabor_sigma", "gabor_gamma", "population_temperature", "dt_ms",
            "membrane_tau_ms", "membrane_resistance_mohm", "current_gain_pa",
            "hrf_duration_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "visual_threshold_quantile", "prototype_learning_rate", "recurrent_damping",
            "inverse_regularization",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and lie in [0, 1]")
        for name in (
            "refractory_ms", "current_gain_pa", "recurrent_gain_pa", "inhibition_gain_pa",
            "eeg_noise_sd", "meg_noise_sd",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not self.rest_mv <= self.reset_mv < self.threshold_mv:
            raise ValueError("require rest_mv <= reset_mv < threshold_mv")
        object.__setattr__(self, "orientations", tuple(float(x) for x in self.orientations))
        object.__setattr__(self, "spatial_frequencies", tuple(float(x) for x in self.spatial_frequencies))

    def to_dict(self) -> dict[str, Any]:
        """Return all model settings as a plain dictionary for reproducibility metadata."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BNBEARequest:
    """One reading sample and any explicitly supplied prototypes or empirical validation evidence.

    Visual images are grayscale or RGB arrays. Audio is a one-dimensional waveform. Lesion, TMS,
    lead-field, fMRI and human validation data are supplied through ``experiment_data`` and are
    never fabricated when absent.
    """

    sample_id: str
    visual_image: Any | None = None
    auditory_waveform: Any | None = None
    audio_sample_rate_hz: float = 16000.0
    grapheme_prototypes: Sequence[GraphemePrototype] = ()
    orthographic_sequence: Sequence[str] = ()
    lexicon: Sequence[LexicalEntry] = ()
    semantic_features: Mapping[str, float] = field(default_factory=dict)
    context_features: Mapping[str, float] = field(default_factory=dict)
    experiment_data: Mapping[str, Any] = field(default_factory=dict)
    language: str = "und"

    def __post_init__(self) -> None:
        """Validate sensory arrays, inventories, feature strengths, and experiment metadata."""
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if self.visual_image is None and self.auditory_waveform is None:
            raise ValueError("at least one of visual_image or auditory_waveform must be supplied")
        if self.visual_image is not None:
            image = _finite_array("visual_image", self.visual_image)
            if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] not in (1, 3, 4)):
                raise ValueError("visual_image must be a 2-D grayscale or 3-D RGB/RGBA array")
            object.__setattr__(self, "visual_image", image)
        if self.auditory_waveform is not None:
            waveform = _finite_array("auditory_waveform", self.auditory_waveform, ndim=1)
            if not math.isfinite(float(self.audio_sample_rate_hz)) or self.audio_sample_rate_hz <= 0:
                raise ValueError("audio_sample_rate_hz must be finite and positive")
            object.__setattr__(self, "auditory_waveform", waveform)
        for name in ("semantic_features", "context_features"):
            mapping = getattr(self, name)
            if not isinstance(mapping, Mapping):
                raise TypeError(f"{name} must be a mapping")
            normalized = {}
            for key, raw in mapping.items():
                value = float(raw)
                if not isinstance(key, str) or not key.strip():
                    raise ValueError(f"{name} keys must be non-empty strings")
                if not math.isfinite(value):
                    raise ValueError(f"{name}[{key!r}] must be finite")
                normalized[key] = value
            object.__setattr__(self, name, normalized)
        object.__setattr__(self, "grapheme_prototypes", tuple(self.grapheme_prototypes))
        sequence = tuple(str(item) for item in self.orthographic_sequence)
        if any(not item for item in sequence):
            raise ValueError("orthographic_sequence items must be non-empty strings")
        object.__setattr__(self, "orthographic_sequence", sequence)
        object.__setattr__(self, "lexicon", tuple(self.lexicon))
        object.__setattr__(self, "experiment_data", dict(self.experiment_data))
        if not isinstance(self.language, str) or not self.language.strip():
            raise ValueError("language must be a non-empty identifier")


@dataclass(slots=True)
class BNBEAResult:
    """Complete BNBEA result with distinct functional, biological and experimental tracks."""

    sample_id: str
    functional: dict[str, Any]
    biological: dict[str, Any]
    experimental: dict[str, Any]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result without merging the three evidence tracks."""
        return _jsonable({
            "schema": "ore.brain_networks_biological_evidence.1.0",
            "sample_id": self.sample_id,
            "functional": self.functional,
            "biological": self.biological,
            "experimental": self.experimental,
            "operation_trace": self.operation_trace,
            "diagnostics": self.diagnostics,
            "metadata": self.metadata,
        })
