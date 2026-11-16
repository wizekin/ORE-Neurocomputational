"""Validated request/result contracts for the six-stage sensory-perception model."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


def _finite_vector(name: str, value: Any) -> np.ndarray:
    """Convert a supplied value to a non-empty finite float vector; reject malformed samples."""
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 1 or result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    return result


def to_jsonable(value: Any, *, max_array_items: int = 4096) -> Any:
    """Convert contracts and NumPy values into bounded JSON-safe values for traces and reports.

    Small arrays become lists; large arrays become shape/statistics summaries so a page image or
    waveform is never copied wholesale into the SQLite execution-event metadata.
    """
    if is_dataclass(value):
        return to_jsonable(asdict(value), max_array_items=max_array_items)
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item, max_array_items=max_array_items) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item, max_array_items=max_array_items) for item in value]
    if isinstance(value, np.ndarray):
        if value.size <= max_array_items:
            return value.tolist()
        finite = value[np.isfinite(value)] if np.issubdtype(value.dtype, np.number) else np.array([])
        return {
            "array_summary": True,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite_count": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else None,
            "std": float(np.std(finite)) if finite.size else None,
            "min": float(np.min(finite)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
        }
    if isinstance(value, np.generic):
        return to_jsonable(value.item(), max_array_items=max_array_items)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


@dataclass(frozen=True, slots=True)
class PhonemeSpec:
    """Describe a language-specific phoneme prototype with broad articulatory feature labels.

    The fields are category metadata used by a transparent decoder, not a complete phonetic theory
    or a claim that a cortical population stores an explicit symbolic feature table.
    """

    symbol: str
    manner: str
    place: str = "unknown"
    voiced: bool = False
    vowel_height: str = ""
    vowel_backness: str = ""
    prior: float = 1.0

    def __post_init__(self) -> None:
        """Validate the phoneme label and positive prior used for category activation."""
        if not self.symbol.strip() or not self.manner.strip():
            raise ValueError("phoneme symbol and manner must be non-empty")
        if not math.isfinite(float(self.prior)) or self.prior <= 0.0:
            raise ValueError("phoneme prior must be finite and positive")


@dataclass(frozen=True, slots=True)
class SensoryConfig:
    """Numerical settings for visual filters, auditory analysis, decision thresholds and LIF dynamics.

    Defaults are reproducible engineering parameters, not participant-specific or universal human
    physiological constants. The LIF values are deliberately exposed for sensitivity experiments.
    """

    epsilon: float = 1e-8
    visual_orientations: tuple[float, ...] = (0.0, 0.7853981634, 1.5707963268, 2.3561944902)
    visual_frequencies: tuple[float, ...] = (0.08, 0.16)
    gabor_size: int = 11
    visual_tile_size: int = 24
    visual_text_threshold: float = 0.58
    max_text_regions: int = 64
    max_character_candidates: int = 256
    cochlear_channels: int = 24
    cochlear_min_hz: float = 80.0
    cochlear_max_hz: float = 8000.0
    auditory_frame_ms: float = 25.0
    auditory_hop_ms: float = 10.0
    lpc_order_per_khz: float = 2.0
    phoneme_temperature: float = 0.85
    selection_confidence: float = 0.42
    selection_margin: float = 0.06
    recurrent_iterations: int = 5
    recurrent_damping: float = 0.55
    lif_population_size: int = 16
    lif_steps_per_operation: int = 4
    dt_ms: float = 1.0
    membrane_tau_ms: float = 20.0
    membrane_resistance_mohm: float = 100.0
    rest_mv: float = -70.0
    reset_mv: float = -68.0
    threshold_mv: float = -50.0
    input_current_pa: float = 220.0
    recurrent_current_pa: float = 40.0
    random_seed: int = 19
    simulate_biology: bool = True

    def __post_init__(self) -> None:
        """Reject invalid filter sizes, rates, thresholds and unstable biological simulation settings."""
        if self.gabor_size < 5 or self.gabor_size % 2 == 0:
            raise ValueError("gabor_size must be an odd integer >= 5")
        if self.visual_tile_size < 4 or self.max_text_regions < 1 or self.max_character_candidates < 1:
            raise ValueError("visual tile and candidate limits must be positive")
        if self.cochlear_channels < 4 or self.cochlear_min_hz <= 0 or self.cochlear_max_hz <= self.cochlear_min_hz:
            raise ValueError("cochlear filterbank settings are invalid")
        if self.auditory_frame_ms <= 0 or self.auditory_hop_ms <= 0 or self.lpc_order_per_khz <= 0:
            raise ValueError("auditory analysis windows and LPC order must be positive")
        if not 0.0 < self.visual_text_threshold < 1.0 or not 0.0 < self.selection_confidence < 1.0:
            raise ValueError("probability thresholds must be between zero and one")
        if self.phoneme_temperature <= 0 or self.recurrent_iterations < 1:
            raise ValueError("temperature and recurrent iteration count must be positive")
        if not 0.0 < self.recurrent_damping <= 1.0:
            raise ValueError("recurrent_damping must be in (0, 1]")
        if self.lif_population_size < 2 or self.lif_steps_per_operation < 1:
            raise ValueError("LIF population size and operation steps are invalid")
        for name in ("epsilon", "dt_ms", "membrane_tau_ms", "membrane_resistance_mohm"):
            number = float(getattr(self, name))
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True, slots=True)
class SensoryRequest:
    """Supply optional visual and/or auditory input plus explicit language and context priors.

    Images are grayscale/RGB(A) arrays. Audio is a mono waveform in floating-point or integer
    sample units. Orthographic and phonetic priors are caller evidence; they are never fabricated.
    """

    sample_id: str
    visual_image: Any | None = None
    auditory_waveform: Any | None = None
    sample_rate_hz: float | None = None
    visual_attention: Mapping[str, float] | None = None
    auditory_attention: Mapping[str, float] | None = None
    orthographic_prior: Mapping[str, float] | None = None
    grapheme_prototypes: Mapping[str, Sequence[float]] | None = None
    phonetic_context: Mapping[str, float] | None = None
    phoneme_inventory: Sequence[PhonemeSpec | Mapping[str, Any]] = ()
    speaker_baseline: Mapping[str, Sequence[float] | float] | None = None
    language: str = "en-general"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate modality arrays, sample-rate metadata, priors and user-supplied phoneme records."""
        if not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if self.visual_image is None and self.auditory_waveform is None:
            raise ValueError("at least one of visual_image or auditory_waveform must be supplied")
        if self.visual_image is not None:
            image = np.asarray(self.visual_image, dtype=np.float64)
            if image.ndim not in (2, 3) or image.size == 0 or not np.all(np.isfinite(image)):
                raise ValueError("visual_image must be a finite non-empty 2-D or 3-D image array")
            if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
                raise ValueError("visual_image must contain 1, 3, or 4 channels")
        if self.auditory_waveform is not None:
            audio = _finite_vector("auditory_waveform", self.auditory_waveform)
            if audio.size < 32:
                raise ValueError("auditory_waveform must contain at least 32 samples")
            if self.sample_rate_hz is None or not math.isfinite(float(self.sample_rate_hz)) or self.sample_rate_hz <= 40:
                raise ValueError("sample_rate_hz must be finite and exceed 40 Hz when audio is supplied")
        for name in ("visual_attention", "auditory_attention", "orthographic_prior", "phonetic_context"):
            value = getattr(self, name)
            if value is not None and any(not math.isfinite(float(item)) for item in value.values()):
                raise ValueError(f"{name} values must be finite")
        inventory = tuple(
            item if isinstance(item, PhonemeSpec) else PhonemeSpec(**dict(item))
            for item in self.phoneme_inventory
        )
        if len({item.symbol for item in inventory}) != len(inventory):
            raise ValueError("phoneme_inventory symbols must be unique")
        object.__setattr__(self, "phoneme_inventory", inventory)
        object.__setattr__(self, "metadata", dict(self.metadata or {}))


@dataclass(frozen=True, slots=True)
class SensoryResult:
    """Return separate functional outputs, 39 operation traces and optional neural-reference states.

    Biological-reference variables are simulation outputs only. They are not EEG, MEG, fMRI,
    electrophysiology, or direct measurements from a human participant.
    """

    sample_id: str
    visual_output: Mapping[str, Any]
    auditory_output: Mapping[str, Any]
    phonetic_output: Mapping[str, Any]
    phoneme_output: Mapping[str, Any]
    operation_trace: Sequence[Mapping[str, Any]]
    biological_trace: Sequence[Mapping[str, Any]]
    integration: Mapping[str, Any]
    annotation_report: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result while summarizing large arrays to keep logs and reports bounded."""
        return to_jsonable(self)
