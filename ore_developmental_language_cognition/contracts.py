"""Contracts and persistent learner state for developmental language/cognition (Point 16).

All developmental values are model estimates from supplied experience. They are not diagnostic
scores, normative age tests, or direct neural measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence


def _finite(name: str, value: float) -> float:
    """Convert a numeric input to a finite float or reject invalid values."""
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(name + " must be finite")
    return result


def _unit(name: str, value: float) -> float:
    """Validate a value in the closed interval [0, 1]."""
    result = _finite(name, value)
    if not 0.0 <= result <= 1.0:
        raise ValueError(name + " must be in [0, 1]")
    return result


def _jsonable(value: Any) -> Any:
    """Recursively convert dataclasses and common containers to JSON-compatible values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted((_jsonable(item) for item in value), key=repr)
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    return value


@dataclass(frozen=True, slots=True)
class DevelopmentalExperience:
    """Describe one meaningful auditory, social, conceptual, memory, or control experience.

    Acoustic frames are precomputed feature mappings or numeric vectors. The optional raw audio
    input is transformed with simple frame-level measurements; this package is not a clinical
    hearing test or a full speech-recognition system.
    """

    experience_id: str
    age_months: float | None = None
    acoustic_frames: Sequence[Mapping[str, Any] | Sequence[float]] = ()
    audio_samples: Sequence[float] = ()
    sample_rate: int = 16000
    syllables: Sequence[str] = ()
    boundary_cues: Sequence[Mapping[str, Any]] = ()
    phonetic_label: str = ""
    native_language: str = "environment"
    phonetic_feedback: Mapping[str, Any] = field(default_factory=dict)
    scene_referents: Sequence[Mapping[str, Any]] = ()
    gaze_target: str = ""
    gesture_target: str = ""
    speaker_intention: str = ""
    social_contingency: float = 0.0
    target_word: str = ""
    syntactic_hints: Mapping[str, Any] = field(default_factory=dict)
    context_tokens: Sequence[str] = ()
    concept_examples: Sequence[Mapping[str, Any]] = ()
    memory_cues: Sequence[str] = ()
    working_memory_items: Sequence[str] = ()
    distractors: Sequence[str] = ()
    current_goal: str = ""
    action_candidates: Sequence[Mapping[str, Any]] = ()
    current_rule: str = ""
    previous_rule: str = ""
    rule_feedback: Mapping[str, Any] = field(default_factory=dict)
    circuit_activations: Mapping[str, float] = field(default_factory=dict)
    active_pathways: Mapping[str, float] = field(default_factory=dict)
    path_lengths: Mapping[str, float] = field(default_factory=dict)
    plan_steps: Sequence[Mapping[str, Any]] = ()
    plan_observations: Mapping[str, Any] = field(default_factory=dict)
    performance_feedback: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate age, audio metadata, normalized evidence, and detached event collections."""
        if not str(self.experience_id).strip():
            raise ValueError("experience_id must be non-empty")
        if self.age_months is not None:
            age = _finite("age_months", self.age_months)
            if not 0.0 <= age <= 240.0:
                raise ValueError("age_months must be between 0 and 240 for this developmental model")
            object.__setattr__(self, "age_months", age)
        if int(self.sample_rate) <= 0:
            raise ValueError("sample_rate must be positive")
        object.__setattr__(self, "sample_rate", int(self.sample_rate))
        object.__setattr__(self, "audio_samples", tuple(_finite("audio_sample", x) for x in self.audio_samples))
        object.__setattr__(self, "acoustic_frames", tuple(
            dict(x) if isinstance(x, Mapping) else tuple(_finite("acoustic_feature", v) for v in x)
            for x in self.acoustic_frames
        ))
        object.__setattr__(self, "syllables", tuple(str(x).strip().lower() for x in self.syllables if str(x).strip()))
        object.__setattr__(self, "boundary_cues", tuple(dict(x) for x in self.boundary_cues))
        object.__setattr__(self, "scene_referents", tuple(dict(x) for x in self.scene_referents))
        object.__setattr__(self, "concept_examples", tuple(dict(x) for x in self.concept_examples))
        object.__setattr__(self, "action_candidates", tuple(dict(x) for x in self.action_candidates))
        object.__setattr__(self, "plan_steps", tuple(dict(x) for x in self.plan_steps))
        for name in (
            "phonetic_feedback", "syntactic_hints", "rule_feedback", "circuit_activations",
            "active_pathways", "path_lengths", "plan_observations", "performance_feedback",
            "metadata",
        ):
            object.__setattr__(self, name, dict(getattr(self, name)))
        for name in ("context_tokens", "memory_cues", "working_memory_items", "distractors"):
            object.__setattr__(self, name, tuple(str(x) for x in getattr(self, name)))
        object.__setattr__(self, "social_contingency", _unit("social_contingency", self.social_contingency))
        object.__setattr__(self, "gaze_target", str(self.gaze_target))
        object.__setattr__(self, "gesture_target", str(self.gesture_target))
        object.__setattr__(self, "speaker_intention", str(self.speaker_intention))
        object.__setattr__(self, "target_word", str(self.target_word).strip().lower())
        object.__setattr__(self, "current_goal", str(self.current_goal).strip())
        object.__setattr__(self, "current_rule", str(self.current_rule).strip())
        object.__setattr__(self, "previous_rule", str(self.previous_rule).strip())


@dataclass(frozen=True, slots=True)
class DevelopmentalConfig:
    """Configure learning rates and computational approximations without fixed age deadlines."""

    prototype_rate: float = 0.16
    category_creation_distance: float = 0.32
    statistical_smoothing: float = 0.5
    segmentation_threshold: float = 0.52
    multimodal_temperature: float = 0.45
    lexical_learning_rate: float = 0.14
    lexical_establishment_exposures: int = 3
    concept_learning_rate: float = 0.12
    memory_learning_rate: float = 0.10
    memory_retention: float = 0.88
    memory_capacity: float = 4.0
    working_memory_decay: float = 0.18
    executive_temperature: float = 0.55
    executive_learning_rate: float = 0.12
    circuit_learning_rate: float = 0.025
    circuit_decay: float = 0.004
    pruning_threshold: float = 0.035
    pruning_patience: int = 5
    myelination_rate: float = 0.015
    calibration_rate: float = 0.08
    target_success: float = 0.78
    maximum_episodes: int = 200

    def __post_init__(self) -> None:
        """Validate normalized learning rates and positive scale parameters."""
        for name in (
            "prototype_rate", "memory_retention", "working_memory_decay",
            "lexical_learning_rate", "concept_learning_rate", "memory_learning_rate",
            "circuit_learning_rate", "circuit_decay", "pruning_threshold",
            "myelination_rate", "calibration_rate", "target_success",
        ):
            object.__setattr__(self, name, _unit(name, getattr(self, name)))
        for name in (
            "category_creation_distance", "statistical_smoothing", "segmentation_threshold",
            "multimodal_temperature", "memory_capacity", "executive_temperature",
            "pruning_patience",
        ):
            value = _finite(name, getattr(self, name))
            if value <= 0:
                raise ValueError(name + " must be positive")
            object.__setattr__(self, name, value)
        if self.lexical_establishment_exposures < 2 or self.maximum_episodes < 1:
            raise ValueError("exposure and episode limits are too small")
        if int(self.pruning_patience) != self.pruning_patience:
            raise ValueError("pruning_patience must be an integer")
        object.__setattr__(self, "pruning_patience", int(self.pruning_patience))


@dataclass
class DevelopmentalLearnerState:
    """Store separate developing representations, memories, controls, and circuit parameters."""

    learner_id: str = "default"
    phonetic_prototypes: dict[str, list[float]] = field(default_factory=dict)
    phonetic_counts: dict[str, int] = field(default_factory=dict)
    broad_sensitivity: float = 0.75
    language_exposure_counts: dict[str, int] = field(default_factory=dict)
    native_contrast_strengths: dict[str, float] = field(default_factory=dict)
    unfamiliar_contrast_strengths: dict[str, float] = field(default_factory=dict)
    syllable_counts: dict[str, int] = field(default_factory=dict)
    transition_counts: dict[str, int] = field(default_factory=dict)
    word_boundary_counts: dict[str, int] = field(default_factory=dict)
    lexicon: dict[str, dict[str, Any]] = field(default_factory=dict)
    concept_store: dict[str, dict[str, Any]] = field(default_factory=dict)
    concept_relations: list[dict[str, Any]] = field(default_factory=list)
    episodic_memory: list[dict[str, Any]] = field(default_factory=list)
    semantic_associations: dict[str, float] = field(default_factory=dict)
    working_memory_items: list[dict[str, Any]] = field(default_factory=list)
    working_memory_capacity: float = 4.0
    executive_weights: dict[str, float] = field(default_factory=lambda: {
        "goal": 0.55, "relevance": 0.35, "distraction": 0.25, "prepotency": 0.20,
        "switching": 0.25,
    })
    current_rule: str = ""
    executive_error_history: list[float] = field(default_factory=list)
    circuit_weights: dict[str, float] = field(default_factory=dict)
    circuit_low_use_counts: dict[str, int] = field(default_factory=dict)
    myelination: dict[str, float] = field(default_factory=dict)
    hippocampal_prefrontal_coupling: float = 0.10
    developmental_parameters: dict[str, float] = field(default_factory=dict)
    task_difficulty: dict[str, float] = field(default_factory=dict)
    performance_history: dict[str, list[float]] = field(default_factory=dict)
    multi_step_state: dict[str, Any] = field(default_factory=dict)
    experience_count: int = 0

    def __post_init__(self) -> None:
        """Validate persistent values and normalize existing learner state."""
        if not str(self.learner_id).strip():
            raise ValueError("learner_id must be non-empty")
        self.broad_sensitivity = _unit("broad_sensitivity", self.broad_sensitivity)
        self.working_memory_capacity = max(0.5, _finite("working_memory_capacity", self.working_memory_capacity))
        self.hippocampal_prefrontal_coupling = _unit(
            "hippocampal_prefrontal_coupling", self.hippocampal_prefrontal_coupling
        )
        self.phonetic_prototypes = {
            str(key): [_finite("prototype_feature", value) for value in vector]
            for key, vector in self.phonetic_prototypes.items()
        }
        for name in ("native_contrast_strengths", "unfamiliar_contrast_strengths",
                     "semantic_associations", "myelination", "developmental_parameters",
                     "task_difficulty"):
            setattr(self, name, {str(key): _unit(name, value) for key, value in getattr(self, name).items()})
        self.phonetic_counts = {str(key): max(0, int(value)) for key, value in self.phonetic_counts.items()}
        self.syllable_counts = {str(key): max(0, int(value)) for key, value in self.syllable_counts.items()}
        self.transition_counts = {str(key): max(0, int(value)) for key, value in self.transition_counts.items()}
        self.word_boundary_counts = {str(key): max(0, int(value)) for key, value in self.word_boundary_counts.items()}
        self.language_exposure_counts = {str(key): max(0, int(value)) for key, value in self.language_exposure_counts.items()}
        self.working_memory_items = list(self.working_memory_items)
        self.episodic_memory = list(self.episodic_memory)
        self.concept_relations = list(self.concept_relations)
        self.executive_error_history = list(self.executive_error_history)
        self.performance_history = {str(k): list(v) for k, v in self.performance_history.items()}
        self.experience_count = max(0, int(self.experience_count))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible representation of the complete learner state."""
        return _jsonable(asdict(self))


@dataclass(frozen=True, slots=True)
class DevelopmentalProcessingResult:
    """Return stage outputs, operation trace, biological diagnostics, and updated learner state."""

    experience_id: str
    broad_perception: Mapping[str, Any]
    acoustic_features: Mapping[str, Any]
    phonetic_categories: Mapping[str, Any]
    statistical_learning: Mapping[str, Any]
    word_segmentation: Mapping[str, Any]
    perceptual_tuning: Mapping[str, Any]
    multimodal_grounding: Mapping[str, Any]
    word_meaning_mapping: Mapping[str, Any]
    vocabulary_growth: Mapping[str, Any]
    concept_formation: Mapping[str, Any]
    memory_stabilization: Mapping[str, Any]
    working_memory: Mapping[str, Any]
    executive_control: Mapping[str, Any]
    circuit_refinement: Mapping[str, Any]
    memory_control_integration: Mapping[str, Any]
    multi_step_reasoning: Mapping[str, Any]
    developmental_calibration: Mapping[str, Any]
    operation_trace: tuple[Mapping[str, Any], ...]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return the complete Point 16 result as a JSON-compatible structure."""
        return _jsonable(asdict(self))
