"""Typed contracts for distributed-only semantic representation and processing.

The model keeps population-specific property patterns separate.  It deliberately
does not define a central or amodal semantic embedding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

POPULATIONS = (
    "visual",
    "auditory",
    "action",
    "tactile",
    "affective",
    "linguistic",
    "contextual",
    "spatial",
)

POPULATION_ALIASES = {
    "vision": "visual",
    "sight": "visual",
    "sound": "auditory",
    "audio": "auditory",
    "phonological": "auditory",
    "motor": "action",
    "motor_action": "action",
    "somatosensory": "tactile",
    "touch": "tactile",
    "emotion": "affective",
    "emotional": "affective",
    "interoceptive": "affective",
    "language": "linguistic",
    "lexical": "linguistic",
    "context": "contextual",
    "episodic_context": "contextual",
    "spatial_motor": "spatial",
}


@dataclass(frozen=True)
class ConceptPattern:
    """Store a concept as separate property maps for each participating population.

    Each population map is an independent property representation.  The
    label/category fields are readout metadata, not a central neural code.
    """

    label: str
    properties: Mapping[str, Mapping[str, float]]
    category: str = "unspecified"
    description: str = ""
    typical_action: str = ""
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class EpisodicExperience:
    """Represent one sourced experience as population-specific observations."""

    concept_label: str
    properties: Mapping[str, Mapping[str, float]]
    context: Mapping[str, float] = field(default_factory=dict)
    familiarity: float = 1.0
    durability: float = 1.0
    provenance: str = "caller_supplied"


@dataclass(frozen=True)
class DistributedSemanticRequest:
    """Describe multimodal evidence, context, task, learning permission, and lesions."""

    modality_inputs: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    linguistic_input: str | None = None
    signal_inputs: Mapping[str, Sequence[float]] = field(default_factory=dict)
    context: Mapping[str, float] = field(default_factory=dict)
    task: str = "recognize"
    task_relevance: Mapping[str, float] = field(default_factory=dict)
    reliability: Mapping[str, float] = field(default_factory=dict)
    episodic_cues: Mapping[str, float] = field(default_factory=dict)
    max_iterations: int = 12
    convergence_tolerance: float = 1e-5
    learning: bool = False
    learning_target: str | None = None
    learning_rate: float = 0.02
    spike_times_ms: Mapping[str, float] = field(default_factory=dict)
    lesion_strengths: Mapping[str, float] = field(default_factory=dict)
    progressive_degradation: float = 0.0


@dataclass
class DistributedSemanticResult:
    """Expose separate population states, candidate evidence, traces, and diagnostics."""

    interpretation: str | None
    category: str | None
    response: str
    confidence: float
    candidate_probabilities: dict[str, float]
    candidate_scores: dict[str, float]
    category_scores: dict[str, float]
    population_activations: dict[str, list[float]]
    population_means: dict[str, float]
    population_reliability: dict[str, float]
    observed_indices: dict[str, list[int]]
    inferred_indices: dict[str, list[int]]
    assembly_membership: dict[str, bool]
    episodic_retrieval: list[dict[str, Any]]
    timing_coordination: dict[str, Any]
    redundancy_metrics: dict[str, Any]
    lesion_analysis: dict[str, Any]
    stage_trace: list[dict[str, Any]]
    operation_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    logging: dict[str, Any] = field(default_factory=dict)
