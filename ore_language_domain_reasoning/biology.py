"""Functional surrogates for interacting language, control, reasoning, and social-cognition systems.

These classes describe network-level roles, not an anatomical connectome or one-region/one-function map.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class NetworkActivity:
    """Track one distributed functional subsystem's participation across a task."""

    name: str
    biological_scope: str
    engagement_count: int = 0
    last_operation: str = ""
    last_load: float = 0.0
    activity_history: list[dict[str, Any]] = field(default_factory=list)

    def engage(self, operation: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Record a bounded functional engagement without claiming a direct neural readout."""
        data = dict(payload or {})
        self.engagement_count += 1
        self.last_operation = str(operation)
        load = data.get("load", data.get("demand_score", data.get("control_allocation", 0.0)))
        try:
            self.last_load = max(0.0, min(1.0, float(load)))
        except (TypeError, ValueError):
            self.last_load = 0.0
        row = {"operation": self.last_operation, "engagement_count": self.engagement_count,
               "load_proxy": self.last_load,
               "evidence_boundary": "functional_surrogate_not_neural_measurement"}
        self.activity_history.append(row)
        return row

    def snapshot(self) -> dict[str, Any]:
        """Return compact subsystem activity diagnostics."""
        return {"network": self.name, "biological_scope": self.biological_scope,
                "engagement_count": self.engagement_count, "last_operation": self.last_operation,
                "last_load_proxy": self.last_load,
                "activity_history": list(self.activity_history[-15:])}


class BiologicalLanguageReasoningNetwork:
    """Coordinate auditory/visual, language, semantic, control, and inference surrogates."""

    def __init__(self) -> None:
        """Initialize interacting functional subsystems used by Point 15."""
        self.perceptual = NetworkActivity("auditory_visual_perceptual",
            "Auditory pathways for speech and visual/orthographic pathways for text")
        self.language_selective = NetworkActivity("frontotemporal_language_selective",
            "Distributed frontal and temporal language-selective circuitry, commonly left-lateralized")
        self.conceptual_semantic = NetworkActivity("distributed_conceptual_semantic",
            "Distributed modality-linked and associative conceptual knowledge systems")
        self.multiple_demand_control = NetworkActivity("multiple_demand_control",
            "Bilateral frontal/parietal and distributed domain-general control systems")
        self.domain_general_reasoning = NetworkActivity("domain_general_reasoning",
            "Distributed systems supporting deliberate inferential operations")
        self.social_pragmatic = NetworkActivity("social_pragmatic_cognition",
            "Medial prefrontal, temporoparietal, posterior-midline, and related systems")
        self.integration_monitoring = NetworkActivity("cross_network_integration_monitoring",
            "Interacting language, semantic, control, and social inference systems")
        self.language_production = NetworkActivity("language_production_task_output",
            "Language production and task-related motor/control systems")
        self.hippocampal_cortical_learning = NetworkActivity("hippocampal_cortical_learning",
            "Medial-temporal episodic binding and experience-dependent cortical learning")
        self._systems = {item.name: item for item in (
            self.perceptual, self.language_selective, self.conceptual_semantic,
            self.multiple_demand_control, self.domain_general_reasoning, self.social_pragmatic,
            self.integration_monitoring, self.language_production,
            self.hippocampal_cortical_learning,
        )}

    def engage(self, system: str, operation: str,
               payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Engage a named subsystem and reject unknown routing targets."""
        if system not in self._systems:
            raise KeyError("unknown biological subsystem: " + system)
        return self._systems[system].engage(operation, payload)

    def snapshot(self) -> dict[str, Any]:
        """Return subsystem diagnostics and explicit biological-scope limitations."""
        return {**{name: network.snapshot() for name, network in self._systems.items()},
                "scientific_scope": (
                    "Distributed network-level approximations, not direct neural measurements, "
                    "exact anatomical localization, or literal biological equations."
                ),
                "functional_separation": (
                    "Language, conceptual retrieval, general control/reasoning, and social-pragmatic "
                    "inference remain separately traceable while exchanging information."
                )}
