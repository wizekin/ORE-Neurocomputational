"""Distributed biological-function surrogates for context-dependent interpretation.

These classes model plausible functional contributions of interacting neural systems. They are
not anatomical digital twins, exclusive one-region/one-operation assignments, or claims that
neurons literally execute the equations used by this software.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import ContextEvidence, ContextState, InputEvent
from .functional import tokenize


class LanguageEncodingNetwork:
    """Model lexical and constructional encoding by a distributed language-responsive network."""

    def __init__(self) -> None:
        """Initialize the event-encoding counter and last encoded representation."""
        self.encoding_count = 0
        self.last_encoding: dict[str, Any] = {}

    def encode(self, event: InputEvent) -> dict[str, Any]:
        """Extract lexical cues, modality, explicit facts, and source reliability."""
        self.encoding_count += 1
        representation = {
            "event_id": event.event_id,
            "tokens": tokenize(event.content),
            "modality": event.modality,
            "explicit_fact_keys": tuple(sorted(str(key) for key in event.explicit_facts)),
            "feature_keys": tuple(sorted(str(key) for key in event.features)),
            "reliability": event.reliability,
        }
        self.last_encoding = representation
        return dict(representation)

    def snapshot(self) -> dict[str, Any]:
        """Return encoding activity and the most recent event representation."""
        return {"model": "distributed_language_encoding_surrogate",
                "encoding_count": self.encoding_count, "last_encoding": dict(self.last_encoding)}


class HippocampalDiscourseMemoryNetwork:
    """Model retrieval of recent discourse episodes and their temporal lineage."""

    def __init__(self) -> None:
        """Initialize discourse retrieval and reactivation counters."""
        self.retrieval_count = 0
        self.reactivated_ids: tuple[str, ...] = ()

    def retrieve(self, records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        """Record the identities of retrieved discourse episodes without changing their content."""
        self.retrieval_count += 1
        self.reactivated_ids = tuple(str(item.get("evidence_id", "")) for item in records)
        return self.reactivated_ids

    def snapshot(self) -> dict[str, Any]:
        """Expose discourse retrieval counts and retained episode identifiers."""
        return {"model": "hippocampal_discourse_retrieval_surrogate",
                "retrieval_count": self.retrieval_count, "reactivated_ids": self.reactivated_ids}


class TemporalSemanticMemoryNetwork:
    """Model distributed activation of conceptual knowledge and conventional associations."""

    def __init__(self) -> None:
        """Initialize semantic activation counters and source-activation state."""
        self.activation_count = 0
        self.last_activation: dict[str, float] = {}

    def activate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        """Pool retrieval scores by source without asserting that sources are independent."""
        self.activation_count += 1
        totals: dict[str, float] = {}
        for row in records:
            source = str(row.get("source_type", "unknown"))
            totals[source] = totals.get(source, 0.0) + float(row.get("retrieval_score", 0.0))
        total = sum(totals.values())
        if total > 0:
            totals = {key: value / total for key, value in totals.items()}
        self.last_activation = dict(sorted(totals.items()))
        return dict(self.last_activation)

    def snapshot(self) -> dict[str, Any]:
        """Return distributed semantic source activation diagnostics."""
        return {"model": "temporal_semantic_context_activation_surrogate",
                "activation_count": self.activation_count,
                "last_activation": dict(self.last_activation)}


class MedialPrefrontalGoalContextNetwork:
    """Model maintenance of task goals and goal-conditioned contextual selection."""

    def __init__(self) -> None:
        """Initialize goal-selection counters and the current active goal."""
        self.selection_count = 0
        self.active_goal = ""

    def select(self, goal: str) -> str:
        """Maintain an explicitly provided task goal for the current interpretation cycle."""
        self.selection_count += 1
        self.active_goal = str(goal)
        return self.active_goal

    def snapshot(self) -> dict[str, Any]:
        """Expose the current task goal as a functional-control state."""
        return {"model": "medial_prefrontal_goal_context_surrogate",
                "selection_count": self.selection_count, "active_goal": self.active_goal}


class PerceptualSceneNetwork:
    """Model the contribution of modality-specific scene and environmental cue extraction."""

    def __init__(self) -> None:
        """Initialize scene encoding and cue-count state."""
        self.encoding_count = 0
        self.last_cues: tuple[str, ...] = ()

    def encode(self, event: InputEvent, records: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
        """Combine explicitly supplied scene cues with retrieved scene-record identifiers."""
        self.encoding_count += 1
        cues = list(event.scene_cues)
        for row in records:
            evidence = row.get("evidence")
            if isinstance(evidence, ContextEvidence):
                cues.extend(evidence.concepts)
        self.last_cues = tuple(dict.fromkeys(str(item).lower() for item in cues))
        return self.last_cues

    def snapshot(self) -> dict[str, Any]:
        """Return scene-encoding counters and the most recent cue set."""
        return {"model": "perceptual_scene_context_surrogate",
                "encoding_count": self.encoding_count, "last_cues": self.last_cues}


class PosteriorMidlineSituationModelNetwork:
    """Model integration of discourse, conceptual, scene, and event information."""

    def __init__(self) -> None:
        """Initialize situation-integration counters and content summary."""
        self.integration_count = 0
        self.last_summary: dict[str, Any] = {}

    def integrate(self, state: ContextState) -> dict[str, Any]:
        """Summarize the current situation state while leaving source provenance intact."""
        self.integration_count += 1
        summary = {
            "entity_count": len(state.situation.entities),
            "fact_count": len(state.situation.facts),
            "event_count": len(state.situation.events),
            "open_ambiguity_count": len(state.situation.unresolved_items),
            "conflict_count": len(state.situation.conflicts),
        }
        self.last_summary = summary
        return dict(summary)

    def snapshot(self) -> dict[str, Any]:
        """Expose situation-model integration activity and the last structural summary."""
        return {"model": "posterior_midline_situation_model_surrogate",
                "integration_count": self.integration_count,
                "last_summary": dict(self.last_summary)}


class PrefrontalSelectionControlNetwork:
    """Model controlled competition, ambiguity monitoring, and provisional commitment."""

    def __init__(self) -> None:
        """Initialize selection and ambiguity monitoring counters."""
        self.selection_count = 0
        self.ambiguity_count = 0
        self.last_status = "not_run"

    def select(self, status: str, candidate_count: int) -> dict[str, Any]:
        """Record whether the current interpretation is selected, ambiguous, or unresolved."""
        self.selection_count += 1
        if status != "selected":
            self.ambiguity_count += 1
        self.last_status = status
        return {"status": status, "candidate_count": candidate_count,
                "commitment": status == "selected"}

    def snapshot(self) -> dict[str, Any]:
        """Return controlled-selection and ambiguity-monitoring diagnostics."""
        return {"model": "prefrontal_selection_control_surrogate",
                "selection_count": self.selection_count,
                "ambiguity_count": self.ambiguity_count, "last_status": self.last_status}


class BiologicalContextNetwork:
    """Coordinate interacting memory, semantic, goal, scene, situation, and control surrogates."""

    def __init__(self) -> None:
        """Initialize the distributed functional network used by Point 12."""
        self.language = LanguageEncodingNetwork()
        self.hippocampal_discourse = HippocampalDiscourseMemoryNetwork()
        self.temporal_semantic = TemporalSemanticMemoryNetwork()
        self.medial_prefrontal = MedialPrefrontalGoalContextNetwork()
        self.perceptual_scene = PerceptualSceneNetwork()
        self.posterior_midline = PosteriorMidlineSituationModelNetwork()
        self.prefrontal_control = PrefrontalSelectionControlNetwork()

    def snapshot(self) -> dict[str, Any]:
        """Return all subsystem states and the biological-model scope limitation."""
        return {
            "language": self.language.snapshot(),
            "hippocampal_discourse": self.hippocampal_discourse.snapshot(),
            "temporal_semantic": self.temporal_semantic.snapshot(),
            "medial_prefrontal": self.medial_prefrontal.snapshot(),
            "perceptual_scene": self.perceptual_scene.snapshot(),
            "posterior_midline": self.posterior_midline.snapshot(),
            "prefrontal_control": self.prefrontal_control.snapshot(),
            "scientific_scope": (
                "Distributed functional surrogates informed by context-sensitive comprehension; "
                "not anatomical replicas or literal neural computations."
            ),
        }
