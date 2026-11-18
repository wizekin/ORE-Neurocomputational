"""Biologically inspired network surrogates for Point 09 schema formation.

The classes model broad functional interactions among episodic binding, distributed cortical
association, prefrontal context/goal control, and outcome-sensitive evaluation. They are not
anatomical digital twins and their values are not measured neural signals.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import EpisodeInput, SchemaConfig, SchemaFrame
from .functional import association_strength_delta


def _softmax(values: Sequence[float]) -> list[float]:
    """Normalize competitive activation scores with a numerically stable softmax."""
    if not values:
        return []
    maximum = max(values)
    exponentials = [math.exp(max(-60.0, min(60.0, value - maximum))) for value in values]
    total = sum(exponentials) or 1.0
    return [value / total for value in exponentials]


class HippocampalRelationalBinder:
    """Preserve separate episode traces and their event/entity/relation bindings."""

    def __init__(self) -> None:
        """Initialize a distinct episode index and binding counter."""
        self.episodes: dict[str, EpisodeInput] = {}
        self.binding_count = 0

    def bind(self, episode: EpisodeInput) -> dict[str, Any]:
        """Index one episode without merging it with similar experiences."""
        existing = self.episodes.get(episode.episode_id)
        if existing is not None and existing.to_dict() != episode.to_dict():
            raise ValueError("episode_id already exists with different episode content")
        is_new = existing is None
        if is_new:
            self.episodes[episode.episode_id] = episode
            self.binding_count += 1
        return {
            "episode_id": episode.episode_id, "new_trace": is_new,
            "event_count": len(episode.events), "entity_count": len(episode.entities),
            "relation_count": len(episode.relations),
            "episode_identity_preserved": True,
            "interpretation": "episodic relational binding surrogate; not a literal hippocampal graph",
        }

    def snapshot(self) -> dict[str, Any]:
        """Return episode-store size and binding activity without exposing raw content."""
        return {"model": "hippocampal_relational_binding_surrogate",
                "episode_count": len(self.episodes), "binding_count": self.binding_count}


class DistributedCorticalSchemaNetwork:
    """Learn gradual pairwise co-activation strengths across event, role, and relation labels."""

    def __init__(self, learning_rate: float = 0.08) -> None:
        """Initialize a sparse association map and bounded learning rate."""
        self.learning_rate = float(learning_rate)
        if not math.isfinite(self.learning_rate) or not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1]")
        self.associations: dict[tuple[str, str], float] = {}
        self.update_count = 0

    def learn_episode(self, episode: EpisodeInput) -> int:
        """Update co-active role/event/relation links using a bounded Hebbian-style rule."""
        updates = 0
        for event in episode.events:
            pattern = {"event:" + event.event_type}
            pattern.update("role:" + role.casefold() for role in event.participants)
            for role, entity_id in event.participants.items():
                entity_type = episode.entities[entity_id].entity_type
                pattern.add("type:" + role.casefold() + ":" + entity_type)
            ordered = sorted(pattern)
            for index, left in enumerate(ordered):
                for right in ordered[index + 1:]:
                    key = (left, right)
                    old = self.associations.get(key, 0.0)
                    self.associations[key] = association_strength_delta(
                        old, episode.reliability, self.learning_rate
                    )
                    updates += 1
        for relation in episode.relations:
            for left, right in (
                ("predicate:" + relation.predicate.casefold(), "evidence:" + relation.evidential_status),
                ("predicate:" + relation.predicate.casefold(), "entity_type:" + episode.entities[relation.subject_id].entity_type),
                ("predicate:" + relation.predicate.casefold(), "object_type:" + episode.entities[relation.object_id].entity_type),
            ):
                key = tuple(sorted((left, right)))
                self.associations[key] = association_strength_delta(
                    self.associations.get(key, 0.0), relation.confidence * episode.reliability,
                    self.learning_rate,
                )
                updates += 1
        self.update_count += updates
        return updates

    def spread_activation(self, active_cues: Mapping[str, float], limit: int = 8) -> dict[str, float]:
        """Propagate a bounded one-step activation from active cues across learned associations."""
        output: dict[str, float] = {}
        for (left, right), weight in self.associations.items():
            if left in active_cues:
                output[right] = max(output.get(right, 0.0), active_cues[left] * weight)
            if right in active_cues:
                output[left] = max(output.get(left, 0.0), active_cues[right] * weight)
        return dict(sorted(output.items(), key=lambda item: (-item[1], item[0]))[:max(0, limit)])

    def snapshot(self) -> dict[str, Any]:
        """Expose association counts and strongest links as inspectable cortical-model state."""
        strongest = sorted(
            ({"left": key[0], "right": key[1], "strength": value}
             for key, value in self.associations.items()),
            key=lambda item: (-item["strength"], item["left"], item["right"]),
        )[:20]
        return {"model": "distributed_cortical_hebbian_schema_surrogate",
                "association_count": len(self.associations), "update_count": self.update_count,
                "strongest_associations": strongest,
                "interpretation": "co-activation weights are model parameters, not synaptic measurements"}


class MedialPrefrontalSchemaNetwork:
    """Represent context- and goal-sensitive selection among candidate schema frames."""

    def __init__(self, temperature: float = 0.25) -> None:
        """Initialize a competitive selection temperature."""
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.activation_count = 0
        self.last_activations: list[dict[str, Any]] = []

    def activate(
        self,
        schemas: Sequence[SchemaFrame],
        context: Mapping[str, Any] | None = None,
        goals: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Rank schema candidates using current context and goal overlap."""
        self.activation_count += 1
        cues = {str(key) + "=" + str(value) for key, value in (context or {}).items()}
        goal_set = {str(goal).casefold() for goal in goals}
        raw_scores: list[float] = []
        for schema in schemas:
            profile = set(schema.context_profile)
            known_goals = set(schema.goals)
            context_score = len(cues & profile) / max(1, len(cues | profile)) if cues or profile else 0.0
            goal_score = len(goal_set & known_goals) / max(1, len(goal_set | known_goals)) if goal_set or known_goals else 0.0
            raw_scores.append(0.55 * context_score + 0.45 * goal_score)
        probabilities = _softmax([value / self.temperature for value in raw_scores])
        self.last_activations = [
            {"schema_id": schema.schema_id, "activation": score,
             "probability": probability}
            for schema, score, probability in zip(schemas, raw_scores, probabilities)
        ]
        return sorted(self.last_activations, key=lambda item: (-item["probability"], item["schema_id"]))

    def snapshot(self) -> dict[str, Any]:
        """Return recent candidate activation and its control-network interpretation."""
        return {"model": "medial_prefrontal_context_goal_selection_surrogate",
                "activation_count": self.activation_count,
                "last_activations": list(self.last_activations)}


class OrbitofrontalGoalOutcomeNetwork:
    """Track goal-conditioned outcome frequencies as a coarse expected-value surrogate."""

    def __init__(self) -> None:
        """Initialize goal-to-outcome evidence counters."""
        self.outcome_counts: dict[str, dict[str, float]] = {}
        self.observation_count = 0

    def learn_episode(self, episode: EpisodeInput) -> int:
        """Update observed goal/outcome associations without inventing unobserved outcomes."""
        updates = 0
        for goal in set(episode.goals):
            counts = self.outcome_counts.setdefault(goal, {})
            for outcome in set(episode.outcomes):
                counts[outcome] = counts.get(outcome, 0.0) + episode.reliability
                updates += 1
        self.observation_count += 1
        return updates

    def expected_outcomes(self, goal: str) -> list[dict[str, Any]]:
        """Return frequency-normalized outcome expectations for a supplied goal."""
        counts = self.outcome_counts.get(str(goal).casefold(), {})
        total = sum(counts.values())
        if total <= 0.0:
            return []
        return sorted(
            ({"outcome": key, "probability": value / total}
             for key, value in counts.items()),
            key=lambda item: (-item["probability"], item["outcome"]),
        )

    def snapshot(self) -> dict[str, Any]:
        """Return goal-outcome coverage without asserting a unique orbitofrontal code."""
        return {"model": "orbitofrontal_goal_outcome_surrogate",
                "goal_count": len(self.outcome_counts),
                "observation_count": self.observation_count,
                "expected_outcomes": {
                    goal: self.expected_outcomes(goal) for goal in sorted(self.outcome_counts)
                }}


class BiologicalSchemaNetwork:
    """Coordinate episodic binding, cortical regularity learning, and goal/context control models."""

    def __init__(self, config: SchemaConfig | None = None) -> None:
        """Initialize the four interacting Point 09 network surrogates."""
        self.config = config or SchemaConfig()
        self.hippocampus = HippocampalRelationalBinder()
        self.cortex = DistributedCorticalSchemaNetwork(self.config.learning_rate)
        self.medial_prefrontal = MedialPrefrontalSchemaNetwork()
        self.orbitofrontal = OrbitofrontalGoalOutcomeNetwork()

    def observe(self, episodes: Sequence[EpisodeInput]) -> dict[str, Any]:
        """Bind each distinct episode and update distributed association and goal evidence."""
        trace: list[dict[str, Any]] = []
        for episode in episodes:
            hippocampal = self.hippocampus.bind(episode)
            cortical_updates = self.cortex.learn_episode(episode)
            goal_updates = self.orbitofrontal.learn_episode(episode)
            trace.append({
                "episode_id": episode.episode_id, "hippocampal": hippocampal,
                "cortical_association_updates": cortical_updates,
                "goal_outcome_updates": goal_updates,
            })
        return {"episode_network_trace": trace, "network_state": self.snapshot()}

    def select_schema(
        self,
        schemas: Sequence[SchemaFrame],
        context: Mapping[str, Any] | None = None,
        goals: Sequence[str] = (),
    ) -> list[dict[str, Any]]:
        """Use goal and context overlap to rank schema activations."""
        return self.medial_prefrontal.activate(schemas, context, goals)

    def snapshot(self) -> dict[str, Any]:
        """Return the interacting network states and their modeling boundaries."""
        return {
            "hippocampus": self.hippocampus.snapshot(),
            "distributed_cortex": self.cortex.snapshot(),
            "medial_prefrontal": self.medial_prefrontal.snapshot(),
            "orbitofrontal": self.orbitofrontal.snapshot(),
            "scientific_scope": (
                "Functional network surrogates only; no class is a one-region/one-function "
                "digital replica or a direct measure of neural activity."
            ),
        }
