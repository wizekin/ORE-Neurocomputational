"""Biologically inspired network surrogates for Point 10 schema-guided inference.

The models represent broad functional interactions among semantic activation, prefrontal control,
hippocampal episodic comparison, attention/conflict, and outcome-sensitive action selection.
They are not anatomical digital twins or direct measurements of neuronal activity.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .contracts import CaseEvidence, InferenceConfig, SchemaProfile
from .functional import entropy, select_action, softmax


class HippocampalEpisodicComparisonNetwork:
    """Retain individual case traces and retrieve similar cases without merging their identities."""

    def __init__(self) -> None:
        """Initialize a case index and retrieval counter."""
        self.cases: dict[str, CaseEvidence] = {}
        self.retrieval_count = 0

    def register(self, case: CaseEvidence) -> bool:
        """Store a new case once while rejecting conflicting reuse of a case identifier."""
        existing = self.cases.get(case.case_id)
        if existing is not None:
            if existing.to_dict() != case.to_dict():
                raise ValueError("case_id already exists with different evidence")
            return False
        self.cases[case.case_id] = case
        return True

    def _similarity(self, current: CaseEvidence, previous: CaseEvidence) -> float:
        """Compare observed feature values, directed relations, event fragments, and context."""
        current_features = {
            key: item for key, item in current.features.items()
            if item.status in {"observed", "contradicted"}
        }
        previous_features = {
            key: item for key, item in previous.features.items()
            if item.status in {"observed", "contradicted"}
        }
        shared = set(current_features) & set(previous_features)
        if shared:
            feature_match = sum(
                float(str(current_features[key].value).casefold()
                      == str(previous_features[key].value).casefold())
                for key in shared
            ) / len(shared)
        else:
            feature_match = 0.0
        current_relations = {
            (item.subject, item.predicate, item.object)
            for item in current.relations if item.status == "observed"
        }
        previous_relations = {
            (item.subject, item.predicate, item.object)
            for item in previous.relations if item.status == "observed"
        }
        union_relations = current_relations | previous_relations
        relation_match = (
            len(current_relations & previous_relations) / len(union_relations)
            if union_relations else 0.0
        )
        current_events, previous_events = set(current.event_history), set(previous.event_history)
        union_events = current_events | previous_events
        event_match = len(current_events & previous_events) / len(union_events) if union_events else 0.0
        current_context = {str(k) + "=" + str(v) for k, v in current.context.items()}
        previous_context = {str(k) + "=" + str(v) for k, v in previous.context.items()}
        union_context = current_context | previous_context
        context_match = len(current_context & previous_context) / len(union_context) if union_context else 0.0
        return 0.45 * feature_match + 0.25 * relation_match + 0.15 * event_match + 0.15 * context_match

    def retrieve(self, current: CaseEvidence, limit: int = 5) -> list[dict[str, Any]]:
        """Reactivate similar case memories and preserve each retrieved episode's source identity."""
        self.retrieval_count += 1
        ranked = [
            {"case_id": case_id, "similarity": self._similarity(current, item),
             "source_reference": item.source_reference}
            for case_id, item in self.cases.items() if case_id != current.case_id
        ]
        return sorted(ranked, key=lambda item: (-item["similarity"], item["case_id"]))[:max(0, limit)]

    def snapshot(self) -> dict[str, Any]:
        """Return memory and retrieval counts without exporting stored sensory details."""
        return {
            "model": "hippocampal_episodic_comparison_surrogate",
            "case_count": len(self.cases), "retrieval_count": self.retrieval_count,
            "episode_identity_preserved": True,
        }


class TemporalSemanticActivationNetwork:
    """Spread activation from observed semantic cues to schema candidates through learned associations."""

    def __init__(self, association_gain: float = 1.0) -> None:
        """Initialize a sparse schema-feature association map."""
        self.association_gain = float(association_gain)
        if not math.isfinite(self.association_gain) or self.association_gain < 0.0:
            raise ValueError("association_gain must be finite and non-negative")
        self.associations: dict[tuple[str, str], float] = {}
        self.activation_count = 0

    def index_profiles(self, profiles: Sequence[SchemaProfile]) -> int:
        """Index schema-conditioned feature likelihoods as reusable semantic associations."""
        added = 0
        for profile in profiles:
            for feature, distribution in profile.feature_probabilities.items():
                for value, probability in distribution.items():
                    key = (feature + "=" + value, profile.schema_id)
                    old = self.associations.get(key, 0.0)
                    self.associations[key] = max(old, float(probability) * self.association_gain)
                    added += 1
            for slot_name, slot in profile.frame.slots.items():
                for entity_type, probability in slot.allowed_type_probabilities.items():
                    key = ("role:" + slot_name + "=" + entity_type.casefold(), profile.schema_id)
                    self.associations[key] = max(
                        self.associations.get(key, 0.0), float(probability) * self.association_gain
                    )
                    added += 1
        return added

    def activate(self, case: CaseEvidence, limit: int = 8) -> dict[str, float]:
        """Accumulate schema activation from reliable observed cues using associative weights."""
        self.activation_count += 1
        activation: dict[str, float] = {}
        for item in case.usable_evidence:
            cue = item.feature + "=" + str(item.value).casefold()
            for (indexed_cue, schema_id), weight in self.associations.items():
                if cue == indexed_cue:
                    activation[schema_id] = activation.get(schema_id, 0.0) + item.reliability * weight
        total = sum(activation.values())
        normalized = {key: value / total for key, value in activation.items()} if total > 0 else {}
        return dict(sorted(normalized.items(), key=lambda item: (-item[1], item[0]))[:max(0, limit)])

    def snapshot(self) -> dict[str, Any]:
        """Expose association volume and strongest links without implying literal semantic units."""
        strongest = sorted(
            ({"cue": cue, "schema_id": schema, "strength": weight}
             for (cue, schema), weight in self.associations.items()),
            key=lambda row: (-row["strength"], row["cue"], row["schema_id"]),
        )[:20]
        return {
            "model": "distributed_temporal_semantic_activation_surrogate",
            "association_count": len(self.associations), "activation_count": self.activation_count,
            "strongest_associations": strongest,
        }


class MedialPrefrontalPriorControlNetwork:
    """Model goal- and context-dependent gain over candidate schema activations."""

    def __init__(self, temperature: float = 0.35) -> None:
        """Initialize the competitive selection temperature."""
        self.temperature = float(temperature)
        if not math.isfinite(self.temperature) or self.temperature <= 0.0:
            raise ValueError("temperature must be finite and positive")
        self.selection_count = 0
        self.last_gain: dict[str, float] = {}

    def modulate(self, case: CaseEvidence, profiles: Sequence[SchemaProfile]) -> dict[str, float]:
        """Compute softmax control gains from context and requested-goal compatibility."""
        self.selection_count += 1
        scores: list[float] = []
        schema_ids: list[str] = []
        for profile in profiles:
            schema_ids.append(profile.schema_id)
            cues = {str(key) + "=" + str(value) for key, value in case.context.items()}
            known = set(profile.frame.context_profile)
            context_score = len(cues & known) / max(1, len(cues | known)) if cues or known else 0.0
            goal_score = (
                sum(float(profile.frame.goals.get(goal, 0.0)) for goal in case.goals)
                / max(1, len(case.goals))
                if case.goals else 0.0
            )
            scores.append(0.55 * context_score + 0.45 * goal_score)
        probabilities = softmax(scores, self.temperature) if scores else []
        self.last_gain = dict(zip(schema_ids, probabilities))
        return dict(self.last_gain)

    def snapshot(self) -> dict[str, Any]:
        """Return recent context/goal gain and the control-model interpretation."""
        return {
            "model": "medial_prefrontal_context_goal_control_surrogate",
            "selection_count": self.selection_count, "last_gain": dict(self.last_gain),
        }


class AttentionConflictNetwork:
    """Estimate ambiguity and prioritize features that discriminate among active schemas."""

    def __init__(self) -> None:
        """Initialize an inspectable conflict and focus-selection state."""
        self.selection_count = 0
        self.last_conflict: dict[str, Any] = {}

    def assess(
        self, posterior: Sequence[float], ranked: Sequence[Mapping[str, Any]]
    ) -> dict[str, Any]:
        """Calculate candidate entropy, top-two margin, and evidence-conflict load."""
        self.selection_count += 1
        probabilities = [float(value) for value in posterior]
        normalized = entropy(probabilities)
        ordered = sorted(probabilities, reverse=True)
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else (ordered[0] if ordered else 0.0)
        conflict_count = sum(len(item.get("conflicting_evidence_ids", ())) for item in ranked)
        conflict = {
            "posterior_entropy": normalized, "top_two_margin": margin,
            "conflicting_evidence_count": conflict_count,
            "ambiguity": normalized > 0.5 or margin < 0.12,
        }
        self.last_conflict = conflict
        return dict(conflict)

    def choose_diagnostic_feature(
        self, profiles: Sequence[SchemaProfile], already_known: set[str]
    ) -> str | None:
        """Select an unknown feature with the greatest disagreement among schema distributions."""
        candidates = sorted({
            feature for profile in profiles for feature in profile.feature_probabilities
            if feature not in already_known
        })
        best_feature: str | None = None
        best_score = -1.0
        for feature in candidates:
            values = {
                value for profile in profiles
                for value in profile.feature_probabilities.get(feature, {})
            }
            if len(values) < 2:
                continue
            rows = [
                profile.feature_probabilities[feature]
                for profile in profiles if feature in profile.feature_probabilities
            ]
            disagreement = sum(
                sum((row.get(value, 0.0) - other.get(value, 0.0)) ** 2 for value in values)
                for index, row in enumerate(rows) for other in rows[index + 1:]
            )
            if disagreement > best_score:
                best_score, best_feature = disagreement, feature
        return best_feature

    def snapshot(self) -> dict[str, Any]:
        """Return conflict-monitor state without asserting a unique neural conflict signal."""
        return {
            "model": "attention_conflict_and_diagnostic_selection_surrogate",
            "selection_count": self.selection_count, "last_conflict": dict(self.last_conflict),
        }


class OrbitofrontalActionSelectionNetwork:
    """Model goal- and risk-sensitive action-value selection from candidate outcomes."""

    def __init__(self, risk_gain: float = 2.0) -> None:
        """Initialize the action-risk penalty gain."""
        self.risk_gain = float(risk_gain)
        if not math.isfinite(self.risk_gain) or self.risk_gain < 0.0:
            raise ValueError("risk_gain must be finite and non-negative")
        self.selection_count = 0
        self.last_action: dict[str, Any] | None = None

    def select(
        self, case: CaseEvidence, ranked: Sequence[Any], config: InferenceConfig
    ) -> dict[str, Any] | None:
        """Select a caller-supplied action using posterior expected utility and action risk."""
        self.selection_count += 1
        decision = select_action(case, ranked, config)
        if decision is None:
            self.last_action = None
            return None
        adjusted = decision.expected_utility + (
            config.action_risk_weight - self.risk_gain
        ) * decision.expected_risk
        self.last_action = {
            "action_id": decision.action_id, "expected_utility": adjusted,
            "expected_risk": decision.expected_risk, "selected": True,
        }
        return dict(self.last_action)

    def snapshot(self) -> dict[str, Any]:
        """Return action-selection state and its computational scope."""
        return {
            "model": "orbitofrontal_goal_outcome_action_surrogate",
            "selection_count": self.selection_count, "last_action": self.last_action,
        }


class BiologicalSchemaInferenceNetwork:
    """Coordinate semantic activation, prefrontal control, episodic comparison, and action models."""

    def __init__(self, config: InferenceConfig | None = None) -> None:
        """Initialize the interacting Point 10 network surrogates."""
        self.config = config or InferenceConfig()
        self.semantic = TemporalSemanticActivationNetwork()
        self.prefrontal = MedialPrefrontalPriorControlNetwork()
        self.hippocampus = HippocampalEpisodicComparisonNetwork()
        self.attention = AttentionConflictNetwork()
        self.orbitofrontal = OrbitofrontalActionSelectionNetwork(self.config.action_risk_weight)

    def prepare(self, profiles: Sequence[SchemaProfile]) -> dict[str, Any]:
        """Index candidate schema cues into the distributed semantic association surrogate."""
        return {"indexed_associations": self.semantic.index_profiles(profiles),
                "network_state": self.snapshot()}

    def process(
        self, case: CaseEvidence, ranked: Sequence[Any], profiles: Sequence[SchemaProfile]
    ) -> dict[str, Any]:
        """Produce network activation, context gain, conflict assessment, episodic recall, and action."""
        semantic_activation = self.semantic.activate(case)
        control_gain = self.prefrontal.modulate(case, profiles)
        posterior = [float(item.posterior) for item in ranked]
        conflict = self.attention.assess(posterior, [
            {"conflicting_evidence_ids": item.conflicting_evidence_ids} for item in ranked
        ])
        recall = self.hippocampus.retrieve(case)
        action = self.orbitofrontal.select(case, ranked, self.config)
        return {
            "semantic_activation": semantic_activation, "prefrontal_gain": control_gain,
            "conflict": conflict, "episodic_recall": recall, "action_selection": action,
            "network_state": self.snapshot(),
        }

    def observe_case(self, case: CaseEvidence) -> bool:
        """Register a case trace once for later episodic comparisons."""
        return self.hippocampus.register(case)

    def snapshot(self) -> dict[str, Any]:
        """Return all network states and a clear boundary on biological interpretation."""
        return {
            "temporal_semantic": self.semantic.snapshot(),
            "medial_prefrontal": self.prefrontal.snapshot(),
            "hippocampus": self.hippocampus.snapshot(),
            "attention_conflict": self.attention.snapshot(),
            "orbitofrontal": self.orbitofrontal.snapshot(),
            "scientific_scope": (
                "Functional network surrogates only; no class is a one-region/one-function "
                "digital replica or a direct measure of neural activity."
            ),
        }
