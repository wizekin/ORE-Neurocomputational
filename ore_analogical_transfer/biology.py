"""Distributed biological-function surrogates for analogical transfer.

The modules model candidate roles for relational retrieval, representation, integration, working memory,
and interference control. They are deliberately described as functional surrogates, not as anatomical
digital twins or claims that neural tissue executes these equations literally.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .contracts import AnalogicalConfig, MappingCandidate, SituationGraph
from .functional import (
    _cue_similarity, _goal_similarity_graph, _relational_relevance,
    canonical_relation_name, clamp, prioritize_working_memory,
)


class HippocampalRelationalRetrievalNetwork:
    """Store distinct relational episodes and retrieve source cases from partial structural cues."""

    def __init__(self) -> None:
        """Initialize the episodic graph store and retrieval counters."""
        self.episodes: dict[str, SituationGraph] = {}
        self.retrieval_count = 0
        self.register_count = 0

    def register(self, graph: SituationGraph) -> bool:
        """Store a graph once while rejecting conflicting reuse of its episode identity."""
        existing = self.episodes.get(graph.case_id)
        if existing is not None:
            if existing.to_dict() != graph.to_dict():
                raise ValueError("case_id already exists with different situation content")
            return False
        self.episodes[graph.case_id] = graph
        self.register_count += 1
        return True

    def retrieve(
        self, target: SituationGraph, config: AnalogicalConfig, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Reactivate stored episodes by relational and goal overlap while retaining source identity."""
        self.retrieval_count += 1
        rows: list[dict[str, Any]] = []
        for graph in self.episodes.values():
            if graph.case_id == target.case_id:
                continue
            cue = _cue_similarity(graph, target, config)
            goal = _goal_similarity_graph(graph, target)
            relation = _relational_relevance(graph, target, config)
            score = 0.50 * cue + 0.20 * goal + 0.30 * relation
            rows.append({
                "case_id": graph.case_id, "domain": graph.domain,
                "score": score, "source_reference": graph.source_reference,
                "retrieval_type": "specific_episode",
            })
        rows.sort(key=lambda item: (-item["score"], item["case_id"]))
        return rows[:max(0, limit)]

    def snapshot(self) -> dict[str, Any]:
        """Return episodic store counts without exposing stored situation contents."""
        return {
            "model": "hippocampal_relational_retrieval_surrogate",
            "episode_count": len(self.episodes), "register_count": self.register_count,
            "retrieval_count": self.retrieval_count,
            "episode_identity_preserved": True,
        }


class PosteriorParietalRelationalRepresentationNetwork:
    """Represent first-order relation signatures and argument-role bindings."""

    def __init__(self) -> None:
        """Initialize the relation representation index."""
        self.signature_counts: dict[str, int] = {}
        self.binding_counts: dict[str, int] = {}
        self.encoding_count = 0

    def encode(self, graph: SituationGraph, config: AnalogicalConfig) -> dict[str, Any]:
        """Index relation labels and their argument positions as a sparse functional representation."""
        encoded: list[dict[str, Any]] = []
        for relation in graph.relations:
            signature = (
                canonical_relation_name(relation.name, config) + "/"
                + str(len(relation.arguments)) + "/"
                + relation.direction + "/" + relation.causal_status
            )
            self.signature_counts[signature] = self.signature_counts.get(signature, 0) + 1
            encoded.append({
                "relation_id": relation.relation_id, "signature": signature,
                "argument_role_ids": [
                    graph.entities[entity_id].role_hints or (graph.entities[entity_id].kind,)
                    for entity_id in relation.arguments
                ],
                "confidence": relation.confidence,
            })
            for index, entity_id in enumerate(relation.arguments):
                key = signature + "|arg" + str(index) + "|" + entity_id
                self.binding_counts[key] = self.binding_counts.get(key, 0) + 1
        self.encoding_count += 1
        return {"relation_count": len(encoded), "relations": encoded}

    def snapshot(self) -> dict[str, Any]:
        """Expose representation-index size and encoding count."""
        return {
            "model": "posterior_parietal_first_order_relation_surrogate",
            "signature_count": len(self.signature_counts),
            "role_binding_count": len(self.binding_counts),
            "encoding_count": self.encoding_count,
        }


class RostrolateralPrefrontalIntegrationNetwork:
    """Integrate several first-order matches into a coherent higher-order relational mapping."""

    def __init__(self) -> None:
        """Initialize higher-order integration counters and recent structural state."""
        self.integration_count = 0
        self.last_integration: dict[str, Any] = {}

    def integrate(
        self, source: SituationGraph, target: SituationGraph,
        mappings: Sequence[MappingCandidate],
    ) -> dict[str, Any]:
        """Emphasize connected edge preservation and higher-order relation consistency."""
        self.integration_count += 1
        if not mappings:
            state = {"integration_score": 0.0, "preserved_relations": 0,
                     "higher_order_links": 0, "coherent": False}
        else:
            best = mappings[0]
            preserved = len(best.preserved_relation_ids)
            source_higher = len(source.higher_order_relations)
            target_higher = len(target.higher_order_relations)
            higher_score = min(source_higher, target_higher) / max(1, source_higher)
            score = clamp(0.55 * best.structural_consistency
                          + 0.25 * best.connectedness + 0.20 * higher_score)
            state = {
                "integration_score": score, "preserved_relations": preserved,
                "higher_order_links": source_higher, "coherent": score >= 0.50,
                "mapping_id": best.mapping_id,
            }
        self.last_integration = state
        return dict(state)

    def snapshot(self) -> dict[str, Any]:
        """Return higher-order integration state without implying a unique analogy center."""
        return {
            "model": "rostrolateral_prefrontal_relational_integration_surrogate",
            "integration_count": self.integration_count,
            "last_integration": dict(self.last_integration),
        }


class DorsolateralPrefrontalWorkingMemoryNetwork:
    """Maintain selected relations and mapping alternatives within a bounded working-memory budget."""

    def __init__(self) -> None:
        """Initialize working-memory update counters and current active set."""
        self.update_count = 0
        self.active_relations: tuple[str, ...] = ()
        self.budget_spent = 0.0

    def maintain(
        self, source: SituationGraph, target: SituationGraph,
        mapping: MappingCandidate, config: AnalogicalConfig,
    ) -> tuple[dict[str, Any], ...]:
        """Select goal-relevant connected relations while limiting concurrent maintenance."""
        self.update_count += 1
        selected = prioritize_working_memory(source, target, mapping, config)
        self.active_relations = tuple(
            item["relation_id"] for item in selected if item["active"]
        )
        self.budget_spent = (
            max((item["budget_spent"] for item in selected), default=0.0)
        )
        return selected

    def snapshot(self) -> dict[str, Any]:
        """Expose maintained relation identities and the configured capacity proxy."""
        return {
            "model": "dorsolateral_prefrontal_working_memory_surrogate",
            "update_count": self.update_count,
            "active_relation_ids": self.active_relations,
            "budget_spent": self.budget_spent,
        }


class VentrolateralPrefrontalInterferenceControlNetwork:
    """Down-weight salient surface matches when their relational positions do not correspond."""

    def __init__(self, suppression_gain: float = 0.65) -> None:
        """Initialize the gain used to suppress surface-only matching interference."""
        self.suppression_gain = float(suppression_gain)
        if not 0.0 <= self.suppression_gain <= 2.0:
            raise ValueError("suppression_gain must be in [0, 2]")
        self.control_count = 0
        self.last_suppression: dict[str, float] = {}

    def suppress(self, correspondences: Sequence[Any]) -> dict[str, float]:
        """Compute per-pair suppression for high surface similarity with low role similarity."""
        self.control_count += 1
        result: dict[str, float] = {}
        for item in correspondences:
            distraction = item.surface_similarity * (1.0 - item.role_similarity)
            result[item.source_entity_id + "->" + item.target_entity_id] = clamp(
                self.suppression_gain * distraction, 0.0, 1.0
            )
        self.last_suppression = result
        return dict(result)

    def snapshot(self) -> dict[str, Any]:
        """Return interference-control state and recent suppression values."""
        return {
            "model": "ventrolateral_prefrontal_interference_control_surrogate",
            "control_count": self.control_count,
            "last_suppression": dict(self.last_suppression),
        }


class TemporalSemanticRelationNetwork:
    """Index domain-independent relation labels across different source domains."""

    def __init__(self) -> None:
        """Initialize distributed relation associations."""
        self.domain_links: dict[tuple[str, str], float] = {}
        self.activation_count = 0

    def index(self, graph: SituationGraph, config: AnalogicalConfig) -> int:
        """Index canonical relation labels and their domains as distributed semantic associations."""
        count = 0
        for relation in graph.relations:
            key = (canonical_relation_name(relation.name, config), graph.domain)
            self.domain_links[key] = max(
                self.domain_links.get(key, 0.0), relation.confidence * relation.importance
            )
            count += 1
        return count

    def activate(self, target: SituationGraph, config: AnalogicalConfig) -> dict[str, float]:
        """Spread target relation cue activation across indexed domains with matching abstractions."""
        self.activation_count += 1
        target_names = {
            canonical_relation_name(relation.name, config)
            for relation in target.relations
        }
        activation: dict[str, float] = {}
        for (relation_name, domain), strength in self.domain_links.items():
            if relation_name in target_names:
                activation[domain] = activation.get(domain, 0.0) + strength
        total = sum(activation.values())
        if total > 0.0:
            activation = {key: value / total for key, value in activation.items()}
        return dict(sorted(activation.items(), key=lambda item: (-item[1], item[0])))

    def snapshot(self) -> dict[str, Any]:
        """Expose relation association volume and activation count."""
        return {
            "model": "distributed_temporal_semantic_relation_surrogate",
            "association_count": len(self.domain_links),
            "activation_count": self.activation_count,
        }


class BiologicalAnalogicalNetwork:
    """Coordinate episodic retrieval, relational encoding, integration, maintenance, and control."""

    def __init__(self) -> None:
        """Initialize the interacting functional network surrogates."""
        self.hippocampus = HippocampalRelationalRetrievalNetwork()
        self.parietal = PosteriorParietalRelationalRepresentationNetwork()
        self.rostrolateral = RostrolateralPrefrontalIntegrationNetwork()
        self.dorsolateral = DorsolateralPrefrontalWorkingMemoryNetwork()
        self.ventrolateral = VentrolateralPrefrontalInterferenceControlNetwork()
        self.temporal_semantic = TemporalSemanticRelationNetwork()

    def observe_graph(self, graph: SituationGraph, config: AnalogicalConfig) -> bool:
        """Register an episodic graph and expose its relations to cortical association surrogates."""
        is_new = self.hippocampus.register(graph)
        self.parietal.encode(graph, config)
        self.temporal_semantic.index(graph, config)
        return is_new

    def process_mapping(
        self, source: SituationGraph, target: SituationGraph,
        correspondences: Sequence[Any], mappings: Sequence[MappingCandidate],
        config: AnalogicalConfig,
    ) -> dict[str, Any]:
        """Produce network-level activation, relation integration, maintenance, and interference output."""
        relation_activation = self.temporal_semantic.activate(target, config)
        parietal = self.parietal.encode(target, config)
        integration = self.rostrolateral.integrate(source, target, mappings)
        working_memory = (
            self.dorsolateral.maintain(source, target, mappings[0], config)
            if mappings else ()
        )
        interference = self.ventrolateral.suppress(correspondences)
        return {
            "temporal_semantic_activation": relation_activation,
            "parietal_relation_encoding": parietal,
            "rostrolateral_integration": integration,
            "working_memory_selection": list(working_memory),
            "interference_suppression": interference,
            "hippocampal_retrieval": self.hippocampus.retrieve(
                target, config, config.retrieval_limit
            ),
            "network_state": self.snapshot(),
        }

    def snapshot(self) -> dict[str, Any]:
        """Return all functional network states and their scientific interpretation limits."""
        return {
            "hippocampus": self.hippocampus.snapshot(),
            "posterior_parietal": self.parietal.snapshot(),
            "rostrolateral_prefrontal": self.rostrolateral.snapshot(),
            "dorsolateral_prefrontal": self.dorsolateral.snapshot(),
            "ventrolateral_prefrontal": self.ventrolateral.snapshot(),
            "temporal_semantic": self.temporal_semantic.snapshot(),
            "scientific_scope": (
                "Distributed functional surrogates informed by analogical-reasoning studies; "
                "not exclusive brain modules, literal neural arithmetic, or anatomical replicas."
            ),
        }
