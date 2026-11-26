"""Read-only bridges from unified USG resources to existing ORE processors."""
from __future__ import annotations

import json
from urllib.parse import quote
from collections import defaultdict
from copy import deepcopy
import re
import unicodedata
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ore_lexical_access import LexicalAccessConfig, LexicalAccessProcessor
    from ore_semantic_composition import (
        PredicateFrame,
        SemanticCompositionConfig,
        SemanticCompositionProcessor,
    )

from .knowledge import USGKnowledgeResource
from .structures import USGPredicateFrame


class USGPredicateFrameConversionError(ValueError):
    """Raised when a preserved USG frame cannot be safely represented by ORE."""


def to_ore_predicate_frame(frame: USGPredicateFrame) -> "PredicateFrame":
    """Convert only explicitly declared USG role data into an ORE predicate frame."""
    from ore_semantic_composition import PredicateFrame

    if not isinstance(frame, USGPredicateFrame):
        raise TypeError("frame must be a USGPredicateFrame")
    try:
        payload = json.loads(frame.payload_json)
        roles_payload = json.loads(frame.roles_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise USGPredicateFrameConversionError(
            f"USG predicate frame {frame.usg_record_id!r} has invalid preserved JSON"
        ) from exc
    if not isinstance(roles_payload, list) or not roles_payload:
        raise USGPredicateFrameConversionError(
            f"USG predicate frame {frame.usg_record_id!r} must declare a non-empty roles array"
        )
    roles: list[str] = []
    required_roles: list[str] = []
    for index, item in enumerate(roles_payload):
        if isinstance(item, str):
            role_name = item
            required = False
        elif isinstance(item, dict):
            role_name = item.get("role", item.get("name"))
            required = item.get("required", False)
            if not isinstance(required, bool):
                raise USGPredicateFrameConversionError(
                    f"role {index} in {frame.usg_record_id!r} has a non-boolean required flag"
                )
        else:
            raise USGPredicateFrameConversionError(
                f"role {index} in {frame.usg_record_id!r} must be a string or object"
            )
        if not isinstance(role_name, str) or not role_name.strip():
            raise USGPredicateFrameConversionError(
                f"role {index} in {frame.usg_record_id!r} has no explicit role/name"
            )
        if role_name in roles:
            raise USGPredicateFrameConversionError(
                f"USG predicate frame {frame.usg_record_id!r} contains duplicate role {role_name!r}"
            )
        roles.append(role_name)
        if required:
            required_roles.append(role_name)

    declared_required = payload.get("required_roles", ())
    if declared_required:
        if not isinstance(declared_required, list) or any(
            not isinstance(item, str) or not item.strip() for item in declared_required
        ):
            raise USGPredicateFrameConversionError(
                f"USG predicate frame {frame.usg_record_id!r} has invalid required_roles"
            )
        required_roles.extend(item for item in declared_required if item not in required_roles)

    preferences = payload.get("selectional_preferences", {})
    if not isinstance(preferences, dict):
        raise USGPredicateFrameConversionError(
            f"USG predicate frame {frame.usg_record_id!r} has invalid selectional_preferences"
        )
    normalized_preferences: dict[str, dict[str, float]] = {}
    for role, features in preferences.items():
        if not isinstance(role, str) or not isinstance(features, dict):
            raise USGPredicateFrameConversionError(
                f"USG predicate frame {frame.usg_record_id!r} has malformed role preferences"
            )
        normalized_preferences[role] = {}
        for feature, raw_value in features.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                raise USGPredicateFrameConversionError(
                    f"selectional preference {role}.{feature} must be numeric"
                )
            normalized_preferences[role][str(feature)] = float(raw_value)

    try:
        return PredicateFrame(
            predicate=frame.frame_key,
            roles=tuple(roles),
            required_roles=tuple(required_roles),
            selectional_preferences=normalized_preferences,
            source_reference=(
                f"{frame.source_reference}&frame_version={quote(frame.frame_version, safe='')}"
            ),
        )
    except (TypeError, ValueError) as exc:
        raise USGPredicateFrameConversionError(
            f"USG predicate frame {frame.usg_record_id!r} violates ORE PredicateFrame constraints: {exc}"
        ) from exc


def build_lexical_access_processor(
    resource: USGKnowledgeResource,
    *,
    config: "LexicalAccessConfig | None" = None,
    morphemes: Iterable[Any] = (),
) -> "LexicalAccessProcessor":
    """Build ORE's lexical-access processor over imported read-only USG entries."""
    from ore_lexical_access import LexicalAccessProcessor

    if not isinstance(resource, USGKnowledgeResource):
        raise TypeError("resource must be a USGKnowledgeResource")
    return LexicalAccessProcessor(
        resource.lexical.entries,
        morphemes=tuple(morphemes),
        config=config,
    )


def build_semantic_composition_processor(
    resource: USGKnowledgeResource,
    *,
    config: "SemanticCompositionConfig | None" = None,
) -> "SemanticCompositionProcessor":
    """Build ORE semantic composition over imported concepts and explicitly converted frames."""
    from ore_semantic_composition import SemanticCompositionProcessor

    if not isinstance(resource, USGKnowledgeResource):
        raise TypeError("resource must be a USGKnowledgeResource")
    frames = tuple(
        to_ore_predicate_frame(frame)
        for frame in resource.structures.predicate_frames
    )
    return SemanticCompositionProcessor(
        concepts=resource.semantic.concepts,
        predicate_frames=frames,
        config=config,
    )


def _semantic_lookup_key(value: str) -> str:
    """Mirror ORE semantic-composition token normalization for exact frame matching."""
    normalized = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    return re.sub(r"^\\W+|\\W+$", "", normalized, flags=re.UNICODE)


class USGSemanticCompositionProcessor:
    """Run ORE semantic composition and attach sentence-relevant explicit USG knowledge.

    Concepts and predicate frames are passed into ORE's existing nine-stage
    processor. Explicit semantic classes, concept mappings, and cross-resource
    relations are then attached to the sentence graph only when their endpoints
    are grounded in selected concepts, exact lexical matches, or matched frames.
    No feature values, inverse relations, class memberships, or truth claims are
    generated by this adapter.
    """

    def __init__(self, resource: USGKnowledgeResource, processor: Any) -> None:
        if not isinstance(resource, USGKnowledgeResource):
            raise TypeError("resource must be a USGKnowledgeResource")
        if not callable(getattr(processor, "process", None)):
            raise TypeError("processor must expose process(request)")
        self.resource = resource
        self._processor = processor

    def __getattr__(self, name: str) -> Any:
        """Expose the wrapped ORE processor's configuration and read-only indexes."""
        return getattr(self._processor, name)

    def process(self, request: Any) -> Any:
        """Compose a sentence, then attach only relevant USG-backed graph evidence."""
        result = self._processor.process(request)
        self._attach_usg_knowledge(result)
        return result

    def _attach_usg_knowledge(self, result: Any) -> None:
        """Add a referentially valid, provenance-preserving USG subgraph to the result."""
        graph = deepcopy(result.sentence_representation)
        graph_nodes = list(graph.get("nodes", ()))
        graph_edges = list(graph.get("edges", ()))
        overlay_nodes: list[dict[str, Any]] = []
        overlay_edges: list[dict[str, Any]] = []
        node_ids: dict[str, str] = {}
        active_record_ids: set[str] = set()
        token_record_edges: set[tuple[int, str, str]] = set()

        def add_record_node(record_id: str, node_type: str, **attributes: Any) -> str:
            node_id = node_ids.get(record_id)
            if node_id is None:
                node_id = f"usg-record:{record_id}"
                node_ids[record_id] = node_id
                overlay_nodes.append({
                    "node_id": node_id,
                    "node_type": node_type,
                    "usg_record_id": record_id,
                    **attributes,
                })
            return node_id

        def bind_token(index: int, record_id: str, node_type: str, **attributes: Any) -> str:
            node_id = add_record_node(record_id, node_type, **attributes)
            active_record_ids.add(record_id)
            binding = (index, record_id, node_type)
            if binding not in token_record_edges:
                token_record_edges.add(binding)
                overlay_edges.append({
                    "edge_type": "grounded_by_usg_record",
                    "source": f"token:{index}",
                    "target": node_id,
                    "record_kind": node_type,
                    "source": "token",
                    "target_record_id": record_id,
                })
            return node_id

        concept_by_id = {
            item.concept_id: item for item in self.resource.semantic.concepts
        }
        for row in result.word_concepts:
            concept_id = row.get("selected_concept_id")
            if not concept_id or concept_id not in concept_by_id:
                continue
            concept = concept_by_id[concept_id]
            bind_token(
                int(row["token_index"]),
                concept_id,
                "usg_semantic_concept",
                label=concept.label,
                aliases=list(concept.aliases),
                language=concept.language,
                lexical_category=concept.lexical_category,
                semantic_features=dict(concept.semantic_features),
                modality_features={
                    key: dict(value) for key, value in concept.modality_features.items()
                },
                source_reference=concept.source_reference,
                verified_by=concept.verified_by,
            )

        language = str(graph.get("language", "und")).casefold()
        normalized_tokens = [
            _semantic_lookup_key(token) for token in result.tokens
        ]
        for index, token_key in enumerate(normalized_tokens):
            if not token_key:
                continue
            for entry in self.resource.lexical.entries:
                if entry.language.casefold() not in {language, "und"}:
                    continue
                if _semantic_lookup_key(entry.orthography) != token_key:
                    continue
                entry_node = bind_token(
                    index,
                    entry.entry_id,
                    "usg_lexical_entry",
                    orthography=entry.orthography,
                    language=entry.language,
                    lexical_category=entry.lexical_category,
                    source_reference=entry.source_reference,
                )
                for meaning in entry.meanings:
                    meaning_node = add_record_node(
                        meaning.meaning_id,
                        "usg_lexical_sense",
                        gloss=meaning.gloss,
                        prior=meaning.prior,
                        source_reference=meaning.source_reference,
                        verified_by=meaning.verified_by,
                    )
                    active_record_ids.add(meaning.meaning_id)
                    overlay_edges.append({
                        "edge_type": "has_usg_lexical_sense",
                        "source": entry_node,
                        "target": meaning_node,
                    })

        matched_frames: list[tuple[int, Any, Any]] = []
        for index, token_key in enumerate(normalized_tokens):
            if not token_key:
                continue
            for frame in self.resource.structures.predicate_frames:
                if _semantic_lookup_key(frame.frame_key) != token_key:
                    continue
                ore_frame = to_ore_predicate_frame(frame)
                frame_node = bind_token(
                    index,
                    frame.frame_id,
                    "usg_predicate_frame",
                    frame_key=frame.frame_key,
                    frame_version=frame.frame_version,
                    roles=list(ore_frame.roles),
                    required_roles=list(ore_frame.required_roles),
                    selectional_preferences={
                        role: dict(values)
                        for role, values in ore_frame.selectional_preferences.items()
                    },
                    source_reference=ore_frame.source_reference,
                    lifecycle_state=frame.lifecycle_state,
                    payload_json=frame.payload_json,
                )
                matched_frames.append((index, frame, ore_frame))
                for proposition in result.propositions:
                    if proposition.get("predicate_index") == index:
                        proposition_id = proposition.get("proposition_id")
                        if proposition_id:
                            overlay_edges.append({
                                "edge_type": "uses_usg_predicate_frame",
                                "source": proposition_id,
                                "target": frame_node,
                                "frame_key": frame.frame_key,
                                "frame_version": frame.frame_version,
                            })

        classes_by_id = {
            item.class_id: item for item in self.resource.structures.classes
        }
        linked_class_ids: set[str] = set()
        for relation in self.resource.relations:
            if relation.subject_id in active_record_ids and relation.object_id in classes_by_id:
                linked_class_ids.add(relation.object_id)
            if relation.object_id in active_record_ids and relation.subject_id in classes_by_id:
                linked_class_ids.add(relation.subject_id)

        for class_id in sorted(linked_class_ids):
            semantic_class = classes_by_id[class_id]
            add_record_node(
                class_id,
                "usg_semantic_class",
                class_key=semantic_class.class_key,
                definition=semantic_class.definition,
                lifecycle_state=semantic_class.lifecycle_state,
                source_reference=semantic_class.source_reference,
                payload_json=semantic_class.payload_json,
            )
            active_record_ids.add(class_id)

        applied_relations = 0
        for relation in self.resource.relations:
            if (
                relation.subject_id not in active_record_ids
                or relation.object_id not in active_record_ids
            ):
                continue
            subject_node = node_ids.get(relation.subject_id)
            object_node = node_ids.get(relation.object_id)
            if not subject_node or not object_node:
                continue
            overlay_edges.append({
                "edge_type": "usg_explicit_relation",
                "source": subject_node,
                "target": object_node,
                "predicate": relation.predicate,
                "direction": "directed",
                "lifecycle_state": relation.lifecycle_state,
                "evidence_ids": list(relation.evidence_ids),
                "context_ids": list(relation.context_ids),
                "source_reference": relation.source_reference,
                "relation_id": relation.relation_id,
            })
            applied_relations += 1

        applied_mappings = 0
        for mapping in self.resource.semantic.mappings:
            if (
                mapping.source_concept_id not in active_record_ids
                or mapping.target_concept_id not in active_record_ids
            ):
                continue
            source_node = node_ids.get(mapping.source_concept_id)
            target_node = node_ids.get(mapping.target_concept_id)
            if not source_node or not target_node:
                continue
            overlay_edges.append({
                "edge_type": "usg_concept_mapping",
                "source": source_node,
                "target": target_node,
                "mapping_type": mapping.mapping_type,
                "lifecycle_state": mapping.lifecycle_state,
                "evidence_ids": list(mapping.evidence_ids),
                "context_ids": list(mapping.context_ids),
                "source_reference": mapping.source_reference,
                "mapping_id": mapping.mapping_id,
            })
            applied_mappings += 1

        graph_nodes.extend(overlay_nodes)
        graph_edges.extend(overlay_edges)
        graph["nodes"] = graph_nodes
        graph["edges"] = graph_edges
        graph["usg_knowledge_graph"] = {
            "schema": "ore.usg_sentence_knowledge_graph.v1",
            "namespace": self.resource.namespace,
            "bundle_id": self.resource.bundle_id,
            "bundle_digest": self.resource.bundle_digest,
            "nodes": overlay_nodes,
            "edges": overlay_edges,
            "counts": {
                "semantic_concepts": sum(
                    node["node_type"] == "usg_semantic_concept" for node in overlay_nodes
                ),
                "lexical_entries": sum(
                    node["node_type"] == "usg_lexical_entry" for node in overlay_nodes
                ),
                "lexical_senses": sum(
                    node["node_type"] == "usg_lexical_sense" for node in overlay_nodes
                ),
                "semantic_classes": sum(
                    node["node_type"] == "usg_semantic_class" for node in overlay_nodes
                ),
                "predicate_frames": sum(
                    node["node_type"] == "usg_predicate_frame" for node in overlay_nodes
                ),
                "explicit_relations": applied_relations,
                "concept_mappings": applied_mappings,
            },
        }

        graph_node_ids = {node.get("node_id") for node in graph_nodes}
        graph_errors = [
            {
                "edge_type": edge.get("edge_type"),
                "source": edge.get("source"),
                "target": edge.get("target"),
            }
            for edge in graph_edges
            if edge.get("source") not in graph_node_ids
            or edge.get("target") not in graph_node_ids
        ]
        valid_references = not graph_errors
        graph.setdefault("validation", {})
        graph["validation"]["valid_references"] = valid_references
        graph["validation"]["graph_errors"] = graph_errors
        result.sentence_representation = graph
        result.diagnostics["graph_errors"] = graph_errors
        result.diagnostics["valid"] = valid_references
        result.diagnostics["usg_semantic_integration"] = {
            "bundle_id": self.resource.bundle_id,
            "bundle_digest": self.resource.bundle_digest,
            "semantic_classes_attached": len(linked_class_ids),
            "explicit_relations_attached": applied_relations,
            "concept_mappings_attached": applied_mappings,
            "predicate_frames_matched": len(matched_frames),
            "graph_references_valid": valid_references,
        }
        result.metadata["usg_semantic_integration"] = {
            "schema": "ore.usg_sentence_knowledge_graph.v1",
            "namespace": self.resource.namespace,
            "bundle_id": self.resource.bundle_id,
            "bundle_digest": self.resource.bundle_digest,
            "read_only": True,
            "semantic_features_inferred": False,
            "inverse_or_transitive_relations_generated": False,
        }


def build_usg_semantic_composition_processor(
    resource: USGKnowledgeResource,
    *,
    config: "SemanticCompositionConfig | None" = None,
) -> USGSemanticCompositionProcessor:
    """Build ORE sentence composition with imported concepts, frames, classes, and relations."""
    processor = build_semantic_composition_processor(resource, config=config)
    return USGSemanticCompositionProcessor(resource, processor)


__all__ = [
    "USGPredicateFrameConversionError",
    "USGSemanticCompositionProcessor",
    "to_ore_predicate_frame",
    "build_lexical_access_processor",
    "build_semantic_composition_processor",
    "build_usg_semantic_composition_processor",
]
