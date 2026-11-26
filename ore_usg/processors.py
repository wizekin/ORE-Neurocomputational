"""Read-only bridges from unified USG resources to existing ORE processors."""
from __future__ import annotations

import json
from urllib.parse import quote
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from ore_lexical_access import LexicalAccessConfig, LexicalAccessProcessor
    from ore_semantic_composition import (
        PredicateFrame,
        SemanticCompositionConfig,
        SemanticCompositionProcessor,
    )

from .knowledge import USGKnowledgeResource
from .access import USGKnowledgeAccess
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
    processor = LexicalAccessProcessor(
        resource.lexical.entries,
        morphemes=tuple(morphemes),
        config=config,
    )
    processor.usg_knowledge = USGKnowledgeAccess(resource)
    return processor


def build_semantic_composition_processor(
    resource: USGKnowledgeResource,
    *,
    config: "SemanticCompositionConfig | None" = None,
) -> "SemanticCompositionProcessor":
    """Build ORE semantic composition over imported concepts and explicitly converted frames."""
    from ore_semantic_composition import SemanticCompositionProcessor

    if not isinstance(resource, USGKnowledgeResource):
        raise TypeError("resource must be a USGKnowledgeResource")
    if resource.semantic_knowledge is not None:
        concepts = resource.semantic_knowledge.active_concepts
        frames = resource.semantic_knowledge.executable_frames
    else:
        concepts = resource.semantic.concepts
        compatible_frames = []
        for frame in resource.structures.predicate_frames:
            try:
                compatible_frames.append(to_ore_predicate_frame(frame))
            except USGPredicateFrameConversionError:
                # One unsupported/incomplete frame must not disable other
                # available semantic knowledge or executable frames.
                continue
        frames = tuple(compatible_frames)
    processor = SemanticCompositionProcessor(
        concepts=concepts,
        predicate_frames=frames,
        config=config,
    )
    processor.usg_knowledge = USGKnowledgeAccess(resource)
    return processor


def build_semantic_cognition_processor(
    resource: USGKnowledgeResource,
    *,
    config: Any | None = None,
    memory_store: Any | None = None,
) -> Any:
    """Build semantic cognition over imported meanings and attach common retrieval."""
    from ore_semantic import SemanticCognitionProcessor

    if not isinstance(resource, USGKnowledgeResource):
        raise TypeError("resource must be a USGKnowledgeResource")
    concepts = (
        resource.semantic_knowledge.active_concepts
        if resource.semantic_knowledge is not None
        else resource.semantic.concepts
    )
    processor = SemanticCognitionProcessor(
        concepts=concepts,
        config=config,
        memory_store=memory_store,
    )
    processor.usg_knowledge = USGKnowledgeAccess(resource)
    return processor


__all__ = [
    "USGPredicateFrameConversionError",
    "to_ore_predicate_frame",
    "build_lexical_access_processor",
    "build_semantic_composition_processor",
    "build_semantic_cognition_processor",
]
