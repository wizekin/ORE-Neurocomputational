"""End-to-end 24-stage Point 09 schema-formation processor.

The processor turns multiple distinct episodes into revisable relational frames. Predictions are
explicitly marked as inferred, temporal association is kept separate from causal evidence, and
new episodes can be integrated, specialized, or stored independently.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .biology import BiologicalSchemaNetwork
from .contracts import (
    EpisodeInput, SchemaConfig, SchemaFrame, SchemaLearningResult, SchemaRelation,
    SchemaSlot, SchemaUpdateResult,
)
from .functional import (
    canonical_role, compose_schema_hierarchy, context_signature, estimate_context_profile,
    estimate_goal_distribution, estimate_outcome_distribution, estimate_transitions,
    event_boundary_scores, episode_similarity, group_related_episodes, induce_relations,
    induce_slots, infer_role_assignments, rank_schema_candidates,
)
from .registry import OPERATION_REGISTRY


def _schema_identifier(episodes: Sequence[EpisodeInput]) -> str:
    """Create a deterministic schema identifier from sorted supporting episode IDs."""
    payload = "|".join(sorted(item.episode_id for item in episodes)).encode("utf-8")
    return "schema-" + hashlib.sha256(payload).hexdigest()[:16]


def _schema_confidence(
    slots: Mapping[str, SchemaSlot], relations: Sequence[SchemaRelation]
) -> float:
    """Summarize support and uncertainty without treating one example as strong regularity."""
    evidence = [item.confidence for item in slots.values()]
    evidence.extend(item.confidence for item in relations)
    return sum(evidence) / len(evidence) if evidence else 0.0


def _build_frame(
    episodes: Sequence[EpisodeInput],
    config: SchemaConfig,
    assignments: Sequence[Any],
    *,
    schema_id: str | None = None,
    name: str | None = None,
    status: str = "learned",
) -> SchemaFrame:
    """Induce one probabilistic relational frame while retaining its supporting episode IDs."""
    if not episodes:
        raise ValueError("at least one episode is required to build a schema")
    episode_ids = {item.episode_id for item in episodes}
    group_assignments = [item for item in assignments if item.episode_id in episode_ids]
    slots = induce_slots(episodes, group_assignments, config)
    relations = induce_relations(episodes, slots, group_assignments, config)
    transitions = estimate_transitions(episodes, config)
    goals = estimate_goal_distribution(episodes)
    outcomes = estimate_outcome_distribution(episodes)
    context_profile = estimate_context_profile(episodes)
    event_types = tuple(sorted({
        event.event_type for episode in episodes for event in episode.events
    }))
    return SchemaFrame(
        schema_id=schema_id or _schema_identifier(episodes),
        name=name or event_types[0].replace("_", " ") + " schema",
        slots=slots,
        relations=relations,
        transitions=transitions,
        goals=goals,
        outcomes=outcomes,
        context_profile=context_profile,
        event_types=event_types,
        supporting_episode_ids=tuple(sorted(episode_ids)),
        confidence=_schema_confidence(slots, relations),
        status=status,
    )


def _observed_roles(episode: EpisodeInput, config: SchemaConfig) -> set[str]:
    """Return canonical participant roles explicitly present in an episode."""
    return {
        canonical_role(role, config.role_aliases)
        for event in episode.events for role in event.participants
    }


def _observed_relation_keys(episode: EpisodeInput, config: SchemaConfig) -> set[tuple[str, str, str]]:
    """Align explicit relation endpoints to observed roles without assigning causal semantics."""
    entity_roles: dict[str, set[str]] = {}
    for event in episode.events:
        for role, entity_id in event.participants.items():
            entity_roles.setdefault(entity_id, set()).add(canonical_role(role, config.role_aliases))
    output: set[tuple[str, str, str]] = set()
    for relation in episode.relations:
        subject_roles = entity_roles.get(relation.subject_id, set())
        object_roles = entity_roles.get(relation.object_id, set())
        if len(subject_roles) == 1 and len(object_roles) == 1:
            output.add((
                next(iter(subject_roles)), relation.predicate.casefold(),
                next(iter(object_roles)),
            ))
    return output


def _transition_probability(
    schema: SchemaFrame, source: str, target: str, context: Mapping[str, Any]
) -> float:
    """Look up a context-specific transition and fall back to the global estimate."""
    key = context_signature(context)
    matching = [
        item for item in schema.transitions
        if item["from_event"] == source and item["to_event"] == target
    ]
    contextual = [item for item in matching if item["context_key"] == key]
    rows = contextual or [item for item in matching if item["context_key"] == "*"]
    return float(rows[0]["probability"]) if rows else 0.0


def _evaluate_mismatch(
    schema: SchemaFrame, episode: EpisodeInput, config: SchemaConfig
) -> dict[str, Any]:
    """Compare an episode with a schema while treating unobserved relations as unknown."""
    observed_roles = _observed_roles(episode, config)
    known_roles = set(schema.slots)
    unexpected_roles = sorted(observed_roles - known_roles)
    missing_core_roles = sorted(
        name for name, slot in schema.slots.items()
        if slot.status == "core" and name not in observed_roles
    )
    role_mismatch = min(
        1.0, (len(unexpected_roles) + 0.25 * len(missing_core_roles))
        / max(1, len(known_roles) + 1)
    )
    known_relations = {
        (item.subject_slot, item.predicate, item.object_slot) for item in schema.relations
    }
    observed_relations = _observed_relation_keys(episode, config)
    novel_relations = sorted(observed_relations - known_relations)
    relation_mismatch = len(novel_relations) / max(1, len(observed_relations))
    event_surprises: list[float] = []
    transition_details: list[dict[str, Any]] = []
    for left, right in zip(episode.events, episode.events[1:]):
        probability = _transition_probability(
            schema, left.event_type, right.event_type, episode.context
        )
        if probability <= 0.0:
            probability = 1.0 / max(1, len(schema.event_types) + 1)
        surprise = min(1.0, -math.log(max(probability, 1e-9)) / 5.0)
        event_surprises.append(surprise)
        transition_details.append({
            "from_event": left.event_type, "to_event": right.event_type,
            "probability": probability, "normalized_surprise": surprise,
        })
    temporal_mismatch = (
        sum(event_surprises) / len(event_surprises) if event_surprises else 0.0
    )
    mismatch = min(1.0, 0.35 * role_mismatch + 0.20 * relation_mismatch
                   + 0.45 * temporal_mismatch)
    return {
        "schema_id": schema.schema_id, "episode_id": episode.episode_id,
        "mismatch": mismatch, "role_mismatch": role_mismatch,
        "relation_mismatch": relation_mismatch, "temporal_mismatch": temporal_mismatch,
        "unexpected_roles": unexpected_roles, "missing_core_roles": missing_core_roles,
        "novel_relations": novel_relations, "transition_details": transition_details,
        "interpretation": (
            "A mismatch is a review signal, not proof that the entire schema is invalid. "
            "Missing relations are not scored as absent unless explicitly observed as negative."
        ),
    }


def _predict_missing_structure(
    schema: SchemaFrame,
    observed: EpisodeInput,
    config: SchemaConfig,
) -> dict[str, Any]:
    """Generate uncertain role, relation, and next-event completions with inferred provenance."""
    observed_roles = _observed_roles(observed, config)
    inferred: list[dict[str, Any]] = []
    for name, slot in sorted(schema.slots.items()):
        if name in observed_roles:
            continue
        if not slot.allowed_type_probabilities:
            continue
        entity_type, type_probability = max(
            slot.allowed_type_probabilities.items(), key=lambda item: (item[1], item[0])
        )
        confidence = min(1.0, slot.coverage * type_probability)
        if confidence >= config.prediction_confidence_threshold:
            inferred.append({
                "kind": "missing_role", "slot": name,
                "likely_entity_type": entity_type, "confidence": confidence,
                "provenance": "inferred", "observed": False,
            })
    for relation in schema.relations:
        if relation.confidence < config.prediction_confidence_threshold:
            continue
        subject_present = relation.subject_slot in observed_roles
        object_present = relation.object_slot in observed_roles
        if subject_present != object_present:
            inferred.append({
                "kind": "slot_relation", "subject_slot": relation.subject_slot,
                "predicate": relation.predicate, "object_slot": relation.object_slot,
                "evidential_status": relation.evidential_status,
                "confidence": relation.confidence, "provenance": "inferred",
                "observed": False,
            })
    next_events: list[dict[str, Any]] = []
    last_event = observed.events[-1].event_type
    key = context_signature(observed.context)
    rows = [
        item for item in schema.transitions
        if item["from_event"] == last_event and item["context_key"] == key
    ]
    if not rows:
        rows = [
            item for item in schema.transitions
            if item["from_event"] == last_event and item["context_key"] == "*"
        ]
    next_events = sorted(
        (
            {"event_type": row["to_event"], "probability": row["probability"],
             "confidence": row["probability"], "provenance": "inferred", "observed": False}
            for row in rows if row["probability"] >= config.prediction_confidence_threshold
        ),
        key=lambda item: (-item["probability"], item["event_type"]),
    )[:config.max_predictions]
    inferred.sort(key=lambda item: (-item["confidence"], item["kind"], str(item.get("slot", ""))))
    return {
        "schema_id": schema.schema_id,
        "context_key": key,
        "inferred_items": inferred[:config.max_predictions],
        "next_event_predictions": next_events,
        "source_episode_id": observed.episode_id,
        "provenance_policy": "predictions are never silently converted into observations",
    }


def _append_operation(
    trace: list[dict[str, Any]], index: int, functional: Mapping[str, Any],
    biological: Mapping[str, Any] | None = None, status: str = "completed",
) -> None:
    """Append an inspectable result for one source-aligned stage."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1, "operation_id": spec.operation_id,
        "operation": spec.name, "status": status, "explanation": spec.purpose,
        "evidence_level": spec.evidence_level, "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional), "biological_output": dict(biological or {}),
    })


class SchemaFormationProcessor:
    """Learn, predict with, evaluate, and revise Point 09 relational schema frames.

    Episodes are retained independently from schemas. The processor implements explicit
    role/slot abstractions, directed relations, context-conditioned transitions, uncertainty,
    hierarchy links, and provenance-aware prediction. Its biological networks are surrogates.
    """

    def __init__(
        self,
        config: SchemaConfig | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize schema state, episode memory, biological surrogates, and logger settings."""
        self.config = config or SchemaConfig()
        self.biological_network = BiologicalSchemaNetwork(self.config)
        self.episodes: dict[str, EpisodeInput] = {}
        self.schemas: dict[str, SchemaFrame] = {}
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )
        self.fit_count = 0

    def _validate_episodes(self, episodes: Sequence[EpisodeInput]) -> list[EpisodeInput]:
        """Reject duplicate IDs with conflicting content and preserve first-seen episode order."""
        if not episodes:
            raise ValueError("at least one episode is required")
        unique: dict[str, EpisodeInput] = {}
        for episode in episodes:
            if not isinstance(episode, EpisodeInput):
                raise TypeError("episodes must contain EpisodeInput records")
            existing = unique.get(episode.episode_id) or self.episodes.get(episode.episode_id)
            if existing is not None:
                if existing.to_dict() != episode.to_dict():
                    raise ValueError(
                        "episode_id " + repr(episode.episode_id) + " already exists with different content"
                    )
                continue
            unique[episode.episode_id] = episode
        return list(unique.values())

    def fit(
        self, episodes: Sequence[EpisodeInput], schema_name_prefix: str = "schema"
    ) -> SchemaLearningResult:
        """Induce episode groups and their schema frames through all 24 registered stages."""
        started = time.perf_counter()
        new_episodes = self._validate_episodes(episodes)
        all_episodes = list(self.episodes.values()) + new_episodes
        assignments = infer_role_assignments(all_episodes, self.config)
        groups = group_related_episodes(
            all_episodes, self.config.similarity_threshold, self.config.role_aliases
        )
        frames: list[SchemaFrame] = []
        for index, group in enumerate(groups, start=1):
            frames.append(_build_frame(
                group, self.config, assignments,
                name=schema_name_prefix + "_" + str(index),
            ))
        hierarchy = compose_schema_hierarchy(
            frames, self.config.hierarchy_similarity_threshold
        )
        for frame in frames:
            links = hierarchy[frame.schema_id]
            frame.parent_schema_ids = links["parents"]
            frame.child_schema_ids = links["children"]
            frame.reusable_components = links["components"]
        for episode in new_episodes:
            self.episodes[episode.episode_id] = episode
        self.schemas = {frame.schema_id: frame for frame in frames}
        self.fit_count += 1
        biological_observation = self.biological_network.observe(new_episodes)
        trace: list[dict[str, Any]] = []

        boundaries = {
            episode.episode_id: event_boundary_scores(episode) for episode in all_episodes
        }
        _append_operation(trace, 0, {
            "episode_count": len(all_episodes), "schema_target": "context, roles, slots, relations, events, goals"
        })
        _append_operation(trace, 1, {
            "episode_boundary_scores": boundaries,
            "event_count": sum(len(item.events) for item in all_episodes),
        })
        _append_operation(trace, 2, {
            "episode_graphs": [
                {"episode_id": item.episode_id, "entities": len(item.entities),
                 "events": len(item.events), "relations": len(item.relations)}
                for item in all_episodes
            ],
            "episode_identity_preserved": True,
        })
        similarity_rows = []
        for index, left in enumerate(all_episodes):
            for right in all_episodes[index + 1:]:
                similarity_rows.append({
                    "left": left.episode_id, "right": right.episode_id,
                    "similarity": episode_similarity(left, right, self.config.role_aliases),
                })
        _append_operation(trace, 3, {
            "group_count": len(groups), "groups": [
                [item.episode_id for item in group] for group in groups
            ], "pairwise_similarity": similarity_rows,
        })
        _append_operation(trace, 4, {
            "role_assignments": [item.to_dict() for item in assignments],
            "ambiguous_assignment_count": sum(item.ambiguous for item in assignments),
        })
        _append_operation(trace, 5, {
            "slots_by_schema": {
                frame.schema_id: {name: slot.to_dict() for name, slot in frame.slots.items()}
                for frame in frames
            }
        })
        _append_operation(trace, 6, {
            "relations_by_schema": {
                frame.schema_id: [item.to_dict() for item in frame.relations] for frame in frames
            },
            "causal_hypothesis_count": sum(
                item.evidential_status == "causal_hypothesis"
                for frame in frames for item in frame.relations
            ),
        })
        _append_operation(trace, 7, {
            "transitions_by_schema": {
                frame.schema_id: list(frame.transitions) for frame in frames
            }
        })
        _append_operation(trace, 8, {
            "evidence_statuses": {
                frame.schema_id: dict(Counter(item.evidential_status for item in frame.relations))
                for frame in frames
            },
            "causal_claims_promoted_to_fact": False,
        })
        _append_operation(trace, 9, {
            "slot_statuses": {
                frame.schema_id: {name: slot.status for name, slot in frame.slots.items()}
                for frame in frames
            },
            "relation_statuses": {
                frame.schema_id: [item.status for item in frame.relations] for frame in frames
            },
        })
        _append_operation(trace, 10, {
            "schema_frame_ids": [frame.schema_id for frame in frames],
            "schema_component_counts": {
                frame.schema_id: {"slots": len(frame.slots), "relations": len(frame.relations),
                                  "transitions": len(frame.transitions)}
                for frame in frames
            },
        })
        _append_operation(trace, 11, {"hierarchy": hierarchy})
        _append_operation(trace, 12, {
            "goals": {frame.schema_id: dict(frame.goals) for frame in frames},
            "outcomes": {frame.schema_id: dict(frame.outcomes) for frame in frames},
        })
        predictions = []
        for episode in all_episodes:
            candidates = rank_schema_candidates(episode, frames, self.config)
            if candidates:
                frame = self.schemas.get(candidates[0]["schema_id"])
                if frame:
                    predictions.append(_predict_missing_structure(frame, episode, self.config))
        _append_operation(trace, 13, {
            "candidate_rankings": [
                {"episode_id": episode.episode_id,
                 "ranked_schemas": rank_schema_candidates(episode, frames, self.config)}
                for episode in all_episodes
            ]
        })
        _append_operation(trace, 14, {"prediction_examples": predictions})
        mismatch_rows = []
        for episode in all_episodes:
            ranked = rank_schema_candidates(episode, frames, self.config)
            if ranked:
                frame = self.schemas[ranked[0]["schema_id"]]
                mismatch_rows.append(_evaluate_mismatch(frame, episode, self.config))
        _append_operation(trace, 15, {"training_episode_mismatch": mismatch_rows})
        _append_operation(trace, 16, {
            "revision_policy": {
                "integrate_below": self.config.update_mismatch_threshold,
                "specialize_below": self.config.specialization_mismatch_threshold,
                "otherwise": "preserve_as_independent_schema",
            },
            "new_evidence_already_integrated": len(new_episodes),
        })
        _append_operation(trace, 17, {
            "exception_count": sum(
                len(row["unexpected_roles"]) + len(row["novel_relations"])
                for row in mismatch_rows
            ),
            "context_profiles": {frame.schema_id: dict(frame.context_profile) for frame in frames},
            "uncertainty_preserved": True,
        })
        _append_operation(trace, 18, {
            "end_to_end_order": [spec.operation_id for spec in OPERATION_REGISTRY[:18]],
            "episode_count": len(all_episodes), "schema_count": len(frames),
        })
        _append_operation(trace, 19, {
            "mathematical_operations": [
                {"operation_id": spec.operation_id, "mathematical_form": spec.mathematical_form}
                for spec in OPERATION_REGISTRY[:18]
            ]
        })
        _append_operation(trace, 20, {
            "episode_network_trace": biological_observation["episode_network_trace"],
            "network_state": biological_observation["network_state"],
        }, biological_observation["network_state"])
        validation = {
            "slot_coverage_mean": (
                sum(slot.coverage for frame in frames for slot in frame.slots.values())
                / max(1, sum(len(frame.slots) for frame in frames))
            ),
            "relation_f1_ready": True,
            "transition_log_loss_ready": True,
            "false_inference_provenance_check": all(
                item.get("provenance") == "inferred" and not item.get("observed")
                for prediction in predictions for item in prediction["inferred_items"]
            ),
            "episode_count": len(all_episodes),
        }
        _append_operation(trace, 21, validation)
        risk_flags = []
        for assignment in assignments:
            if assignment.ambiguous:
                risk_flags.append({
                    "risk": "role_ambiguity", "episode_id": assignment.episode_id,
                    "entity_id": assignment.entity_id, "safeguard": "retain alternatives",
                })
        for frame in frames:
            if any(item.evidential_status == "causal_hypothesis" for item in frame.relations):
                risk_flags.append({
                    "risk": "causal_overclaim", "schema_id": frame.schema_id,
                    "safeguard": "retain causal_hypothesis label",
                })
        _append_operation(trace, 22, {"risk_flags": risk_flags})
        diagnostics = {
            "episode_count": len(all_episodes), "new_episode_count": len(new_episodes),
            "schema_count": len(frames), "role_assignment_count": len(assignments),
            "ambiguous_role_assignment_count": sum(item.ambiguous for item in assignments),
            "total_slots": sum(len(frame.slots) for frame in frames),
            "total_relations": sum(len(frame.relations) for frame in frames),
            "total_transition_rows": sum(len(frame.transitions) for frame in frames),
            "risk_flag_count": len(risk_flags), "duration_ms": round((time.perf_counter() - started) * 1000, 3),
        }
        _append_operation(trace, 23, {
            "schema_summaries": [
                {"schema_id": frame.schema_id, "name": frame.name,
                 "slot_count": len(frame.slots), "relation_count": len(frame.relations),
                 "confidence": frame.confidence, "supporting_episode_ids": list(frame.supporting_episode_ids)}
                for frame in frames
            ],
            "diagnostics": diagnostics,
        })
        result = SchemaLearningResult(
            schemas=tuple(frames), role_assignments=tuple(assignments),
            episode_groups=tuple(tuple(item.episode_id for item in group) for group in groups),
            operation_trace=tuple(trace),
            biological=biological_observation["network_state"],
            diagnostics=diagnostics,
        )
        if self.database_path:
            record_result_to_sqlite(
                result, self.database_path, repository_root=self.repository_root
            )
        return result

    def predict(
        self,
        observed: EpisodeInput,
        schema_id: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Select or rank schemas and produce gap completions with explicit inferred provenance."""
        if not self.schemas:
            raise RuntimeError("fit at least one schema before requesting predictions")
        context_values = dict(observed.context)
        context_values.update(dict(context or {}))
        ranked = rank_schema_candidates(observed, list(self.schemas.values()), self.config)
        if schema_id is not None:
            if schema_id not in self.schemas:
                raise KeyError(schema_id)
            schema = self.schemas[schema_id]
            ranked = [row for row in ranked if row["schema_id"] == schema_id]
        else:
            schema = self.schemas[ranked[0]["schema_id"]]
        prediction_episode = replace(observed, context=context_values)
        result = _predict_missing_structure(schema, prediction_episode, self.config)
        result["ranked_candidates"] = ranked
        activation = self.biological_network.select_schema(
            list(self.schemas.values()), context_values, observed.goals
        )
        result["biological_schema_activation"] = activation
        return result

    def evaluate(self, schema_id: str, episode: EpisodeInput) -> dict[str, Any]:
        """Return structural and temporal mismatch diagnostics for one schema/episode pair."""
        if schema_id not in self.schemas:
            raise KeyError(schema_id)
        return _evaluate_mismatch(self.schemas[schema_id], episode, self.config)

    def update_schema(self, schema_id: str, episode: EpisodeInput) -> SchemaUpdateResult:
        """Integrate, specialize, or preserve a new episode based on calibrated mismatch evidence."""
        if schema_id not in self.schemas:
            raise KeyError(schema_id)
        existing = self.schemas[schema_id]
        stored = self.episodes.get(episode.episode_id)
        if stored is not None:
            if stored.to_dict() != episode.to_dict():
                raise ValueError("episode_id already exists with different episode content")
            mismatch = _evaluate_mismatch(existing, episode, self.config)
            trace = self._update_trace("duplicate_ignored", existing, mismatch)
            return SchemaUpdateResult("duplicate_ignored", existing.schema_id, mismatch, existing, tuple(trace))
        mismatch = _evaluate_mismatch(existing, episode, self.config)
        source_episodes = [
            self.episodes[item] for item in existing.supporting_episode_ids if item in self.episodes
        ]
        if mismatch["mismatch"] <= self.config.update_mismatch_threshold:
            revised_episodes = source_episodes + [episode]
            assignments = infer_role_assignments(revised_episodes, self.config)
            updated = _build_frame(
                revised_episodes, self.config, assignments,
                schema_id=existing.schema_id, name=existing.name, status="learned",
            )
            updated.version = existing.version + 1
            updated.parent_schema_ids = existing.parent_schema_ids
            updated.child_schema_ids = existing.child_schema_ids
            updated.reusable_components = existing.reusable_components
            self.schemas[schema_id] = updated
            action = "integrated"
        else:
            specialized = mismatch["mismatch"] <= self.config.specialization_mismatch_threshold
            status = "specialized" if specialized else "independent"
            assignments = infer_role_assignments([episode], self.config)
            new_frame = _build_frame(
                [episode], self.config, assignments, status=status,
                name=existing.name + (" variant" if specialized else " independent case"),
            )
            if specialized:
                new_frame.parent_schema_ids = (existing.schema_id,)
                existing.child_schema_ids = tuple(sorted(set(existing.child_schema_ids) | {new_frame.schema_id}))
                action = "specialized"
            else:
                action = "preserved_independently"
            self.schemas[new_frame.schema_id] = new_frame
            updated = new_frame
        self.episodes[episode.episode_id] = episode
        trace = self._update_trace(action, updated, mismatch)
        return SchemaUpdateResult(action, updated.schema_id, mismatch, updated, tuple(trace))

    def _update_trace(
        self, action: str, schema: SchemaFrame, mismatch: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        """Create source-aligned trace records for schema comparison, revision, and safeguards."""
        trace: list[dict[str, Any]] = []
        _append_operation(trace, 15, {"mismatch": dict(mismatch)})
        _append_operation(trace, 16, {
            "action": action, "schema_id": schema.schema_id,
            "schema_version": schema.version, "schema_status": schema.status,
        })
        _append_operation(trace, 17, {
            "exceptions_preserved": True, "supporting_episode_ids": list(schema.supporting_episode_ids),
            "uncertainty_preserved": True,
        })
        return trace


def register_annotations(
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index Point 09 classes and callables in the reusable ORE SQLite annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    database = LoggerDatabase(database_path)
    return sync_repository(database, root, include_paths={"ore_schema_formation"})


def record_result_to_sqlite(
    result: SchemaLearningResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the 24-stage operation trace and refresh source annotations in ORE SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    database = LoggerDatabase(database_path)
    catalog = sync_repository(database, root, include_paths={"ore_schema_formation"})
    run_id = database.create_run(
        script_path="ore_schema_formation/point09",
        module_name="ore_schema_formation.pipeline",
        entry_kind="schema_learning",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                source_path="ore_schema_formation/pipeline.py", elapsed_ms=0.0,
                metadata={
                    "operation_id": item.get("operation_id"), "operation": item.get("operation"),
                    "status": item.get("status"), "evidence_level": item.get("evidence_level"),
                    "mathematical_form": item.get("mathematical_form"),
                    "functional_output": item.get("functional_output"),
                    "biological_output": item.get("biological_output"),
                },
            )
        database.finish_run(
            run_id, "completed", duration_ms=round((time.perf_counter() - started) * 1000, 3)
        )
    except Exception as exc:
        database.finish_run(
            run_id, "failed", duration_ms=round((time.perf_counter() - started) * 1000, 3),
            error_summary=type(exc).__name__ + ": " + str(exc),
        )
        raise
    return {"run_id": run_id, "annotation_catalog": catalog, "database_summary": database.summary()}
