"""Persistent 16-stage processor for episodic-to-semantic transformation and gist extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .biology import DistributedSemanticNetwork, HippocampalIndexNetwork
from .contracts import (
    EpisodeInput,
    FEATURE_GROUPS,
    MemoryTransformationConfig,
    MemoryTransformationRequest,
    MemoryTransformationResult,
)
from .functional import (
    accessibility_decay,
    brier_score,
    cue_similarity,
    episode_similarity,
    estimate_feature_statistics,
    estimate_relation_statistics,
    leave_one_out_gist_similarity,
    relation_signature,
    semantic_context_weight,
    softmax,
    source_provenance,
    weighted_feature_similarity,
)
from .registry import OPERATION_REGISTRY


@dataclass(slots=True)
class EpisodicMemoryRecord:
    """Retain an immutable event, hippocampal index, per-feature access, and retrieval metadata."""

    episode: EpisodeInput
    concept_id: str
    hippocampal_index: np.ndarray
    feature_accessibility: dict[str, float]
    last_access_update_ms: float | None = None
    retrieval_count: int = 0
    reactivation_count: int = 0


class EpisodicToSemanticTransformationProcessor:
    """Transform experience across linked episodic and semantic stores without overwriting either.

    The processor stores event-specific feature bindings and separate concept-level statistics.
    Repeated and related episodes can strengthen a semantic gist; low-accessibility event details
    remain present in the episodic record. Query type determines whether event reconstruction,
    semantic retrieval, or a combination is returned.
    """

    _ACCESS_DECAY_PER_HOUR = {
        "content": 0.00012,
        "causal": 0.00014,
        "affective": 0.00024,
        "spatial": 0.00072,
        "temporal": 0.00082,
        "sensory": 0.00068,
    }

    def __init__(
        self,
        config: MemoryTransformationConfig | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize hippocampal and cortical models, dual stores, counters, and optional SQLite logging."""
        self.config = config or MemoryTransformationConfig()
        self.hippocampus = HippocampalIndexNetwork(self.config)
        self.cortex = DistributedSemanticNetwork(self.config)
        self.episodic_store: dict[str, EpisodicMemoryRecord] = {}
        self.episode_concept_ids: dict[str, str] = {}
        self.semantic_store: dict[str, dict[str, Any]] = {}
        self._next_concept_number = 1
        self.request_count = 0
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )

    def add_episode(self, episode: EpisodeInput) -> dict[str, Any]:
        """Add one immutable episode, assign a concept cluster, and create its hippocampal index."""
        existing = self.episodic_store.get(episode.episode_id)
        if existing is not None:
            if existing.episode.to_dict() != episode.to_dict():
                raise ValueError("episode_id already exists with different event content")
            return {"episode_id": episode.episode_id, "status": "already_registered",
                    "concept_id": existing.concept_id,
                    "retrieval_count": existing.retrieval_count}
        if len(self.episodic_store) >= self.config.max_episodes:
            raise ValueError("configured episodic store capacity has been reached")
        concept_id, assignment = self._assign_concept(episode)
        biological = self.hippocampus.register(episode)
        access = {f"{group}:{feature}": 1.0
                  for group, features in episode.feature_groups().items()
                  for feature in features}
        self.episodic_store[episode.episode_id] = EpisodicMemoryRecord(
            episode=episode,
            concept_id=concept_id,
            hippocampal_index=self.hippocampus.event_indices[episode.episode_id].copy(),
            feature_accessibility=access,
        )
        self.episode_concept_ids[episode.episode_id] = concept_id
        return {"episode_id": episode.episode_id, "status": "registered",
                "concept_id": concept_id, "concept_assignment": assignment,
                "hippocampal_index": biological}

    def _assign_concept(self, episode: EpisodeInput) -> tuple[str, dict[str, Any]]:
        """Assign a hinted concept or a similarity-based cluster without forcing ambiguous merges."""
        if episode.concept_hint is not None:
            concept_id = f"hint:{episode.concept_hint.casefold()}"
            return concept_id, {"method": "caller_supplied_concept_hint",
                                "confidence": 1.0, "ambiguous": False}
        concept_episodes: dict[str, list[EpisodeInput]] = {}
        for episode_id, record in self.episodic_store.items():
            concept_episodes.setdefault(record.concept_id, []).append(record.episode)
        scores: list[tuple[str, float]] = []
        for concept_id, members in concept_episodes.items():
            member_scores = [episode_similarity(episode, member)["overall"] for member in members]
            scores.append((concept_id, float(np.mean(sorted(member_scores, reverse=True)[:3]))))
        scores.sort(key=lambda item: item[1], reverse=True)
        if scores and scores[0][1] >= self.config.concept_assignment_threshold:
            second = scores[1][1] if len(scores) > 1 else -1.0
            margin = scores[0][1] - second
            if len(scores) == 1 or margin >= self.config.concept_ambiguity_margin:
                return scores[0][0], {"method": "similarity_to_existing_episode_cluster",
                                      "confidence": scores[0][1], "runner_up_score": second,
                                      "margin": margin, "ambiguous": False}
            ambiguity = True
        else:
            second = scores[1][1] if len(scores) > 1 else None
            margin = (scores[0][1] - second) if scores and second is not None else None
            ambiguity = False
        concept_id = f"concept-{self._next_concept_number:05d}"
        self._next_concept_number += 1
        return concept_id, {"method": "new_separate_concept_cluster",
                            "best_existing_score": scores[0][1] if scores else None,
                            "runner_up_score": second, "margin": margin,
                            "ambiguous": ambiguity,
                            "automatic_merge_performed": False}

    def _log_sqlite(self, rows: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
        """Persist all 16 operation annotations and their outputs through the repository SQLite logger."""
        if self.database_path is None:
            return {"enabled": False, "reason": "database_path_not_configured"}
        from ore_logger.recorder import SQLiteRunRecorder

        run_id = uuid.uuid4().hex
        recorder = SQLiteRunRecorder(
            database_path=self.database_path,
            repository_root=self.repository_root,
            run_id=run_id,
            component="ore_memory_transformation",
            callable_name="EpisodicToSemanticTransformationProcessor.process",
            started_at_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            entry_kind="neurocomputational_component",
        )
        for index, (spec, row) in enumerate(zip(OPERATION_REGISTRY, rows), start=1):
            recorder.log_operation({
                "step": index, "operation": spec.operation_id, "name": spec.name,
                "status": row.get("status", "completed"),
                "phase": "episodic_to_semantic_transformation",
                "explanation": spec.purpose, "mathematical_form": spec.mathematical_form,
                "biological_reference": spec.biological_reference,
                "evidence_level": spec.evidence_level, "inputs": list(spec.inputs),
                "outputs": list(spec.outputs),
                "functional_output": row.get("functional_output", {}),
                "biological_output": row.get("biological_output", {}),
            })
        return {"enabled": True, **recorder.finish(
            "completed", elapsed_ms, output_type="MemoryTransformationResult"
        )}

    def process(self, request: MemoryTransformationRequest) -> MemoryTransformationResult:
        """Run all 16 source-ordered operations over new and previously stored experiences."""
        started = time.perf_counter()
        self.request_count += 1
        rows: list[dict[str, Any]] = []

        def record(
            index: int,
            functional_output: dict[str, Any],
            biological_output: dict[str, Any] | None = None,
            status: str = "completed",
        ) -> None:
            """Append one inspectable operation result in canonical source order."""
            spec = OPERATION_REGISTRY[index]
            rows.append({
                "operation_id": spec.operation_id,
                "operation": spec.name,
                "status": status,
                "evidence_level": spec.evidence_level,
                "functional_output": functional_output,
                "biological_output": biological_output or {},
            })

        added: list[dict[str, Any]] = []
        for episode in request.episodes:
            added.append(self.add_episode(episode))

        # ES01: use query cues and event indices to reactivate candidates without forcing certainty.
        if request.target_episode_id is not None:
            if request.target_episode_id not in self.episodic_store:
                raise KeyError(f"unknown target_episode_id: {request.target_episode_id}")
            candidate_ids = [request.target_episode_id]
        elif request.candidate_episode_ids:
            missing = set(request.candidate_episode_ids) - set(self.episodic_store)
            if missing:
                raise KeyError(f"unknown candidate episode IDs: {sorted(missing)}")
            candidate_ids = list(request.candidate_episode_ids)
        else:
            candidate_ids = list(self.episodic_store)
        cue_scores: dict[str, dict[str, float]] = {}
        for episode_id in candidate_ids:
            episode = self.episodic_store[episode_id].episode
            cue_scores[episode_id] = cue_similarity(
                request.cue_features, request.cue_relations, episode
            )
        raw_scores = []
        for episode_id in candidate_ids:
            record_state = self.episodic_store[episode_id]
            access_mean = (float(np.mean(list(record_state.feature_accessibility.values())))
                           if record_state.feature_accessibility else 0.5)
            cue_score = cue_scores[episode_id]["overall"]
            score = (0.82 * cue_score + 0.18 * access_mean) if (
                request.cue_features or request.cue_relations
            ) else 0.5 * access_mean + 0.5
            raw_scores.append(score)
        probabilities = softmax(raw_scores, self.config.cue_selection_temperature) if raw_scores else (
            np.asarray([], dtype=np.float64))
        order = sorted(range(len(candidate_ids)), key=lambda idx: raw_scores[idx], reverse=True)
        selected_ids = [candidate_ids[idx] for idx in order[:self.config.maximum_reactivated_episodes]]
        if request.target_episode_id is not None:
            selected_ids = [request.target_episode_id]
        selection_report = {
            "candidate_count": len(candidate_ids),
            "selected_episode_ids": selected_ids,
            "cue_scores": cue_scores,
            "selection_probabilities": {
                episode_id: float(probabilities[index])
                for index, episode_id in enumerate(candidate_ids)
            } if candidate_ids else {},
            "selection_method": ("targeted_episode" if request.target_episode_id else
                                 "cue_competition" if request.cue_features or request.cue_relations
                                 else "accessibility_weighted_candidate_selection"),
        }
        selected_patterns: dict[str, list[float]] = {}
        completion_reports: dict[str, dict[str, Any]] = {}
        for episode_id in selected_ids:
            completed, report = self.hippocampus.complete(
                episode_id, request.cue_features, request.cue_relations
            )
            selected_patterns[episode_id] = completed.tolist()
            completion_reports[episode_id] = report
        record(0, selection_report, {
            "reactivated_event_count": len(selected_ids),
            "hippocampal_pattern_completion": completion_reports,
        })

        # ES02: preserve feature groups and confidence; absent values are not fabricated.
        reconstructed: dict[str, Any] = {}
        for episode_id in selected_ids:
            episode = self.episodic_store[episode_id].episode
            groups = episode.feature_groups()
            reconstructed[episode_id] = {
                "feature_groups": groups,
                "feature_confidence": episode.feature_confidence,
                "source_type": episode.source_type,
                "source_confidence": episode.source_confidence,
                "overall_confidence": episode.overall_confidence,
                "unknown_features_are_absent_from_maps": True,
            }
        record(1, {"reconstructed_episode_count": len(reconstructed),
                  "feature_group_names": list(FEATURE_GROUPS),
                  "reconstructed": reconstructed})

        # ES03: preserve per-event graph bindings and source-supported relations.
        event_graphs: dict[str, dict[str, Any]] = {}
        for episode_id in selected_ids:
            episode = self.episodic_store[episode_id].episode
            nodes = []
            for group, values in episode.feature_groups().items():
                for feature, value in values.items():
                    nodes.append({"id": f"{group}:{feature}", "group": group,
                                  "feature": feature, "value": value,
                                  "source_type": episode.source_type})
            edges = [{
                "subject": relation.subject, "predicate": relation.predicate,
                "object": relation.object, "confidence": relation.confidence,
                "source_type": relation.source_type,
            } for relation in episode.relations]
            event_graphs[episode_id] = {
                "episode_id": episode_id, "nodes": nodes, "relations": edges,
                "binding_quality": float(np.mean([item.confidence for item in episode.relations]))
                if episode.relations else None,
                "event_identity_preserved": True,
            }
        record(2, {"event_graphs": event_graphs,
                  "binding_rule": "relations remain attached to original episode_id"})

        # ES04: collect all members of selected concepts and align their feature/relation patterns.
        concept_ids = sorted({self.episode_concept_ids[item] for item in selected_ids})
        related_ids = sorted(
            episode_id for episode_id, record_state in self.episodic_store.items()
            if record_state.concept_id in concept_ids
        )
        related_episodes = [self.episodic_store[item].episode for item in related_ids]
        similarity_matrix: dict[str, dict[str, float]] = {}
        for left in related_episodes:
            similarity_matrix[left.episode_id] = {}
            for right in related_episodes:
                if left.episode_id == right.episode_id:
                    similarity_matrix[left.episode_id][right.episode_id] = 1.0
                elif right.episode_id in similarity_matrix and left.episode_id in similarity_matrix[right.episode_id]:
                    similarity_matrix[left.episode_id][right.episode_id] = (
                        similarity_matrix[right.episode_id][left.episode_id])
                else:
                    similarity_matrix[left.episode_id][right.episode_id] = (
                        episode_similarity(left, right)["overall"])
        record(3, {"selected_episode_ids": selected_ids, "related_episode_ids": related_ids,
                   "concept_ids": concept_ids, "similarity_matrix": similarity_matrix,
                   "related_episode_count": len(related_ids)})

        # ES05-ES06: estimate invariant features, contrastive diagnosticity, and recurring relations.
        semantic_representations: list[dict[str, Any]] = []
        concept_statistics: dict[str, dict[str, Any]] = {}
        concept_relations: dict[str, dict[str, Any]] = {}
        all_episodes = [record_state.episode for record_state in self.episodic_store.values()]
        for concept_id in concept_ids:
            members = [self.episodic_store[item].episode for item in related_ids
                       if self.episode_concept_ids[item] == concept_id]
            contrasts = [item for item in all_episodes
                         if self.episode_concept_ids[item.episode_id] != concept_id]
            stats = estimate_feature_statistics(
                members, [concept_id] * len(members), self.config,
                contrasts, [self.episode_concept_ids[item.episode_id] for item in contrasts],
            )
            relations = estimate_relation_statistics(members)
            concept_statistics[concept_id] = stats
            concept_relations[concept_id] = relations
        record(4, {"feature_statistics": concept_statistics,
                   "diagnostic_contrast_count": len(all_episodes) - len(related_episodes),
                   "feature_interpretation": "missing_is_unknown_explicit_zero_is_observed_absence"})
        record(5, {"recurring_relation_statistics": concept_relations,
                   "relation_types_remain_typed": True})

        # ES07: construct one semantic gist per selected concept from well-supported features/relations.
        gist_by_concept: dict[str, dict[str, Any]] = {}
        for concept_id in concept_ids:
            members = [self.episodic_store[item].episode for item in related_ids
                       if self.episode_concept_ids[item] == concept_id]
            stats = concept_statistics[concept_id]
            gist_groups: dict[str, dict[str, float]] = {group: {} for group in FEATURE_GROUPS}
            feature_support: dict[str, dict[str, Any]] = {}
            provisional: dict[str, Any] = {}
            for key, row in stats.items():
                group, feature = key.split(":", 1)
                supported = (row["is_invariant"] and
                             row["effective_support"] >= self.config.minimum_effective_support)
                if supported:
                    gist_groups[group][feature] = float(row["prevalence"])
                else:
                    provisional[key] = row
                feature_support[key] = row
            relation_gist = {
                key: row for key, row in concept_relations[concept_id].items()
                if row["effective_support"] >= self.config.minimum_effective_support
                and row["prevalence"] >= 0.50
            }
            gist_by_concept[concept_id] = {
                "concept_id": concept_id, "features": gist_groups,
                "relations": relation_gist, "feature_support": feature_support,
                "provisional_or_unstable_features": provisional,
                "supporting_episode_ids": [item.episode_id for item in members],
                "supporting_episode_count": len(members),
                "semantic_status": "supported_gist" if (
                    any(gist_groups[group] for group in FEATURE_GROUPS) or relation_gist
                ) else "provisional_insufficient_repeated_evidence",
            }
        record(6, {"semantic_gists": gist_by_concept,
                   "gist_requires_repeated_reliable_support": True})

        # ES08: retain context when useful; down-weight only low-utility spatial/temporal features.
        context_weight_reports: dict[str, dict[str, float]] = {}
        for concept_id, gist in gist_by_concept.items():
            weights: dict[str, float] = {}
            for key, row in gist["feature_support"].items():
                group, _ = key.split(":", 1)
                if group in {"spatial", "temporal"}:
                    weight = semantic_context_weight(
                        row["diagnosticity"], row["context_importance"],
                        row["prevalence"], self.config,
                    )
                else:
                    weight = 1.0
                weights[key] = weight
            gist["context_weights"] = weights
            context_weight_reports[concept_id] = weights
        record(7, {"context_weights": context_weight_reports,
                   "context_is_selectively_abstracted": True})

        # ES09: keep full event records and semantic summaries as independent linked stores.
        episodic_representation = {
            "episode_count": len(self.episodic_store),
            "event_records": {
                episode_id: self._episodic_snapshot(episode_id)
                for episode_id in related_ids
            },
            "event_specific_bindings_preserved": True,
        }
        record(8, {"episodic_store_count": len(self.episodic_store),
                   "semantic_concept_count": len(self.semantic_store) + len(concept_ids),
                   "episodic_records_mutated_by_abstraction": False})

        # ES10: project semantic gist into distributed cortical-like populations.
        integration_reports: dict[str, dict[str, Any]] = {}
        for concept_id, gist in gist_by_concept.items():
            pattern, population_report = self.cortex.encode_gist(
                gist["features"], gist["relations"], gist["context_weights"]
            )
            integration = self.cortex.integrate(concept_id, pattern)
            integration_reports[concept_id] = {
                "population": population_report, "integration": integration,
                "cortical_pattern": pattern.tolist(),
            }
            gist["distributed_cortical_pattern"] = pattern.tolist()
        record(9, {"distributed_semantic_integration": integration_reports},
               {"semantic_population": self.cortex.snapshot(),
                "semantic_model_scope": "distributed_rate_population_surrogate"})

        # ES11: preserve counterexamples, uncertain features, and distinct concept identities.
        distinctions: dict[str, dict[str, Any]] = {}
        for concept_id, gist in gist_by_concept.items():
            members = [self.episodic_store[item].episode for item in related_ids
                       if self.episode_concept_ids[item] == concept_id]
            exceptions: dict[str, Any] = {}
            conflicts: dict[str, Any] = {}
            for key, row in gist["provisional_or_unstable_features"].items():
                group, feature = key.split(":", 1)
                feature_evidence = {
                    item.episode_id: getattr(item, f"{group}_features").get(feature)
                    for item in members if feature in getattr(item, f"{group}_features")
                }
                if row["is_conflicting"]:
                    conflicts[key] = {"statistics": row, "episode_evidence": feature_evidence}
                else:
                    exceptions[key] = {"statistics": row, "episode_evidence": feature_evidence}
            gist["exceptions"] = exceptions
            gist["conflicts"] = conflicts
            distinctions[concept_id] = {
                "exceptions": exceptions, "conflicts": conflicts,
                "automatic_cross_concept_merge": False,
                "distinct_episode_ids_preserved": [item.episode_id for item in members],
            }
            self.semantic_store[concept_id] = gist
        record(10, {"concept_distinctions": distinctions,
                    "ambiguous_features_omitted_from_core_gist": True,
                    "concepts_auto_merged": False})

        # ES12: model access decay separately from trace existence and strengthen reactivated traces.
        accessibility_report = self._update_accessibility(request, selected_ids)
        record(11, accessibility_report, {
            "episode_record_count": len(self.episodic_store),
            "episode_content_deleted": False,
            "accessibility_model": "separate_feature_level_accessibility_state",
        })

        # ES13: choose the memory track requested by the task.
        primary_episode_id = request.target_episode_id or (
            selected_ids[0] if selected_ids else None
        )
        primary_concept_id = request.target_concept_id
        if primary_concept_id is None and primary_episode_id is not None:
            primary_concept_id = self.episode_concept_ids.get(primary_episode_id)
        if primary_concept_id is None and concept_ids:
            primary_concept_id = max(
                concept_ids,
                key=lambda item: len(self.semantic_store.get(item, {}).get("supporting_episode_ids", [])),
            )
        if primary_concept_id is not None and primary_concept_id not in self.episode_concept_ids.values():
            raise KeyError(f"unknown target_concept_id: {primary_concept_id}")
        retrieval_policy = {
            "retrieval_type": request.retrieval_type,
            "primary_episode_id": primary_episode_id,
            "primary_concept_id": primary_concept_id,
            "episodic_track_used": request.retrieval_type in {"episodic", "mixed"},
            "semantic_track_used": request.retrieval_type in {"semantic", "mixed"},
        }
        record(12, retrieval_policy, {
            "episodic_support": "hippocampal_index_and_event_specific_binding",
            "semantic_support": "distributed_cortical_concept_pattern",
        })

        # ES14: reconstruct only supported details and attach their event/source provenance.
        retrieved_result = self._build_retrieval_result(
            request, primary_episode_id, primary_concept_id, selected_ids
        )
        provenance = self._attach_provenance(retrieved_result)
        record(13, {"retrieved_result": retrieved_result,
                    "provenance": provenance,
                    "unsupported_gap_filling_performed": False})

        # ES15: reactivation and newly supplied evidence refine access and linked knowledge, not old facts.
        self._record_reactivation(selected_ids)
        refinement = {
            "new_episode_ids": [row["episode_id"] for row in added
                                if row["status"] == "registered"],
            "reactivated_episode_ids": selected_ids,
            "reactivation_counts": {
                episode_id: self.episodic_store[episode_id].reactivation_count
                for episode_id in selected_ids
            },
            "semantic_concept_support": {
                concept_id: self.semantic_store[concept_id]["supporting_episode_ids"]
                for concept_id in concept_ids
            },
            "earlier_event_content_rewritten": False,
            "updates_are_evidence_additions_or_access_changes": True,
        }
        record(14, refinement)

        # ES16: evaluate generalization, specificity, source integrity, and externally grounded calibration.
        quality = self._evaluate_quality(
            concept_ids, related_ids, selected_ids, retrieved_result, provenance
        )
        record(15, quality)
        if len(rows) != len(OPERATION_REGISTRY):
            raise RuntimeError("operation trace does not cover all 16 registered stages")

        uncertainty = {
            "cue_selection_entropy": float(entropy(list(selection_report["selection_probabilities"].values())))
            if selection_report["selection_probabilities"] else None,
            "provisional_features": {
                concept_id: gist["provisional_or_unstable_features"]
                for concept_id, gist in gist_by_concept.items()
            },
            "conflicting_features": {
                concept_id: gist.get("conflicts", {})
                for concept_id, gist in gist_by_concept.items()
            },
            "source_confidence_is_not_retrieval_accuracy": True,
            "missing_features_interpreted_as": "unknown_not_absent",
        }
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger = self._log_sqlite(rows, elapsed_ms)
        biological = {
            "hippocampal_index_network": self.hippocampus.snapshot(),
            "distributed_semantic_network": self.cortex.snapshot(),
            "model_scope": "dual_population_surrogate_not_cell_level_digital_twin",
            "literal_neuronal_arithmetic_claimed": False,
            "episodic_trace_erasure_inferred_from_low_accessibility": False,
            "semantic_and_episodic_tracks_coexist": True,
        }
        return MemoryTransformationResult(
            request_id=request.request_id,
            operation_trace=rows,
            selected_episode_ids=selected_ids,
            related_episode_ids=related_ids,
            concept_ids=concept_ids,
            episodic_representation=episodic_representation,
            semantic_representations=[self.semantic_store[item] for item in concept_ids],
            schema_links=refinement.get("schema_links", []),
            retrieved_result=retrieved_result,
            uncertainty=uncertainty,
            accessibility=accessibility_report,
            quality=quality,
            biological=biological,
            logger=logger,
        )

    def _episodic_snapshot(self, episode_id: str) -> dict[str, Any]:
        """Return an event's complete original evidence and separate current access estimates."""
        state = self.episodic_store[episode_id]
        return {
            "episode": state.episode.to_dict(),
            "concept_id": state.concept_id,
            "hippocampal_index": state.hippocampal_index.tolist(),
            "feature_accessibility": dict(state.feature_accessibility),
            "retrieval_count": state.retrieval_count,
            "reactivation_count": state.reactivation_count,
            "stored_content_retained": True,
        }

    def _update_accessibility(
        self, request: MemoryTransformationRequest, selected_ids: Sequence[str]
    ) -> dict[str, Any]:
        """Apply time/interference decay and reactivation gain without deleting episodic details."""
        per_episode: dict[str, Any] = {}
        now = request.now_ms
        for episode_id, state in self.episodic_store.items():
            episode = state.episode
            elapsed_hours = 0.0
            if now is not None and state.last_access_update_ms is not None:
                elapsed_hours = max(0.0, now - state.last_access_update_ms) / 3_600_000.0
            before = dict(state.feature_accessibility)
            if elapsed_hours > 0.0:
                for feature_key, access in list(state.feature_accessibility.items()):
                    group, feature = feature_key.split(":", 1)
                    competing = 0
                    for other_id, other in self.episodic_store.items():
                        if other_id == episode_id or other.concept_id != state.concept_id:
                            continue
                        if feature in getattr(other.episode, f"{group}_features"):
                            competing += 1
                    interference = competing / max(1, sum(
                        1 for item in self.episodic_store.values() if item.concept_id == state.concept_id
                    ) - 1)
                    state.feature_accessibility[feature_key] = accessibility_decay(
                        access, elapsed_hours, self._ACCESS_DECAY_PER_HOUR[group],
                        interference, self.config.interference_rate_per_hour,
                    )
            if now is not None:
                state.last_access_update_ms = max(now, state.last_access_update_ms or now)
            is_reactivated = episode_id in selected_ids or episode_id in request.rehearsal_episode_ids
            if is_reactivated:
                boost = self.config.accessibility_reactivation_boost * request.reactivation_gain
                for feature_key, access in list(state.feature_accessibility.items()):
                    state.feature_accessibility[feature_key] = float(np.clip(access + boost, 0.0, 1.0))
            per_episode[episode_id] = {
                "elapsed_hours": elapsed_hours,
                "reactivated": is_reactivated,
                "before": before,
                "after": dict(state.feature_accessibility),
                "stored_feature_count": sum(len(values) for values in episode.feature_groups().values()),
            }
        return {"per_episode": per_episode, "time_advanced": now is not None,
                "trace_content_deleted": False,
                "accessibility_is_not_trace_existence": True}

    def _build_retrieval_result(
        self,
        request: MemoryTransformationRequest,
        episode_id: str | None,
        concept_id: str | None,
        selected_ids: Sequence[str],
    ) -> dict[str, Any]:
        """Build episodic, semantic, or combined retrieval output without inventing missing details."""
        result: dict[str, Any] = {
            "retrieval_type": request.retrieval_type,
            "status": "completed",
            "episodic_detail": None,
            "semantic_knowledge": None,
            "supporting_episode_ids": [],
        }
        if request.retrieval_type in {"episodic", "mixed"}:
            if episode_id is None or episode_id not in self.episodic_store:
                result["status"] = "no_episode_available"
            else:
                state = self.episodic_store[episode_id]
                result["episodic_detail"] = {
                    "episode_id": episode_id,
                    "feature_groups": state.episode.feature_groups(),
                    "relations": [relation.to_dict() for relation in state.episode.relations],
                    "source_type": state.episode.source_type,
                    "source_id": state.episode.source_id,
                    "source_confidence": state.episode.source_confidence,
                    "feature_accessibility": dict(state.feature_accessibility),
                    "reconstruction_label": "event_specific_reinstatement_candidate",
                }
                result["supporting_episode_ids"].append(episode_id)
        if request.retrieval_type in {"semantic", "mixed"}:
            concept = self.semantic_store.get(concept_id) if concept_id else None
            if concept is None:
                result["status"] = "semantic_gist_not_available"
            else:
                result["semantic_knowledge"] = {
                    "concept_id": concept_id,
                    "features": concept["features"],
                    "relations": concept["relations"],
                    "semantic_status": concept["semantic_status"],
                    "feature_support": concept["feature_support"],
                    "exceptions": concept.get("exceptions", {}),
                    "conflicts": concept.get("conflicts", {}),
                }
                result["supporting_episode_ids"] = sorted(set(
                    result["supporting_episode_ids"] + concept["supporting_episode_ids"]
                ))
        result["query_task_goal"] = request.task_goal
        result["source_separation_rule"] = (
            "semantic_regularities do not establish exact details of a particular occurrence"
        )
        result["candidate_episode_ids"] = list(selected_ids)
        return result

    def _attach_provenance(self, result: dict[str, Any]) -> dict[str, Any]:
        """Attach source labels and confidence to episodic and generalized retrieved properties."""
        report: dict[str, Any] = {"episodic_features": {}, "semantic_features": {}}
        episode_detail = result.get("episodic_detail")
        if episode_detail:
            episode_id = episode_detail["episode_id"]
            episode = self.episodic_store[episode_id].episode
            for group, features in episode.feature_groups().items():
                for feature, value in features.items():
                    confidence = episode.feature_confidence.get(group, {}).get(feature, 1.0)
                    key = f"{group}:{feature}"
                    report["episodic_features"][key] = {
                        "source_type": episode.source_type,
                        "source_id": episode.source_id,
                        "confidence": episode.overall_confidence * episode.source_confidence * confidence,
                        "accessibility": self.episodic_store[episode_id].feature_accessibility.get(key, 1.0),
                        "evidence_episode_ids": [episode_id],
                    }
        semantic = result.get("semantic_knowledge")
        if semantic:
            concept_id = semantic["concept_id"]
            members = [self.episodic_store[item].episode
                       for item in self.semantic_store[concept_id]["supporting_episode_ids"]]
            for key, support in semantic["feature_support"].items():
                group, feature = key.split(":", 1)
                supporting = [item for item in members
                              if feature in getattr(item, f"{group}_features")]
                report["semantic_features"][key] = {
                    **source_provenance(supporting, key),
                    "prevalence": support["prevalence"],
                    "diagnosticity": support["diagnosticity"],
                    "effective_support": support["effective_support"],
                    "generalized_claim_is_not_single_episode_fact": len(supporting) > 1,
                }
            report["semantic_relations"] = {
                key: {
                    "status": "generalized_relation" if len(row["supporting_episode_ids"]) > 1
                    else "single_episode_relation",
                    "supporting_episode_ids": row["supporting_episode_ids"],
                    "source_types": row["source_types"],
                    "mean_reliability": row["mean_reliability"],
                } for key, row in semantic["relations"].items()
            }
        result["provenance"] = report
        return report

    def _record_reactivation(self, selected_ids: Sequence[str]) -> None:
        """Increment retrieval metadata while leaving immutable event content unchanged."""
        for episode_id in selected_ids:
            state = self.episodic_store[episode_id]
            state.retrieval_count += 1
            state.reactivation_count += 1

    def _evaluate_quality(
        self,
        concept_ids: Sequence[str],
        related_ids: Sequence[str],
        selected_ids: Sequence[str],
        retrieved_result: Mapping[str, Any],
        provenance: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Evaluate generalization, event discrimination, provenance, and available calibration evidence."""
        group_scores: dict[str, Any] = {}
        specificity_scores: list[float] = []
        for concept_id in concept_ids:
            members = [self.episodic_store[item].episode for item in related_ids
                       if self.episode_concept_ids[item] == concept_id]
            group_scores[concept_id] = leave_one_out_gist_similarity(members)
            pairs = [
                episode_similarity(members[i], members[j])["overall"]
                for i in range(len(members)) for j in range(i + 1, len(members))
            ]
            if pairs:
                specificity_scores.append(float(np.mean([1.0 - value for value in pairs])))
        semantic_features = provenance.get("semantic_features", {})
        episodic_features = provenance.get("episodic_features", {})
        labelled_count = len(semantic_features) + len(episodic_features)
        source_labelled_count = sum(
            bool(row.get("status") or row.get("source_type") or row.get("source_types"))
            for row in list(semantic_features.values()) + list(episodic_features.values())
        )
        source_coverage = (source_labelled_count / labelled_count) if labelled_count else 1.0
        generalization_values = [
            row["mean_similarity"] for row in group_scores.values()
            if row["mean_similarity"] is not None
        ]
        generalization = float(np.mean(generalization_values)) if generalization_values else None
        specificity = float(np.mean(specificity_scores)) if specificity_scores else None
        # No external correctness labels are part of this request, so calibration is not fabricated.
        calibration = brier_score([], [])
        status = "evaluated" if related_ids else "insufficient_memory_evidence"
        return {
            "status": status,
            "held_out_generalization": generalization,
            "leave_one_out_by_concept": group_scores,
            "episodic_distinguishability": specificity,
            "source_provenance_coverage": float(source_coverage),
            "confidence_brier_score": calibration,
            "calibration_status": "not_measured_no_external_outcomes",
            "selected_episode_count": len(selected_ids),
            "related_episode_count": len(related_ids),
            "retrieval_track": retrieved_result.get("retrieval_type"),
            "refinement_signal": bool(
                any(self.semantic_store.get(item, {}).get("conflicts")
                    or self.semantic_store.get(item, {}).get("exceptions")
                    for item in concept_ids)
            ),
            "causal_or_permanent_storage_claimed": False,
        }

    def snapshot(self) -> dict[str, Any]:
        """Return dual-store sizes, retrieval metadata, concepts, and biological model diagnostics."""
        return {
            "episode_count": len(self.episodic_store),
            "concept_count": len(self.semantic_store),
            "episode_concept_ids": dict(self.episode_concept_ids),
            "retrieval_counts": {
                episode_id: state.retrieval_count
                for episode_id, state in self.episodic_store.items()
            },
            "biological": {
                "hippocampal_index_network": self.hippocampus.snapshot(),
                "distributed_semantic_network": self.cortex.snapshot(),
            },
        }
