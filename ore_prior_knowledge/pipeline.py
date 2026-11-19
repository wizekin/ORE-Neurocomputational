"""End-to-end 14-stage Point 02 prior-knowledge integration processor."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from .biology import BiologicalPriorKnowledgeNetwork
from .contracts import (
    KnowledgeNode, KnowledgeRelation, KnowledgeSchema, PriorKnowledgeConfig,
    PriorKnowledgeRequest, PriorKnowledgeResult, PriorKnowledgeState,
)
from .functional import (
    align_analogies, build_integrated_representation, build_retrieval_routes, choose_integration_strategy,
    compare_knowledge, evaluate_schema_fit, extract_new_representation, generate_retrieval_cues,
    rank_candidates, retrieve_candidates, stable_id, test_retrieval_routes,
)
from .registry import OPERATION_REGISTRY


def _append_operation(
    trace: list[dict[str, Any]], index: int, functional_output: Mapping[str, Any],
    biological_output: Mapping[str, Any] | None = None,
) -> None:
    """Append a source-ordered operation with its mathematical and biological metadata."""
    spec = OPERATION_REGISTRY[index]
    trace.append({"step": index + 1, "source_step": spec.source_step,
                  "operation_id": spec.operation_id, "operation": spec.name,
                  "status": "completed", "purpose": spec.purpose,
                  "evidence_level": spec.evidence_level,
                  "mathematical_form": spec.mathematical_form,
                  "functional_output": dict(functional_output),
                  "biological_output": dict(biological_output or {})})


def _merge_seed_knowledge(state: PriorKnowledgeState, request: PriorKnowledgeRequest) -> None:
    """Add supplied seed graph items while rejecting conflicting definitions of existing IDs."""
    for node in request.knowledge_nodes:
        existing = state.knowledge_nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"conflicting prior-knowledge node definition: {node.node_id}")
        state.knowledge_nodes[node.node_id] = node
    known_ids = set(state.knowledge_nodes)
    for relation in request.knowledge_relations:
        if relation.source_id not in known_ids or relation.target_id not in known_ids:
            raise ValueError("knowledge relation references an unknown node")
        if relation not in state.knowledge_relations:
            state.knowledge_relations.append(relation)
    for schema in request.schemas:
        existing_schema = state.schemas.get(schema.schema_id)
        if existing_schema is not None and existing_schema != schema:
            if existing_schema.version > schema.version:
                # A repeated baseline seed must not overwrite a schema learned by this processor.
                continue
            if schema.version > existing_schema.version:
                # An explicitly newer input schema can advance the learner's stored version.
                state.schemas[schema.schema_id] = schema
                continue
            raise ValueError(f"conflicting schema definition: {schema.schema_id}")
        state.schemas[schema.schema_id] = schema


class PriorKnowledgeProcessor:
    """Coordinate retrieval, comparison, schema control, episodic binding, and consolidation surrogates."""

    def __init__(
        self, learner_id: str, config: PriorKnowledgeConfig | None = None,
        state: PriorKnowledgeState | None = None,
        biological: BiologicalPriorKnowledgeNetwork | None = None,
    ) -> None:
        """Initialize one learner's knowledge graph, configuration, and biological network."""
        if not learner_id or not learner_id.strip():
            raise ValueError("learner_id must be non-empty")
        self.learner_id = learner_id.strip()
        self.config = config or PriorKnowledgeConfig()
        self.state = state or PriorKnowledgeState(self.learner_id)
        if self.state.learner_id != self.learner_id:
            raise ValueError("processor learner_id must match state learner_id")
        self.biological = biological or BiologicalPriorKnowledgeNetwork(
            self.config.dimensions, self.config.learning_rate
        )
        if self.biological.dimensions != self.config.dimensions:
            raise ValueError("biological dimensions must match configuration")

    def process(
        self, request: PriorKnowledgeRequest, state: PriorKnowledgeState | None = None,
    ) -> PriorKnowledgeResult:
        """Connect new material to prior knowledge while preserving novelty, contradictions, and provenance."""
        if request.learner_id != self.learner_id:
            raise ValueError("request learner_id must match processor learner_id")
        active = state if state is not None else self.state
        if active.learner_id != self.learner_id:
            raise ValueError("supplied state learner_id must match processor learner_id")
        _merge_seed_knowledge(active, request)
        unit = request.new_input
        context = {**dict(request.current_context), **dict(unit.context)}
        trace: list[dict[str, Any]] = []
        bio = self.biological
        config = self.config
        context_frame = {
            "learner_id": self.learner_id, "modality": unit.modality,
            "task": context.get("task", "understand_and_integrate"),
            "goal": context.get("goal", "connect new information to prior knowledge"),
            "topic": context.get("topic"), "active_context": context,
            "attention_available": min(1.0, max(0.0, float(request.learner_state.get("attention", 0.8)))),
            "working_memory_available": min(1.0, max(0.0, float(request.learner_state.get("working_memory", 0.8)))),
            "input_load": min(1.0, len(unit.text) / 1200.0),
            "context_is_provisional": True,
        }
        capacity_gate = max(0.0, min(1.0,
            0.45 * context_frame["attention_available"]
            + 0.35 * context_frame["working_memory_available"]
            - 0.20 * context_frame["input_load"]))
        _append_operation(trace, 0, {"context_frame": context_frame,
                                     "processing_gate": capacity_gate},
                          bio.process_step(1, [str(x) for x in context.values() if x is not None],
                                           0.0, unit.reliability))

        representation = extract_new_representation(unit)
        representation["context_frame"] = context_frame
        _append_operation(trace, 1, {"new_representation": representation},
                          bio.process_step(2, representation["content_tokens"], 0.0, unit.reliability))

        cues = generate_retrieval_cues(representation, context)
        _append_operation(trace, 2, {"retrieval_cues": cues,
                                     "cue_count": len(cues),
                                     "cue_types": sorted({item["cue_type"] for item in cues})},
                          bio.process_step(3, [item["cue"] for item in cues], 0.05, unit.reliability))

        candidates = retrieve_candidates(
            cues, active.knowledge_nodes, active.knowledge_relations,
            limit=config.retrieval_limit, max_hops=config.max_spread_hops,
            temperature=config.retrieval_temperature,
        )
        _append_operation(trace, 3, {"candidate_memories": candidates,
                                     "candidate_count": len(candidates)},
                          bio.process_step(4, [item["label"] for item in candidates],
                                           0.15, unit.reliability))

        ranked = rank_candidates(candidates, representation, context, active.knowledge_nodes)
        _append_operation(trace, 4, {"ranked_candidates": ranked,
                                     "ranking_factors": ["activation", "relevance", "reliability", "context_fit"]},
                          bio.process_step(5, [item["label"] for item in ranked[:5]],
                                           0.10, unit.reliability))

        comparisons = compare_knowledge(
            representation, ranked, active.knowledge_nodes, active.knowledge_relations
        )
        _append_operation(trace, 5, {"comparison_profile": comparisons,
                                     "contradiction_count": sum(len(x["contradictions"]) for x in comparisons)},
                          bio.process_step(6, representation["content_tokens"],
                                           0.20, unit.reliability, prediction_error=0.1))

        analogy_maps = align_analogies(representation, comparisons, ranked)
        _append_operation(trace, 6, {"analogy_maps": analogy_maps,
                                     "analogy_limits_preserved": True},
                          bio.process_step(7, representation["content_tokens"],
                                           0.20, unit.reliability))

        fit = evaluate_schema_fit(
            representation, active.schemas,
            conflict_threshold=config.conflict_threshold,
            novelty_threshold=config.novelty_threshold,
            integration_threshold=config.integration_threshold,
        )
        comparison_conflicts = sum(len(item["contradictions"]) for item in comparisons)
        if comparison_conflicts:
            fit["contradiction"] = max(
                fit["contradiction"], min(1.0, 0.60 * comparison_conflicts)
            )
            if fit["contradiction"] >= config.conflict_threshold:
                fit["fit_state"] = "schema_conflict"
        prediction_error = max(fit["novelty"], fit["contradiction"])
        schema_selection = bio.mPFC.select([item["fit"] for item in fit["schema_candidates"]])
        fit["biological_schema_selection"] = schema_selection
        _append_operation(trace, 7, {"schema_fit": fit,
                                     "prediction_error": prediction_error},
                          bio.process_step(8, representation["content_tokens"],
                                           fit["novelty"], unit.reliability, prediction_error))

        plan = choose_integration_strategy(
            fit, unit.reliability, config.minimum_reliability,
            revision_reliability_threshold=config.schema_update_threshold,
        )
        _append_operation(trace, 8, {"integration_plan": plan},
                          bio.process_step(9, representation["content_tokens"],
                                           fit["novelty"], unit.reliability, prediction_error))

        integrated = build_integrated_representation(representation, ranked, comparisons,
                                                     plan["strategy"])
        concept_id = stable_id("p02-knowledge-", unit.unit_id + "|" + unit.text)
        features = dict(representation["features"])
        separate_episode = plan["strategy"] in {"store_separately", "revise_candidate_schema", "defer"}
        active.knowledge_nodes.setdefault(concept_id, KnowledgeNode(
            node_id=concept_id, label="; ".join(representation["concepts"][:4]),
            kind="episode" if separate_episode else "concept",
            features=features, context=context, reliability=unit.reliability,
            memory_type="episodic" if separate_episode else "semantic",
            provenance=unit.provenance,
        ))
        bound_edges: list[dict[str, Any]] = []
        if plan["strategy"] in {"integrate", "complement_schema"}:
            relation_label = "related_to" if plan["strategy"] == "integrate" else "complements"
            for candidate in ranked[:5]:
                weight = max(candidate["rank_score"], 0.05)
                edge = KnowledgeRelation(
                    source_id=concept_id, predicate=relation_label,
                    target_id=candidate["node_id"], strength=weight,
                    confidence=unit.reliability, evidential_status="inferred",
                    context=context,
                )
                if edge not in active.knowledge_relations:
                    active.knowledge_relations.append(edge)
                bound_edges.append({"target_id": candidate["node_id"],
                                    "relation": relation_label, "strength": weight,
                                    "evidential_status": "inferred"})
        integrated["knowledge_node_id"] = concept_id
        integrated["bound_relations"] = bound_edges
        _append_operation(trace, 9, {"bound_memory": integrated,
                                     "new_association_count": len(bound_edges)},
                          bio.process_step(10, representation["content_tokens"],
                                           fit["novelty"], unit.reliability, prediction_error))

        schema_updates: list[dict[str, Any]] = []
        best = fit.get("best_schema")
        if best and plan["strategy"] in {"integrate", "complement_schema"}:
            schema = active.schemas[best["schema_id"]]
            if unit.reliability >= config.schema_update_threshold:
                updated_expectations = dict(schema.expectations)
                for feature, value in features.items():
                    old = updated_expectations.get(feature, 0.0)
                    updated_expectations[feature] = min(1.0, (1.0 - config.learning_rate) * old
                                                        + config.learning_rate * value)
                revised = replace(schema, expectations=updated_expectations,
                                  confidence=max(schema.confidence, min(1.0, 0.8 * schema.confidence
                                                                       + 0.2 * unit.reliability)),
                                  version=schema.version + 1,
                                  supporting_episode_ids=tuple(dict.fromkeys(
                                      schema.supporting_episode_ids + (unit.unit_id,))))
                active.schemas[schema.schema_id] = revised
                schema_updates.append({"schema_id": schema.schema_id,
                                       "update_type": "cautious_supported_refinement",
                                       "previous_version": schema.version,
                                       "new_version": revised.version,
                                       "automatic_overwrite": False,
                                       "changed_expectations": sorted(features)})
            else:
                schema_updates.append({"schema_id": schema.schema_id,
                                       "update_type": "candidate_only_insufficient_evidence",
                                       "automatic_overwrite": False})
        elif plan["strategy"] == "revise_candidate_schema":
            schema_updates.append({"schema_id": best["schema_id"] if best else None,
                                   "update_type": "revision_proposal_requires_corroboration",
                                   "proposal": {"new_features": features,
                                                "new_relations": representation["relations"]},
                                   "automatic_overwrite": False})
        _append_operation(trace, 10, {"schema_updates": schema_updates,
                                      "schema_state_count": len(active.schemas)},
                          bio.process_step(11, representation["content_tokens"],
                                           fit["novelty"], unit.reliability, prediction_error))

        new_episode: dict[str, Any] | None = None
        if plan["strategy"] in {"store_separately", "revise_candidate_schema", "defer"}:
            episode_id = stable_id("p02-episode-", unit.unit_id + "|" + unit.text + "|" + str(len(active.episodes)))
            new_episode = {
                "episode_id": episode_id, "unit_id": unit.unit_id,
                "content": unit.text, "concepts": list(representation["concepts"]),
                "context": context, "provenance": unit.provenance,
                "schema_expectation": best if best else None,
                "fit_state": fit["fit_state"], "strategy": plan["strategy"],
                "reliability": unit.reliability,
                "status": "distinct_episode_pending_later_integration",
                "observed_inference_distinction_preserved": True,
            }
            active.episodes.append(new_episode)
        _append_operation(trace, 11, {"new_episode": new_episode,
                                      "separate_episode_created": new_episode is not None},
                          bio.process_step(12, representation["content_tokens"],
                                           fit["novelty"], unit.reliability, prediction_error))

        gate = bio.learning_gate.compute(fit["novelty"], unit.reliability, prediction_error)
        cue_names = [item["cue"] for item in cues]
        target_names = [row["node_id"] for row in ranked[:5]] + [concept_id]
        binding_episode_id = new_episode["episode_id"] if new_episode else stable_id(
            "p02-bound-", unit.unit_id + "|" + unit.text + "|" + str(active.process_count)
        )
        binding = bio.bind_and_learn(binding_episode_id, representation["content_tokens"],
                                     {"context": context, "unit_id": unit.unit_id,
                                      "fit_state": fit["fit_state"]},
                                     cue_names, target_names, gate["learning_gate"])
        for cue in cue_names:
            for target in target_names:
                key = f"{cue}::{target}"
                old = active.associative_weights.get(key, 0.0)
                active.associative_weights[key] = min(1.0, old + config.learning_rate
                                                       * gate["learning_gate"] * (1.0 - old))
        _append_operation(trace, 12, {"consolidation": {"binding": binding, "learning_gate": gate,
                                                          "associative_weight_count": len(active.associative_weights)},
                                      "consolidation_is_gradual": True},
                          bio.process_step(13, representation["content_tokens"],
                                           fit["novelty"], unit.reliability, prediction_error))
        active.consolidation_count += 1

        integrated["retrieval_routes"] = build_retrieval_routes(representation, cues, ranked, analogy_maps)
        retrieval_test = test_retrieval_routes(representation, integrated["retrieval_routes"], ranked)
        active.retrieval_history.append({"unit_id": unit.unit_id, **retrieval_test})
        confidence = {
            "fit_state": fit["fit_state"], "schema_fit": best["fit"] if best else None,
            "novelty": fit["novelty"], "contradiction": fit["contradiction"],
            "retrieval_route_coverage": retrieval_test["route_coverage"],
            "candidate_entropy": fit["candidate_entropy"],
            "uncertainty_sources": ["candidate competition", "schema mismatch", "source reliability"],
            "inferences_are_not_observations": True,
        }
        _append_operation(trace, 13, {"retrieval_diagnostics": retrieval_test,
                                      "retrieval_routes": integrated["retrieval_routes"],
                                      "confidence_and_uncertainty": confidence},
                          bio.process_step(14, representation["content_tokens"],
                                           fit["novelty"], unit.reliability, prediction_error))

        active.process_count += 1
        output = {
            "unit_id": unit.unit_id, "strategy": plan["strategy"],
            "fit_state": fit["fit_state"], "candidate_count": len(ranked),
            "integrated_link_count": len(bound_edges),
            "new_episode_created": new_episode is not None,
            "schema_updates": len(schema_updates),
            "retrieval_route_count": len(integrated["retrieval_routes"]),
            "biological_fidelity": "computational surrogate",
        }
        return PriorKnowledgeResult(
            learner_id=self.learner_id, operation_trace=tuple(trace),
            integrated_representation=integrated, new_episode=new_episode,
            schema_updates=tuple(schema_updates), analogy_maps=tuple(analogy_maps),
            retrieval_routes=tuple(integrated["retrieval_routes"]),
            confidence_and_uncertainty=confidence, biological=bio.snapshot(),
            final_state=active.to_dict(), output=output,
        )


def register_annotations(
    database_path: str | Path, repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index every Point 02 class and callable through the shared AST annotation catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    return sync_repository(LoggerDatabase(database_path), root, include_paths={"ore_prior_knowledge"})


def record_result_to_sqlite(
    result: PriorKnowledgeResult, database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist all 14 operation records and index Point 02 class/function annotations in SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase
    root = Path(repository_root).expanduser().resolve() if repository_root else Path(__file__).resolve().parents[1]
    database = LoggerDatabase(database_path)
    catalogue = sync_repository(database, root, include_paths={"ore_prior_knowledge"})
    run_id = database.create_run(
        script_path="ore_prior_knowledge/point02", module_name="ore_prior_knowledge.pipeline",
        entry_kind="prior_knowledge_integration", working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                source_path="ore_prior_knowledge/pipeline.py",
                elapsed_ms=round((time.perf_counter() - started) * 1000.0, 3),
                metadata={"operation_id": item["operation_id"], "operation": item["operation"],
                          "status": item["status"], "source_step": item["source_step"],
                          "evidence_level": item["evidence_level"],
                          "mathematical_form": item["mathematical_form"],
                          "functional_output": item["functional_output"],
                          "biological_output": item["biological_output"]},
            )
        database.finish_run(run_id, "completed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3))
    except Exception as exc:
        database.finish_run(run_id, "failed",
                            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
                            error_summary=type(exc).__name__ + ": " + str(exc))
        raise
    return {"run_id": run_id, "annotation_catalogue": catalogue,
            "database_summary": database.summary()}
