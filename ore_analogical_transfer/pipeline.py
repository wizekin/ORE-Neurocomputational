"""End-to-end 18-stage Point 11 analogical transfer processor and feedback logger."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

from .biology import BiologicalAnalogicalNetwork
from .contracts import (
    AnalogicalConfig, AnalogicalTransferResult, AnalogySchema, AnalogicalEntity,
    AnalogicalRelation, HigherOrderRelation, MappingCandidate, ParameterBridge,
    SituationGraph, TransferHypothesis, TransferValidation,
)
from .functional import (
    abstract_relation_templates, build_transfer_hypotheses, candidate_entity_correspondences,
    generate_partial_mappings, prioritize_working_memory, rank_mappings,
    recurrent_relation_activation, retrieve_sources, score_mapping,
    translate_parameters, validate_transfer,
)
from .registry import OPERATION_REGISTRY


def _trace_operation(
    trace: list[dict[str, Any]], index: int,
    functional: Mapping[str, Any],
    biological: Mapping[str, Any] | None = None,
    status: str = "completed",
) -> None:
    """Append a canonical operation record with its mathematical and biological rationale."""
    spec = OPERATION_REGISTRY[index]
    trace.append({
        "step": index + 1, "operation_id": spec.operation_id,
        "operation": spec.name, "status": status,
        "explanation": spec.purpose, "evidence_level": spec.evidence_level,
        "mathematical_form": spec.mathematical_form,
        "functional_output": dict(functional),
        "biological_output": dict(biological or {}),
    })


def _candidate_schema(
    source: SituationGraph,
    target: SituationGraph,
    mapping: MappingCandidate,
    config: AnalogicalConfig,
) -> AnalogySchema:
    """Propose a conditional domain-independent relation schema without committing it to memory."""
    templates = abstract_relation_templates(source, mapping, config)
    digest = hashlib.sha256(
        repr((source.case_id, target.domain, templates)).encode("utf-8")
    ).hexdigest()[:12]
    conditions = tuple(
        str(key) + "=" + str(value) for key, value in sorted(source.context.items())
    )
    return AnalogySchema(
        schema_id="analogy:" + digest,
        name=source.domain + "-to-" + target.domain + " relational pattern",
        relation_templates=templates, conditions=conditions,
        source_domains=(source.domain,), target_domains=(target.domain,),
    )


def _build_evidence_ledger(
    source: SituationGraph | None,
    target: SituationGraph,
    hypotheses: Sequence[TransferHypothesis],
    retrieved: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Keep source memories, target observations, and generated transfer hypotheses separately labeled."""
    rows: list[dict[str, Any]] = []
    if source is not None:
        rows.append({
            "content_type": "source_episode", "case_id": source.case_id,
            "domain": source.domain, "source": source.source_reference,
            "status": "retrieved_source", "provenance": "specific_episode",
        })
        for relation in source.relations:
            rows.append({
                "content_type": "source_relation",
                "relation": relation.to_dict(),
                "status": relation.evidential_status,
                "provenance": relation.provenance,
            })
    rows.append({
        "content_type": "target_episode", "case_id": target.case_id,
        "domain": target.domain, "source": target.source_reference,
        "status": "current_target_representation", "provenance": "target_observation",
    })
    for relation in target.relations:
        rows.append({
            "content_type": "target_relation", "relation": relation.to_dict(),
            "status": relation.evidential_status, "provenance": relation.provenance,
        })
    for item in retrieved:
        rows.append({
            "content_type": "episodic_retrieval", "case_id": item["case_id"],
            "domain": item["domain"], "score": item["score"],
            "source": item["source_reference"], "status": "retrieved_not_observed",
            "provenance": "specific_episode",
        })
    for hypothesis in hypotheses:
        rows.append({
            "content_type": "transfer_hypothesis",
            "hypothesis_id": hypothesis.hypothesis_id,
            "relation": hypothesis.relation.to_dict(),
            "confidence": hypothesis.confidence,
            "status": "inferred_not_observed",
            "provenance": "analogical_transfer",
            "support_chain": hypothesis.support_chain,
        })
    return tuple(rows)


class AnalogicalTransferProcessor:
    """Find, evaluate, adapt, and learn from relational analogies across source and target domains.

    Source and target graphs must encode observed or explicitly hypothesized relations. The
    processor never equates surface resemblance with structural validity and never treats a
    transferred hypothesis as target observation. Reusable analogy schemas are updated only
    after explicit authorization and a caller-supplied set of distinct validated case IDs.
    """

    def __init__(
        self,
        sources: Sequence[SituationGraph] = (),
        schemas: Sequence[AnalogySchema] = (),
        config: AnalogicalConfig | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize source memory, analogy schemas, biological surrogates, and logger settings."""
        self.config = config or AnalogicalConfig()
        self.source_memory: dict[str, SituationGraph] = {}
        self.schemas: dict[str, AnalogySchema] = {}
        self.pending_results: dict[str, AnalogicalTransferResult] = {}
        self.pending_schema_candidates: dict[str, AnalogySchema] = {}
        self.profile_lineage: list[dict[str, Any]] = []
        self.biological_network = BiologicalAnalogicalNetwork()
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )
        for graph in sources:
            self.register_source(graph)
        for schema in schemas:
            self.register_analogy_schema(schema)

    def register_source(self, graph: SituationGraph) -> None:
        """Add one source graph to episodic memory and index its cortical relation cues."""
        if not isinstance(graph, SituationGraph):
            raise TypeError("graph must be a SituationGraph")
        previous = self.source_memory.get(graph.case_id)
        if previous is not None and previous.to_dict() != graph.to_dict():
            raise ValueError("case_id already exists with different source graph content")
        self.source_memory[graph.case_id] = graph
        self.biological_network.observe_graph(graph, self.config)

    def register_analogy_schema(self, schema: AnalogySchema) -> None:
        """Register an existing conditional analogy schema without altering its reliability history."""
        if not isinstance(schema, AnalogySchema):
            raise TypeError("schema must be an AnalogySchema")
        existing = self.schemas.get(schema.schema_id)
        if existing is not None and existing.to_dict() != schema.to_dict():
            raise ValueError("schema_id already exists with different analogy schema content")
        self.schemas[schema.schema_id] = schema

    def _select_source_and_mapping(
        self,
        target: SituationGraph,
        supplied_sources: Sequence[SituationGraph] = (),
    ) -> tuple[tuple[Any, ...], SituationGraph | None, MappingCandidate | None,
               tuple[MappingCandidate, ...], tuple[Any, ...], tuple[Any, ...]]:
        """Search multiple retrieved sources and mappings, preferring coherent relation preservation."""
        for graph in supplied_sources:
            self.register_source(graph)
        candidates, retrieved_graphs = retrieve_sources(
            target, tuple(self.source_memory.values()), self.config
        )
        all_candidates: list[tuple[float, SituationGraph, MappingCandidate,
                                   tuple[MappingCandidate, ...], tuple[Any, ...]]] = []
        for source in retrieved_graphs:
            pairs = candidate_entity_correspondences(source, target, self.config)
            mappings = rank_mappings(source, target, self.config)
            if mappings:
                for mapping in mappings:
                    all_candidates.append((
                        mapping.total_score + 0.20 * mapping.structural_consistency,
                        source, mapping, mappings, pairs,
                    ))
        all_candidates.sort(key=lambda item: (
            -item[0], -item[2].structural_consistency,
            item[1].case_id, item[2].mapping_id,
        ))
        if not all_candidates:
            return candidates, None, None, (), (), retrieved_graphs
        _, selected_source, selected_mapping, mapping_alternatives, selected_pairs = all_candidates[0]
        return (
            candidates, selected_source, selected_mapping,
            mapping_alternatives, selected_pairs, retrieved_graphs,
        )

    def transfer(
        self,
        target: SituationGraph,
        source_cases: Sequence[SituationGraph] = (),
        parameter_bridges: Sequence[ParameterBridge] = (),
        *,
        persist_to_sqlite: bool = True,
    ) -> AnalogicalTransferResult:
        """Execute all 18 Point 11 stages and return qualified hypotheses with complete provenance."""
        started = time.perf_counter()
        if not isinstance(target, SituationGraph):
            raise TypeError("target must be a SituationGraph")
        trace: list[dict[str, Any]] = []

        cue_relevance = {
            "relation_types": sorted({item.name for item in target.relations}),
            "goal_count": len(target.goals),
            "domain": target.domain,
            "relation_count": len(target.relations),
            "transfer_is_optional": True,
        }
        _trace_operation(trace, 0, cue_relevance)

        (source_candidates, selected_source, selected_mapping,
         mapping_alternatives, selected_pairs, retrieved_graphs) = self._select_source_and_mapping(
            target, source_cases
        )
        _trace_operation(trace, 1, {
            "source_candidates": [item.to_dict() for item in source_candidates],
            "selected_source_case_id": selected_source.case_id if selected_source else None,
            "retrieval_limit": self.config.retrieval_limit,
        }, {"hippocampus": self.biological_network.hippocampus.snapshot()})

        _trace_operation(trace, 2, {
            "target_graph": {
                "case_id": target.case_id, "domain": target.domain,
                "entity_count": len(target.entities), "relation_count": len(target.relations),
                "higher_order_relation_count": len(target.higher_order_relations),
                "constraints": [item.to_dict() for item in target.constraints],
                "goals": target.goals,
            },
            "unknowns_remain_explicit": True,
        }, {"posterior_parietal": self.biological_network.parietal.snapshot()})

        source_summary = (
            {
                "case_id": selected_source.case_id, "domain": selected_source.domain,
                "entity_count": len(selected_source.entities),
                "relation_count": len(selected_source.relations),
                "higher_order_relation_count": len(selected_source.higher_order_relations),
                "evidence_statuses": sorted({item.evidential_status for item in selected_source.relations}),
                "source_reference": selected_source.source_reference,
            } if selected_source else {}
        )
        _trace_operation(trace, 3, {
            "reconstructed_source": source_summary,
            "source_episode_kept_separate": True,
        })

        if selected_source is not None:
            source_relation_signatures = [
                {
                    "relation_id": item.relation_id, "name": item.name,
                    "canonical_name": self.config.relation_aliases.get(item.name, item.name),
                    "arity": len(item.arguments), "direction": item.direction,
                    "causal_status": item.causal_status,
                    "importance": item.importance,
                } for item in selected_source.relations
            ]
            surface_relation_scores = [
                {
                    "source_entity": pair.source_entity_id,
                    "target_entity": pair.target_entity_id,
                    "surface_similarity": pair.surface_similarity,
                    "relational_similarity": pair.relational_similarity,
                } for pair in selected_pairs
            ]
        else:
            source_relation_signatures = []
            surface_relation_scores = []
        _trace_operation(trace, 4, {
            "source_target_similarity_channels": surface_relation_scores,
            "surface_channel_is_not_the_primary_mapping_criterion": True,
            "source_relation_signatures": source_relation_signatures,
        })

        _trace_operation(trace, 5, {
            "canonical_relation_aliases": dict(self.config.relation_aliases),
            "source_signatures": source_relation_signatures,
            "target_signatures": [
                {
                    "relation_id": item.relation_id, "name": item.name,
                    "canonical_name": self.config.relation_aliases.get(item.name, item.name),
                    "arity": len(item.arguments), "direction": item.direction,
                    "causal_status": item.causal_status,
                } for item in target.relations
            ],
            "argument_order_and_causal_status_preserved": True,
        })

        candidate_correspondences = (
            candidate_entity_correspondences(selected_source, target, self.config)
            if selected_source else ()
        )
        generated_mappings = (
            rank_mappings(selected_source, target, self.config)
            if selected_source else ()
        )
        _trace_operation(trace, 6, {
            "candidate_correspondences": [item.to_dict() for item in candidate_correspondences],
            "partial_mapping_count": len(generated_mappings),
            "one_to_one_default_partial_assignment": True,
        })

        _trace_operation(trace, 7, {
            "ranked_mappings": [item.to_dict() for item in generated_mappings],
            "selected_mapping": selected_mapping.to_dict() if selected_mapping else None,
            "structural_confidence_threshold": self.config.minimum_structural_consistency,
        })

        biological: dict[str, Any] = {}
        if selected_source and selected_mapping:
            integration = self.biological_network.rostrolateral.integrate(
                selected_source, target, (selected_mapping,)
            )
            biological = self.biological_network.process_mapping(
                selected_source, target, candidate_correspondences,
                (selected_mapping,), self.config,
            )
        else:
            integration = {"integration_score": 0.0, "coherent": False}
            biological = self.biological_network.snapshot()
        _trace_operation(trace, 8, {
            "higher_order_integration": integration,
            "source_higher_order_relations": (
                [item.to_dict() for item in selected_source.higher_order_relations]
                if selected_source else []
            ),
            "target_higher_order_relations": [item.to_dict() for item in target.higher_order_relations],
        }, {"rostrolateral_prefrontal": integration})

        retrieved = biological.get("hippocampal_retrieval", [])
        _trace_operation(trace, 9, {
            "retrieved_relational_fragments": retrieved,
            "retrieval_provenance_preserved": True,
        }, {"hippocampus": self.biological_network.hippocampus.snapshot()})

        working_memory: tuple[dict[str, Any], ...] = ()
        recurrent_activation: dict[str, float] = {}
        if selected_source and selected_mapping:
            working_memory = prioritize_working_memory(
                selected_source, target, selected_mapping, self.config
            )
            active_ids = [item["relation_id"] for item in working_memory if item["active"]]
            recurrent_activation = recurrent_relation_activation(
                active_ids, selected_source, selected_mapping, iterations=3
            )
            biological["recurrent_relation_activation"] = recurrent_activation
        _trace_operation(trace, 10, {
            "working_memory_selection": list(working_memory),
            "recurrent_relation_activation": recurrent_activation,
            "budget": self.config.working_memory_budget,
        }, {
            "dorsolateral_prefrontal": self.biological_network.dorsolateral.snapshot(),
            "ventrolateral_prefrontal": self.biological_network.ventrolateral.snapshot(),
        })

        if (
            selected_source and selected_mapping
            and selected_mapping.structural_consistency >= self.config.minimum_structural_consistency
            and selected_mapping.confidence >= self.config.minimum_mapping_confidence
        ):
            hypotheses, rejected, disanalogies = build_transfer_hypotheses(
                selected_source, target, selected_mapping, self.config
            )
        else:
            hypotheses, rejected, disanalogies = (), (), ()
        _trace_operation(trace, 11, {
            "candidate_transfer_hypotheses": [item.to_dict() for item in hypotheses],
            "rejected_low_confidence_or_unmapped_relations": [
                item.to_dict() for item in rejected
            ],
            "each_hypothesis_retains_a_support_chain": True,
        })

        _trace_operation(trace, 12, {
            "target_constraints": [item.to_dict() for item in target.constraints],
            "rejected_hypotheses": [item.to_dict() for item in rejected],
            "disanalogies": [item.to_dict() for item in disanalogies],
            "hard_constraint_violations_block_transfer": True,
            "surface_kind_differences_are_not_by_themselves_hard_failures": True,
        })

        translations = (
            translate_parameters(selected_source, parameter_bridges)
            if selected_source else ()
        )
        explanation = tuple(
            {
                "source_entity": selected_source.entities[pair.source_entity_id].label,
                "target_entity": target.entities[pair.target_entity_id].label,
                "source_role": selected_source.entities[pair.source_entity_id].role_hints,
                "target_role": target.entities[pair.target_entity_id].role_hints,
                "mapping_confidence": pair.confidence,
                "explanation_provenance": "analogy_mapping_not_identity",
            }
            for pair in selected_mapping.correspondences
        ) if selected_source and selected_mapping else ()
        _trace_operation(trace, 13, {
            "target_explanation": list(explanation),
            "parameter_translations": list(translations),
            "target_specific_units_and_laws_required": True,
        })

        predictions = tuple({
            "hypothesis_id": item.hypothesis_id,
            "relation": item.relation.to_dict(),
            "confidence": item.confidence,
            "status": "prediction_candidate_not_observed",
            "source_relation_id": item.source_relation_id,
            "scope_caveats": item.caveats,
        } for item in hypotheses)
        _trace_operation(trace, 14, {
            "target_predictions": list(predictions),
            "outcome_validation_pending": True,
            "unmeasured_target_consequences_are_not_false": True,
        })

        alternatives = [
            {
                "mapping_id": item.mapping_id,
                "confidence": item.confidence,
                "structural_consistency": item.structural_consistency,
                "total_score": item.total_score,
            } for item in mapping_alternatives[1:]
        ]
        _trace_operation(trace, 15, {
            "alternative_mapping_count": len(alternatives),
            "alternative_mappings": alternatives,
            "revision_strategy": "compare alternative source cases and mappings before abstaining",
            "validated_outcome_available": False,
        })

        candidate_schema = (
            _candidate_schema(selected_source, target, selected_mapping, self.config)
            if selected_source and selected_mapping else None
        )
        _trace_operation(trace, 16, {
            "candidate_analogy_schema": candidate_schema.to_dict() if candidate_schema else None,
            "committed_to_long_term_schema_memory": False,
            "schema_induction_requires_validated_repeated_transfers": True,
        })

        status = "no_source_found"
        if source_candidates and not selected_mapping:
            status = "no_valid_mapping"
        elif selected_mapping is not None:
            if selected_mapping.structural_consistency < self.config.minimum_structural_consistency:
                status = "mapping_below_structural_threshold"
            elif rejected and not hypotheses:
                status = "all_candidate_transfers_rejected"
            elif hypotheses:
                status = "transfer_generated"
            else:
                status = "mapping_only"
        ledger = _build_evidence_ledger(selected_source, target, hypotheses, retrieved)
        self.biological_network.observe_graph(target, self.config)
        duration_ms = round((time.perf_counter() - started) * 1000.0, 3)
        diagnostics = {
            "source_memory_size": len(self.source_memory),
            "analogy_schema_count": len(self.schemas),
            "source_candidate_count": len(source_candidates),
            "mapping_count": len(mapping_alternatives),
            "selected_source_case_id": selected_source.case_id if selected_source else None,
            "selected_mapping_id": selected_mapping.mapping_id if selected_mapping else None,
            "transferred_hypothesis_count": len(hypotheses),
            "rejected_hypothesis_count": len(rejected),
            "disanalogy_count": len(disanalogies),
            "duration_ms": duration_ms,
            "scientific_scope": (
                "Graph matching, similarity, and confidence equations are computational hypotheses; "
                "biological components are distributed functional surrogates, not literal neural code."
            ),
        }
        _trace_operation(trace, 17, {
            "target_episode_retained": target.case_id in self.biological_network.hippocampus.episodes,
            "candidate_schema": candidate_schema.to_dict() if candidate_schema else None,
            "automatic_schema_update": False,
            "diagnostics": diagnostics,
        }, self.biological_network.snapshot())
        result = AnalogicalTransferResult(
            target_case_id=target.case_id, status=status,
            source_candidates=source_candidates,
            selected_source_case_id=selected_source.case_id if selected_source else None,
            mappings=mapping_alternatives, selected_mapping=selected_mapping,
            transferred_hypotheses=hypotheses, rejected_hypotheses=rejected,
            disanalogies=disanalogies, target_explanation=explanation,
            target_predictions=predictions, parameter_translations=translations,
            evidence_ledger=ledger, operation_trace=tuple(trace),
            biological=self.biological_network.snapshot(), diagnostics=diagnostics,
        )
        self.pending_results[target.case_id] = result
        if candidate_schema is not None:
            self.pending_schema_candidates[target.case_id] = candidate_schema
        if self.database_path and persist_to_sqlite:
            record_result_to_sqlite(result, self.database_path, self.repository_root)
        return result

    def evaluate_transfer(
        self,
        target_case_id: str,
        observed_relations: Sequence[AnalogicalRelation],
        *,
        learning_authorized: bool = False,
        validated_case_ids: Sequence[str] = (),
    ) -> TransferValidation:
        """Score later target evidence and consolidate an analogy schema only under explicit authorization."""
        if target_case_id not in self.pending_results:
            raise KeyError("no pending analogical transfer exists for this target case")
        result = self.pending_results[target_case_id]
        outcomes = validate_transfer(result.transferred_hypotheses, observed_relations, self.config)
        validated = sum(bool(item["validated"]) for item in outcomes)
        contradicted = len(outcomes) - validated
        untested = len(result.transferred_hypotheses) - len(outcomes)
        brier = [float(item["brier_score"]) for item in outcomes]
        surprise = [float(item["surprise"]) for item in outcomes]
        supported = validated > contradicted and validated > 0
        schema_id: str | None = None
        update_status = "not_authorized"
        if learning_authorized:
            unique_ids = sorted(set(str(item) for item in validated_case_ids if str(item).strip()))
            candidate = self.pending_schema_candidates.get(target_case_id)
            if len(unique_ids) < self.config.schema_update_minimum_successes:
                update_status = "insufficient_distinct_validated_cases"
            elif candidate is None:
                update_status = "no_candidate_analogy_schema"
            elif not outcomes:
                update_status = "no_tested_transfer_outcomes"
            else:
                existing = self.schemas.get(candidate.schema_id)
                if existing is None:
                    existing = candidate
                success_delta = validated * self.config.schema_update_learning_rate
                failure_delta = contradicted * self.config.schema_update_learning_rate
                revised = AnalogySchema(
                    schema_id=existing.schema_id, name=existing.name,
                    relation_templates=existing.relation_templates,
                    conditions=existing.conditions,
                    exceptions=existing.exceptions + (
                        ("counterexample observed in " + target_case_id,) if contradicted else ()
                    ),
                    source_domains=tuple(sorted(set(existing.source_domains) | {
                        item.domain for item in self.source_memory.values()
                        if item.case_id == result.selected_source_case_id
                    })),
                    target_domains=tuple(sorted(set(existing.target_domains) | {
                        self.source_memory.get(target_case_id, SituationGraph.__new__(SituationGraph)).domain
                        if target_case_id in self.source_memory else ""
                    } - {""})),
                    successful_transfers=existing.successful_transfers + int(round(success_delta)),
                    failed_transfers=existing.failed_transfers + int(round(failure_delta)),
                    alpha=existing.alpha, beta=existing.beta, version=existing.version + 1,
                )
                if existing.schema_id not in self.schemas or revised.to_dict() != existing.to_dict():
                    self.schemas[revised.schema_id] = revised
                    schema_id = revised.schema_id
                    update_status = "analogy_schema_updated_from_authorized_validation"
                    self.profile_lineage.append({
                        "schema_id": revised.schema_id, "validated_case_ids": unique_ids,
                        "validated_transfers": validated, "contradicted_transfers": contradicted,
                        "version": revised.version,
                    })
        return TransferValidation(
            target_case_id=target_case_id, tested_hypotheses=outcomes,
            validated_count=validated, contradicted_count=contradicted,
            untested_count=untested,
            mean_brier_score=(sum(brier) / len(brier) if brier else None),
            mean_surprise=(sum(surprise) / len(surprise) if surprise else None),
            supported=supported, schema_update_authorized=learning_authorized,
            updated_schema_id=schema_id, update_status=update_status,
        )


def register_annotations(
    database_path: str | Path, repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Index all Point 11 package classes and callable docstrings in the ORE SQLite catalogue."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    return sync_repository(
        LoggerDatabase(database_path), root,
        include_paths={"ore_analogical_transfer"},
    )


def record_result_to_sqlite(
    result: AnalogicalTransferResult,
    database_path: str | Path,
    repository_root: str | Path | None = None,
) -> dict[str, Any]:
    """Persist the 18-stage transfer trace and index package annotations in ORE SQLite."""
    from ore_logger.catalog import sync_repository
    from ore_logger.database import LoggerDatabase

    root = (Path(repository_root).expanduser().resolve() if repository_root
            else Path(__file__).resolve().parents[1])
    database = LoggerDatabase(database_path)
    catalog = sync_repository(
        database, root, include_paths={"ore_analogical_transfer"}
    )
    run_id = database.create_run(
        script_path="ore_analogical_transfer/point11",
        module_name="ore_analogical_transfer.pipeline",
        entry_kind="analogical_transfer",
        working_directory=str(root),
    )
    started = time.perf_counter()
    try:
        for sequence, item in enumerate(result.operation_trace, start=1):
            database.record_event(
                run_id=run_id, sequence_no=sequence, event_kind="operation",
                source_path="ore_analogical_transfer/pipeline.py", elapsed_ms=0.0,
                metadata={
                    "operation_id": item.get("operation_id"), "operation": item.get("operation"),
                    "status": item.get("status"), "evidence_level": item.get("evidence_level"),
                    "mathematical_form": item.get("mathematical_form"),
                    "functional_output": item.get("functional_output"),
                    "biological_output": item.get("biological_output"),
                },
            )
        database.finish_run(
            run_id, "completed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
        )
    except Exception as exc:
        database.finish_run(
            run_id, "failed",
            duration_ms=round((time.perf_counter() - started) * 1000.0, 3),
            error_summary=type(exc).__name__ + ": " + str(exc),
        )
        raise
    return {
        "run_id": run_id, "annotation_catalog": catalog,
        "database_summary": database.summary(),
    }
