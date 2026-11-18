"""Focused tests for Point 11 analogical transfer and relational generalization."""

from __future__ import annotations

from pathlib import Path

import pytest

from ore_logger.database import LoggerDatabase
from ore_analogical_transfer import (
    AnalogicalConfig,
    AnalogicalEntity,
    AnalogicalRelation,
    AnalogicalTransferProcessor,
    AnalogySchema,
    HigherOrderRelation,
    ParameterBridge,
    SituationGraph,
    TransferConstraint,
    OPERATION_REGISTRY,
    candidate_entity_correspondences,
    rank_mappings,
    relation_similarity,
    register_annotations,
)


def make_config(**overrides) -> AnalogicalConfig:
    """Configure domain-specific labels to share only their structural relation roles."""
    values = {
        "relation_aliases": {
            "pressure_produces": "drives",
            "voltage_causes": "drives",
            "water_moves_through": "flows_through",
            "current_passes_through": "flows_through",
            "resistance_limits": "constrains",
        },
        "minimum_structural_consistency": 0.50,
        "minimum_mapping_confidence": 0.40,
        "transfer_confidence_threshold": 0.30,
        "mapping_threshold": 0.20,
        "working_memory_budget": 4.0,
    }
    values.update(overrides)
    return AnalogicalConfig(**values)


def make_hydraulic_source(case_id: str = "hydraulic-source") -> SituationGraph:
    """Create a hydraulic source where pressure drives flow and resistance limits it."""
    entities = {
        "pressure": AnalogicalEntity(
            "pressure", "pressure difference", "hydraulic_pressure",
            attributes={"domain": "fluid"}, role_hints=("driving_difference",),
        ),
        "water": AnalogicalEntity(
            "water", "water flow", "liquid", attributes={"medium": "water"},
            role_hints=("flow",),
        ),
        "pipe": AnalogicalEntity(
            "pipe", "pipe", "tube", attributes={"material": "metal"},
            role_hints=("pathway",),
        ),
        "resistance": AnalogicalEntity(
            "resistance", "pipe resistance", "hydraulic_resistance",
            attributes={"medium": "water"}, role_hints=("resistance", "constraint"),
        ),
    }
    relations = (
        AnalogicalRelation(
            "hyd-drive", "pressure_produces", ("pressure", "water"),
            conditions={"steady_state": True}, causal_status="causal",
            confidence=0.95, importance=1.0, quantitative_form="direct",
        ),
        AnalogicalRelation(
            "hyd-path", "water_moves_through", ("water", "pipe"),
            causal_status="functional", confidence=0.90, importance=0.9,
        ),
        AnalogicalRelation(
            "hyd-limit", "resistance_limits", ("resistance", "water"),
            conditions={"steady_state": True}, causal_status="functional",
            confidence=0.90, importance=1.0,
        ),
    )
    higher = (
        HigherOrderRelation(
            "hyd-higher", "hyd-path", "supports", "hyd-drive",
            confidence=0.8, importance=0.5,
        ),
    )
    return SituationGraph(
        case_id=case_id, domain="hydraulics", entities=entities,
        relations=relations, higher_order_relations=higher,
        context={"steady_state": True, "pressure_gap": 5.0},
        goals=("explain_flow",), source_reference="hydraulic_case_memory",
    )


def make_electrical_target(
    case_id: str = "electrical-target",
    constraints=(),
    include_resistor: bool = True,
    conflict: bool = False,
) -> SituationGraph:
    """Create an electrical target that shares two relations but lacks the resistance rule."""
    entities = {
        "voltage": AnalogicalEntity(
            "voltage", "voltage difference", "electrical_potential",
            attributes={"domain": "electric"}, role_hints=("driving_difference",),
        ),
        "current": AnalogicalEntity(
            "current", "electric current", "charge_flow",
            attributes={"medium": "charge"}, role_hints=("flow",),
        ),
        "wire": AnalogicalEntity(
            "wire", "conductor", "conductive_pathway",
            attributes={"material": "copper"}, role_hints=("pathway",),
        ),
    }
    if include_resistor:
        entities["resistor"] = AnalogicalEntity(
            "resistor", "electrical resistance", "electrical_component",
            attributes={"medium": "charge"}, role_hints=("resistance", "constraint"),
        )
    relations = [
        AnalogicalRelation(
            "elec-drive", "voltage_causes", ("voltage", "current"),
            conditions={"steady_state": True}, causal_status="causal",
            confidence=0.95, importance=1.0, quantitative_form="direct",
        ),
        AnalogicalRelation(
            "elec-path", "current_passes_through", ("current", "wire"),
            causal_status="functional", confidence=0.9, importance=0.9,
        ),
    ]
    higher = (
        HigherOrderRelation(
            "elec-higher", "elec-path", "supports", "elec-drive",
            confidence=0.8, importance=0.5,
        ),
    )
    if conflict and include_resistor:
        relations.append(AnalogicalRelation(
            "elec-conflict", "resistance_limits", ("resistor", "current"),
            direction="undirected", causal_status="functional",
            evidential_status="observed", confidence=0.8,
        ))
    return SituationGraph(
        case_id=case_id, domain="electricity", entities=entities,
        relations=tuple(relations), higher_order_relations=higher,
        constraints=tuple(constraints),
        context={"steady_state": True, "voltage_difference": 12.0},
        goals=("explain_flow",), source_reference="electrical_target_observation",
    )


def processor(**kwargs) -> AnalogicalTransferProcessor:
    """Register the hydraulic source and Point 11 configuration."""
    return AnalogicalTransferProcessor(
        sources=(make_hydraulic_source(),), config=make_config(), **kwargs
    )


def test_registry_contains_exact_18_source_ordered_operations():
    """Ensure the entire Point 11 sequence is represented once and in order."""
    assert len(OPERATION_REGISTRY) == 18
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P11-{index:02d}" for index in range(1, 19)
    ]


def test_hydraulic_to_electrical_transfer_maps_roles_not_surface_names():
    """Transfer a missing resistance relation between different object types/domains."""
    model = processor()
    target = make_electrical_target()
    result = model.transfer(target)
    assert result.status == "transfer_generated"
    assert result.selected_source_case_id == "hydraulic-source"
    assert result.selected_mapping is not None
    mapping = result.selected_mapping.entity_map
    assert mapping["pressure"] == "voltage"
    assert mapping["water"] == "current"
    assert mapping["pipe"] == "wire"
    assert mapping["resistance"] == "resistor"
    transferred = next(item for item in result.transferred_hypotheses
                       if item.source_relation_id == "hyd-limit")
    assert transferred.relation.arguments == ("resistor", "current")
    assert transferred.relation.evidential_status == "hypothesized"
    assert transferred.target_status in {"candidate", "qualified"}
    assert all(item["status"] != "observed" for item in result.target_predictions)


def test_surface_similarity_is_separate_from_relational_role_similarity():
    """Prefer functionally corresponding entities even when words and materials differ."""
    source = make_hydraulic_source()
    target = make_electrical_target()
    pairs = candidate_entity_correspondences(source, target, make_config())
    best = {(item.source_entity_id, item.target_entity_id): item for item in pairs}
    assert best[("water", "current")].relational_similarity > 0.7
    assert best[("water", "current")].surface_similarity < 0.5
    assert best[("pipe", "wire")].relational_similarity > 0.7


def test_relations_preserve_arity_direction_and_causal_status():
    """Reject superficial relation matches when their argument type or direction is incompatible."""
    source = AnalogicalRelation(
        "r1", "drives", ("a", "b"), direction="directed", causal_status="causal"
    )
    same_role = AnalogicalRelation(
        "r2", "drives", ("x", "y"), direction="directed", causal_status="causal"
    )
    reversed = AnalogicalRelation(
        "r3", "drives", ("x", "y"), direction="undirected", causal_status="causal"
    )
    wrong_cause = AnalogicalRelation(
        "r4", "drives", ("x", "y"), direction="directed", causal_status="correlational"
    )
    wrong_arity = AnalogicalRelation(
        "r5", "drives", ("x", "y", "z"), direction="directed", causal_status="causal"
    )
    assert relation_similarity(source, same_role, make_config()) == pytest.approx(1.0)
    assert relation_similarity(source, reversed, make_config()) == 0.0
    assert relation_similarity(source, wrong_cause, make_config()) < 1.0
    assert relation_similarity(source, wrong_arity, make_config()) == 0.0


def test_higher_order_relations_increase_structural_integration():
    """Preserve dependencies between first-order relations, not isolated edge matches only."""
    source = make_hydraulic_source()
    target = make_electrical_target()
    mappings = rank_mappings(source, target, make_config())
    assert mappings
    best = mappings[0]
    assert best.preserved_relation_ids == ("hyd-drive", "hyd-path")
    assert best.structural_consistency > 0.5
    assert best.connectedness == pytest.approx(1.0)


def test_target_domain_hard_constraint_rejects_transfer_and_records_disanalogy():
    """Block transfer when target domain explicitly declares a relation unsupported."""
    constraint = TransferConstraint(
        "no-resistance-rule", "unsupported_relation", predicate="resistance_limits",
        explanation="This simplified target model does not include that component law",
        severity="hard",
    )
    result = processor().transfer(make_electrical_target(constraints=(constraint,)))
    assert not result.transferred_hypotheses
    assert any(item.rejected_reason == "hard_target_constraint"
               for item in result.rejected_hypotheses)
    assert any(item.severity == "hard_stop" for item in result.disanalogies)


def test_same_endpoints_with_incompatible_direction_are_not_silently_transferred():
    """Treat a role-reversed or direction-mismatched target edge as a disanalogy."""
    result = processor().transfer(make_electrical_target(conflict=True))
    assert any("hyd-limit" == item.source_relation_id
               for item in result.rejected_hypotheses)
    assert any(item.category == "relational_disanalogy" for item in result.disanalogies)


def test_partial_mapping_keeps_unmapped_source_roles_and_reduces_coverage():
    """Allow an unmatched source role without inventing a target entity for it."""
    source, target = make_hydraulic_source(), make_electrical_target(include_resistor=False)
    mappings = rank_mappings(source, target, make_config())
    assert mappings
    best = mappings[0]
    assert "resistance" not in best.entity_map
    assert "hyd-limit" in best.missing_relation_ids
    full_best = rank_mappings(source, make_electrical_target(), make_config())[0]
    assert full_best.entity_map["resistance"] == "resistor"
    assert best.confidence < full_best.confidence


def test_target_condition_mismatch_qualifies_transfer_instead_of_asserting_it():
    """Retain uncertain transfers with caveats when source boundary conditions are unspecified."""
    target = make_electrical_target()
    target_graph = SituationGraph(
        case_id=target.case_id, domain=target.domain, entities=target.entities,
        relations=target.relations, higher_order_relations=target.higher_order_relations,
        constraints=target.constraints, context={"voltage_difference": 12.0},
        goals=target.goals, source_reference=target.source_reference,
    )
    result = processor().transfer(target_graph)
    if result.transferred_hypotheses:
        assert all(
            item.target_status == "qualified" or any("does not specify" in caveat for caveat in item.caveats)
            for item in result.transferred_hypotheses
        )
    else:
        assert result.status in {
            "mapping_only", "mapping_below_structural_threshold",
            "all_candidate_transfers_rejected",
        }


def test_parameter_bridge_translates_only_dimensionally_compatible_quantities():
    """Use a declared conversion rather than copying numerical values between domains."""
    bridge = ParameterBridge(
        "pressure_gap", "voltage_difference", scale=2.4,
        source_unit="Pa", target_unit="V",
        source_dimension="potential_difference", target_dimension="potential_difference",
    )
    result = processor().transfer(make_electrical_target(), parameter_bridges=(bridge,))
    row = next(item for item in result.parameter_translations
               if item["target_parameter"] == "voltage_difference")
    assert row["target_value"] == pytest.approx(12.0)
    assert row["source_value"] == pytest.approx(5.0)
    assert row["dimension"] == "potential_difference"
    with pytest.raises(ValueError):
        ParameterBridge(
            "pressure", "voltage", source_dimension="pressure",
            target_dimension="electrical_potential",
        )


def test_working_memory_selects_high_priority_relations_under_budget():
    """Keep active relations within the configured maintenance capacity."""
    result = processor().transfer(make_electrical_target())
    step = result.operation_trace[10]["functional_output"]
    rows = step["working_memory_selection"]
    assert rows
    assert step["budget"] == pytest.approx(4.0)
    assert all(row["active"] or row["cost"] <= step["budget"] for row in rows)
    assert all(row["budget_spent"] <= step["budget"] for row in rows)


def test_retrieval_can_use_a_structurally_related_source_from_another_domain():
    """Retrieve by relational overlap rather than requiring same-domain object labels."""
    alternate = SituationGraph(
        "alternate-source", "thermal_system",
        {
            "heat": AnalogicalEntity("heat", "thermal gradient", "thermal", role_hints=("driving_difference",)),
            "flow": AnalogicalEntity("flow", "heat transfer", "energy", role_hints=("flow",)),
            "path": AnalogicalEntity("path", "thermal path", "conductor", role_hints=("pathway",)),
        },
        (
            AnalogicalRelation("a1", "pressure_produces", ("heat", "flow"),
                               causal_status="causal"),
            AnalogicalRelation("a2", "water_moves_through", ("flow", "path"),
                               causal_status="functional"),
        ),
        goals=("explain_flow",),
    )
    model = processor()
    model.register_source(alternate)
    result = model.transfer(make_electrical_target())
    assert result.source_candidates
    assert result.source_candidates[0].case_id in {"hydraulic-source", "alternate-source"}


def test_prediction_evaluation_distinguishes_validation_from_contradiction():
    """Compare later target evidence with transfer hypotheses rather than assuming predictions are true."""
    model = processor()
    result = model.transfer(make_electrical_target())
    hypothesis = next(item for item in result.transferred_hypotheses
                      if item.source_relation_id == "hyd-limit")
    observed = AnalogicalRelation(
        "later-observation", hypothesis.relation.name,
        hypothesis.relation.arguments, direction=hypothesis.relation.direction,
        conditions=hypothesis.relation.conditions,
        evidential_status="observed", causal_status=hypothesis.relation.causal_status,
        confidence=0.95,
    )
    feedback = model.evaluate_transfer(result.target_case_id, (observed,))
    assert feedback.validated_count == 1
    assert feedback.contradicted_count == 0
    assert feedback.mean_brier_score is not None
    assert feedback.supported
    contradiction = AnalogicalRelation(
        "refutation", hypothesis.relation.name, hypothesis.relation.arguments,
        direction=hypothesis.relation.direction, conditions=hypothesis.relation.conditions,
        evidential_status="contradicted", causal_status=hypothesis.relation.causal_status,
    )
    failed = model.evaluate_transfer(result.target_case_id, (contradiction,))
    assert failed.contradicted_count == 1
    assert not failed.supported


def test_validated_repeated_transfer_updates_schema_only_after_authorization():
    """Require both explicit permission and multiple distinct validated target cases."""
    model = processor()
    result = model.transfer(make_electrical_target())
    hypothesis = next(item for item in result.transferred_hypotheses
                      if item.source_relation_id == "hyd-limit")
    observed = AnalogicalRelation(
        "later-observation", hypothesis.relation.name, hypothesis.relation.arguments,
        evidential_status="observed", causal_status=hypothesis.relation.causal_status,
    )
    blocked = model.evaluate_transfer(
        result.target_case_id, (observed,), learning_authorized=True,
        validated_case_ids=("target-1", "target-2"),
    )
    assert blocked.update_status == "insufficient_distinct_validated_cases"
    updated = model.evaluate_transfer(
        result.target_case_id, (observed,), learning_authorized=True,
        validated_case_ids=("target-1", "target-2", "target-3"),
    )
    assert updated.update_status == "analogy_schema_updated_from_authorized_validation"
    assert updated.updated_schema_id is not None
    schema = model.schemas[updated.updated_schema_id]
    assert schema.successful_transfers == 1
    assert "hydraulics" in schema.source_domains
    assert "electricity" in schema.target_domains
    assert schema.reliability > 0.5


def test_failed_transfer_is_preserved_as_a_counterexample_when_authorized():
    """Retain failure counts and exceptions instead of strengthening an invalid analogy."""
    model = processor()
    result = model.transfer(make_electrical_target())
    hypothesis = next(item for item in result.transferred_hypotheses
                      if item.source_relation_id == "hyd-limit")
    contradicted = AnalogicalRelation(
        "later-refutation", hypothesis.relation.name, hypothesis.relation.arguments,
        evidential_status="contradicted", causal_status=hypothesis.relation.causal_status,
    )
    feedback = model.evaluate_transfer(
        result.target_case_id, (contradicted,), learning_authorized=True,
        validated_case_ids=("case-1", "case-2", "case-3"),
    )
    assert feedback.update_status == "analogy_schema_updated_from_authorized_validation"
    schema = model.schemas[feedback.updated_schema_id]
    assert schema.failed_transfers == 1
    assert any("counterexample" in item for item in schema.exceptions)


def test_missing_prediction_outcomes_are_not_counted_as_false():
    """Keep hypotheses untested when the target supplies no later relation evidence."""
    model = processor()
    result = model.transfer(make_electrical_target())
    feedback = model.evaluate_transfer(result.target_case_id, ())
    assert feedback.untested_count == len(result.transferred_hypotheses)
    assert feedback.validated_count == 0
    assert feedback.contradicted_count == 0
    assert feedback.mean_brier_score is None


def test_repeated_source_and_target_ids_preserve_episode_identity():
    """Reject conflicting re-use of an episode identifier in the biological episodic store."""
    model = processor()
    graph = make_hydraulic_source("same-case")
    model.register_source(graph)
    assert model.biological_network.hippocampus.register(graph) is False
    changed = SituationGraph(
        "same-case", "other_domain", graph.entities, graph.relations,
        graph.higher_order_relations, graph.constraints, graph.context,
        graph.goals, graph.source_reference,
    )
    with pytest.raises(ValueError):
        model.register_source(changed)


def test_biological_network_has_distinct_retrieval_relation_integration_and_control_surrogates():
    """Inspect functional network states without claiming one brain region performs the whole analogy."""
    model = processor()
    result = model.transfer(make_electrical_target())
    state = result.biological
    assert state["hippocampus"]["episode_count"] >= 2
    assert state["posterior_parietal"]["encoding_count"] > 0
    assert state["rostrolateral_prefrontal"]["integration_count"] > 0
    assert state["dorsolateral_prefrontal"]["update_count"] > 0
    assert state["ventrolateral_prefrontal"]["control_count"] > 0
    assert "surrogates" in state["scientific_scope"]


def test_source_and_target_evidence_remain_separate_from_inferred_transfer():
    """Retain separate source memory, target observation, and transfer-hypothesis ledger entries."""
    result = processor().transfer(make_electrical_target())
    labels = {row["provenance"] for row in result.evidence_ledger}
    assert {"specific_episode", "target_observation", "analogical_transfer"} <= labels
    inferred = [row for row in result.evidence_ledger
                if row["content_type"] == "transfer_hypothesis"]
    assert inferred and all(row["status"] == "inferred_not_observed" for row in inferred)


def test_sqlite_indexes_point11_symbols_and_records_all_18_operations(tmp_path):
    """Index package annotations and persist every canonical operation in the ORE logger database."""
    database_path = tmp_path / "point11.sqlite3"
    root = Path(__file__).resolve().parents[1]
    catalog = register_annotations(database_path, root)
    assert catalog["symbols_written"] > 0
    model = processor(database_path=database_path, repository_root=root)
    result = model.transfer(make_electrical_target())
    database = LoggerDatabase(database_path)
    annotations = database.search_annotations("analogical", limit=2000)
    assert any(row["module_name"] == "ore_analogical_transfer.pipeline"
               and row["symbol_kind"] == "class" for row in annotations)
    stored = database.list_runs(limit=10)
    run = next(item for item in stored if item["entry_kind"] == "analogical_transfer")
    events = database.list_events(run["run_id"])
    assert len(events) == 18
    assert events[0]["metadata"]["operation_id"] == "P11-01"
    assert events[-1]["metadata"]["operation_id"] == "P11-18"
    assert len(result.operation_trace) == 18


def test_public_result_serialization_contains_the_complete_trace_and_provenance():
    """Serialize the transfer output with mapping alternatives, disanalogies, and 18 stages."""
    result = processor().transfer(make_electrical_target())
    data = result.to_dict()
    assert data["target_case_id"] == "electrical-target"
    assert len(data["operation_trace"]) == 18
    assert data["operation_trace"][0]["operation_id"] == "P11-01"
    assert data["operation_trace"][-1]["operation_id"] == "P11-18"
    assert data["selected_source_case_id"] == "hydraulic-source"
    assert data["transferred_hypotheses"]


def test_no_source_found_returns_a_complete_trace_and_abstains():
    """Do not fabricate a source analogy when episodic and semantic memory are empty."""
    model = AnalogicalTransferProcessor(config=make_config())
    result = model.transfer(make_electrical_target())
    assert result.status == "no_source_found"
    assert result.selected_source_case_id is None
    assert result.transferred_hypotheses == ()
    assert len(result.operation_trace) == 18
