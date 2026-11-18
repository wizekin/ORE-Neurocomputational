"""Tests for Point 09 schema formation, relations, biological models, and SQLite records."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ore_logger.database import LoggerDatabase
from ore_schema_formation import (
    EntityInput,
    EpisodeInput,
    EventInput,
    RelationInput,
    SchemaConfig,
    SchemaFormationProcessor,
    BiologicalSchemaNetwork,
    OPERATION_REGISTRY,
    register_annotations,
    record_result_to_sqlite,
)
from ore_schema_formation.functional import (
    compose_schema_hierarchy,
    estimate_transitions,
    group_related_episodes,
    induce_relations,
    induce_slots,
    infer_role_assignments,
)
from ore_schema_formation.contracts import SchemaFrame
from ore_schema_formation.functional import context_signature


def restaurant_episode(
    prefix: str,
    *,
    include_relations: bool = True,
    reverse_order_relation: bool = False,
    causal_hypothesis: bool = False,
) -> EpisodeInput:
    """Build a distinct restaurant experience whose surface entities vary but roles persist."""
    customer = prefix + "-customer"
    server = prefix + "-server"
    venue = prefix + "-venue"
    meal = prefix + "-meal"
    payment = prefix + "-payment"
    entities = {
        customer: EntityInput(customer, "A customer " + prefix, "person"),
        server: EntityInput(server, "A server " + prefix, "person"),
        venue: EntityInput(venue, "A restaurant " + prefix, "place"),
        meal: EntityInput(meal, "A meal " + prefix, "food"),
        payment: EntityInput(payment, "A payment instrument " + prefix, "payment_method"),
    }
    events = (
        EventInput("arrive", {"customer": customer, "restaurant": venue},
                   event_id=prefix + "-event-1", timestamp=0.0),
        EventInput("order", {"customer": customer, "waiter": server, "meal": meal},
                   event_id=prefix + "-event-2", timestamp=2.0),
        EventInput("serve", {"waiter": server, "customer": customer, "meal": meal},
                   event_id=prefix + "-event-3", timestamp=5.0),
        EventInput("pay", {"customer": customer, "payment_method": payment},
                   event_id=prefix + "-event-4", timestamp=8.0),
    )
    relations = []
    if include_relations:
        if reverse_order_relation:
            relations.append(RelationInput(
                meal, "orders", customer, "observed", confidence=0.9
            ))
        else:
            relations.append(RelationInput(
                customer, "orders", meal, "causal_hypothesis" if causal_hypothesis else "observed",
                confidence=0.8 if causal_hypothesis else 1.0,
            ))
        relations.append(RelationInput(server, "serves", meal, "functional", confidence=0.9))
    return EpisodeInput(
        episode_id=prefix, entities=entities, events=events, relations=tuple(relations),
        context={"setting": "restaurant", "service": "dine_in"},
        goals=("eat_meal",), outcomes=("meal_received", "bill_settled"),
        source_reference="test_fixture:" + prefix,
    )


def unrelated_episode() -> EpisodeInput:
    """Build a strongly incongruent aircraft episode for schema specialization tests."""
    pilot, aircraft, tower = "pilot-1", "aircraft-1", "tower-1"
    entities = {
        pilot: EntityInput(pilot, "pilot", "person"),
        aircraft: EntityInput(aircraft, "aircraft", "vehicle"),
        tower: EntityInput(tower, "control tower", "facility"),
    }
    events = (
        EventInput("launch", {"pilot": pilot, "aircraft": aircraft}),
        EventInput("navigate", {"pilot": pilot, "aircraft": aircraft, "tower": tower}),
        EventInput("land", {"pilot": pilot, "aircraft": aircraft, "tower": tower}),
    )
    relations = (RelationInput(pilot, "controls", aircraft, "observed"),)
    return EpisodeInput(
        episode_id="aircraft-case", entities=entities, events=events, relations=relations,
        context={"setting": "airport"}, goals=("navigate_aircraft",),
        outcomes=("arrived_safely",),
    )


def trained_processor(tmp_path: Path | None = None, **config_overrides):
    """Fit a two-episode restaurant schema with aliases for role labels."""
    config = SchemaConfig(role_aliases={"waiter": "service_provider"}, **config_overrides)
    processor = SchemaFormationProcessor(config=config)
    result = processor.fit([restaurant_episode("r1"), restaurant_episode("r2")])
    assert len(result.schemas) == 1
    return processor, result


def test_registry_has_exact_24_ordered_source_stages():
    """Verify source alignment and stable operation identifiers for Point 09."""
    assert len(OPERATION_REGISTRY) == 24
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P09-{index:02d}" for index in range(1, 25)
    ]


def test_contracts_preserve_episode_identity_and_reject_dangling_references():
    """Validate relation and participant references against the episode entity table."""
    customer = EntityInput("c", "customer", "person")
    with pytest.raises(ValueError, match="unknown entities"):
        EpisodeInput(
            "bad", {"c": customer},
            [EventInput("order", {"customer": "missing"})],
        )
    with pytest.raises(ValueError, match="evidential_status"):
        RelationInput("c", "knows", "c", "certain_cause")
    episode = restaurant_episode("unique")
    assert episode.to_dict()["episode_id"] == "unique"
    assert [row["event_type"] for row in episode.to_dict()["events"]] == [
        "arrive", "order", "serve", "pay"
    ]


def test_episode_grouping_abstracts_common_structure_not_specific_entity_ids():
    """Group related episodes with shared goals and relational structure while retaining their IDs."""
    episodes = [restaurant_episode("g1"), restaurant_episode("g2")]
    groups = group_related_episodes(episodes, threshold=0.42, aliases={"waiter": "service_provider"})
    assert len(groups) == 1
    assert {item.episode_id for item in groups[0]} == {"g1", "g2"}
    assert groups[0][0].entities != groups[0][1].entities


def test_role_assignment_does_not_merge_reused_entity_ids_across_episodes():
    """Treat entity IDs as episode-local so a reused label cannot blend different roles."""
    episodes = [
        EpisodeInput(
            "local-a",
            {"shared": EntityInput("shared", "customer", "person"),
             "meal-a": EntityInput("meal-a", "meal A", "food")},
            [EventInput("order", {"customer": "shared", "meal": "meal-a"})],
        ),
        EpisodeInput(
            "local-b",
            {"shared": EntityInput("shared", "service provider", "person"),
             "meal-b": EntityInput("meal-b", "meal B", "food")},
            [EventInput("serve", {"service_provider": "shared", "meal": "meal-b"})],
        ),
    ]
    assignments = infer_role_assignments(episodes, SchemaConfig())
    shared = {(item.episode_id, item.entity_id): item for item in assignments if item.entity_id == "shared"}
    assert shared[("local-a", "shared")].selected_role == "customer"
    assert shared[("local-b", "shared")].selected_role == "service_provider"
    assert not shared[("local-a", "shared")].ambiguous
    assert not shared[("local-b", "shared")].ambiguous


def test_role_induction_maps_surface_synonym_to_reusable_schema_slot():
    """Map waiter and service-provider mentions to one role while retaining different fillers."""
    episodes = [restaurant_episode("s1"), restaurant_episode("s2")]
    config = SchemaConfig(role_aliases={"waiter": "service_provider"})
    assignments = infer_role_assignments(episodes, config)
    slots = induce_slots(episodes, assignments, config)
    assert "service_provider" in slots
    assert set(slots["service_provider"].observed_fillers) == {"s1-server", "s2-server"}
    assert slots["service_provider"].coverage == 1.0
    assert slots["service_provider"].status == "core"


def test_relation_eligibility_counts_episodes_even_when_relation_not_observed():
    """Use the correct role-compatible denominator instead of counting positive episodes only."""
    episodes = [restaurant_episode("eligible-absent", include_relations=False),
                restaurant_episode("eligible-present", include_relations=False)]
    positive = restaurant_episode("positive", include_relations=True)
    episodes.append(positive)
    config = SchemaConfig(role_aliases={"waiter": "service_provider"})
    assignments = infer_role_assignments(episodes, config)
    slots = induce_slots(episodes, assignments, config)
    relations = induce_relations(episodes, slots, assignments, config)
    orders = next(item for item in relations if item.predicate == "orders")
    assert orders.support_count == 1
    assert orders.eligible_count == 3
    assert orders.supporting_episode_ids == ("positive",)
    assert orders.confidence == pytest.approx(1.5 / 4.0)


def test_directed_relations_and_causal_hypotheses_remain_distinct():
    """Do not collapse reversed edges or promote a causal hypothesis into observed fact."""
    episodes = [
        restaurant_episode("dir-forward"),
        restaurant_episode("dir-reverse", reverse_order_relation=True),
        restaurant_episode("dir-causal", causal_hypothesis=True),
    ]
    config = SchemaConfig(role_aliases={"waiter": "service_provider"})
    assignments = infer_role_assignments(episodes, config)
    slots = induce_slots(episodes, assignments, config)
    relations = induce_relations(episodes, slots, assignments, config)
    orders = [item for item in relations if item.predicate == "orders"]
    assert {item.subject_slot for item in orders} == {"customer", "meal"}
    assert {item.object_slot for item in orders} == {"meal", "customer"}
    causal = next(item for item in orders if item.evidential_status == "causal_hypothesis")
    assert causal.evidential_status == "causal_hypothesis"


def test_context_conditioned_transition_probabilities_are_normalized():
    """Estimate probabilistic next-event alternatives without imposing one rigid event chain."""
    episodes = [restaurant_episode("t1"), restaurant_episode("t2")]
    config = SchemaConfig(role_aliases={"waiter": "service_provider"})
    rows = estimate_transitions(episodes, config)
    key = context_signature(episodes[0].context)
    conditioned = [
        row for row in rows
        if row["context_key"] == key and row["from_event"] == "arrive"
    ]
    assert len(conditioned) == 4
    assert sum(row["probability"] for row in conditioned) == pytest.approx(1.0)
    order = next(row for row in conditioned if row["to_event"] == "order")
    assert order["count"] == 2
    assert order["probability"] > 0.5
    assert any(row["context_key"] == "*" for row in rows)


def test_fit_builds_slots_relations_transitions_biology_and_24_stage_trace():
    """Run the integrated processor and validate its structural and biological outputs."""
    processor, result = trained_processor()
    schema = result.schemas[0]
    assert len(result.operation_trace) == 24
    assert result.operation_trace[0]["operation_id"] == "P09-01"
    assert result.operation_trace[-1]["operation_id"] == "P09-24"
    assert set(schema.slots) >= {
        "customer", "restaurant", "service_provider", "meal", "payment_method"
    }
    assert any(item.predicate == "orders" for item in schema.relations)
    assert any(item["from_event"] == "arrive" and item["to_event"] == "order"
               for item in schema.transitions)
    assert schema.supporting_episode_ids == ("r1", "r2")
    assert result.biological["hippocampus"]["episode_count"] == 2
    assert result.biological["distributed_cortex"]["association_count"] > 0
    assert result.biological["medial_prefrontal"]["model"].startswith("medial_prefrontal")
    assert result.diagnostics["schema_count"] == 1


def test_predictions_fill_gaps_but_mark_every_completion_as_inferred():
    """Predict missing roles, relations, and transitions without claiming they were observed."""
    processor, result = trained_processor()
    schema = result.schemas[0]
    partial = EpisodeInput(
        "partial", {
            "partial-customer": EntityInput("partial-customer", "visitor", "person"),
            "partial-venue": EntityInput("partial-venue", "venue", "place"),
        },
        [EventInput("arrive", {"customer": "partial-customer", "restaurant": "partial-venue"})],
        context={"setting": "restaurant", "service": "dine_in"}, goals=("eat_meal",),
    )
    prediction = processor.predict(partial, schema_id=schema.schema_id)
    inferences = prediction["inferred_items"] + prediction["next_event_predictions"]
    assert inferences
    assert all(item["provenance"] == "inferred" for item in inferences)
    assert all(item["observed"] is False for item in inferences)
    assert any(item["kind"] == "slot_relation" for item in prediction["inferred_items"])
    assert prediction["next_event_predictions"][0]["event_type"] == "order"


def test_evaluate_and_integrate_schema_consistent_new_episode():
    """Integrate another matching restaurant event while retaining all episode IDs."""
    processor, result = trained_processor()
    schema = result.schemas[0]
    new_episode = restaurant_episode("r3")
    mismatch = processor.evaluate(schema.schema_id, new_episode)
    assert mismatch["mismatch"] < processor.config.update_mismatch_threshold
    update = processor.update_schema(schema.schema_id, new_episode)
    assert update.action == "integrated"
    assert update.updated_schema.version == 2
    assert set(update.updated_schema.supporting_episode_ids) == {"r1", "r2", "r3"}
    assert update.operation_trace[0]["operation_id"] == "P09-16"
    assert processor.biological_network.hippocampus.snapshot()["episode_count"] == 3


def test_incongruent_new_episode_can_be_preserved_as_a_specialized_schema():
    """Create a child schema for a structurally distinct but potentially related experience."""
    processor, result = trained_processor(
        update_mismatch_threshold=0.10, specialization_mismatch_threshold=0.80
    )
    parent = result.schemas[0]
    update = processor.update_schema(parent.schema_id, unrelated_episode())
    assert update.action == "specialized"
    assert update.updated_schema.parent_schema_ids == (parent.schema_id,)
    assert update.updated_schema.status == "specialized"
    assert update.updated_schema.schema_id in processor.schemas
    assert update.updated_schema.schema_id in processor.schemas[parent.schema_id].child_schema_ids


def test_duplicate_episode_cannot_retrain_or_change_existing_memory():
    """Ignore identical repeated episode IDs rather than double-counting their evidence."""
    processor, result = trained_processor()
    schema = result.schemas[0]
    duplicate = restaurant_episode("r1")
    before = processor.biological_network.hippocampus.snapshot()["binding_count"]
    update = processor.update_schema(schema.schema_id, duplicate)
    after = processor.biological_network.hippocampus.snapshot()["binding_count"]
    assert update.action == "duplicate_ignored"
    assert before == after
    assert set(update.updated_schema.supporting_episode_ids) == {"r1", "r2"}


def test_biological_networks_keep_episodic_binding_separate_from_cortical_learning():
    """Verify duplicate episode protection and bounded associative plasticity surrogates."""
    episodes = [restaurant_episode("bio1"), restaurant_episode("bio2")]
    network = BiologicalSchemaNetwork(SchemaConfig(role_aliases={"waiter": "service_provider"}))
    result = network.observe(episodes)
    before = network.hippocampus.snapshot()["binding_count"]
    cortical_updates = network.cortex.update_count
    goal_observations = network.orbitofrontal.observation_count
    network.observe([episodes[0]])
    assert network.hippocampus.snapshot()["binding_count"] == before
    assert network.cortex.update_count == cortical_updates
    assert network.orbitofrontal.observation_count == goal_observations
    assert result["network_state"]["distributed_cortex"]["association_count"] > 0
    assert all(0.0 <= value <= 1.0 for value in network.cortex.associations.values())


def test_schema_frame_hierarchy_links_narrower_slots_to_broader_frames():
    """Link strict slot subsets as parent-child relations without creating duplicate memberships."""
    broad = SchemaFrame(
        "broad", "broad", {"customer": None, "meal": None, "payment_method": None}, (),
        (), {}, {}, {}, ("order",), (), 0.5,
    )
    narrow = SchemaFrame(
        "narrow", "narrow", {"customer": None, "meal": None}, (),
        (), {}, {}, {}, ("order",), (), 0.5,
    )
    hierarchy = compose_schema_hierarchy([broad, narrow], similarity_threshold=0.5)
    assert hierarchy["narrow"]["parents"] == ("broad",)
    assert hierarchy["broad"]["children"] == ("narrow",)
    assert hierarchy["narrow"]["components"] == ("customer", "meal")


def test_sqlite_indexes_every_point09_class_and_records_all_24_operations(tmp_path):
    """Index Point 09 docstrings and record a full 24-stage run in the ORE logger database."""
    processor, result = trained_processor()
    database_path = tmp_path / "point09.sqlite3"
    repository_root = Path(__file__).resolve().parents[1]
    catalog = register_annotations(database_path, repository_root)
    assert catalog["symbols_written"] > 0
    db = LoggerDatabase(database_path)
    annotation_rows = db.search_annotations("schema", limit=1000)
    assert any(row["module_name"] == "ore_schema_formation.contracts"
               and row["symbol_kind"] == "class" for row in annotation_rows)
    stored = record_result_to_sqlite(result, database_path, repository_root)
    assert stored["database_summary"]["events"] == 24
    runs = db.list_runs(limit=5)
    run = next(item for item in runs if item["run_id"] == stored["run_id"])
    assert run["status"] == "completed"
    events = db.list_events(stored["run_id"])
    assert len(events) == 24
    assert events[0]["metadata"]["operation_id"] == "P09-01"
    assert events[-1]["metadata"]["operation_id"] == "P09-24"
