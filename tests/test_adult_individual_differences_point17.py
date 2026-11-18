"""Focused validation tests for the 15-stage adult individual-differences model."""
from __future__ import annotations

import sqlite3

import pytest

from ore_adult_individual_differences import (
    AdultComprehensionBiologicalNetwork,
    AdultComprehensionExperience,
    AdultDifferencesConfig,
    AdultIndividualDifferencesProcessor,
    AdultLearnerState,
    OPERATION_REGISTRY,
    adapt_processing_support,
    estimate_current_resources,
    incorporate_cultural_and_lived_experience,
    monitor_interference_and_hypotheses,
    retrieve_gist_detail_source_memory,
    update_knowledge_without_false_assimilation,
)


def experience(**overrides):
    """Build a task with domain knowledge, language cues, context, memory, and feedback."""
    values = {
        "experience_id": "p17-episode-001",
        "content_units": ("Thermal", "expansion", "occurs", "when", "metal", "is", "heated", "."),
        "domain": "physics",
        "task_goal": {"detail": 0.35, "gist": 0.20, "relation": 0.30, "procedure": 0.05, "source": 0.10},
        "desired_output": "explain_cause",
        "language": "language-a",
        "language_cues": {"word_order": 1.0, "case_marking": 0.9},
        "interpretation_candidates": (
            {
                "hypothesis_id": "thermal-expansion",
                "label": "heating raises particle motion and average separation",
                "prior": 0.55,
                "cue_likelihoods": {"word_order": 0.95, "case_marking": 0.85},
            },
            {
                "hypothesis_id": "contraction",
                "label": "heating reduces average separation",
                "prior": 0.45,
                "cue_likelihoods": {"word_order": 0.15, "case_marking": 0.25},
            },
        ),
        "dependency_pairs": ((0, 1), (1, 5), (5, 6)),
        "pace_pressure": 0.15,
        "distraction": 0.10,
        "fatigue": 0.10,
        "stress": 0.05,
        "motivation": 0.8,
        "external_support": 0.2,
        "time_budget_ms": 800.0,
        "task_context": {
            "goal_focus": 0.9,
            "wrong_commitment_cost": 0.9,
            "unresolved_references": 1,
            "simultaneous_alternatives": 2,
            "proposition_count": 3,
        },
        "cultural_cues": ("shared-laboratory-practice",),
        "cultural_conventions": {
            "lab-report-convention": {
                "cues": ("shared-laboratory-practice",),
                "interpretation": "report the measured result before the explanation",
                "reliability": 0.9,
                "applicability": 0.9,
                "source": "task-specific-practice",
                "explicitly_demonstrated": True,
            }
        },
        "context_observations": {"shared-laboratory-practice": 0.9},
        "memory_request": "balanced",
        "age_years": 42,
        "memory_evidence": {
            "gist_cue_quality": 0.9,
            "detail_cue_quality": 0.75,
            "source_cue_quality": 0.65,
            "encoding_time_ms": 900.0,
            "interference": 0.1,
            "distinctiveness": 0.8,
        },
        "new_facts": ({
            "fact_id": "thermal-expansion-rule",
            "value": "positive-temperature-coefficient",
            "reliability": 0.95,
            "repetitions": 2,
            "novelty": 0.8,
            "source_id": "lesson-source-1",
        },),
        "evidence_quality": 0.9,
        "provenance_completeness": 0.95,
        "performance_feedback": {"accuracy": 0.9},
    }
    values.update(overrides)
    return AdultComprehensionExperience(**values)


def learner_state(**overrides):
    """Create an individual learner profile with distinct domain and language history."""
    values = {
        "learner_id": "learner-p17",
        "attention_control": 0.8,
        "working_memory_support": 0.72,
        "processing_speed": 0.78,
        "prior_knowledge": {
            "physics": [{
                "schema_id": "thermal-expansion-schema",
                "schema": "thermal expansion",
                "features": ("thermal", "expansion", "metal", "heated", "temperature"),
                "reliability": 0.95,
                "goal_fit": 0.95,
                "recency": 0.9,
                "conflict": 0.0,
                "source_id": "prior-lesson",
                "chunks": ({
                    "chunk_id": "thermal-expansion-pattern",
                    "label": "thermal expansion relation",
                    "features": ("thermal", "expansion"),
                    "schema_cost": 0.5,
                    "exception_cost": 0.1,
                    "exceptions": ("some materials have unusual thermal behavior",),
                },),
                "relations": ({"relation": "temperature-increase-causes-expansion"},),
            }],
            "music": [{
                "schema_id": "music-pattern",
                "schema": "music",
                "features": ("rhythm", "melody"),
                "reliability": 0.99,
            }],
        },
        "expertise_by_domain": {"physics": 0.85, "music": 0.9},
        "language_proficiency": {"language-a": 0.95, "language-b": 0.45},
        "language_cue_reliability": {
            "language-a": {"word_order": 0.95, "case_marking": 0.9},
            "language-b": {"word_order": 0.6, "case_marking": 0.5},
        },
        "cultural_experience": {
            "demonstrated-lab-context": {
                "conventions": {
                    "lab-report-convention": {
                        "cues": ("shared-laboratory-practice",),
                        "interpretation": "report measurements before conclusions",
                        "reliability": 0.95,
                        "source": "observed-practice",
                    }
                }
            }
        },
        "component_speed": {
            "perception": 0.8, "retrieval": 0.75, "integration": 0.7, "decision": 0.85,
        },
    }
    values.update(overrides)
    return AdultLearnerState(**values)


def test_registry_has_exactly_fifteen_source_ordered_stages():
    """Keep the operation registry aligned with all 15 source steps."""
    assert len(OPERATION_REGISTRY) == 15
    assert [item.operation_id for item in OPERATION_REGISTRY] == [
        f"P17-{index:02d}" for index in range(1, 16)
    ]


def test_task_goal_is_explicit_normalized_and_feature_prioritized():
    """The task objective should determine feature priorities rather than content alone."""
    result = AdultIndividualDifferencesProcessor().process(experience(
        metadata={"feature_goal_support": {
            "numeric-detail": {"detail": 1.0, "gist": 0.0},
            "causal-link": {"relation": 1.0, "detail": 0.2},
        }}
    ))
    goal = result.task_goal
    assert sum(goal["objective_vector"].values()) == pytest.approx(1.0)
    assert goal["feature_priorities"]["numeric-detail"] > goal["feature_priorities"]["causal-link"] * 0.5
    assert goal["goal_is_explicit_not_inferred_from_content"] is True


def test_current_resources_change_with_interference_and_external_support():
    """Resource estimates should be task-sensitive and external structure can support maintenance."""
    learner = learner_state()
    config = AdultDifferencesConfig()
    calm = estimate_current_resources(
        experience(distraction=0.0, fatigue=0.0, stress=0.0, external_support=0.0),
        learner, config,
    )
    overloaded = estimate_current_resources(
        experience(distraction=0.95, fatigue=0.9, stress=0.8, external_support=0.0),
        learner, config,
    )
    supported = estimate_current_resources(
        experience(distraction=0.95, fatigue=0.9, stress=0.8, external_support=1.0),
        learner, config,
    )
    assert overloaded["attention_control"] < calm["attention_control"]
    assert overloaded["working_memory_availability"] < calm["working_memory_availability"]
    assert supported["working_memory_availability"] > overloaded["working_memory_availability"]
    assert overloaded["permanent_trait_or_diagnosis_inferred"] is False


def test_segmentation_preserves_order_source_and_boundary_uncertainty():
    """Tokenization should preserve original source links and uncertain boundaries."""
    result = AdultIndividualDifferencesProcessor().process(experience(
        content_units=({"text": "The metal", "source_id": "page-2", "confidence": 0.9},
                       "expanded slowly"),
        metadata={"boundary_cues": ({"strength": 0.5, "syntax": 0.5},)},
    ))
    units = result.segmented_units["units"]
    assert [row["content"] for row in units[:4]] == ["The", "metal", "expanded", "slowly"]
    assert all(row["source_id"] in {"page-2", "input-1"} for row in units)
    assert result.segmented_units["source_order_preserved"] is True
    assert result.segmented_units["provenance_retained"] is True


def test_single_string_input_is_treated_as_one_input_segment_not_characters():
    """A raw string is normalized as one input item before tokenization."""
    result = AdultIndividualDifferencesProcessor().process(experience(
        content_units="Thermal expansion occurs.",
    ))
    assert [row["content"] for row in result.segmented_units["units"]] == [
        "Thermal", "expansion", "occurs", "."
    ]


def test_schema_retrieval_is_domain_specific_and_reliability_weighted():
    """The physics schema should be selected without importing the unrelated music expertise."""
    result = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience())
    selected = result.retrieved_knowledge["selected_schema"]
    assert selected["schema_id"] == "thermal-expansion-schema"
    assert selected["reliability"] == pytest.approx(0.95)
    assert result.retrieved_knowledge["expertise_is_domain_specific"] is True


def test_expertise_chunking_reduces_cost_without_discarding_exceptions():
    """A matching schema can compress familiar details while retaining its exceptions."""
    result = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience())
    chunking = result.chunking
    assert chunking["chunks"]
    assert chunking["effective_unit_count"] < chunking["original_unit_count"]
    assert chunking["chunks"][0]["exceptions_retained"]
    assert chunking["exceptions_and_novel_details_preserved"] is True


def test_dependency_interactivity_increases_working_memory_load_pressure():
    """More dependencies should increase coordination demand for the same input."""
    processor = AdultIndividualDifferencesProcessor(state=learner_state())
    low = processor.process(experience(dependency_pairs=()))
    high = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience(
        dependency_pairs=((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)),
    ))
    assert high.working_memory_load["load_estimate"] > low.working_memory_load["load_estimate"]
    assert high.working_memory_load["dependency_count"] > low.working_memory_load["dependency_count"]


def test_processing_schedule_estimates_components_and_slower_speed_increases_duration():
    """The model should expose component time rather than a single global speed score."""
    slow_state = learner_state(processing_speed=0.25, component_speed={
        "perception": 0.2, "retrieval": 0.2, "integration": 0.2, "decision": 0.2,
    })
    fast_state = learner_state(processing_speed=0.9, component_speed={
        "perception": 0.9, "retrieval": 0.9, "integration": 0.9, "decision": 0.9,
    })
    slow = AdultIndividualDifferencesProcessor(state=slow_state).process(experience(time_budget_ms=300))
    fast = AdultIndividualDifferencesProcessor(state=fast_state).process(experience(time_budget_ms=300))
    assert set(slow.processing_schedule["component_times_ms"]) == {
        "perception", "retrieval", "integration", "decision"
    }
    assert slow.processing_schedule["total_processing_time_ms"] > fast.processing_schedule["total_processing_time_ms"]
    assert slow.processing_schedule["temporal_pressure"] > fast.processing_schedule["temporal_pressure"]
    assert slow.processing_schedule["processing_speed_is_not_intelligence_or_knowledge"] is True


def test_language_cue_weights_depend_on_proficiency_and_observed_cue_strength():
    """Language cue reliability should be learned and proficiency-sensitive."""
    high = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience())
    low_state = learner_state(language_proficiency={"language-a": 0.2})
    low = AdultIndividualDifferencesProcessor(state=low_state).process(experience())
    high_probs = [row["probability"] for row in high.language_interpretation["ranked_interpretations"]]
    low_probs = [row["probability"] for row in low.language_interpretation["ranked_interpretations"]]
    assert high_probs[0] > low_probs[0]
    assert high.language_interpretation["cue_weights_are_experience_specific"] is True
    assert high.language_interpretation["no_universal_bilingual_advantage_or_disadvantage_assumed"] is True


def test_neutral_language_evidence_does_not_force_candidate_selection():
    """Neutral cue evidence leaves candidate ranking close to the supplied priors."""
    result = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience(
        language_cues={"word_order": 0.5, "case_marking": 0.5},
    ))
    ranked = result.language_interpretation["ranked_interpretations"]
    assert ranked[0]["probability"] == pytest.approx(0.55)
    assert ranked[1]["probability"] == pytest.approx(0.45)


def test_cultural_conventions_require_individual_demonstrated_context():
    """The system should use matching learned conventions and avoid filling absent cultural knowledge."""
    state = learner_state()
    result = AdultIndividualDifferencesProcessor(state=state).process(experience())
    assert result.cultural_interpretation["context_hypotheses"][0]["context_id"] == "lab-report-convention"
    unknown = AdultIndividualDifferencesProcessor(state=AdultLearnerState()).process(experience(
        cultural_cues=(), cultural_conventions={}, context_observations={},
    ))
    assert unknown.cultural_interpretation["context_hypotheses"] == []
    assert unknown.cultural_interpretation["stereotype_based_inference_used"] is False


def test_integrated_representation_separates_observed_retrieved_and_inferred_content():
    """A retrieved schema or cultural convention must not become an observed utterance."""
    result = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience())
    integrated = result.integrated_representation
    assert integrated["observed"]
    assert integrated["retrieved"]
    assert integrated["inferred"]
    assert integrated["source_and_inference_boundaries_preserved"] is True
    assert all(row["provenance"] != "observed" for row in integrated["inferred"])


def test_interference_monitor_maintains_alternatives_when_evidence_is_close():
    """Close candidate probabilities and costly errors should defer interpretation commitment."""
    result = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience(
        interpretation_candidates=(
            {"hypothesis_id": "a", "label": "interpretation A", "prior": 0.5,
             "cue_likelihoods": {"word_order": 0.5, "case_marking": 0.5}},
            {"hypothesis_id": "b", "label": "interpretation B", "prior": 0.5,
             "cue_likelihoods": {"word_order": 0.5, "case_marking": 0.5}},
        ),
        task_context={"wrong_commitment_cost": 1.0},
    ))
    monitor = result.interference_monitor
    assert monitor["defer_commitment"] is True
    assert len(monitor["live_hypotheses"]) == 2
    assert monitor["normalized_entropy"] == pytest.approx(1.0)


def test_adaptive_support_changes_with_expertise_and_load():
    """Novices receive more scaffolding while experts receive less redundant guidance."""
    event = experience()
    config = AdultDifferencesConfig()
    novice = learner_state(expertise_by_domain={"physics": 0.0}, prior_knowledge={})
    expert = learner_state(expertise_by_domain={"physics": 1.0})
    processor = AdultIndividualDifferencesProcessor(config=config)
    result = processor.process(event, state=novice)
    novice_support = result.adaptive_support
    expert_result = AdultIndividualDifferencesProcessor(config=config, state=expert).process(
        experience(content_units=("Expansion",), dependency_pairs=(), external_support=1.0,
                   distraction=0.0, fatigue=0.0, stress=0.0,
                   task_context={"goal_focus": 0.95, "wrong_commitment_cost": 0.1,
                                 "unresolved_references": 0, "simultaneous_alternatives": 1,
                                 "proposition_count": 1})
    )
    assert novice_support["support_level"] > expert_result.adaptive_support["support_level"]
    assert novice_support["strategy"] == "explicit_scaffolding"
    assert expert_result.adaptive_support["expertise_is_domain_specific"] is True


def test_gist_detail_and_source_memory_are_separate_and_not_age_deterministic():
    """Age metadata alone should not alter memory estimates when actual cue evidence is matched."""
    evidence = {
        "gist_cue_quality": 0.9, "detail_cue_quality": 0.65, "source_cue_quality": 0.4,
        "encoding_time_ms": 800.0, "interference": 0.2, "distinctiveness": 0.7,
    }
    young = retrieve_gist_detail_source_memory(experience(age_years=30, memory_evidence=evidence))
    older = retrieve_gist_detail_source_memory(experience(age_years=85, memory_evidence=evidence))
    assert young["gist_probability"] == pytest.approx(older["gist_probability"])
    assert young["episodic_detail_probability"] == pytest.approx(older["episodic_detail_probability"])
    assert young["source_context_probability"] == pytest.approx(older["source_context_probability"])
    assert len({
        young["gist_probability"], young["episodic_detail_probability"],
        young["source_context_probability"],
    }) > 1
    assert older["age_is_not_a_deterministic_memory_penalty"] is True


def test_knowledge_update_preserves_conflict_and_records_novel_exception():
    """A single conflicting source should be retained as an exception without overwriting prior knowledge."""
    state = learner_state(knowledge_estimates={
        "physics": {
            "thermal-law": {
                "value": "linear-expansion",
                "confidence": 0.95,
                "value_counts": {"'linear-expansion'": 4},
                "value_examples": {"'linear-expansion'": "linear-expansion"},
                "sources": ["trusted-source"],
                "exceptions": [],
                "update_count": 0,
            }
        }
    })
    result = update_knowledge_without_false_assimilation(
        experience(new_facts=({
            "fact_id": "thermal-law", "value": "unusual-nonlinear-expansion",
            "reliability": 0.95, "repetitions": 2, "novelty": 0.95,
            "source_id": "new-experiment",
        },)),
        state, AdultDifferencesConfig(),
    )
    assert result["updates"][0]["classification"] == "novel_exception"
    assert result["knowledge_estimates"]["thermal-law"]["value"] == "linear-expansion"
    assert len(result["exceptions"]) == 1
    assert result["source_provenance_preserved"] is True


def test_confidence_calibration_accounts_for_evidence_provenance_and_uncertainty():
    """Strong evidence and source completeness should improve confidence, all else equal."""
    strong = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience(
        evidence_quality=1.0, provenance_completeness=1.0,
    ))
    weak = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience(
        evidence_quality=0.3, provenance_completeness=0.3,
    ))
    assert strong.calibrated_decision["calibrated_confidence"] > weak.calibrated_decision["calibrated_confidence"]
    assert strong.calibrated_decision["next_action"]
    assert strong.calibrated_decision["no_individual_ability_or_age_diagnosis"] is True


def test_full_pipeline_has_fifteen_ordered_operations_and_separate_networks():
    """Run all source stages and return biological outputs for the relevant distributed systems."""
    result = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience())
    assert len(result.operation_trace) == 15
    assert [row["operation_id"] for row in result.operation_trace] == [
        item.operation_id for item in OPERATION_REGISTRY
    ]
    assert "goal_attention_control" in result.biological
    assert "distributed_language_network" in result.biological
    assert "gist_detail_source_memory" in result.biological
    assert result.biological["scientific_scope"].startswith("Bounded rate-coded")


def test_rate_coded_network_uses_recurrence_and_inhibition():
    """Network activation should carry prior activity into later processing events."""
    network = AdultComprehensionBiologicalNetwork()
    first = network.engage("frontoparietal_working_memory", "maintain", 0.7)
    second = network.engage("frontoparietal_working_memory", "update", 0.1)
    assert second["activation_before"] == pytest.approx(first["activation_after"])
    assert second["activation_after"] != pytest.approx(first["activation_after"])
    assert "recurrent_gain" in network.snapshot()["frontoparietal_working_memory"]["parameters"]


def test_sqlite_registers_package_annotations_and_logs_all_fifteen_steps(tmp_path):
    """Index classes/functions and persist all 15 operation outputs in the ORE SQLite logger."""
    from ore_logger.database import LoggerDatabase
    from ore_adult_individual_differences import register_annotations, record_result_to_sqlite

    path = tmp_path / "point17.sqlite3"
    result = AdultIndividualDifferencesProcessor(state=learner_state()).process(experience())
    catalogue = register_annotations(path)
    assert catalogue["files_seen"] >= 7
    assert LoggerDatabase(path).summary()["symbols"] > 0
    logged = record_result_to_sqlite(result, path)
    with sqlite3.connect(path) as connection:
        run = connection.execute(
            "SELECT status, event_count FROM runs WHERE run_id = ?", (logged["run_id"],)
        ).fetchone()
        count = connection.execute(
            "SELECT COUNT(*) FROM events WHERE run_id = ?", (logged["run_id"],)
        ).fetchone()[0]
    assert run == ("completed", 15)
    assert count == 15


def test_contract_rejects_invalid_probabilities_and_invalid_adult_age():
    """Reject malformed goals, resource values, and non-adult age metadata."""
    with pytest.raises(ValueError):
        AdultComprehensionExperience(experience_id="bad", content_units=("text",),
                                     domain="x", task_goal={"detail": -0.2, "gist": 0.1})
    with pytest.raises(ValueError):
        AdultComprehensionExperience(experience_id="bad-age", content_units=("text",),
                                     domain="x", age_years=12)
    with pytest.raises(ValueError):
        AdultLearnerState(attention_control=1.5)
    with pytest.raises(ValueError):
        AdultDifferencesConfig(speed_decay_tau_ms=0.0)
