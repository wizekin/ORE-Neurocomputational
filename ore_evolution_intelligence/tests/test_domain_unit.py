"""Domain-layer unit tests (steps 2.1, 5-8; section 11 lifecycle states)."""
from __future__ import annotations

import pytest

from ore_evolution_intelligence.domain.actions import (
    ACTION_VOCABULARY, ORE_MODULES, PROHIBITED_ACTIONS, ActionCall,
    validate_action,
)
from ore_evolution_intelligence.domain.behavior_descriptor import (
    DEFAULT_CALIBRATION_SAMPLES, DESCRIPTOR_DIMENSIONS, BehaviorDescriptor,
    DescriptorCalibration, DescriptorDimension, default_calibration,
)
from ore_evolution_intelligence.domain.evaluation_record import (
    OUTCOME_CREDIT, OUTCOME_KINDS, AggregateMetrics, HardConstraints,
    ScalarFitness, TaskOutcomeRecord, t95,
)
from ore_evolution_intelligence.domain.genome import (
    ParameterGenome, PolicyTableGenome, default_search_space, table_context_key,
)
from ore_evolution_intelligence.domain.strategy import (
    LIFECYCLE_STATES, LineageRecord, StrategyCandidate,
)

from .fixtures import make_baseline_genome


class TestActionVocabulary:
    """Step 8: explicit action vocabulary with typed parameters."""

    def test_approved_actions_validate(self):
        call = ActionCall("activate_module", {"module": "ore_semantic"})
        assert validate_action(call.name, call.params) == []

    def test_unknown_action_rejected(self):
        errors = validate_action("deploy_to_production", {})
        assert any("not in the approved vocabulary" in e for e in errors)

    def test_privileged_actions_rejected(self):
        for name in PROHIBITED_ACTIONS:
            errors = validate_action(name, {})
            assert errors, name

    def test_typed_parameter_violations_rejected(self):
        errors = validate_action("activate_module", {"module": "not_a_module"})
        assert errors
        errors = validate_action("request_retrieval", {"depth": 9})
        assert errors

    def test_action_call_hashable_and_deduplicated(self):
        a = ActionCall("activate_module", {"module": "ore_syntax"})
        b = ActionCall("activate_module", {"module": "ore_syntax"})
        assert len({a, b}) == 1


class TestParameterGenome:
    """Step 7, 2.1: bounded numeric parameter vectors, serializable and schema-valid."""

    def test_defaults_are_valid(self):
        genome = make_baseline_genome()
        assert genome.validate() == []
        assert genome.kind == "parameters"

    def test_out_of_range_rejected(self):
        genome = make_baseline_genome()
        bad = genome.clone_with({"cpu_budget_share": 0.6})
        bad.values["cpu_budget_share"] = 5.0
        assert bad.validate() != []

    def test_unknown_parameter_rejected(self):
        space = default_search_space()
        values = {d.name: d.default_value() for d in space}
        values["surprise__param"] = 1.0
        with pytest.raises(ValueError):
            ParameterGenome(values, space)

    def test_serialization_round_trip(self):
        genome = make_baseline_genome()
        restored = ParameterGenome.from_dict(genome.as_dict())
        assert restored.content_payload() == genome.content_payload()

    def test_continuous_names_for_cmaes_guard(self):
        genome = make_baseline_genome()
        # The default space has one integer and one choice parameter.
        assert len(genome.continuous_names()) < len(genome.values)


class TestPolicyTableGenome:
    """Step 7: finite policy tables over the bounded context schema."""

    def test_valid_table(self):
        key = table_context_key("reading", "high", True, "normal")
        table = PolicyTableGenome({
            key: {ActionCall("activate_module", {"module": "ore_reading"})},
        })
        assert table.validate() == []

    def test_invalid_context_key_rejected(self):
        table = PolicyTableGenome({("unknown", "high", "no", "normal"): set()})
        assert table.validate() != []

    def test_prohibited_action_in_table_rejected(self):
        with pytest.raises(ValueError):
            PolicyTableGenome({
                table_context_key("reading", "high", False, "normal"):
                    {ActionCall("shell_execute", {})},
            })

    def test_table_round_trip(self):
        key = table_context_key("discourse", "medium", False, "pressured")
        table = PolicyTableGenome({
            key: {
                ActionCall("activate_module", {"module": "ore_discourse"}),
                ActionCall("request_retrieval", {"depth": 2}),
            },
        })
        restored = PolicyTableGenome.from_dict(table.as_dict())
        assert restored.content_payload() == table.content_payload()


class TestStrategyCandidate:
    """Section 2.1/11: candidate representation and strict lifecycle states."""

    def test_candidate_construction(self):
        candidate = StrategyCandidate.from_genome(
            make_baseline_genome(),
            lineage=LineageRecord(operation="initial", generation=0),
        )
        assert candidate.state == "GENERATED"
        assert candidate.candidate_id.startswith("cand_")
        assert len(candidate.content_hash) == 64

    def test_state_must_be_known(self):
        with pytest.raises(ValueError):
            candidate = StrategyCandidate.from_genome(make_baseline_genome())
            object.__setattr__(candidate, "state", "MYSTICAL")
            StrategyCandidate(
                genome=make_baseline_genome(),
                capabilities=frozenset(),
                state="MYSTICAL",
            )

    def test_lifecycle_states_match_specification(self):
        assert LIFECYCLE_STATES == (
            "GENERATED", "SANDBOX_TESTED", "VALIDATED", "APPROVED",
            "CANARY", "ACTIVE", "DEPRECATED", "REJECTED",
        )

    def test_candidate_serialization_round_trip(self):
        candidate = StrategyCandidate.from_genome(make_baseline_genome())
        restored = StrategyCandidate.from_dict(candidate.as_dict())
        assert restored.content_hash == candidate.content_hash
        assert restored.state == candidate.state

    def test_identical_genomes_share_hash(self):
        a = StrategyCandidate.from_genome(make_baseline_genome())
        b = StrategyCandidate.from_genome(make_baseline_genome())
        assert a.candidate_id == b.candidate_id


class TestBehaviorDescriptor:
    """Steps 39-40: behavior descriptors and versioned calibration."""

    def test_calibration_covers_all_dimensions(self):
        calibration = default_calibration()
        assert set(d.name for d in calibration.dimensions) == set(DESCRIPTOR_DIMENSIONS)

    def test_discretization_reproducible(self):
        calibration = default_calibration()
        again = DescriptorCalibration.from_samples(
            DEFAULT_CALIBRATION_SAMPLES, version=calibration.version,
        )
        assert again.boundaries() == calibration.boundaries()

    def test_missing_measurements_handled_explicitly(self):
        calibration = default_calibration()
        descriptor = BehaviorDescriptor(
            {"speed": None, "reasoning_depth": 4.0, "resource_tier": 0.5,
             "reliability": 0.8, "module_cooperation": 0.6},
            calibration.version,
        )
        coordinates, problems = descriptor.cell_coordinates(calibration)
        assert coordinates[0] is None
        assert problems and "missing measurement" in problems[0]

    def test_out_of_range_handled_explicitly(self):
        dimension = DescriptorDimension("speed", 100.0, 500.0, 4)
        _bin, problem = dimension.discretize(900.0)
        assert problem and "out-of-range" in problem

    def test_distance_symmetric(self):
        a = BehaviorDescriptor({"speed": 200.0, "reasoning_depth": 4.0,
                                "resource_tier": 0.5, "reliability": 0.8,
                                "module_cooperation": 0.6}, "v1")
        b = BehaviorDescriptor({"speed": 300.0, "reasoning_depth": 4.0,
                                "resource_tier": 0.5, "reliability": 0.8,
                                "module_cooperation": 0.6}, "v1")
        assert a.distance(b) == b.distance(a) > 0.0


class TestEvaluationRecordTypes:
    """Sections 2.2, step 5: outcome semantics, gates, scalar fitness."""

    def test_outcome_kinds_match_specification(self):
        assert OUTCOME_KINDS == (
            "SUCCESS", "PARTIAL", "FAILURE", "ABSTAIN", "TIMEOUT", "INVALID",
        )

    def test_aggregate_from_outcomes(self):
        outcomes = [
            TaskOutcomeRecord(
                task_id="reading_01", trial=0, seed=1, outcome="SUCCESS",
                quality=0.9, latency_ms=100.0, memory_mb=20.0, cpu_ms=90.0,
                cost=12.0, safety_violations=0,
                evidence={"active_modules": 3},
            ),
            TaskOutcomeRecord(
                task_id="reading_02", trial=1, seed=2, outcome="PARTIAL",
                quality=0.5, latency_ms=150.0, memory_mb=22.0, cpu_ms=120.0,
                cost=14.0, safety_violations=0,
                evidence={"active_modules": 2},
            ),
        ]
        metrics = AggregateMetrics.from_outcomes(outcomes, n_tasks=2)
        assert metrics.quality == pytest.approx(0.7)
        assert metrics.reliability == 1.0
        assert metrics.cost_mean == pytest.approx(13.0)
        assert metrics.n_trials == 2

    def test_hard_constraints_gate_safety_regardless_of_quality(self):
        metrics = AggregateMetrics(
            quality=0.99, reliability=1.0, latency_mean_ms=50.0,
            latency_p95_ms=60.0, memory_peak_mb=30.0, cpu_total_ms=100.0,
            cost=20.0, cost_mean=10.0, safety_violations=2, failure_rate=0.0,
            timeout_rate=0.0, invalid_rate=0.0, abstain_rate=0.0,
            quality_std=0.01, quality_ci95=0.01, worst_quality=0.9,
            n_trials=9, n_tasks=3, mean_active_modules=3.0, cooperation=0.9,
        )
        gate = HardConstraints().check(metrics)
        assert not gate.feasible
        assert any("safety" in reason for reason in gate.reasons)

    def test_scalar_fitness_documented_weights(self):
        fitness = ScalarFitness()
        assert fitness.w_quality == 1.0  # documented policy choice

    def test_t95_known_values(self):
        assert t95(1) == pytest.approx(12.706, rel=1e-3)
        assert t95(100) == pytest.approx(1.96)
