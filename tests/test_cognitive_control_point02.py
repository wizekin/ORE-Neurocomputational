from datetime import datetime, timedelta, timezone

import pytest

from ore_cognitive_control import (
    CognitiveContext, CognitiveEngine, DecisionBudget, OperatorRegistry,
    OperatorResult, RetrievalCandidate, RetrievalQuery, StepStatus,
    base_level_activation, rank_candidates,
)


class EchoOperator:
    operator_id = "echo"
    version = "1.0"
    required_capabilities = frozenset({"write"})
    estimated_cost = 1.0

    def precondition(self, context):
        return bool(context.goal)

    def execute(self, context):
        return OperatorResult(True, progress=0.5, state_patch={"done": True})


def test_activation_increases_with_recent_access():
    now = datetime.now(timezone.utc)
    old = base_level_activation([now - timedelta(days=10)], now)
    recent = base_level_activation([now - timedelta(seconds=10)], now)
    assert recent > old


def test_empty_activation_history_is_negative_infinity():
    assert base_level_activation([], datetime.now(timezone.utc)) == float("-inf")


def test_ranking_omits_unavailable_and_respects_limit():
    now = datetime.now(timezone.utc)
    candidates = [
        RetrievalCandidate("a", "A", (now,), frozenset({"alpha"}), frozenset({"parse"})),
        RetrievalCandidate("b", "B", available=False),
        RetrievalCandidate("c", "C", (now - timedelta(days=1),)),
    ]
    result = rank_candidates(candidates, RetrievalQuery(("alpha",), limit=1, now=now), frozenset({"parse"}))
    assert len(result) == 1
    assert result[0].candidate.memory_id == "a"


def test_registry_rejects_duplicate_version():
    registry = OperatorRegistry()
    registry.register(EchoOperator())
    with pytest.raises(ValueError):
        registry.register(EchoOperator())


def test_engine_executes_at_most_one_operator_and_traces():
    registry = OperatorRegistry()
    registry.register(EchoOperator())
    traces = []
    engine = CognitiveEngine(registry, trace_sink=traces.append)
    context = CognitiveContext("ctx-1", "complete task", available_capabilities=frozenset({"write"}))
    trace = engine.step(context)
    assert trace.status == StepStatus.EXECUTED
    assert trace.operations_used == 1
    assert trace.result.state_patch == {"done": True}
    assert traces == [trace]


def test_engine_does_not_execute_without_capability():
    registry = OperatorRegistry()
    registry.register(EchoOperator())
    trace = CognitiveEngine(registry).step(CognitiveContext("ctx-2", "goal"))
    assert trace.status == StepStatus.NO_OPERATOR
    assert trace.operations_used == 0


def test_zero_budget_prevents_execution():
    registry = OperatorRegistry()
    registry.register(EchoOperator())
    engine = CognitiveEngine(registry, budget=DecisionBudget(max_operations=0))
    trace = engine.step(CognitiveContext("ctx-3", "goal", available_capabilities=frozenset({"write"})))
    assert trace.status == StepStatus.BUDGET_EXHAUSTED
