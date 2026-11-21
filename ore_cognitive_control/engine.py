"""One-step Soar-inspired selection/execution cycle with hard resource bounds."""
from __future__ import annotations

from dataclasses import replace
from time import monotonic
from typing import Iterable

from .models import (
    CognitiveContext, CognitiveTrace, DecisionBudget, Operator, OperatorResult,
    StepStatus, TraceSink,
)
from .registry import OperatorRegistry


class CognitiveEngine:
    """Select and execute at most one applicable operator per invocation.

    This is deliberately a single decision step, not a hidden planning loop.
    Callers own persistence, scheduling, retries and integration adapters.
    """
    def __init__(
        self,
        registry: OperatorRegistry,
        *,
        budget: DecisionBudget | None = None,
        trace_sink: TraceSink | None = None,
    ) -> None:
        self.registry = registry
        self.budget = budget or DecisionBudget()
        self.trace_sink = trace_sink

    def _emit(self, trace: CognitiveTrace) -> CognitiveTrace:
        if self.trace_sink is not None:
            self.trace_sink(trace)
        return trace

    def step(self, context: CognitiveContext) -> CognitiveTrace:
        started = monotonic()
        if self.budget.max_operations == 0 or self.budget.max_elapsed_ms == 0:
            return self._emit(CognitiveTrace(
                context.context_id, StepStatus.BUDGET_EXHAUSTED, None, 0, 0,
                (monotonic() - started) * 1000, "decision budget is zero"
            ))

        candidates: list[tuple[float, str, str, Operator]] = []
        considered = 0
        for operator in self.registry.list_operators():
            if (monotonic() - started) * 1000 > self.budget.max_elapsed_ms:
                return self._emit(CognitiveTrace(
                    context.context_id, StepStatus.BUDGET_EXHAUSTED, None, considered, 0,
                    (monotonic() - started) * 1000, "time budget exhausted during selection"
                ))
            considered += 1
            if not operator.required_capabilities.issubset(context.available_capabilities):
                continue
            try:
                if not operator.precondition(context):
                    continue
            except Exception as exc:
                continue
            # Prefer operators explicitly relevant to the current task, then lower cost.
            relevance = float(context.task_relevance.get(operator.operator_id, 0.0))
            if not 0 <= relevance <= 1:
                relevance = 0.0
            score = relevance - (0.01 * float(operator.estimated_cost))
            candidates.append((score, operator.operator_id, operator.version, operator))

        if not candidates:
            return self._emit(CognitiveTrace(
                context.context_id, StepStatus.NO_OPERATOR, None, considered, 0,
                (monotonic() - started) * 1000, "no operator passed capability and precondition checks"
            ))

        candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
        selected = candidates[0][3]
        if (monotonic() - started) * 1000 > self.budget.max_elapsed_ms:
            return self._emit(CognitiveTrace(
                context.context_id, StepStatus.BUDGET_EXHAUSTED, selected.operator_id,
                considered, 0, (monotonic() - started) * 1000,
                "time budget exhausted before execution"
            ))
        try:
            result = selected.execute(context)
            status = StepStatus.EXECUTED if result.success else StepStatus.ERROR
            reason = result.message or ("operator completed" if result.success else "operator reported failure")
        except Exception as exc:
            result = OperatorResult(False, message=f"{type(exc).__name__}: {exc}")
            status = StepStatus.ERROR
            reason = result.message
        elapsed = (monotonic() - started) * 1000
        if elapsed > self.budget.max_elapsed_ms:
            status = StepStatus.BUDGET_EXHAUSTED
            reason = "operator exceeded elapsed-time budget; result is reported but not automatically committed"
        return self._emit(CognitiveTrace(
            context.context_id, status, selected.operator_id, considered, 1,
            elapsed, reason, result
        ))
