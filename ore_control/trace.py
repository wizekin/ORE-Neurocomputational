"""Decision-trace building and persistence for the control layer (Point 02 §3.4).

Each cycle records the eligible and rejected operators with machine-readable
reasons, retrieval candidates and scores, resource estimates, the selected
action with rationale, execution and validation results, and the next-cycle
reason. Traces persist through the workspace observer's ``ore_logger``-backed
store; persistence failures are recorded but never erase the in-run decision
provenance (§7 reliability tests), and payloads stay compact: hashes, IDs,
and reasons rather than large inputs.
"""
from __future__ import annotations

from typing import Any, Mapping

from .contracts import (
    CandidateRecord,
    DecisionTrace,
    SelectionRecord,
    new_control_id,
)
from .registry import OperationSpec, operation_by_step


class DecisionTraceBuilder:
    """Accumulate one cycle's decision trace (Point 02 §3.4)."""

    def __init__(
        self,
        *,
        cycle: int,
        context_id: str,
        workspace_revision: int,
        goal_id: str,
        open_goal_status: str,
    ) -> None:
        """Open one trace for the cycle being executed."""
        self.cycle = cycle
        self.context_id = context_id
        self.workspace_revision = workspace_revision
        self.goal_id = goal_id
        self.open_goal_status = open_goal_status
        self.open_question_ids: list[str] = []
        self.candidates: list[CandidateRecord] = []
        self.retrieval_candidates: list[Mapping[str, Any]] = []
        self.retrieval_shortlist_ids: list[str] = []
        self.retrieval_ambiguities: list[Mapping[str, Any]] = []
        self.resource_estimates: dict[str, Mapping[str, Any]] = {}
        self.selection: SelectionRecord | None = None
        self.execution_result: dict[str, Any] = {}
        self.validation_result: dict[str, Any] = {}
        self.integration_decisions: list[Mapping[str, Any]] = []
        self.state_transition: str = ""
        self.next_cycle_reason: str = ""
        self.timing_ms: float = 0.0
        self.cost_measurements: dict[str, float] = {}
        self.metacognitive_report: dict[str, Any] = {}
        self.operation_steps: list[Mapping[str, Any]] = []
        self.persistence_errors: list[str] = []

    # ---------------------------------------------------------------- steps

    def record_step(
        self,
        step: int,
        details: Mapping[str, Any],
        *,
        child_context_id: str | None = None,
    ) -> None:
        """Append one operation trace record for an executed algorithm step."""
        spec: OperationSpec = operation_by_step(step)
        self.operation_steps.append({
            "operation_id": spec.operation_id,
            "step": spec.step,
            "title": spec.title,
            "cycle": self.cycle,
            "child_context_id": child_context_id,
            "details": dict(details),
        })

    # ------------------------------------------------------------ finalize

    def build(self) -> DecisionTrace:
        """Freeze the cycle's decision trace record."""
        return DecisionTrace(
            trace_id=new_control_id("trace"),
            cycle=self.cycle,
            context_id=self.context_id,
            workspace_revision=self.workspace_revision,
            goal_id=self.goal_id,
            open_goal_status=self.open_goal_status,
            open_question_ids=tuple(self.open_question_ids),
            candidates=tuple(self.candidates),
            retrieval_candidates=tuple(dict(item) for item in self.retrieval_candidates),
            retrieval_shortlist_ids=tuple(self.retrieval_shortlist_ids),
            retrieval_ambiguities=tuple(
                dict(item) for item in self.retrieval_ambiguities
            ),
            resource_estimates=dict(self.resource_estimates),
            selection=self.selection,
            execution_result=dict(self.execution_result),
            validation_result=dict(self.validation_result),
            integration_decisions=tuple(
                dict(item) for item in self.integration_decisions
            ),
            state_transition=self.state_transition,
            next_cycle_reason=self.next_cycle_reason,
            timing_ms=self.timing_ms,
            cost_measurements=dict(self.cost_measurements),
            metacognitive_report=dict(self.metacognitive_report),
            operation_steps=tuple(dict(item) for item in self.operation_steps),
        )


def persist_trace(trace: DecisionTrace, observer: Any) -> bool:
    """Persist one decision trace through the workspace observer (§3.4).

    The observer is the repository's ``ore_logger``-backed recording path. A
    persistence failure is reported as ``False`` with the error retained by
    the caller; the in-run trace itself always survives.
    """
    if observer is None:
        return False
    try:
        observer.record(
            "control_cycle_decision",
            trace.to_dict(),
            class_name="CognitiveControlCoordinator",
            module_name="ore_control.cycle",
        )
        return True
    except Exception as error:  # noqa: BLE001 - reliability boundary (§7)
        return _persistence_failed(error)


def _persistence_failed(error: BaseException) -> bool:
    """Record that persistence failed without erasing in-run provenance."""
    import logging

    logging.getLogger(__name__).debug(
        "decision-trace persistence failed: %s", error,
    )
    return False


__all__ = ["DecisionTraceBuilder", "persist_trace"]
