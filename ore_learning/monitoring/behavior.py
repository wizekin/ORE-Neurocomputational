"""Behavioral and operational regression detection (Point 14, Step 50).

Schema violations, invalid outputs, safety events, verifier failures, resource
overruns, and unexpected tool/action proposals are counted per model; the monitor
distinguishes model degradation from upstream data or infrastructure failures
through correlated failure signatures.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..config import LearningConfig

__all__ = ("BehaviorEvent", "BehaviorMonitor", "RegressionVerdict")


@dataclass(slots=True)
class BehaviorEvent:
    """One observed behavioural or operational event."""

    kind: str  # schema_violation | invalid_output | safety_event | verifier_failure
             # | resource_overrun | unexpected_action | upstream_data_error
             # | infrastructure_error
    model_id: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"kind": self.kind, "model_id": self.model_id, "detail": self.detail}


@dataclass(slots=True)
class RegressionVerdict:
    """The monitor's verdict on what regressed and where the cause likely sits."""

    model_id: str
    regression_detected: bool
    likely_cause: str  # model_degradation | upstream_data | infrastructure | none
    evidence: list[str]
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"model_id": self.model_id,
                "regression_detected": self.regression_detected,
                "likely_cause": self.likely_cause, "evidence": list(self.evidence),
                "recommendation": self.recommendation}


class BehaviorMonitor:
    """Step 50 -- aggregate events and separate model from upstream failures."""

    MODEL_EVENT_KINDS = ("schema_violation", "invalid_output", "safety_event",
                         "verifier_failure", "resource_overrun", "unexpected_action")
    UPSTREAM_KINDS = ("upstream_data_error", "infrastructure_error")

    def __init__(self, config: LearningConfig | None = None) -> None:
        """Configure thresholds."""
        self.config = config or LearningConfig()
        self.events: list[BehaviorEvent] = []

    def record(self, kind: str, model_id: str, detail: str = "") -> BehaviorEvent:
        """Record one event."""
        if kind not in self.MODEL_EVENT_KINDS + self.UPSTREAM_KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        event = BehaviorEvent(kind=kind, model_id=model_id, detail=detail)
        self.events.append(event)
        return event

    def verdict(self, model_id: str, *, total_calls: int) -> RegressionVerdict:
        """Classify the regression and its likely cause (Step 50)."""
        model_events = [event for event in self.events if event.model_id == model_id]
        upstream_events = [event for event in self.events
                           if event.kind in self.UPSTREAM_KINDS]
        schema_violations = sum(1 for event in model_events
                                if event.kind == "schema_violation")
        rate = schema_violations / max(total_calls, 1)
        evidence: list[str] = []
        regression = False
        cause = "none"
        if rate > self.config.behavior_max_schema_violation_rate:
            regression = True
            # correlated upstream errors shift the blame to data or infrastructure
            if len(upstream_events) >= schema_violations and upstream_events:
                cause = "upstream_data"
                evidence.append(f"{len(upstream_events)} upstream errors correlate "
                                "with the schema violations")
            else:
                cause = "model_degradation"
                evidence.append(f"schema violation rate {rate:.3f} exceeds "
                                f"{self.config.behavior_max_schema_violation_rate}")
        for kind in ("safety_event", "unexpected_action"):
            count = sum(1 for event in model_events if event.kind == kind)
            if count:
                regression = True
                if cause == "none":
                    cause = "model_degradation"
                evidence.append(f"{count} {kind} events")
        recommendation = ("quarantine and roll back (Step 51)" if regression
                          else "no action; keep monitoring")
        return RegressionVerdict(model_id=model_id, regression_detected=regression,
                                 likely_cause=cause, evidence=evidence,
                                 recommendation=recommendation)

    def describe(self) -> dict[str, Any]:
        """Summarize recorded events by kind."""
        counts: dict[str, int] = {}
        for event in self.events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        return {"events": len(self.events), "by_kind": counts}
