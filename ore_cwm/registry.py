"""Registered operation inventory for the Cognitive World Model's 52 algorithmic steps.

The identifiers follow the repository's point-module convention (for example
P04-01..P04-22) and enumerate the steps of the Point 04 Cognitive World Model
algorithm specification, sections A through G, in source order.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CWMOperationSpec:
    """Describe one registered CWM algorithm step with its section and primary computation."""

    operation_id: str
    name: str
    section: str
    computation: str

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible operation specification."""
        return {
            "operation_id": self.operation_id,
            "name": self.name,
            "section": self.section,
            "computation": self.computation,
        }


_ROWS: tuple[CWMOperationSpec, ...] = (
    CWMOperationSpec("CWM-01", "Acquire a consistent snapshot",
                     "A. State and operation contracts",
                     "Read one workspace revision atomically without mixing records across revisions."),
    CWMOperationSpec("CWM-02", "Extract state features",
                     "A. State and operation contracts",
                     "Derive the compact versioned cognitive-state feature vector from the snapshot."),
    CWMOperationSpec("CWM-03", "Label provenance",
                     "A. State and operation contracts",
                     "Mark every feature as observed, deterministically derived, or predicted."),
    CWMOperationSpec("CWM-04", "Define CognitiveOperation",
                     "A. State and operation contracts",
                     "Validate the typed operation contract with preconditions and resource envelope."),
    CWMOperationSpec("CWM-05", "Define outcome dimensions",
                     "A. State and operation contracts",
                     "Fix goal, evidence, uncertainty, validity, cost, and side-effect metric definitions."),
    CWMOperationSpec("CWM-06", "Define PredictionRecord",
                     "A. State and operation contracts",
                     "Tie each prediction to revision, goal, operation, feature hash, and model version."),
    CWMOperationSpec("CWM-07", "Define ObservedOutcome",
                     "A. State and operation contracts",
                     "Capture actual results with identical metrics; unavailable stays missing, not zero."),
    CWMOperationSpec("CWM-08", "Preserve lineage",
                     "A. State and operation contracts",
                     "Link prediction, dispatch, execution, artifacts, validation, and observed outcome."),
    CWMOperationSpec("CWM-09", "Obtain candidates from ORE",
                     "B. Candidate generation and eligibility",
                     "Ask the existing module registry and coordinator contracts for feasible operations."),
    CWMOperationSpec("CWM-10", "Check preconditions",
                     "B. Candidate generation and eligibility",
                     "Verify required artifacts, schemas, dependencies, and module availability."),
    CWMOperationSpec("CWM-11", "Apply capability and policy checks",
                     "B. Candidate generation and eligibility",
                     "Respect sandbox, permissions, resource limits, and task constraints."),
    CWMOperationSpec("CWM-12", "Build operation context",
                     "B. Candidate generation and eligibility",
                     "Encode operation type, module, goal class, input size, and resource load."),
    CWMOperationSpec("CWM-13", "Detect unsupported candidates",
                     "B. Candidate generation and eligibility",
                     "Compare candidate features with historical coverage; widen uncertainty for novel cases."),
    CWMOperationSpec("CWM-14", "Establish conservative defaults",
                     "C. Statistical baseline prediction",
                     "Define per-operation-class fallback estimates used before learning begins."),
    CWMOperationSpec("CWM-15", "Estimate success probability",
                     "C. Statistical baseline prediction",
                     "Beta-Binomial smoothing p=(s+alpha)/(s+f+alpha+beta) with credible intervals."),
    CWMOperationSpec("CWM-16", "Estimate resource cost",
                     "C. Statistical baseline prediction",
                     "Track duration, CPU, memory, and I/O with EMA and robust dispersion statistics."),
    CWMOperationSpec("CWM-17", "Estimate information gain",
                     "C. Statistical baseline prediction",
                     "Prior entropy minus expected posterior entropy, or labeled empirical proxies."),
    CWMOperationSpec("CWM-18", "Estimate goal progress",
                     "C. Statistical baseline prediction",
                     "Reuse the existing goal evaluator's criteria, weights, and critical requirements."),
    CWMOperationSpec("CWM-19", "Estimate risk dimensions",
                     "C. Statistical baseline prediction",
                     "Predict failure, conflict, invalidation, timeout, and policy risks separately."),
    CWMOperationSpec("CWM-20", "Quantify uncertainty",
                     "C. Statistical baseline prediction",
                     "Return intervals, variance, sample counts, calibration, and OOD indicators."),
    CWMOperationSpec("CWM-21", "Define transition targets",
                     "D. Predictive transition model",
                     "Approximate observable consequences with f_theta(state, action, context)."),
    CWMOperationSpec("CWM-22", "Condition on relevant context",
                     "D. Predictive transition model",
                     "Include operation type, module, goal class, input size, and resource load."),
    CWMOperationSpec("CWM-23", "Predict outcome classes",
                     "D. Predictive transition model",
                     "Represent success, partial, no-progress, recoverable, and hard failure separately."),
    CWMOperationSpec("CWM-24", "Predict typed artifact effects",
                     "D. Predictive transition model",
                     "Predict artifact categories, schemas, and quantities, never semantic content."),
    CWMOperationSpec("CWM-25", "Simulate in isolation",
                     "D. Predictive transition model",
                     "Produce provisional state deltas in planning objects; never mutate the workspace."),
    CWMOperationSpec("CWM-26", "Apply confidence gates",
                     "D. Predictive transition model",
                     "Require minimum sample support and calibration; else conservative fallback."),
    CWMOperationSpec("CWM-27", "Bound the horizon",
                     "E. Short-horizon planning and MPC",
                     "Limit rollout depth and branching under strict candidate and time budgets."),
    CWMOperationSpec("CWM-28", "Generate feasible sequences",
                     "E. Short-horizon planning and MPC",
                     "Reuse dependency-aware enumeration satisfying module contracts and task constraints."),
    CWMOperationSpec("CWM-29", "Roll the model forward",
                     "E. Short-horizon planning and MPC",
                     "Predict each next state in sequence while propagating uncertainty."),
    CWMOperationSpec("CWM-30", "Score sequences",
                     "E. Short-horizon planning and MPC",
                     "Minimize the documented objective over cost, benefit, risk, uncertainty, and delay."),
    CWMOperationSpec("CWM-31", "Enforce hard constraints",
                     "E. Short-horizon planning and MPC",
                     "Reject sequences violating budgets, permissions, safety, or mandatory verification."),
    CWMOperationSpec("CWM-32", "Include verify and stop actions",
                     "E. Short-horizon planning and MPC",
                     "Allow verification when evidence is weak and stopping when the goal is achieved."),
    CWMOperationSpec("CWM-33", "Select a safe feasible sequence",
                     "E. Short-horizon planning and MPC",
                     "Choose the best-supported sequence with conservative risk bounds or defer."),
    CWMOperationSpec("CWM-34", "Execute only the first action",
                     "E. Short-horizon planning and MPC",
                     "Dispatch exactly one operation through the existing coordinator."),
    CWMOperationSpec("CWM-35", "Observe and replan",
                     "E. Short-horizon planning and MPC",
                     "Read the new authoritative revision, regenerate candidates, and plan again."),
    CWMOperationSpec("CWM-36", "Align predictions and outcomes",
                     "F. Prediction-error learning",
                     "Match using stable IDs and lineage across compatible metric definitions."),
    CWMOperationSpec("CWM-37", "Calculate prediction errors",
                     "F. Prediction-error learning",
                     "Track MAE, RMSE, interval coverage, Brier score, log loss, and calibration."),
    CWMOperationSpec("CWM-38", "Update estimators safely",
                     "F. Prediction-error learning",
                     "Update only validated observations with bounded updates and historical versions."),
    CWMOperationSpec("CWM-39", "Detect drift",
                     "F. Prediction-error learning",
                     "Monitor rolling error and calibration; lower confidence and isolate classes."),
    CWMOperationSpec("CWM-40", "Prevent feedback contamination",
                     "F. Prediction-error learning",
                     "Never train on predictions; label aborted, duplicated, and cancelled executions."),
    CWMOperationSpec("CWM-41", "Version models and policies",
                     "F. Prediction-error learning",
                     "Record feature schema, training window, parameters, metrics; support rollback."),
    CWMOperationSpec("CWM-42", "Learn from accepted outcomes",
                     "F. Prediction-error learning",
                     "Count success only after validation and integration status are known."),
    CWMOperationSpec("CWM-43", "Expose a narrow prediction API",
                     "G. ORE integration and safeguards",
                     "predict_outcomes(snapshot, candidates, horizon, budget) over audited contracts."),
    CWMOperationSpec("CWM-44", "Advise AdaptiveResourceCoordinator",
                     "G. ORE integration and safeguards",
                     "Supply predicted cost, benefit, and risk as additional decision evidence."),
    CWMOperationSpec("CWM-45", "Advise CognitiveTaskCoordinator",
                     "G. ORE integration and safeguards",
                     "Return ranked candidates while the coordinator stays authoritative."),
    CWMOperationSpec("CWM-46", "Use workspace revisions correctly",
                     "G. ORE integration and safeguards",
                     "Read through workspace APIs; store predictions as provisional objects only."),
    CWMOperationSpec("CWM-47", "Instrument through existing logger",
                     "G. ORE integration and safeguards",
                     "Record predictions, actions, outcomes, and model versions in SQLite."),
    CWMOperationSpec("CWM-48", "Implement deterministic fallback",
                     "G. ORE integration and safeguards",
                     "Defer to coordinator policies on timeout, stale revision, or low confidence."),
    CWMOperationSpec("CWM-49", "Bound compute",
                     "G. ORE integration and safeguards",
                     "Enforce candidate, horizon, memory, and runtime budgets with cancellation."),
    CWMOperationSpec("CWM-50", "Explain recommendations",
                     "G. ORE integration and safeguards",
                     "Report benefits, costs, risks, uncertainty, constraints, and historical support."),
    CWMOperationSpec("CWM-51", "Run shadow mode first",
                     "G. ORE integration and safeguards",
                     "Predict without changing execution; compare forecasts with actual outcomes."),
    CWMOperationSpec("CWM-52", "Introduce controlled advisory influence",
                     "G. ORE integration and safeguards",
                     "Let CWM rank candidates while ORE control retains final authority."),
)

OPERATION_REGISTRY: tuple[CWMOperationSpec, ...] = _ROWS


def registry_as_dicts() -> tuple[dict[str, Any], ...]:
    """Return every registered CWM operation as a detached dictionary."""
    return tuple(item.to_dict() for item in OPERATION_REGISTRY)


__all__ = ["OPERATION_REGISTRY", "CWMOperationSpec", "registry_as_dicts"]
