"""Stable operation registry for the ORE integrated blackboard coordination algorithm.

Every operation ID below corresponds to one numbered step of Algorithm 1 in
``ORE_Integrated_Algorithm_Point_01.md`` (Priority 1 / Source Point 1: blackboard
architecture and shared cognitive workspace). The IDs follow the repository-wide
traceability convention: stable, never recycled, and usable for runtime tracing.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one blackboard coordination operation and its source step."""

    operation_id: str
    step: int
    title: str
    purpose: str


_BLACKBOARD_OPERATIONS: tuple[tuple[int, str, str, str], ...] = (
    (1, "Validate the task request and establish a run boundary",
     "Normalize the request, fingerprint it, and treat the task identity as an idempotency key."),
    (2, "Create one isolated shared workspace for the task",
     "Separate run identity from workspace identity and create revision-zero shared state."),
    (3, "Build the logical blackboard layers",
     "Classify records across eight logical layers with one-hot layer vectors and lifecycle order."),
    (4, "Preserve the original input as the immutable evidence root",
     "Digest the input, validate source spans, and preserve source-group independence."),
    (5, "Attach a verified USG snapshot as a read-only knowledge resource",
     "Verify the bundle digest and publish only a compact read-only reference to the workspace."),
    (6, "Register specialist modules and their authority boundaries",
     "Validate manifests, capabilities, and the module-field write authority matrix."),
    (7, "Construct the dependency and compatibility graph",
     "Plan a DAG with exact contract versions and return blocked results for mandatory gaps."),
    (8, "Create task-specific module subscriptions",
     "Register the subscription matrix with field sensitivity weights and activation thresholds."),
    (9, "Establish the baseline goal requirements and completion evaluator",
     "Configure weighted coverage plus explicit critical requirement hard gates."),
    (10, "Take a detached workspace snapshot for each module computation",
     "Give every specialist an immutable revision snapshot with an invocation identity."),
    (11, "Measure what changed in the workspace",
     "Compute field-specific non-negative change magnitudes with recorded metrics."),
    (12, "Compute module relevance from subscriptions and changed fields",
     "Apply R_m = sum_f A_mf * w_mf * Delta_f and order eligible modules deterministically."),
    (13, "Dispatch a committed workspace event to relevant subscribers",
     "Deliver committed events to subscribed modules with deduplicated, bounded callbacks."),
    (14, "Estimate the value and cost of further processing",
     "Combine reliability history, heuristic information gain, and a labelled comparison score."),
    (15, "Allocate a bounded execution plan",
     "Select modules under worker, call, activation, time, and memory budgets."),
    (16, "Execute specialist computations against immutable inputs",
     "Invoke registered backends under timeouts and validate typed contract outputs."),
    (17, "Convert module outputs into provenance-complete findings",
     "Bridge typed artifacts into findings with lineage, spans, rules, and USG references."),
    (18, "Validate every proposed publication before it enters shared state",
     "Enforce the nine publication invariants through the workspace state manager."),
    (19, "Defer findings whose prerequisites have not arrived",
     "Keep well-formed findings pending and reconsider them when dependencies are accepted."),
    (20, "Merge compatible findings using the target field's merge policy",
     "Apply field-specific merge operators and record merge inputs deterministically."),
    (21, "Preserve contradictions and competing hypotheses",
     "Register conflicts and aggregate transparent evidence scores without forced resolution."),
    (22, "Commit accepted changes atomically and advance the workspace revision",
     "Commit validated merges under the workspace lock and publish post-commit events."),
    (23, "Route the committed change to the next set of specialists",
     "Activate only relevant subscribers and suppress redundant recomputation."),
    (24, "Execute another bounded blackboard cycle",
     "Repeat snapshot, selection, execution, publication, and commit within budgets."),
    (25, "Evaluate the goal after every meaningful revision",
     "Combine weighted coverage with critical-requirement hard gates."),
    (26, "Recover safely from timeouts, errors, and unavailable specialists",
     "Classify failures, retry within a shared budget, and respect circuit breakers."),
    (27, "Reconstruct the reasoning and provenance trace",
     "Build the auditable explanation graph and measure trace completeness."),
    (28, "Produce the task result without collapsing epistemic categories",
     "Return observations, imported knowledge, hypotheses, alternatives, and conclusions separately."),
    (29, "Measure integration performance against a sequential baseline",
     "Compare task quality, activations, latency, recovery, and provenance metrics."),
    (30, "Run ablations to identify which integration links actually help",
     "Disable one integration mechanism at a time and measure its contribution."),
    (31, "Keep learned policy updates outside ordinary inference",
     "Record measurements and policy proposals as separate transactions with gates."),
    (32, "Close the workspace/run cleanly and retain reproducibility metadata",
     "Flush executors, retain state, and expose the reproducibility tuple."),
)


def _operation_id(step: int) -> str:
    """Format the stable operation ID for one source step."""
    return f"BBA-{step:03d}"


BLACKBOARD_OPERATIONS: tuple[OperationSpec, ...] = tuple(
    OperationSpec(
        operation_id=_operation_id(step), step=step, title=title, purpose=purpose,
    )
    for step, title, purpose in _BLACKBOARD_OPERATIONS
)

BLACKBOARD_OPERATION_IDS: tuple[str, ...] = tuple(
    item.operation_id for item in BLACKBOARD_OPERATIONS
)


def operation_by_step(step: int) -> OperationSpec:
    """Return the operation specification for one algorithm step number."""
    if not isinstance(step, int) or not 1 <= step <= len(BLACKBOARD_OPERATIONS):
        raise ValueError(f"unknown blackboard algorithm step: {step!r}")
    return BLACKBOARD_OPERATIONS[step - 1]


def operation_by_id(operation_id: str) -> OperationSpec:
    """Return the operation specification for one stable operation ID."""
    for item in BLACKBOARD_OPERATIONS:
        if item.operation_id == operation_id:
            return item
    raise ValueError(f"unknown blackboard operation ID: {operation_id!r}")


__all__ = [
    "BLACKBOARD_OPERATION_IDS",
    "BLACKBOARD_OPERATIONS",
    "OperationSpec",
    "operation_by_id",
    "operation_by_step",
]
