"""Source-ordered operation definitions for the 50-step Point 15 algorithm.

Each of the specification's 50 steps becomes one auditable operation with an
identifier (P15-01 .. P15-50), a purpose, its mathematical form where the
specification cites one, and the safeguard that motivates it.  The engine
records exactly one trace entry per operation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = (
    "OPERATION_REGISTRY",
    "OperationSpec",
    "PHASES",
    "PHASE_OF_STEP",
    "registry_as_dicts",
    "step_by_id",
)


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Describe one source step, its equation, and its safeguard."""

    operation_id: str
    source_step: int
    phase: str
    name: str
    purpose: str
    mathematical_form: str
    safeguard: str
    evidence_level: str = "software_reference_implementation"

    def to_dict(self) -> dict[str, Any]:
        """Return the operation definition as a plain dictionary."""
        return asdict(self)


_ROWS: tuple[tuple[str, str, str, str], ...] = (
    # Phase I -- Goal intake and workflow compilation (Steps 1-8)
    ("receive_objective",
     "Accept the goal with request ID, requester, timestamp, priority, deadline and "
     "constraints; preserve the original request as immutable input; reject malformed "
     "or empty requests.",
     "valid(request) -> workflow_id + idempotency_key | rejection",
     "Reject malformed requests before any allocation."),
    ("interpret_intent",
     "Resolve entities, classes, actions and desired outcome through USG; represent "
     "G = (O, C, I, D); mark inferred values as hypotheses, not facts; request "
     "clarification when interpretations materially change execution.",
     "G = (O, C, I, D)",
     "Hypotheses never silently become authoritative USG facts."),
    ("classify_risk",
     "Classify operations as read-only, reversible write, externally visible or "
     "irreversible; decide automatic, approval-gated or prohibited execution from "
     "policy rather than request wording.",
     "R = sum(w_i * r_i)",
     "High risk triggers review, not automatic permission."),
    ("check_existing_state",
     "Search for matching active, completed or failed workflows; reuse results only "
     "when inputs, versions, permissions, freshness and validation remain applicable.",
     "match(fingerprint, versions, permissions, freshness) -> reuse | new",
     "Idempotency keys prevent duplicate work and duplicate side effects."),
    ("decompose_subtasks",
     "Create a directed typed task graph; each task declares schemas, preconditions, "
     "postconditions, capability, cost, deadline, retry policy and side-effect class; "
     "detect unbounded cycles.",
     "W = (V, E), edge (i, j) <=> task j depends on task i",
     "Unbounded dependency cycles are rejected at compile time."),
    ("select_pattern",
     "Choose the simplest suitable pattern: sequential, supervisor/worker, map-reduce, "
     "evaluator-optimizer, conditional graph, approval gate or hybrid.",
     "pattern = simplest(patterns satisfying constraints)",
     "Avoid multi-agent complexity when a module call suffices."),
    ("estimate_feasibility",
     "Estimate CPU, memory, I/O, network, model calls, storage, elapsed time and "
     "external costs; reject, defer or downscale work exceeding capacity.",
     "c_i = (cpu, mem, io, net, time); sum(c_i) <= budget",
     "Admission control bounds aggregate and peak resource use."),
    ("persist_workflow_spec",
     "Validate graph structure, schemas, permissions, dependencies, deadlines, "
     "budgets and terminal conditions; persist the immutable plan version and hash.",
     "hash(plan) -> plan_version",
     "Workflow state is operational state, not a competing cognitive workspace."),
    # Phase II -- Registry, delegation and parallel execution (Steps 9-16)
    ("maintain_registry",
     "Store stable ID, versions, capabilities, accepted schemas, permissions, resource "
     "profiles, health, reliability history and evaluation evidence; expired or "
     "revoked entries are ineligible.",
     "registry = {id -> manifest}",
     "Least privilege applies to every worker registration."),
    ("discover_workers",
     "Filter candidates by mandatory capability, schema, permission, health, policy "
     "and resource predicates; a worker failing any mandatory predicate is excluded.",
     "eligible = AND(mandatory predicates)",
     "Ineligible workers are excluded regardless of score."),
    ("rank_candidates",
     "Score eligible workers on task fit, measured success, latency, cost, "
     "uncertainty, load and trust with deterministic tie-breaks.",
     "S(a, t) = w_f*F + w_q*Q - w_l*L - w_c*C - w_u*U - w_r*R",
     "Selection features and tie-breaks are logged."),
    ("assign_responsibilities",
     "Issue a task lease with workflow/task/worker IDs, input references, permitted "
     "tools, budget, deadline, cancellation token and output contract; expose no "
     "unrelated workspace data.",
     "lease = (ids, refs, tools, budget, deadline, contract)",
     "Supervisor owns logical assignment; ModuleScheduler owns execution scheduling."),
    ("execute_map_reduce",
     "Dispatch independent subtasks to a bounded pool; validate contracts before "
     "aggregation; preserve partial results when workers fail.",
     "parallel(V_independent) -> reduce(results)",
     "Contract validation precedes aggregation."),
    ("enforce_backpressure",
     "Set per-workflow, per-worker and global concurrency limits; use bounded queues; "
     "defer or reject work at capacity.",
     "rho = lambda / mu; rho >= 1 => unstable",
     "Sustained overload activates admission deferral."),
    ("track_leases",
     "Track heartbeat and progress; expired leases trigger reconciliation before "
     "reassignment; fencing tokens prevent stale overwrites.",
     "epoch(task) monotonic; accept iff epoch == current",
     "Worker split-brain is prevented by fencing."),
    ("classify_failures",
     "Classify errors as transient, permanent, policy, resource or unknown; retry "
     "only transient failures with bounded exponential backoff and jitter.",
     "d_k = min(d_max, d_0 * 2^k) + J_k",
     "Authorization failures are never retried as transient."),
    # Phase III -- Workflow state, recovery and side-effect safety (Steps 17-23)
    ("execute_state_machine",
     "Execute the validated state machine over pending, ready, leased, running, "
     "waiting, succeeded, failed, cancelled and compensated; persist transitions "
     "transactionally.",
     "transition in TABLE[current]",
     "Invalid transitions are rejected and recorded."),
    ("evaluate_branches",
     "Evaluate predicates against typed, provenance-bearing workspace values; record "
     "predicate inputs and rule version; route uncertain values to clarification.",
     "branch = rule(inputs@revision, rule_version)",
     "Missing values never route to silent guessing."),
    ("coordinate_joins",
     "Define each join as all, any, quorum or weighted evidence quorum; distinguish "
     "missing from negative results; preserve dissenting outputs.",
     "join(votes) per semantics",
     "Dissent is preserved for evaluation."),
    ("persist_checkpoints",
     "Checkpoint workflow state, completed references, pending work, plan version, "
     "workspace revision, leases and recovery metadata; exclude secrets.",
     "snapshot = txn(state, refs, leases)",
     "Checkpoint corruption is detected by integrity checks."),
    ("resume_checkpoint",
     "Validate integrity, workflow version, permissions, deadlines, artifacts and "
     "external state; reconcile in-flight leases and side effects; recheck "
     "preconditions before resuming.",
     "resume = verify(checkpoint) then reconcile",
     "Resume revalidates permissions and preconditions."),
    ("cancel_compensate",
     "Stop new dispatches, signal running workers, mark unfinished tasks; invoke "
     "declared compensation only when safe and authorized; disclose irreversible "
     "partial effects.",
     "cancel -> compensate(if declared and authorized)",
     "Compensation is not guaranteed rollback."),
    ("prevent_duplicate_effects",
     "Use idempotency keys, operation receipts, outbox/inbox patterns and "
     "deduplication records; reconcile uncertain outcomes before any retry.",
     "execute(key) once; receipt(key) thereafter",
     "Retries cannot duplicate externally visible side effects."),
    # Phase IV -- Evaluator-optimizer and evidence validation (Steps 24-30)
    ("declare_contracts",
     "Specify correctness, completeness, evidence, safety, format and performance "
     "criteria before execution; version evaluator rules and fixtures.",
     "contract = {criteria}, versioned",
     "Criteria are declared in advance, not fitted afterwards."),
    ("validate_structure",
     "Check schemas, required fields, types, ranges, domain constraints and "
     "invariants; schema validity alone is not correctness.",
     "schema(payload) and invariants(payload)",
     "Semantic mismatches are detected, not just syntax."),
    ("retrieve_evidence",
     "Query USG, documents, graph relations and permitted external resources; "
     "preserve source identity, timestamp, version and location; distinguish primary "
     "evidence from model-generated claims.",
     "evidence = retrieve(query) with provenance",
     "Primary evidence is distinguished from summaries."),
    ("assess_evidence",
     "Assess reliability, directness, recency, independence and applicability; "
     "collapse correlated copies; preserve material disagreement.",
     "independent_confirmations = |distinct groups|",
     "Correlated sources are not multiplied as independent."),
    ("run_critic",
     "Review proposed output against the predeclared contract and evidence; return "
     "structured defects with severity, location, rationale and correction.",
     "critic(artifact, contract) -> defects",
     "Prefer a different implementation to reduce shared failure modes."),
    ("bounded_revision",
     "Allow revisions for actionable defects within iteration, time and cost limits; "
     "each revision creates a new artifact version.",
     "stop when pass | budget end | stall",
     "The critic loop cannot run forever."),
    ("final_verification",
     "Run required tests, deterministic checks and policy checks; require human "
     "approval for consequential actions; evaluators cannot grant denied permissions.",
     "verify = AND(required checks)",
     "Failed deterministic tests reject the output."),
    # Phase V -- MCP and A2A interoperability (Steps 31-36)
    ("expose_mcp",
     "Publish only explicitly approved read-only tools, resources and prompts with "
     "stable schemas, versions, errors and authorization.",
     "catalog = allowlist(capabilities)",
     "MCP overexposure is prevented by allowlist and least privilege."),
    ("consume_external_mcp",
     "Treat external servers as untrusted providers; validate identity, protocol "
     "version, schemas, scopes and output size; enforce policy allowlists.",
     "validate(server, schema, size) -> outcome",
     "Tool descriptions do not authorize execution."),
    ("normalize_protocol_calls",
     "Wrap calls in typed envelopes with request ID, caller, capability, schema "
     "version, arguments, authorization context, deadline, idempotency key and trace "
     "ID; normalize success, structured error, timeout and unknown outcomes.",
     "envelope -> outcome in {OK, ERROR, TIMEOUT, UNKNOWN}",
     "Unknown outcomes are first-class states."),
    ("secure_invocation",
     "Use least-privilege credentials, secret isolation, input validation, output "
     "sanitization, network restrictions, rate limits and audit logs; separate "
     "read/write scopes.",
     "authorize -> sandbox -> audit",
     "Credentials are never exposed to model context."),
    ("justify_a2a",
     "Use A2A only for independently operated agents requiring discovery, "
     "delegation, status exchange or artifact handoff; internal modules use simpler "
     "typed interfaces.",
     "a2a iff independent operator and justified",
     "Unnecessary A2A complexity is avoided."),
    ("validate_external_agents",
     "Check identity, task ID, capability, artifact schema, provenance, evidence and "
     "policy; treat remote assertions as proposals until locally checked; handle "
     "duplicate, delayed, out-of-order and contradictory messages.",
     "remote_result -> proposal until validated",
     "External results remain untrusted until validated."),
    # Phase VI -- Three-tier cognitive memory and RAG (Steps 37-42)
    ("store_semantic_memory",
     "Semantic memory holds durable concepts and relations; USG remains canonical; "
     "agent output cannot directly modify authoritative knowledge.",
     "semantic = USG (canonical)",
     "Updates follow the canonical validation and promotion path."),
    ("store_episodic_memory",
     "Record tasks, observations, decisions, transitions, outcomes, failures and "
     "snapshots; link each episode to workflow, task, worker, time, evidence and "
     "versions; apply privacy and retention policy.",
     "episode = (workflow, task, worker, t, outcome, versions)",
     "Retention and privacy policy bound what is stored."),
    ("store_procedural_memory",
     "Store reusable algorithms, workflows, procedures and tool routines with "
     "preconditions, inputs, outputs, dependencies, versions, originating episodes "
     "and test results; promote only after validation.",
     "procedure: DRAFT -> (tests pass) -> VALIDATED",
     "Procedures cannot be promoted without tests."),
    ("link_memory_types",
     "Link semantic, episodic and procedural memory with stable IDs and typed "
     "relations without conflating them.",
     "relations: derived_from, uses_concept, validated_by, observed_in, supersedes",
     "An observation is not automatically a fact."),
    ("hybrid_rag",
     "Combine exact IDs, lexical retrieval, USG semantic relations, graph traversal "
     "and metadata filters; rerank by task intent; return evidence with provenance.",
     "R(d, q) = a*R_lex + b*R_sem + g*R_graph + d*R_fresh + e*R_auth",
     "Generated summaries are not returned as evidence."),
    ("memory_governance",
     "Track validity intervals, source versions, confidence, supersession and "
     "retention class; resolve conflicts by authority, scope and time; support "
     "required deletion and redaction.",
     "freshness/conflict/retention policies",
     "Stale memory is never authoritative evidence."),
    # Phase VII -- Production reliability and resource governance (Steps 43-47)
    ("distributed_tracing",
     "Propagate trace, workflow, task and parent-span IDs across coordinators, "
     "workers and tools; record latency, status, retries, queue wait and provenance; "
     "redact secrets.",
     "span = (trace_id, parent, operation, timing)",
     "Secrets and sensitive payloads are redacted."),
    ("durable_queues",
     "Operate bounded durable queues, justified priority classes and worker pools "
     "sized by measured throughput; define dead-letter, poison-message and graceful "
     "shutdown behavior.",
     "queue = bounded(durable)",
     "Poison messages are dead-lettered, not dropped silently."),
    ("semantic_cache",
     "Cache only when inputs, permissions, versions and freshness match; include "
     "model/tool version and workspace revision in keys; speculation must be "
     "bounded, cancellable and side-effect-free.",
     "key = (capability, inputs, versions, revision)",
     "Stale cache is not authoritative evidence."),
    ("monitor_drift",
     "Track success, calibration, latency, cost, failure classes and distribution "
     "shifts by task and version; trigger review at validated thresholds; quarantine "
     "only under explicit policy.",
     "drift(recent, baseline) vs threshold",
     "Memory is never silently rewritten."),
    ("enforce_quotas",
     "Set CPU, memory, wall-clock, network, filesystem, process and output limits; "
     "generated or untrusted code runs behind a real isolation boundary.",
     "limits = (cpu, mem, time, io, net, output)",
     "Language-level restrictions alone are insufficient."),
    # Phase VIII -- Governance, lifecycle and evaluation (Steps 48-50)
    ("version_approve_rollback",
     "Version workflows, registry entries, tool schemas, procedures and evaluator "
     "contracts; promote through staged tests and approval; roll back to known-good "
     "versions on failure.",
     "version(pin) -> staged tests -> approve",
     "Rollback preserves migration paths."),
    ("evaluate_system",
     "Measure goal completion, correctness, evidence coverage, provenance, recovery, "
     "duplicate side effects, latency, throughput, resources, cost, security and "
     "human intervention; compare with simpler baselines.",
     "metrics vs baselines",
     "Fault injection validates recovery claims."),
    ("close_workflow",
     "Close only after terminal conditions and required checks pass; persist "
     "deliverables, evaluation, limitations, provenance and workspace revision; "
     "store the episode; propose procedures only after validation; report completed, "
     "partial, failed and unverified work distinctly.",
     "close = terminal_conditions AND required_checks",
     "Learning artifacts require independent validation."),
)

PHASES: dict[str, tuple[int, int]] = {
    "I": (1, 8),
    "II": (9, 16),
    "III": (17, 23),
    "IV": (24, 30),
    "V": (31, 36),
    "VI": (37, 42),
    "VII": (43, 47),
    "VIII": (48, 50),
}

_PHASE_NAMES = {
    "I": "Phase I -- Goal intake and workflow compilation",
    "II": "Phase II -- Registry, delegation and parallel execution",
    "III": "Phase III -- Workflow state, recovery and side-effect safety",
    "IV": "Phase IV -- Evaluator-optimizer and evidence validation",
    "V": "Phase V -- MCP and A2A interoperability",
    "VI": "Phase VI -- Three-tier cognitive memory and RAG",
    "VII": "Phase VII -- Production reliability and resource governance",
    "VIII": "Phase VIII -- Governance, lifecycle and evaluation",
}


def _phase_for(step: int) -> str:
    """Return the phase label for a source step."""
    for label, (start, end) in PHASES.items():
        if start <= step <= end:
            return label
    raise ValueError(f"step {step} outside the 50-step range")  # pragma: no cover


OPERATION_REGISTRY: tuple[OperationSpec, ...] = tuple(
    OperationSpec(
        operation_id=f"P15-{index:02d}",
        source_step=index,
        phase=_phase_for(index),
        name=name,
        purpose=purpose,
        mathematical_form=formula,
        safeguard=safeguard,
    )
    for index, (name, purpose, formula, safeguard) in enumerate(_ROWS, start=1)
)

PHASE_OF_STEP = {spec.source_step: spec.phase for spec in OPERATION_REGISTRY}
STEP_BY_NAME = {spec.name: spec for spec in OPERATION_REGISTRY}


def step_by_id(operation_id: str) -> OperationSpec:
    """Return the operation specification for a P15-NN identifier."""
    for spec in OPERATION_REGISTRY:
        if spec.operation_id == operation_id:
            return spec
    raise KeyError(f"unknown operation id: {operation_id}")


def registry_as_dicts() -> list[dict[str, Any]]:
    """Return the complete 50-operation registry in source order."""
    return [spec.to_dict() for spec in OPERATION_REGISTRY]
