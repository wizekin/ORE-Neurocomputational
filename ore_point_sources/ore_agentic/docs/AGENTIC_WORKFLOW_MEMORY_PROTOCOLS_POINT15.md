# ORE Agentic Workflow and Memory Layer — Point 15 Architecture

Canonical relevance rank: 4 of 24

Source: "ORE Integrated Algorithm — Point 15: Agentic Systems, Cognitive
Memory Architectures and Communication Protocols" (source family E —
Specialized Learning and Agentic Systems; sections 15.1-15.11).

Status: Implemented as the `ore_agentic` package, software-first reference
implementation.  Workers are deterministic local callables behind typed
adapters; no external LLM, MCP server or agent is required at runtime.  This
document maps the specification to the implementation, tests, and
integration seams in the live repository.

---

## 1. Objective and design principle

`ore_agentic` implements the Persistent Agentic Workflow and Memory Layer: a
workflow graph references the authoritative Shared Cognitive Workspace; a
supervisor delegates bounded tasks; three logically distinct memory types
link durable knowledge, experience and validated procedures; a protocol
gateway exposes authorized capabilities through controlled MCP and optional
A2A adapters; and a reliability layer provides queues, tracing, quotas,
checkpoints, recovery and rollback.

The authority chain is preserved end to end:

- The **Shared Cognitive Workspace** (ore_workspace) stays the authoritative
  cognitive state; `ore_agentic` stores only operational workflow state and
  references workspace revisions.
- **USG** (ore_usg) stays the canonical semantic memory; agent output is
  recorded as lifecycle candidates and can only enter canonical knowledge
  through the USG bundle import path.
- Hypotheses and observations never silently become authoritative facts;
  retries never duplicate externally visible side effects.

## 2. Package boundaries (specification Section 4 → implementation)

| Proposed boundary | Implementation |
|---|---|
| `ore_agentic/workflow/` | `workflow/graph.py` (DAG schema, topological order, critical path), `workflow/machine.py` (validated state machine, branch evaluation), `workflow/checkpoints.py` (transactional snapshots, integrity-verified restore), `workflow/side_effects.py` (idempotency, receipts, outbox/inbox, compensation) |
| `ore_agentic/delegation/` | `delegation/supervisor.py` (eligibility, S(a,t) ranking, bounded leases, classified retries, joins), `delegation/workers.py` (Worker protocol, LocalWorker, MapReduceExecutor, bounded queue), `delegation/leases.py` (fencing epochs, heartbeats) |
| `ore_agentic/registry/` | `registry/agents.py` (capability/permission/health registry), `registry/operations.py` (the 50-step operation registry `P15-01..P15-50`) |
| `ore_agentic/memory/` | `memory/semantic.py` (USG-backed semantic memory + canonical promotion gate), `memory/episodic.py` (linked, redacted, retention-governed episodes), `memory/procedural.py` (test-gated procedures), `memory/links.py` (typed relations between tiers) |
| `ore_agentic/retrieval/` | `retrieval/hybrid.py` (lexical + semantic + graph + freshness + authority with rerank and ranking metrics), `retrieval/evidence.py` (evidence quality, correlation groups, preserved conflict) |
| `ore_agentic/protocols/mcp/` | `protocols/mcp/server.py` (approved read-only capability catalog, authorization, size caps), `protocols/mcp/client.py` (allowlisted untrusted provider, normalized outcomes) |
| `ore_agentic/protocols/a2a/` | `protocols/a2a/agent.py` (agent cards, justified delegation, untrusted-result validation, duplicate/out-of-order/contradiction handling) |
| `ore_agentic/evaluation/` | `evaluation/contracts.py` (criteria declared before execution), `evaluation/validators.py` (schema subset + deterministic checks), `evaluation/critic.py` (bounded revision loop, final verifier, approval gates) |
| `ore_agentic/observability/` | `observability/tracing.py` (trace/span propagation, secret redaction), `observability/metrics.py` (queue metrics, rho utilization, drift monitor, semantic cache) |
| `ore_agentic/security/` | `security/policy.py` (default-deny authorization, risk classification), `security/sandbox.py` (quota manager, sandbox profiles, audit log) |
| `ore_agentic/persistence/` | `persistence/database.py` (agentic SQLite schema), `persistence/workflow_store.py` (create/transition/checkpoint/load), `persistence/memory_store.py` (episodes, procedures, links, proposals), `persistence/workspace_bridge.py` (workspace publication) |
| `logging/` (spec role) | `annotations.py` — the repository's established convention: `register_annotations` (ore_logger catalogue), `record_result_to_sqlite` (one event per step), `AgenticAnnotationRecorder` (engine-time structured logging) |
| central registration | `registration.py` — ModuleRegistry/ContractCatalog integration following `ore_usg`/`ore_temporal`/`ore_learning` |
| 50-step engine | `pipeline.py` — `AgenticEngine` executes all eight phases in order with exactly one trace entry per source step |

Root modules: `config.py` (governed parameters, `AgenticConfig`),
`contracts.py` (goal/task/lease/result/provenance types),
`functional.py` (every cited formula as a pure function), `__main__.py`
(`python -m ore_agentic` demonstration).

## 3. The 50-step algorithm

`registry/operations.py` defines the operation registry
(`P15-01`..`P15-50`) with each step's purpose, mathematical form and
safeguard.  `AgenticEngine.run()` records **exactly one trace entry per
source step, in order** (asserted by
`test_engine_collects_system_evaluation_metrics`):

- Phase I (Steps 1-8): intake with immutable request reference and
  idempotency keys; USG-resolved intent G=(O,C,I,D) with hypotheses marked
  non-authoritative; risk R=Σ(wᵢrᵢ) with automatic/approval/prohibited
  decisions from policy (not wording); duplicate-work reuse; typed task
  graph with cycle rejection; simplest-pattern selection; quota-checked
  admission; immutable plan version + hash.
- Phase II (Steps 9-16): registry with health/versions; mandatory
  eligibility predicates; S(a,t)=w_f F+w_q Q−w_l L−w_c C−w_u U−w_r R with
  deterministic tie-breaks; bounded leases; map-reduce over a bounded pool
  with contract validation before aggregation; backpressure with ρ=λ/μ;
  fencing epochs; dₖ=min(dmax,d₀·2ᵏ)+Jₖ retries for transient failures only.
- Phase III (Steps 17-23): validated transition table; recorded branch
  evaluations with rule version; all/any/quorum/weighted joins with dissent
  preservation; transactional checkpoints excluding secrets;
  integrity-verified resume; cancellation with declared+authorized
  compensation only; idempotency keys, receipts, outbox/inbox and
  reconciliation of uncertain outcomes.
- Phase IV (Steps 24-30): contracts declared before execution; schema +
  domain-invariant validation; independent evidence retrieval with
  provenance; correlated-source suppression and preserved conflict; a
  separate deterministic critic with structured defects; revision bounded by
  iteration/cost/stall budgets; final verification that rejects failed
  deterministic tests and gates consequential actions on approval.
- Phase V (Steps 31-36): read-only approved MCP catalog; untrusted external
  providers with allowlists and size caps; typed envelopes with normalized
  success/error/timeout/unknown outcomes; least-privilege secure invocation;
  A2A only when justified; remote results remain proposals until locally
  validated.
- Phase VI (Steps 37-42): semantic proposals as lifecycle candidates (USG
  canonical); linked episodic records with redaction and retention;
  test-gated procedural promotion; typed cross-tier links; hybrid
  R(d,q)=αR_lex+βR_sem+γR_graph+δR_fresh+εR_auth with provenance and
  ranking evaluation; freshness/conflict/retention governance.
- Phase VII (Steps 43-47): trace/span propagation with redaction; bounded
  queues with dead-letter semantics; version-and-revision-keyed semantic
  cache (stale is never authoritative); drift monitoring that recommends
  review (quarantine only under explicit policy); quotas and sandbox
  profiles with an honest limitation: in-process profiles are guard rails,
  generated code needs a real isolation boundary.
- Phase VIII (Steps 48-50): version pinning and staged promotion; system
  evaluation metrics (including duplicate-side-effect prevention and
  unauthorized-call counts); close-out that stores the outcome episode and
  reports completed/partial/failed/unverified work distinctly.

## 4. Integration seams

| Existing component | Point 15 responsibility | How it is used |
|---|---|---|
| Shared Cognitive Workspace | Authoritative cognitive state | `persistence/workspace_bridge.py` publishes typed findings via `FindingPublisher`; denials are reported, never bypassed |
| USG | Canonical semantic memory | Duck-typed index (`lookup`, `relations_for`) keeps the engine decoupled; proposals go through `USGPromotionGate` |
| ModuleRegistry / ContractCatalog | Central registration | `register_ore_agentic_module(registry)` registers contracts, manifest and backend (`ORE.AGENTIC.*` contracts, `ore.agentic_workflow` module) |
| ore_logger | Execution history | `annotations.py` follows the repository convention; `ore_logger.catalog.sync_repository(include_paths={"ore_agentic"})` |
| ModuleScheduler | Module execution scheduling | Preserved: the supervisor owns only logical assignment |

Quick start:

```python
from ore_agentic import AgenticEngine, GoalRequest, LocalWorker, WorkerManifest

engine = AgenticEngine()  # optionally database_path=... for persistence
engine.register_worker(
    WorkerManifest(worker_id="worker-1", version="1.0.0",
                   capabilities=("analysis.summarize",)),
    LocalWorker("worker-1", "1.0.0", ("analysis.summarize",), handler))
result = engine.run(GoalRequest(
    objective="Summarize the evidence",
    requester="operator",
    subgoals=(SubgoalSpec(name="summarize",
                          capability="analysis.summarize"),)))
assert [entry["step"] for entry in result.trace] == list(range(1, 51))
```

## 5. Test map (40 acceptance tests)

| Test file | Specification acceptance tests |
|---|---|
| `tests/test_agentic_workflow_point15.py` | 1-7 (intake, clarification, prohibition, idempotency, graph, cycles), 17-23 (state machine, branches, joins, checkpoints, resume, cancellation, compensation, uncertain-side-effect reconciliation) |
| `tests/test_agentic_delegation_point15.py` | 7-16 (eligibility, capability/schema/permission selection, admission, concurrency, partial results, backpressure, fencing, retries, non-retry of authorization failures) |
| `tests/test_agentic_evaluation_point15.py` | 24-30 (contracts, schemas, evidence, correlated sources, critic, bounded revision, final verification) |
| `tests/test_agentic_memory_point15.py` | 34-38 (USG promotion path, episodic linkage, procedure promotion, hybrid retrieval provenance, secret redaction) |
| `tests/test_agentic_protocols_point15.py` | 31-33 (MCP authorization, malformed/oversized responses, untrusted A2A results) |
| `tests/test_agentic_reliability_point15.py` | 39-40 (overload/worker-loss/stale-checkpoint robustness, distinct end-to-end status) plus quotas, sandbox, cache, drift, metrics |
| `tests/test_agentic_integration_point15.py` | Registration, workspace publication, ore_logger annotations, USG integration, demo entry point, 50-step operation registry |

## 6. Limitations (honest disclosure)

- Workers are deterministic local callables; the specification's
  "implementation and benchmark results are not claimed" caveat applies to
  any performance characterization.
- Vector retrieval is omitted until evaluation shows measurable benefit over
  the lexical + USG-semantic + graph channels (per the implementation
  roadmap).
- The sandbox executor enforces wall-clock and output-size limits in
  process; language-level restrictions are insufficient for untrusted
  generated code, which requires a real isolation boundary.
- The in-memory semantic cache and deferral paths are calibrated for the
  reference configuration; production values require measured workloads.
