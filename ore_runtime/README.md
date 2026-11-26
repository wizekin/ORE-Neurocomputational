# ORE Runtime — Point 19 Concurrent and Incremental Cognitive Execution Engine

This package implements the "ORE Integrated Algorithm — Point 19: Concurrent and
Incremental Cognitive Execution Engine" specification for the
ORE-Neurocomputational repository. It coordinates independent cognitive
modules as one safe, efficient, reproducible computational system:
versioned shared state, bounded worker pools, dependency-aware scheduling,
durable event publication, incremental invalidation, and deterministic testing.

Integration boundaries (spec section 6):

- **Shared cognitive workspace** — `ore_workspace.SharedWorkspace` receives
  validated task results as `Finding` envelopes through
  `ore_runtime.adapters.WorkspaceRuntimeAdapter`; the workspace stays
  authoritative for its own fields.
- **USG** — versioned USG source records are registered through a validated
  write contract (`register_usg_source` / `declare_usg_change`); dependent
  conclusions are invalidated and recomputed incrementally. No concurrency
  mechanism bypasses USG validation.
- **ORE module registries** — `ore_runtime.registration.register_runtime_module`
  registers the engine as module `ore.runtime.engine` with capabilities
  `runtime.concurrent_execution`, `runtime.event_coordination`, and
  `runtime.incremental_recompute` in `ore_contracts.ModuleRegistry`.

## Quick start

    python -m ore_runtime          # synthetic three-task demo
    python -m pytest tests/runtime -q

```python
from ore_runtime import CognitiveExecutionEngine, EngineConfig, ExecutionRequest, TaskSpec

engine = CognitiveExecutionEngine(EngineConfig(max_workers=4))
report = engine.execute_goal(
    ExecutionRequest(goal_id="g", description="parse the sentence",
                    expected_outputs=("conclude",)),
    (
        TaskSpec(task_id="perceive", name="perceive", priority=0.8),
        TaskSpec(task_id="analyze", name="analyze", dependencies=("perceive",)),
        TaskSpec(task_id="conclude", name="conclude", dependencies=("analyze",)),
    ),
    {
        "perceive": lambda ctx: {"tokens": [...]},
        "analyze":  lambda ctx: {"parse": [...]},
        "conclude": lambda ctx: {"summary": "done"},
    },
)
engine.close()
```

## Package layout (spec section 6.1 boundaries)

| Module | Responsibility (spec step) |
| --- | --- |
| `contracts.py` | Task states, side-effect classes, retry safety, engine config (steps 2, 4, 5) |
| `concurrency/task_graph.py` | Task nodes, edges, cycle detection, readiness, bounded iterative regions, critical path (steps 2, 3, 3.2) |
| `concurrency/versioned_state.py` | Immutable snapshots, optimistic commits, atomic publication, RCU reclamation, transactional commit hooks (steps 6, 10, 11, 12, 13) |
| `concurrency/scheduler.py` | Bounded ready queues, scoring policy, admission control with watermark hysteresis, aging, quotas, conservative work stealing (steps 14, 16, 17, 18, 19, 3.3) |
| `concurrency/worker_pool.py` | Bounded worker lifecycle, capability matching, cancellation tokens, cooperative checkpoints, crash recovery state (steps 4, 15, 20) |
| `events/schemas.py` | Typed event envelopes with schema versions, the event catalogue, evolution rules (step 21, 3.5) |
| `events/bus.py` | Local event routing with predicates, honest delivery guarantees (steps 22, 25) |
| `events/idempotency.py` | Durable `(consumer_id, event_id)` deduplication (step 24) |
| `events/outbox.py` | Transactional outbox with delivery retry and backoff (step 28) |
| `events/sagas.py` | Sagas with ordered local transactions and explicit compensation (step 29) |
| `history/event_store.py` | Durable event log, per-aggregate sequences, replay/reduce, snapshots, projections with staleness (steps 26, 27) |
| `incremental/dependency_graph.py` | Artifact dependency records, content hashes, reverse dependency indexes (steps 34, 35) |
| `incremental/invalidation.py` | Stale propagation, topological recompute planning, freshness checks (steps 35, 36, 38) |
| `engine.py` | End-to-end `EXECUTE_GOAL` loop integrating every subsystem (section 5 pseudocode) |
| `adapters.py` | Shared-workspace publication adapter and USG source adapter (section 6) |
| `testing/deterministic_scheduler.py` | Controlled interleavings, replay manifests, invariant suite, fuzzing (steps 39, 40) |
| `metrics.py` | Operational metrics: throughput, queue-wait percentiles, worker utilization, conflict rate, event lag, outbox backlog, stale counts, recompute ratio (section 9) |
| `registration.py` | ORE module/contract registration into `ore_contracts` (section 6) |
| `annotations.py` | Indexing into the shared ORE SQLite logger |

## Execution guarantees implemented

- Each task reads a coherent workspace snapshot at a pinned revision.
- State changes commit atomically at defined boundaries with optimistic
  version checks; conflicts never silently lose updates.
- Events are durable, per-aggregate ordered, causally linked, and safe under
  redelivery through the idempotency store.
- A committed state change and its outbound event are written in the same
  transaction (outbox), so publication failures are retried, never lost.
- Failed workers retry per side-effect class or compensate through auditable
  sagas; irreversible steps record residual effects.
- Scheduling enforces capability match, dependency completion, active-task
  ceilings, per-module quotas, and watermark backpressure with an idle-pool
  bypass so bursts degrade throughput without deadlock.
- USG or workspace changes invalidate exactly the affected downstream
  artifacts; stale conclusions are never presented as current.
- Deterministic schedules replay invariant outcomes; fuzzed interleavings
  (failures, timeouts, cancellations, duplicate events) keep invariants.

## Scope disclaimer

This is the conservative single-process implementation the specification
recommends: established synchronization primitives, bounded thread pools, and
deterministic testing. Lock-free structures, RCU for every structure, stream
processing, and distributed orchestration are deliberately deferred until
workload measurements justify them.
