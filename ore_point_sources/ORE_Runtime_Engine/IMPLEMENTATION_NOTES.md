# ORE Point 19 — Concurrent and Incremental Cognitive Execution Engine

Implementation of the "ORE Integrated Algorithm — Point 19: Concurrent and
Incremental Cognitive Execution Engine" specification, integrated with the
ORE-Neurocomputational repository.

## Contents of this archive

```
ore_runtime/            The algorithm implementation (Python package)
  README.md             Full documentation incl. step-by-spec mapping
  contracts.py           Typed records, task states, engine configuration
  engine.py             End-to-end EXECUTE_GOAL execution engine
  adapters.py           ore_workspace publication + USG source adapters
  registration.py       ore_contracts ModuleRegistry/ContractCatalog integration
  annotations.py        Shared ORE SQLite logger indexing
  metrics.py            Operational metrics (spec section 9)
  exceptions.py         Typed engine exceptions
  concurrency/          task_graph, versioned_state, scheduler, worker_pool
  events/               schemas, bus, idempotency, outbox, sagas
  history/              event_store (durable log, replay, projections)
  incremental/          dependency_graph, invalidation
  testing/              deterministic_scheduler (replay, invariants, fuzzing)
tests/runtime/          Acceptance tests for spec section 8
  test_task_graph.py            dependency ordering, cycles, iterative regions
  test_versioned_state.py       snapshot consistency, optimistic conflicts,
                                atomic publication, RCU reclamation
  test_scheduler_pool.py       capability safety, bounded execution,
                                backpressure, fairness/aging, work stealing,
                                cancellation, checkpoints
  test_events.py                schema validation/evolution, idempotent
                                consumption, causal traces, per-aggregate
                                order, event sourcing, projection staleness
  test_outbox_sagas.py          outbox reliability under publisher crash,
                                saga compensation, crash recovery
  test_incremental.py          invalidation exactness, incremental ==
                                full recomputation, topological recompute
  test_deterministic_scheduler.py  deterministic replay, invariant fuzzing
  test_engine_integration.py   end-to-end engine, ore_workspace integration,
                                USG invalidation + recompute, ModuleRegistry
                                registration, backpressure, audit trace
```

## Installing into the ORE-Neurocomputational repository

1. Copy `ore_runtime/` to the repository root (next to `ore_workspace/`).
2. Copy `tests/runtime/` into the repository's `tests/` directory.
3. Add `"ore_runtime*"` to the `include` list under
   `[tool.setuptools.packages.find]` in `pyproject.toml` (one line).
4. No other repository files need to change. The algorithm integrates with
   the existing `ore_workspace`, `ore_contracts`, and `ore_usg` packages.

## Running

    python -m pytest tests/runtime -q      # 76 acceptance tests
    python -m ore_runtime                 # synthetic three-task demo

Requires Python 3.10+; the algorithm itself depends only on the standard
library. Repository integrations (`adapters.py`, `registration.py`,
`annotations.py`) import `ore_workspace`, `ore_contracts`, and `ore_logger`,
which are part of the ORE-Neurocomputational repository.

## What is implemented (spec traceability)

- Phase A (steps 1-5): goal normalization, task DAG decomposition, cycle
  rejection with diagnostic paths, bounded iterative regions, capability and
  resource attachment, side-effect classification with retry-safety scoring.
- Phase B (steps 6-13): immutable versioned snapshots, optimistic commits
  with conflict detection, atomic pointer-swap publication, RCU grace-period
  reclamation, linearizable commit/claim points.
- Phase C (steps 14-20): ready queues, capability-fit assignment, scheduling
  score (alpha*priority + beta/slack - gamma*cost + delta*aging), watermark
  admission control, aging against starvation, conservative work stealing,
  cancellation tokens with cooperative checkpoints.
- Phase D (steps 21-30): typed event catalogue with schema versions and
  migrations, local event bus with predicates and honest delivery guarantees,
  correlation/causation identifiers, idempotent consumption, event sourcing
  with snapshots, CQRS projections with staleness reporting, transactional
  outbox, sagas with compensation. Stream processing is deliberately deferred
  (documented scale-up path).
- Phase E (steps 31-33): established concurrent structures only (stdlib locks,
  bounded queues, ThreadPoolExecutor); no custom lock-free algorithms, and
  the ABA/reclamation concerns are documented rather than reimplemented.
- Phase F (steps 34-38): artifact dependency records with content hashes,
  reverse dependency indexes, transitive invalidation, topological recompute
  with fresh-cache reuse, stale marking that preserves historical hashes.
- Phase G (steps 39-40): replay manifests, deterministic schedules with
  fault injection, invariant suite, fuzzing harness with race reproduction
  tracking.

All 26 acceptance tests of spec section 8 are covered by `tests/runtime/`.
