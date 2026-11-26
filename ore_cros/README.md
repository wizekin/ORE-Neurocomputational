# ORE Point 23 — LDOS / CROS Implementation (`ore_cros`)

Full implementation of the ORE Integrated Algorithm **Point 23: Learning-Directed
Operating Systems and Adaptive Cognitive Resource Management** as a new
`ore_cros` package for the [ORE-Neurocomputational](https://github.com/wizekin/ORE-Neurocomputational)
repository.

CROS (Cognitive Resource-Oriented System) is a higher-level resource intelligence
and coordination layer that works **with** existing ORE components
(AdaptiveResourceCoordinator, CognitiveTaskCoordinator, ModuleScheduler,
SharedWorkspace, ore_usg, ore_logger) — it does not replace the host operating
system and never writes kernel settings.

## What is implemented

All 52 steps of the integrated algorithm across phases A-G:

| Phase | Steps | Coverage |
| --- | --- | --- |
| A — Observation and resource accounting | 1-8 | host discovery (unknown ≠ zero), scope topology, unit-safe telemetry schemas, /proc + cgroup v2 + os collectors with simulated fallback, baselines, task/module attribution, pressure detection, time-windowed bounded history, quality gates |
| B — MAPE-K autonomic loop | 9-15 | Monitor (versioned snapshots), Analyze (bottleneck ranking with evidence), Plan (candidates with effects/costs/rollback), Knowledge (versioned, provenance-separated), hard-constraint validation, narrow typed executor with audit + rollback, outcome residuals |
| C — Deterministic CPU/module scheduling | 16-26 | deterministic baseline, weighted fairness (virtual runtime, CFS/EEVDF as references), EDF with feasibility, Rate Monotonic with Liu-Layland bound, MLFQ with aging, gang scheduling, quotas/borrowing, starvation guards, shadow evaluation, constrained adaptation with fallback |
| D — Intelligent memory management | 27-35 | working sets, hot/warm/cold tiers (provenance-preserving), LRU/CLOCK baselines, MGLRU-inspired replacer, read-only DAMON adapter, NUMA placement costs, compression tiering, GPU admission with transfer batching, CXL extension point |
| E — Distributed placement and migration | 36-41 | expiring node advertisements, capability AND budget AND policy AND data-access AND health eligibility, cost+locality scoring with explanations, Borg/K8s/Decima reference notes, cost-aware migration with checkpoint requirement, side-effect-aware failure recovery |
| F — I/O, networking and storage | 42-46 | I/O profiling, io_uring selection with portable fallback, eBPF/XDP/DPDK/Demikernel evidence gates, storage capability matrix, durability contracts that optimization cannot downgrade |
| G — Real-time, criticality and learning | 47-52 | criticality matrix (policy-owned, no self-promotion), WCET feasibility and hard-real-time evidence requirements, demand/completion predictors with calibrated uncertainty, constrained contextual bandit with propensity logging + offline replay + canary, RL/MPC prerequisites-gated research path, watchdog/rollback/deterministic fallback with hysteresis and dwell times |

Plus the end-to-end `ADAPTIVE_RESOURCE_CYCLE` (spec section 5) in
`ore_cros/cycle.py` and the ORE integration boundaries (spec section 6) in
`ore_cros/integration.py`.

## Repository placement

```
ore_cros/                     the package (25 modules, stdlib-only)
tests/test_cros_*.py          127 acceptance tests mapping spec section 8
docs/POINT_23_LDOS_CROS.md    repository-style documentation
pyproject.toml                one-line addition: "ore_cros*" in packages.find.include
```

Install into a clone of ORE-Neurocomputational by copying `ore_cros/` next to
the other `ore_*` packages, the `tests/test_cros_*.py` files into `tests/`,
and adding `"ore_cros*"` to `[tool.setuptools.packages.find] include` in
`pyproject.toml` (already done in the repository copy this zip was built from).

## Quick start

```python
from ore_cros import CROSCoordinator, CROSConfig, ResourceDemand, TaskRecord, WorkloadClass

coordinator = CROSCoordinator(config=CROSConfig())

tasks = [TaskRecord(
    task_id="verify-1", parent_goal_id="goal", module_id="workspace-verify",
    workload_class=WorkloadClass.INTEGRITY_CRITICAL,
    demand=ResourceDemand(cpu=0.2), remaining_work=1.0,
)]

result = coordinator.adaptive_resource_cycle(tasks)
print(result.diagnosis.condition)        # e.g. CPU_BOUND / HEALTHY
print(result.action.action_type)         # e.g. DEFER_BACKGROUND
print(result.watchdog_triggered)         # False unless a safety limit tripped
print(result.decision.to_dict())         # full provenance record
```

Deterministic replay benchmark:

```python
from ore_cros.cycle import run_replay_benchmark
from ore_cros.evaluation.benchmarks import TaskTraceEvent

trace = [TaskTraceEvent(f"t{i}", "m", WorkloadClass.BATCH, i * 0.1, 0.5)
         for i in range(30)]
report = run_replay_benchmark(trace, worker_count=4)
print(report.to_dict())  # throughput, p50/p95/p99, queue delay, fairness, overhead
```

## Tests

```
python -m pytest tests/ -k cros        # 127 acceptance tests, all passing
python -m pytest tests/                # full repository suite (no regressions)
```

The suite covers every acceptance item in section 8 of the specification; see
`TRACEABILITY.md` for the step-by-step and item-by-item mapping.

## Dependencies

Python 3.10+ standard library only — no third-party imports. Platform-specific
features (cgroup v2, DAMON, io_uring, eBPF, NUMA) are capability-detected with
portable fallbacks and never required.

## Design guarantees

1. Learning may never override hard safety, integrity, isolation, dependency,
   fairness or real-time constraints.
2. The policy engine cannot invoke arbitrary OS commands; it only reaches the
   runtime control state through the narrow typed execution manager.
3. Every adaptive policy has activation thresholds, watchdogs, rollback
   triggers, known-good settings and a deterministic fallback.
4. Unknown capacities are `None`, never zero; estimates are labelled as
   estimates; scopes are never silently mixed.
5. No hard real-time claim without bounded execution and platform evidence.
