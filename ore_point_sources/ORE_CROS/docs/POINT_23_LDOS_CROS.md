# Point 23 — Learning-Directed Operating Systems and Adaptive Resource Management (CROS)

Implementation status: IMPLEMENTED (software-tested). This document describes the `ore_cros` package, the CROS (Cognitive Resource-Oriented System) adaptive resource-management layer of ORE-Neurocomputational, implemented from the Point 23 integrated algorithm specification ("Learning-Directed Operating Systems and Adaptive Resource Management", source point 23, overall relevance rank 6 of 24).

> Scientific scope: this is a biologically unconstrained engineering layer. It does not replace the host operating system, does not write kernel settings, and its default parameters are engineering starting points that require calibration against measured workloads. CFS/EEVDF, DAMON, io_uring, eBPF/XDP/DPDK, MGLRU, Borg/Kubernetes/Decima and CXL are treated as references or capability-detected options with portable fallbacks — never as mandatory dependencies.

## 1. Purpose and architectural thesis

Point 23 asks how a computational system can learn to allocate CPU time, memory, storage, network bandwidth, accelerators and module capacity according to observed workloads, dependencies, predicted requirements and system objectives. `ore_cros` implements the recommended architecture:

MAPE-K autonomic loop + typed resource-state model + hierarchical deterministic allocation + feedback control + predictive models + constrained learning + verified execution + continuous measurement.

Learning may improve resource decisions, but it can never override hard safety, integrity, isolation, dependency, fairness or real-time constraints (Step 13, Step 52). CROS is a higher-level resource intelligence coordinating existing ORE components — not a duplicate cognitive architecture and not a host-OS replacement.

## 2. Package layout

| Module | Spec section | Responsibility |
| --- | --- | --- |
| `ore_cros/resource_model.py` | 2, Steps 1-2, 47 | Resource vector R_t, demand vectors, workload/criticality classes, versioned resource state, scope topology |
| `ore_cros/config.py` | 2.4, section 7 | Objective weights per class, windows, quality gates, adaptation limits |
| `ore_cros/actions.py` | Steps 11, 14 | Typed resource actions with validated parameters; runtime control state |
| `ore_cros/telemetry/schemas.py` | Step 3 | Unit-safe measurement records; scope-mixing rejection |
| `ore_cros/telemetry/collectors.py` | Steps 1, 4 | Host discovery (`/proc`, cgroup v2, `os` counters) with simulated fallback; collector isolation; baselines |
| `ore_cros/telemetry/quality.py` | Step 8 | Freshness, missingness, source reliability, counter resets; high-impact gates |
| `ore_cros/telemetry/history.py` | Step 7 | Short/medium/long windows; bounded retention with compacted aggregates |
| `ore_cros/accounting/task_costs.py` | Step 5 | Task/module attribution; rate integration; estimated labelling |
| `ore_cros/mape/monitor.py` | Step 9 | Coherent, versioned observation snapshots |
| `ore_cros/mape/analyze.py` | Steps 6, 10 | Pressure (sustained vs spike), Little's Law checks, bottleneck diagnosis with evidence |
| `ore_cros/mape/plan.py` | Steps 11-12 | Candidate generation with effects/costs/rollback; versioned knowledge base |
| `ore_cros/mape/execute.py` | Steps 14-15 | Narrow typed execution manager (audit + rollback); outcome measurement with residuals |
| `ore_cros/policies/baseline.py` | Step 16 | Deterministic baseline scheduler + admission control |
| `ore_cros/policies/cpu.py` | Steps 17-26 | Weighted fairness (virtual runtime), EDF, Rate Monotonic, MLFQ with aging, gang scheduling, quotas, starvation guards, shadow mode, constrained adaptation |
| `ore_cros/policies/memory.py` | Steps 27-35 | Working sets, tiered cache, LRU/CLOCK/MGLRU replacers, DAMON adapter, NUMA, compression tiering, GPU admission, CXL extension point |
| `ore_cros/policies/io_network.py` | Steps 42-46 | I/O profiling, async-I/O backend fallback, eBPF/XDP/DPDK gates, storage matrix, durability contracts |
| `ore_cros/prediction/demand.py` | Step 49 | Demand and completion-time predictors with calibrated uncertainty |
| `ore_cros/learning/bandit.py` | Steps 50-51 | Constrained contextual bandit with propensity logging and offline replay; RL/MPC research path |
| `ore_cros/safety/constraints.py` | Steps 13, 47-48 | Hard-constraint filter, criticality matrix, quota ledger, WCET analysis |
| `ore_cros/safety/fallback.py` | Step 52 | Hysteresis, oscillation detector, PID anti-windup, watchdog, rollback manager |
| `ore_cros/distributed/placement.py` | Steps 36-41 | Node advertisements, eligibility, placement scoring, migration costs, failure recovery |
| `ore_cros/evaluation/benchmarks.py` | sections 8-9 | Replay benchmark, policy comparison, long-replay stability |
| `ore_cros/cycle.py` | section 5 | The end-to-end `ADAPTIVE_RESOURCE_CYCLE` |
| `ore_cros/integration.py` | section 6 | ORE integration boundaries, workload-class mapper, SQLite decision store |

Modules beyond the spec's conceptual boundaries (section 6.1) — `actions.py`, `config.py`, `cycle.py`, `integration.py`, `telemetry/history.py` — hold shared types the specified modules require; the spec's "inspect the repository and reuse existing abstractions" guidance applies.

## 3. The adaptive resource cycle

`CROSCoordinator.adaptive_resource_cycle()` implements section 5 of the specification step for step: discover environment → collect typed telemetry → assess quality → update versioned resource state → collect ready/running tasks → attribute usage → detect pressure and queueing → MAPE-analyze → retrieve knowledge → generate candidates → filter by hard constraints → keep the current safe policy when nothing safe remains, else select baseline-or-constrained → estimate effects, costs, rollback → validate → execute through the restricted resource interface → measure latency/throughput/fairness/pressure → store decision provenance and outcome → update predictors and policy evaluation → roll back and activate the baseline on watchdog or invariant failure. The loop reacts to events, and hysteresis with minimum dwell times prevents oscillation.

```python
from ore_cros import CROSCoordinator, CROSConfig

coordinator = CROSCoordinator(config=CROSConfig())
result = coordinator.adaptive_resource_cycle(tasks, now=timestamp)
print(result.diagnosis.condition, result.action.action_type, result.watchdog_triggered)
```

## 4. Safety model

- Hard constraints are separate from the soft objective J = wL·Latency + wD·DeadlineMisses + wC·Cost + wE·Energy + wF·Unfairness + wS·InstabilityRisk (weights per workload class).
- The constrained contextual bandit selects only from the approved safe action set; shadow mode records proposals without execution; a canary fraction bounds live use; the RL/MPC path refuses to train until every prerequisite is attested.
- Every adaptive change is reversible through the execution manager's undo tokens or the rollback manager's registered baseline; watchdog triggers (invariant failure, prediction error, tail-latency regression, oscillation, timeout) disable adaptation and restore the deterministic baseline for a cooldown period.
- No hard real-time claim is accepted without bounded WCETs, accounted overhead and explicit platform evidence (`WcetAnalyzer.hard_real_time_claim` raises otherwise).
- Resource optimization can never downgrade a storage durability contract.

## 5. Integration with existing ORE components

CROS keeps the section-6 boundaries: AdaptiveResourceCoordinator remains the central policy authority for module selection; CognitiveTaskCoordinator publishes task demand, deadlines and dependencies (projected via `cros_task_from_metadata`); ModuleScheduler applies capability and execution constraints; SharedWorkspace is not a telemetry database; ore_usg facts are never mutated by resource decisions; ore_logger-compatible SQLite stores decision provenance with bounded retention (`SqliteDecisionStore`, table `cros_decisions`). Workload-class assignment is policy-owned (`OreWorkloadClassMapper`); tasks cannot promote their own criticality.

## 6. Acceptance tests

`tests/test_cros_*.py` (127 cases) map every acceptance item of spec section 8: unit-safe telemetry, scope separation, stale-telemetry gating, deterministic replay equivalence, attribution labelling, budget exceptions, integrity-critical reservations, admission control, weighted fairness, aging, EDF/RM validation, oscillation resistance, PID overshoot bounds, independent MAPE stages, constraint filtering, execution-manager exclusivity, baseline restoration, shadow mode, predictor calibration, working-set responsiveness, tier-preserving retrieval, NUMA locality, GPU admission, io_uring fallback, durability contracts, node eligibility, migration costs, side-effect-aware recovery, deferral ordering, WCET evidence, bandit safety, canary limits, fairness floors, benchmark metrics, decision provenance, quota isolation, long-replay stability, failure-safe behavior, and baseline availability when learning fails.

Run them with:

    python -m pytest tests/ -k cros

## 7. Roadmap alignment

The implementation covers roadmap stages 1-5 (observability and accounting; deterministic coordination; MAPE-K and feedback control; predictive adaptation; constrained learning) with stage-6 elements (cgroup v2 reads, DAMON detection, NUMA, io_uring selection, cluster placement) implemented as capability-detected adapters with portable fallbacks and documented evidence gates — no extension is enabled without a measured workload requirement, per the spec's non-goals.

## 8. Limitations

- Telemetry collectors read `/proc`, cgroup v2 and `os.times` where available and fall back to a deterministic simulated source; no kernel settings are ever written.
- The replay benchmark is a discrete-event approximation for policy comparison, not a claim about real operating-system behaviour.
- CFS/EEVDF are design references: `WeightedFairScheduler` implements virtual-runtime proportional sharing but does not reproduce kernel semantics.
- CXL remains a registry extension point with no CXL-specific behaviour.
