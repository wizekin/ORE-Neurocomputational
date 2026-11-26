# ORE Point 9 — Evolutionary Intelligence, Genetic Programming & Quality-Diversity

**Package:** `ore_evolution_intelligence` v1.0.0
**Implements:** "ORE Integrated Algorithm — Point 9: Evolutionary Intelligence, Genetic Programming and Quality-Diversity Algorithms" (all 44 steps of the integrated algorithm, Phases A-E, plus the architecture, integration map, acceptance tests and safeguards of the specification)
**Target:** ORE-Neurocomputational framework
**Status:** fully implemented; 159 dedicated tests passing (`pytest ore_evolution_intelligence/tests/`); full repository suite (1412 tests) green

## What this package is

An isolated **Adaptive Cognitive Strategy Evolution Layer** for ORE-Neurocomputational:
it discovers, evaluates, preserves and selects improved cognitive strategies through
bounded evolutionary optimization over coordinator parameters, finite policy tables and
constrained genetic programming, with a MAP-Elites quality-diversity archive,
independent verification, a validated strategy registry and canary rollout with
rollback.

**Core invariant:** evolution produces candidates, not authority. No evolved program,
policy or USG rule becomes active merely because it scores well on its training
benchmark. Hard correctness and safety conditions are gates, never small penalties.

## The implemented pipeline (section 11 of the specification)

```
Bounded Evolutionary Optimization -> Sandboxed Evaluation -> MAP-Elites Archive ->
Independent Verification -> Versioned Validated Strategy Registry ->
Contextual Coordinator Selection -> Live Monitoring and Rollback
```

Three bounded targets are supported end-to-end:

| target | genome kind | search operators |
|---|---|---|
| module-selection weights / thresholds / routing / resource budgets | `parameters` | group crossover + gaussian mutation (GA), (mu+lambda) ES, CMA-ES |
| task-routing policy (finite context table) | `policy_table` | per-cell crossover, add/drop/defer mutation |
| planning heuristic (typed expression tree) | `policy_ir` | subtree crossover/mutation over a typed grammar |

## Package contents

```
ore_evolution_intelligence/
├── __init__.py            # public API, v1.0.0
├── domain/                # candidate strategy model (section 2.1)
│   ├── actions.py             # approved action vocabulary + typed params (step 8)
│   ├── genome.py               # bounded parameter vectors, finite policy tables
│   ├── strategy.py             # StrategyCandidate, strict lifecycle states (s. 11)
│   ├── behavior_descriptor.py   # descriptor dimensions + versioned calibration (steps 39-40)
│   └── evaluation_record.py    # F=(Q,R,T,M,C,S), hard gates, scalar fitness J (s. 2.2)
├── evaluation/            # Phase B sandboxed execution + measurement
│   ├── benchmark_manifest.py  # versioned protocol, sealed train/val/test split (steps 4-10)
│   ├── task_simulators.py      # deterministic ORE coordinator simulator (section 5)
│   ├── sandbox.py              # capability grants, budgets, traces, invalid detection (steps 11-17)
│   ├── evaluator.py            # independent multi-seed evaluator, hardcoding detection
│   └── heldout_validation.py   # sealed-test comparison, validation envelope (step 29)
├── search/               # Phase C evolutionary optimization + Phase D GP
│   ├── operators.py            # bounded crossover/mutation: valid or rejected (steps 24-25)
│   ├── genetic_optimizer.py    # GA with gates, elites, diversity guard, stopping (steps 18-29)
│   ├── evolution_strategy.py   # (mu+lambda) ES with gaussian perturbation
│   ├── cmaes.py                # CMA-ES, continuous correlated vectors only (step 30)
│   ├── genetic_programming.py # typed grammar, bounded trees, declarative policy IR (steps 31-34)
│   └── novelty_search.py       # kNN novelty + Pareto archive (section 2.5)
├── archive/               # Phase E quality-diversity
│   ├── cell_index.py           # versioned cell boundaries (step 40)
│   ├── map_elites.py            # MAP-Elites archive + QD search driver (steps 41-44)
│   └── archive_metrics.py      # coverage, spread, turnover, family robustness
├── verification/          # independent verification chain
│   ├── static_policy_check.py  # types, bounds, unreachable branches (step 36)
│   ├── semantic_rule_check.py  # contradictions, defer conflicts, USG schema (step 37)
│   ├── regression_gate.py      # deployment-blocking regressions (section 7)
│   └── promotion_policy.py     # strict lifecycle state machine (section 11)
├── registry/              # validated strategy registry
│   ├── artifact_store.py       # content-addressed artifacts + manifests (s. 4)
│   ├── versioning.py           # version history + immutable rollback target (step 3)
│   └── strategy_registry.py    # approval-required registry, dedup, production selection
├── observability/         # evidence and audit
│   ├── events.py               # structured events + ore_logger SQLite persistence
│   ├── lineage.py              # ancestry reconstruction, lineage diversity (step 27)
│   └── repository.py           # append-only SQLite repository (runs/candidates/evals)
├── orchestration/         # the end-to-end service
│   ├── evolution_service.py    # StrategyEvolutionCoordinator.evolve() (section 4 contract)
│   ├── coordinator_adapter.py  # controlled configuration interface (step 38)
│   ├── resource_budget.py      # budgets + CROS-style admission (step 12)
│   ├── rollout_controller.py   # canary rollout, drift monitoring, rollback (s. 7)
│   └── workspace_bridge.py     # ore_workspace finding publication (integration map)
└── tests/                 # 159 tests: unit, property, adversarial, e2e, reproducibility
```

## Strict lifecycle states (section 11)

`GENERATED -> SANDBOX_TESTED -> VALIDATED -> APPROVED -> CANARY -> ACTIVE -> DEPRECATED`
(with `REJECTED` reachable from any pre-ACTIVE state). Every transition requires typed
evidence (evaluation record, static/semantic checks, held-out improvement, explicit
approval, canary plan, canary decision, deprecation/rejection reason). The production
coordinator can only select APPROVED / CANARY / ACTIVE strategies — enforced by the
controlled configuration interface, which raises `PermissionError` for anything else.

## Safety boundaries

- Genotypes are *data interpreted by a capability-enforcing sandbox*: restricted to an
  explicit action vocabulary with typed parameters; no filesystem, network, shell,
  database mutation or unrestricted imports are expressible.
- The held-out test set is sealed; only the post-search held-out validator opens it.
- The baseline is captured as an immutable rollback target (step 3) and restored on
  canary regression or live drift (rollback events recorded).
- Benchmark-hardcoding detection, evaluator independence from candidate data,
  step/output/memory/latency budgets and nontermination termination guards (step 16).
- Evolution failures never interrupt ordinary ORE tasks: `evolve()` always returns a
  report (acceptance test 34).

## Dependencies

- **Runtime:** Python 3.9+ and `numpy` (pure NumPy/stdlib; no DEAP/pymoo/pyribs needed —
  GA, ES, CMA-ES, GP and MAP-Elites are implemented natively per the spec's
  dependency-free integration style)
- **Tests:** `pytest>=7.4`
- **Framework:** `ore_workspace` (finding publication — lazy import), `ore_logger`
  (SQLite event persistence — lazy import). The package imports cleanly in isolation;
  workspace/logger integration tests skip if those modules are unavailable.

## Integration into ORE-Neurocomputational

1. Copy `ore_evolution_intelligence/` into the repository root (next to the other
   `ore_*` packages).

2. `pyproject.toml`:

   - `[tool.setuptools.packages.find]` — add `"ore_evolution_intelligence*"` to `include`
   - `[tool.pytest.ini_options]` — set
     `testpaths = ["tests", "ore_graph_intelligence/tests", "ore_evolution_intelligence/tests"]`

3. Verify:

   ```bash
   python -c "import ore_evolution_intelligence as m; print(m.__version__)"
   # -> 1.0.0
   pytest ore_evolution_intelligence/tests/ -q
   # -> 159 passed
   ```

## Running an evolution experiment

```python
from ore_evolution_intelligence import (
    EvolutionBudgetSpec, EvolutionObjective, StrategyEvolutionCoordinator,
)

coordinator = StrategyEvolutionCoordinator()
report = coordinator.evolve(
    EvolutionObjective.module_selection(),          # narrow, measurable objective
    EvolutionBudgetSpec(total_evaluations=400),     # bounded budget (step 12)
)
print(report.status)                                # "completed" (never raises)
print(report.as_dict()["best"]["state"])            # e.g. "ACTIVE" via canary promotion
print(report.as_dict()["verification"]["heldout"])  # sealed-test improvement evidence
```

The run report records objective, benchmark version, split digests, evaluator version,
seeds, baseline metrics + rollback-target digest, GA/QD search statistics, verification
outcomes, registry records, canary result, lineage summary and the emitted event kinds
(acceptance test 3: every run records objective, task split, evaluator version, seed).

## Evaluation metrics (section 8)

Multi-objective aggregates (Q, R, T, M, C, S) with mean, dispersion, 95% t-confidence
intervals and worst-case behavior; per-task-family quality; latency p50/p95; cost per
task; evolution-efficiency metrics (evaluations to improvement, duplicate rate,
mutation acceptance, lineage diversity); quality-diversity metrics (coverage, per-cell
quality, descriptor spread, turnover); generalization (train-validation gap, sealed
test performance). Raw component metrics and traces are always preserved (step 17).
