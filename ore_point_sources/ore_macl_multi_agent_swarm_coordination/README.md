# ORE Point 11 — Multi-Agent and Swarm Coordination (MACL)

Algorithm implementation for **"ORE Neurocomputational — Point 11 - Multi-Agent
and Swarm Coordination Algorithm"** (ranked Point 14, source Point 11, sections
11.1–11.5, Steps 1–66): the ORE Multi-Agent Cognitive Coordination Layer (MACL)
— hierarchical multi-agent cognitive coordination and selective swarm
optimization, integrated with the existing ORE-Neurocomputational repository.

## Contents

```
├── ore_macl/                          # The algorithm package (22 modules)
│   ├── __init__.py                    # Public API and exports
│   ├── contracts.py                   # AgentDescriptor, AgentTask, AgentMessage, AgentResult,
│   │                                  #   ProvenanceTrail, SchemaSpec, ResourceEnvelope, RetryPolicy
│   ├── registry.py                    # AgentRegistry (lifecycle, health, safe replacement),
│   │                                  #   autonomy classification, 9-stage operation registry
│   ├── decomposition.py               # TaskDAG with cycle detection, TaskDecomposer
│   ├── matching.py                    # EligibilityGate, AgentTaskMatcher, AssignmentSolver,
│   │                                  #   TopologySelector, assign_owners
│   ├── messages.py                    # AgentMessageBus (idempotency, TTL, quotas, lineage)
│   ├── workspace_adapter.py           # ScopedSnapshot, versioned proposals, evidence lineage
│   ├── agents.py                      # BaseBoundedAgent protocol, 5 specialist agents,
│   │                                  #   TaskExecutor, deterministic fallbacks
│   ├── supervisor.py                  # MACLSupervisor: end-to-end coordination loop
│   ├── adjudication.py                # ResultAdjudicator (schema/revision/provenance/criteria)
│   ├── debate.py                      # DebateOrchestrator, independence measurement,
│   │                                  #   independent verification
│   ├── aggregation.py                 # ResultAggregator, keyed publication, goal evaluation
│   ├── peer.py                        # PeerCoordinator (bounded delegation, escalation)
│   ├── recovery.py                    # RecoveryManager (retries, reconciliation, leases,
│   │                                  #   quarantine)
│   ├── reliability.py                 # Beta-Binomial posteriors, uncertainty-aware scoring
│   ├── ledger.py                      # PerformanceLedger (quality, calibration, rewards)
│   ├── pso.py                         # Bounded Particle Swarm Optimization (isolated)
│   ├── aco.py                         # Ant Colony Optimization (isolated)
│   ├── biology.py                     # Rate-coded coordination-network surrogate
│   ├── errors.py                      # MACL exception hierarchy
│   ├── pipeline.py                    # 9-stage traced pipeline, swarm experiments,
│   │                                  #   SQLite annotation + trace persistence
│   └── annotations.py                 # SQLite annotation entry point
├── tests/test_macl_point11.py         # 38 acceptance tests (spec Section 9)
├── docs/MULTI_AGENT_SWARM_COORDINATION_POINT11.md
├── examples/multi_agent_swarm_coordination.py
└── pyproject.toml                     # Adds "ore_macl*" to package discovery
```

## Integration with ORE-Neurocomputational

Copy `ore_macl/`, `tests/test_macl_point11.py`, `docs/…`, and
`examples/…` into the repository root (merging directories), and add
`"ore_macl*"` to the `[tool.setuptools.packages.find] include` list in
`pyproject.toml` (the included `pyproject.toml` already contains this one-line
change on top of the repository's version).

MACL extends — never duplicates — the existing infrastructure:

- agents read versioned snapshots and publish proposals through the existing
  `SharedWorkspace` / `FindingPublisher` validation pipeline;
- deterministic module tasks route through the existing
  `CognitiveTaskCoordinator` gateway (the supervisor never calls module
  backends directly);
- goal completion is evaluated by the existing `GoalEvaluator` from accepted
  outcomes;
- every class and function is indexed into the ORE SQLite logger
  (`register_annotations`), and every pipeline run persists its nine-stage
  operation trace (`record_result_to_sqlite`).

## Verification

- `python3 -m pytest tests/test_macl_point11.py` — 38 tests covering every
  acceptance criterion of the specification (identity/contracts, coordinator
  routing, DAG cycles, idempotent publication, stale revisions, least
  privilege, timeout reconciliation, visible conflicts, bounded critic loops,
  independent verification, uncertainty-aware reliability, quarantine,
  budgets, bounded peer depth, isolated swarm experiments, PSO/ACO constraint
  enforcement, full SQLite lineage).
- Full repository suite: 1234 passed with this implementation in place.
- `ORE_AUTO_LOG=0 PYTHONPATH=. python3 examples/multi_agent_swarm_coordination.py`
  runs the complete nine-stage loop (goal completes; PSO and ACO experiments
  run isolated from production control).

## Architectural guarantees (spec Sections 1, 9, 10)

- MACL is not a second scheduler or second blackboard.
- Autonomy is bounded, justified, and contract-bound; deterministic modules
  stay deterministic.
- Eligibility gates are Boolean hard constraints; a high score never
  overrides a failed gate.
- Conflicts remain visible; no majority-vote truth; minority findings with
  evidence are preserved.
- Reliability uses Beta-Binomial posteriors with credible intervals; tiny
  samples cannot justify precise ranking.
- Timeouts are reconciled (unknown, not failed) before any retry or
  reassignment; leases guard reassignment; repeated violations quarantine.
- PSO/ACO run on fixed replay objectives only; they propose parameters and
  paths, they never alter permissions or bypass validation; MARL production
  control remains prohibited without separate approval.
