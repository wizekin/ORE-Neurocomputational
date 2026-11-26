# Multi-Agent and Swarm Coordination — Point 11 (Ranked Point 14)

Hierarchical Multi-Agent Cognitive Coordination and Selective Swarm Optimization for
ORE, implemented as the `ore_macl` package (Multi-Agent Cognitive Coordination Layer).

Source: *Most Intelligent Systems 01-24.txt*, Point 11, sections 11.1–11.5, Steps 1–66.
Status: contract-driven hierarchical agent layer with isolated swarm experiments;
no MARL production control, no online learning.

## 1. Objective and architectural boundary

MACL lets selected autonomous cognitive agents cooperate on complex goals while
preserving the existing shared workspace, task coordinator, module contracts,
resource coordinator, USG knowledge boundary, and evidence-validation process.
An ORE agent is a bounded decision-making participant with an explicit role,
capability contract, local state, permitted observations/actions, communication
rules, measurable result contract, resource/time/retry limits, and provenance
trail. Deterministic modules remain deterministic modules; only components that
benefit from autonomous action selection, evidence requests, negotiation, or
adaptive delegation are promoted to agents.

**MACL is not a second scheduler or a second blackboard.** Agent work is bounded
by the supervisor, deterministic module work is routed through the existing
`CognitiveTaskCoordinator` gateway, and every accepted contribution is published
through the existing shared workspace (`SharedWorkspace`), whose state manager
remains the sole authority for acceptance, conflict, deferral, and duplication.

Recommended pattern (11.5): hierarchical multi-agent architecture + blackboard +
selective decentralized coordination.

## 2. Package layout

| Module | Spec section | Responsibility |
|---|---|---|
| `contracts.py` | Section 2 | `AgentDescriptor`, `AgentTask`, `AgentMessage`, `AgentResult`, `ProvenanceTrail`, `SchemaSpec`, `ResourceEnvelope`, `RetryPolicy`, lifecycle/decision enums |
| `registry.py` | Steps 1–7 | `AgentRegistry` (identity, lifecycle, health, safe replacement), autonomy classification, nine-stage operation registry |
| `decomposition.py` | Steps 8–11 | `TaskDAG` with cycle detection, `TaskDecomposer` producing typed task envelopes |
| `matching.py` | Steps 12–16 | `EligibilityGate` (Boolean hard constraints), `AgentTaskMatcher` (weighted scoring), `AssignmentSolver` (greedy / maximum-weight), `TopologySelector` |
| `messages.py` | Section 2 | `AgentMessageBus`: idempotency keys, TTL expiry, per-sender quotas, lineage |
| `workspace_adapter.py` | Steps 24–30 | `ScopedSnapshot` (least-privilege reads), versioned proposals through the existing `FindingPublisher`, evidence lineage resolution |
| `agents.py` | Steps 17–23 | `BaseBoundedAgent` protocol, specialist agents (lexical, syntactic, semantic, verifier, critic), `TaskExecutor` with timeouts, deterministic fallbacks |
| `supervisor.py` | Section 3 | `MACLSupervisor`: the end-to-end coordination loop with budget/deadline/attempt bounds, revision, verification, quarantine, and fallbacks |
| `adjudication.py` | Steps 44–45 | `ResultAdjudicator`: schema, identity, revision, provenance, evidence, criteria checks; duplicate fingerprints |
| `debate.py` | Steps 37–43 | `DebateOrchestrator`: independent proposals, measured independence, evidence-based selection (never vote count), bounded rounds, preserved dissent |
| `aggregation.py` | Steps 46–50 | `ResultAggregator`: compatible aggregation, keyed lexical publication, explicit conflicts, dependency activation, goal evaluation |
| `peer.py` | Steps 31–36 | `PeerCoordinator`: capability discovery without broadcast, ancestry-tracked delegation depth, partition reconciliation, escalation |
| `recovery.py` | Steps 51–54 | `RecoveryManager`: transient-only retries, timeout reconciliation (unknown ≠ failed), leases before reassignment, quarantine |
| `reliability.py` | Section 5 | Beta-Binomial posteriors with credible intervals; uncertainty-aware scoring |
| `ledger.py` | Steps 55–57 | `PerformanceLedger`: accepted-result rate, calibration (Brier), latency/cost EMA, task-class coverage, validated-outcome reward |
| `pso.py` | Steps 63, 65 | Bounded, reproducible Particle Swarm Optimization over fixed replay objectives |
| `aco.py` | Steps 64–65 | Ant Colony Optimization over discrete operation graphs with dependency and budget feasibility |
| `biology.py` | — | `MultiAgentCoordinationNetwork`: rate-coded systems-level reference (DLPFC supervisor, ACC conflict, frontoparietal control, TPJ/mPFC agent modelling, basal ganglia selection, cerebellar timing) — an interpretive reference, not a neural claim |
| `pipeline.py` | Sections 3, 8 | Nine-stage traced pipeline, isolated swarm experiments, SQLite annotation registration and trace persistence |
| `annotations.py` | — | Package annotation entry point for the ORE SQLite catalogue |

## 3. The nine-stage registry

| Stage | Source steps | Focus |
|---|---|---|
| MACL-1 | 1–7 | Agent lifecycle and registry |
| MACL-2 | 8–16 | Task decomposition and allocation |
| MACL-3 | 17–23 | Supervisor/worker protocol |
| MACL-4 | 24–30 | Blackboard and shared evidence |
| MACL-5 | 31–36 | Peer-to-peer cooperation |
| MACL-6 | 37–43 | Debate, critic, and verification |
| MACL-7 | 44–50 | Aggregation and integration |
| MACL-8 | 51–58 | Recovery and performance learning |
| MACL-9 | 59–66 | Optional MARL and swarm |

## 4. Mathematical formulation

**Agent-task matching.** Eligibility is a Boolean gate: permissions, dependency
order, verifier availability, and hard budget conditions must all pass before any
scoring. For eligible pairs the score combines capability coverage, schema
compatibility, uncertainty-aware reliability, latency, cost, load, and conflict
risk:

```
score(a, t) = sum_j w_j f_j(a, t) / sum_j w_j
reliability(a) = E[p_a] - lambda * (CI_width(p_a)),   p_a | data ~ Beta(alpha + s_a, beta + f_a)
```

A high score never overrides a failed hard constraint. Assignment maximizes the
sum of scores subject to one accountable owner per task, agent capacity,
dependency order, and budgets (greedy by default; maximum-weight matching as the
alternative).

**Reliability.** Beta-Binomial posteriors return a posterior mean and a credible
interval; the interval width penalizes tiny samples so agents can never be ranked
precisely from insufficient evidence (Step 56).

**Debate.** Independence between two proposals decreases with shared evidence
artifacts, shared model/policy lineage, and shared tools. Selection is by
evidence score — evidence coverage, provenance, consistency, and uncertainty —
never by the number of supporting agents (Step 39).

**PSO.** `v <- w v + c1 r1 (pbest - x) + c2 r2 (gbest - x)`, positions clamped to
declared bounds, hard constraints gate evaluation, iterations and evaluations are
bounded, and runs are reproducible through an explicit seed. Particles represent
module-selection weights or resource-policy coefficients evaluated against fixed
replay objectives.

**ACO.** Edge sampling probability is proportional to `tau^alpha * eta^beta`;
paths are reinforced only when they pass validation, and dependency and budget
feasibility are enforced at every extension. Ants construct operation or
knowledge-graph paths.

Both optimizers are fully isolated from production control (Step 65): they
propose parameter sets or paths; existing ORE policies validate and authorize
them. They cannot alter permissions or bypass validation.

## 5. ORE integration map

| Existing component | MACL usage |
|---|---|
| `SharedWorkspace` / state manager | Authoritative blackboard: versioned snapshots, findings, revisions, conflicts |
| `CognitiveTaskCoordinator` | Deterministic module tasks route through `CoordinatorGateway`; the supervisor never calls module backends directly |
| Module registry / contracts | Capability and schema source for specialist descriptors |
| `FindingPublisher` / publication | Every agent proposal is published through the existing validation pipeline |
| `GoalEvaluator` | Canonical success criteria; stopping uses accepted outcomes, not completion counts |
| `ore_logger` / SQLite observer | Class/callable annotations, per-stage operation records, message/result lineage |

## 6. Usage

```python
from ore_macl import MACLRunConfig, run_multi_agent_coordination

result = run_multi_agent_coordination(MACLRunConfig(
    goal_id="demo",
    input_text="the reporter ignored the scientist and argued about the study",
))
print(result.goal_report.goal_evaluation.outcome.value)   # completed
print(result.stage_ids)                                   # MACL-1 ... MACL-9
```

Lower-level control:

```python
from ore_workspace import SharedWorkspace
from ore_macl import MACLConfig, MACLSupervisor, GoalRequest
from ore_macl.agents import LogicalClock, build_default_agents

workspace = SharedWorkspace(current_input="...", current_goal="understand_sentence")
supervisor = MACLSupervisor(
    workspace, build_default_agents(clock=LogicalClock()),
    config=MACLConfig(max_rounds=8), clock=LogicalClock(),
)
report = supervisor.run_goal(GoalRequest("g1", "understand_sentence", "..."))
supervisor.close()
```

SQLite annotations and operation traces:

```python
from ore_macl import register_annotations, record_result_to_sqlite

register_annotations("ore.db")                     # index classes/functions
record_result_to_sqlite(result, "ore.db")          # persist the 9-stage trace
```

## 7. Acceptance verification

`tests/test_macl_point11.py` (38 tests) covers every acceptance criterion from
Section 9 of the specification: identity/contract completeness, deterministic
module routing through the coordinator, DAG cycle detection, idempotent
publication, stale-revision rejection, out-of-scope access denial, timeout
reconciliation before unsafe retry, visible conflicts without majority voting,
bounded critic loops, independent verification of critical outputs,
sample-supported reliability, quarantine behaviour, concurrency and budget
enforcement, bounded peer delegation and message volume, isolated (non-production)
swarm experiments, PSO/ACO constraint enforcement, full SQLite lineage, and
goal completion from accepted outcomes through the existing evaluator.

## 8. Failure modes and safeguards

| Failure | Safeguard implemented in |
|---|---|
| Too many agents | Small default role set; concurrency budgets (`ResourceEnvelope`, registry capacity) |
| Agents duplicate modules | Autonomy classification with justification (Step 1) |
| Supervisor bottleneck | Bounded peer coordination only where enabled (`MACLConfig.enable_peer_topology`) |
| Message explosion | Idempotency keys, TTL expiry, quotas, delegation depth (messages, peer) |
| Duplicate execution | Idempotency keys, status reconciliation, leases (recovery) |
| Stale workspace state | Revision-bound tasks and adjudication checks (adjudication, workspace_adapter) |
| False consensus | Evidence adjudication and preserved dissent (debate, aggregation) |
| Correlated outputs | Measured independence, independent verifier (debate) |
| Reward hacking | Accepted-outcome rewards, hard constraints, penalties (ledger) |
| Unstable PSO/ACO | Fixed replay objectives, bounded parameters, constraints (pso, aco) |
| Fabricated provenance | Artifact-reference validation and quarantine (adjudication, recovery) |
| Endless debate | Maximum rounds, budget, stop conditions (debate) |
| Duplicate coordinator | The existing coordinator and workspace remain the only authorities |
