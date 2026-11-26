# Distributed Coordination Layer — Point 17 Architecture

Implementation of the ORE Integrated Algorithm "Point 17: Distributed Systems,
Consensus and Coordination Algorithms" (source sections 1–10, 48 algorithmic
steps in Phases A–G) for ORE-Neurocomputational, integrated with USG semantic
knowledge and the CROS resource-management layer.

This document maps every step of the algorithm to its implementation and states
the same scientific caution the source document states: the package implements
distributed coordination as an infrastructure capability beneath ORE's
cognitive architecture, not a new reasoning engine. Protocol agreement is not
epistemic truth; USG keeps that responsibility.

## Package layout

```
ore_distributed/
├── core/                Phase A — deployment, state classification, identity, protocol
│   ├── deployment.py    Step 1   topology detection and descriptors
│   ├── state_class.py   Steps 2–4 state classes, consistency, threat models
│   ├── identity.py      Step 6   durable node identities and credentials
│   └── protocol.py      Step 5   versioned message envelopes and gates
├── membership/          Phase B — registry, admission, authenticated transport
│   ├── registry.py      Steps 7, 9   node registry and capability claims
│   ├── admission.py     Steps 8, 12  authorized membership changes with audit
│   └── transport.py     Steps 10, 11 authenticated fabric, retries, backpressure
├── consensus/           Phase C — Raft authoritative coordination
│   ├── log.py           Steps 14, 16 replicated log and quorum arithmetic
│   ├── state_machine.py Steps 13, 18, 22 deterministic state machine, narrow scope
│   ├── raft.py          Steps 14–19 election, replication, commit, recovery
│   ├── reconfiguration.py Step 20 safe membership changes, config epochs
│   └── leases.py        Step 21 leases with fencing tokens and drift bounds
├── replication/         Phase D — selective CRDTs and semantic conflict handling
│   ├── crdts.py         Steps 23, 24 G-Set, OR-Set, PN-Counter, LWW-Register
│   ├── replica.py       Steps 25, 27 observation identity, statuses
│   ├── conflicts.py     Step 26 semantic conflict records routed to USG
│   ├── reconciliation.py Step 28 anti-entropy reconciliation
│   ├── tombstones.py   Step 29 causal-stability tombstone collection
│   └── promotion.py     Step 30 USG promotion gate
├── gossip/              Phase E — dissemination and suspicion
│   ├── peer_sampling.py Step 31 bounded, diverse peer views
│   ├── dissemination.py Step 32 digest exchange and selective repair
│   └── suspicion.py     Step 33 unreachable → suspected → removed
├── time/                Phase E — ordering and causality
│   ├── hlc.py           Step 34 Hybrid Logical Clocks with persistence
│   └── causal.py        Step 35 parents, vector clocks, dependency checks
├── provenance/          Phase E — immutable event history
│   ├── envelope.py      Step 36 envelopes with digests and signatures
│   └── journal.py       Step 37 audit trail, verification, quarantine
├── placement/           Phase F — distributed placement
│   ├── consistent_hashing.py Step 38 hash ring with virtual nodes
│   └── router.py        Step 39 shard routing per data class
├── execution/           Phase F — task execution
│   ├── tasks.py          Step 40 leases, fencing, ownership validation
│   ├── idempotency.py    Step 41 idempotency keys, outbox/inbox
│   └── checkpoints.py    Step 42 workflow checkpoints with digests
├── recovery/            Phase F — recovery and partitions
│   ├── recovery.py      Step 43 restart orchestration
│   └── partitions.py    Step 44 majority/minority policy, degraded status
├── integration/         Phase G — ORE integration and observability
│   ├── workspace.py      Step 46 typed workspace event routing
│   ├── usg_adapter.py    Step 47 USG validation over ore_usg contracts
│   ├── cros_adapter.py   Step 47 bounded resource adaptation
│   └── observability.py  Step 48 metrics and degraded modes
├── adapters.py          Step 45 interfaces + single-process in-memory adapters
├── pipeline.py          Deployment assembly (Phases A–G)
├── registry.py          The 48-operation registry
└── __main__.py          CLI summary
```

## Core architectural decisions

1. **State separation (Step 2).** Authoritative operational state lives only
   behind the Raft-replicated state machine; mergeable collaborative state
   lives in CRDT replicas; event/provenance history is immutable and append-only;
   agent-local transient state never leaves the node. Unclassified schemas are
   rejected by `StateSchemaRegistry.require`.

2. **Consensus scope is narrow (Step 22).** `COMMAND_TYPES` is the closed set of
   control-plane operations (task assignment, workflow transitions, policy
   versions, coordinator epochs, checkpoints, configuration, semantic
   approvals). High-volume observations and scratch state cannot enter the log.

3. **Commit rules (Step 17).** `RaftNode._advance_commit_index` counts
   replication of a *current-term* entry on a majority before advancing the
   durable commit index; client acknowledgment happens only after that point
   (`ProposalResult.acknowledged`).

4. **Fencing over timeouts (Steps 21, 40).** Lease expiry alone never
   authorizes writes. `LeaseManager.validate_write` and
   `TaskCoordinator.commit_output` check a monotonically increasing fencing
   epoch at the resource boundary; reassignment happens only through the
   authoritative state machine.

5. **Convergence is not truth (Steps 26, 30, 47).** CRDT merges preserve
   contradictory claims as `SemanticConflictRecord`s routed to USG; promotion
   to an assertion requires ore_usg schema validation plus the protocol-level
   promotion gate (schema, provenance, confidence, context, contradiction
   status). Consensus may *order* an approval decision without establishing it.

6. **Explicit causality (Step 35).** HLC provides a practical total-order key
   only; causal parents, vector clocks, and dependency quarantine keep
   causality separate from timestamp ordering.

7. **Single-process deployments stay consensus-free (Steps 1, 45, 48).**
   `build_pipeline` classifies the deployment mode and the single-process
   capabilities disable consensus, CRDT dissemination, gossip, and recovery;
   `LocalDeployment` supplies in-memory adapters behind the Step 45
   interfaces.

## USG and CROS integration

- `integration/usg_adapter.py` builds a USG-KNOWLEDGE/0.1.0 bundle from a
  distributed observation and validates it with the repository's own
  `ore_usg.validation.validate_knowledge_bundle`, preserving evidence lineage
  and rejecting invalid semantic schemas before promotion.
- `integration/cros_adapter.py` converts resource telemetry into the
  repository's `ore_contracts.resource_profiles.CapabilityDemand` and adapts
  gossip frequency, replication batch size, and placement within explicit
  bounds; consensus safety parameters (quorum, election timeouts, commit rule,
  membership) are protected and outside the adaptable set.
- `integration/workspace.py` routes typed `ore_workspace.models.WorkspaceEvent`
  records onto the authoritative, mergeable, and local paths without exposing
  workspace internals as a replicated database.

## Acceptance coverage

The test suite (`tests/test_distributed_*_point17.py`) covers the acceptance set
of the source document section 6, including: single-process operation without a
consensus cluster; authenticated multi-process messaging with rejection of
unknown identities; discovery that cannot create voters; majority election;
minority write blocking; split-vote safety; stale-term rejection; conflicting
suffix reconciliation; committed-entry survival across crash/restart; safe
membership add/remove; duplicate command idempotence; deterministic replay;
fencing-token rejection of stale lease holders and delayed workers; CRDT
associativity/commutativity/idempotence under generated states; reordered and
duplicated delivery convergence; offline replica reconciliation; explicit
semantic contradictions; convergence that never marks assertions true;
tombstone collection that cannot resurrect deleted values; gossip fanout and
message budgets; suspicion before removal; HLC monotonicity across restarts;
causal dependencies independent of HLC ordering; digest and signature
verification; quarantine and repair of missing dependencies; consistent-hash
churn that moves only expected keys; failure-domain rebalancing; idempotent
task effects; checkpoint artifact verification; recovery restoration;
partition status exposure; USG rejection of invalid schemas; CROS consensus
safety enforcement; observable replication lag, commit latency, and recovery
time; and the safe degraded mode when quorum is unavailable.

## Status and limits

- The transport is a deterministic in-memory fabric (partitions, delays,
  duplication, drops, bounded payloads) suitable for protocol testing; a real
  network transport plugs in behind the same `ProtocolGate` interfaces.
- BFT is intentionally out of scope until a deployment threat model demands it
  (Step 4 gate): crash-fault-tolerant consensus is the implemented baseline.
- Membership changes are implemented as serialized single-server
  configuration changes with state transfer, quorum validation, and
  configuration epochs; joint consensus remains an available upgrade path.
