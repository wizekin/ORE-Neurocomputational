# ORE Point 17 — Distributed Systems, Consensus and Coordination Algorithms

Full implementation of the ORE Integrated Algorithm "Point 17: Distributed
Systems, Consensus and Coordination Algorithms" as the `ore_distributed`
package for ORE-Neurocomputational, integrated with USG semantic knowledge
(`ore_usg`) and the CROS resource-management layer (`ore_contracts`).

## Contents of this archive

```
ore_distributed/                       The algorithm package (35 modules)
├── core/        Phase A  deployment, state classification, identity, protocol (Steps 1–6)
├── membership/  Phase B  registry, admission, authenticated transport (Steps 7–12)
├── consensus/   Phase C  Raft: log, state machine, election, commit,
│                         reconfiguration, leases/fencing (Steps 13–22)
├── replication/ Phase D  CRDTs, replicas, conflicts, reconciliation,
│                         tombstones, USG promotion gate (Steps 23–30)
├── gossip/      Phase E  peer sampling, digest dissemination, suspicion (Steps 31–33)
├── time/        Phase E  Hybrid Logical Clocks, causal metadata (Steps 34–35)
├── provenance/  Phase E  immutable envelopes, audit journal, quarantine (Steps 36–37)
├── placement/   Phase F  consistent hashing, shard routing (Steps 38–39)
├── execution/   Phase F  task leases/fencing, idempotency, checkpoints (Steps 40–42)
├── recovery/    Phase F  restart orchestration, partition policy (Steps 43–44)
├── integration/ Phase G  workspace connector, USG adapter, CROS adapter,
│                         observability/degraded modes (Steps 45–48)
├── adapters.py          Step 45 interfaces + single-process in-memory adapters
├── pipeline.py          Deployment assembly across Phases A–G
├── registry.py          The 48-operation registry (Phases A–G)
└── __main__.py          CLI summary (`python -m ore_distributed`)

tests/test_distributed_core_point17.py                      Steps 1–6 acceptance tests
tests/test_distributed_membership_point17.py                Steps 7–12 acceptance tests
tests/test_distributed_raft_point17.py                      Steps 13–22 acceptance tests
tests/test_distributed_crdt_point17.py                     Steps 23–30 acceptance tests
tests/test_distributed_gossip_provenance_point17.py         Steps 31–37 acceptance tests
tests/test_distributed_placement_execution_point17.py       Steps 38–43 acceptance tests
tests/test_distributed_integration_point17.py               Steps 44–48 acceptance tests

docs/DISTRIBUTED_COORDINATION_POINT17_ARCHITECTURE.md      Step-to-code architecture map
examples/distributed_coordination.py                       End-to-end demonstration
```

## Installing into ORE-Neurocomputational

1. Copy `ore_distributed/` into the repository root (next to `ore_usg/`,
   `ore_workspace/`, `ore_contracts/`).
2. Copy the seven `tests/test_distributed_*_point17.py` files into `tests/`.
3. Copy `docs/DISTRIBUTED_COORDINATION_POINT17_ARCHITECTURE.md` into `docs/`.
4. Copy `examples/distributed_coordination.py` into `examples/`.
5. In `pyproject.toml`, append `"ore_distributed*"` to the
   `[tool.setuptools.packages.find] include` list (already done in the
   repository this archive was built against).

The package is dependency-light: it uses only the Python standard library
plus the repository's own `ore_usg`, `ore_workspace`, and `ore_contracts`
modules (the integration adapters import them; the core coordination layer
does not require them).

## Running

```bash
python -m pytest tests/test_distributed_*_point17.py -q   # 107 acceptance tests
python -m ore_distributed                                # deployment + registry summary
python examples/distributed_coordination.py              # end-to-end demonstration
```

## Scope notes

- Raft is the implemented crash-fault-tolerant consensus baseline; BFT stays
  behind the Step 4 threat-model gate, exactly as the source specifies.
- The transport is a deterministic in-memory authenticated fabric (partitions,
  delays, duplication, bounded payloads) so every acceptance scenario in
  source section 6 is testable; real transports plug in behind the same
  protocol-gate interfaces.
- Consensus ordering never establishes epistemic truth: USG validation gates
  every semantic promotion, and CRDT convergence never marks an assertion true.
