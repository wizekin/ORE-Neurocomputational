# ORE Runtime — Point 22: Advanced Kernel, Isolation and Operating-System Architectures

Implementation of "ORE Integrated Algorithm — Point 22: Advanced Kernel,
Isolation and Operating-System Architectures" for the
ORE-Neurocomputational repository: a CROS-inspired, microkernel-influenced
user-space execution and resource-governance layer (`ore_runtime`).

## What is in this package

- `ore_runtime/` — the algorithm implementation (17 modules):
  contracts (manifests, typed envelopes, outcome taxonomy), registry
  (identity, health leases, dependency graph, repository inventory),
  capabilities (sealed grants, attenuation, generations, revocation),
  lifecycle (state machine, supervised restarts, quarantine), ipc (bounded
  queues, back-pressure, deadlines, cancellation, dedupe), resources
  (reservations, admission control, hierarchical budgets, leak
  reconciliation), scheduling (EDF/RMS with assumption guard, budget
  scheduler with reserved classes), isolation (in-process, subprocess with
  scrubbed env + POSIX rlimits, container/microVM/remote records, isolation
  score with hard floors), worker (supervised subprocess bootstrap with
  trust-boundary validation), artifacts (content-addressed immutable
  store), audit (hash-chained append-only SQLite sink, redaction,
  annotations, replay recovery), policy (versioned policies, staged
  rollout with automatic rollback, operator overrides), workspace_guards
  (capability-scoped workspace/USG access), runtime (ComponentRuntime
  facade), testing (fault injection, model-based checks, security probes,
  benchmarks), plus `__main__` demo CLI.
- `tests/test_runtime_point22.py` — 60 acceptance tests mapping the
  spec's section-8 list.
- `docs/ADVANCED_KERNEL_ISOLATION_OS_ARCHITECTURES_POINT22.md` —
  architecture and integration documentation.

## Usage

```bash
# inside the ORE-Neurocomputational repository root
pip install -e .[dev]
python -m pytest tests/test_runtime_point22.py
python -m ore_runtime --demo        # scripted governance demo
python -m ore_runtime --self-test   # model checks + security probes
python -m ore_runtime --benchmark   # in-process vs subprocess overhead
python -m ore_runtime --inventory   # repository component inventory
```

Standalone (without the repository): the package runs with in-memory audit
and a minimal blackboard; the subprocess tier needs the package importable
on `PYTHONPATH`. Repository integration activates `ore_workspace` guards,
`ore_usg` validation, and the shared `ore_logger` SQLite catalogue
automatically when those packages are importable.

The package requires Python 3.10+ and no third-party dependencies.
