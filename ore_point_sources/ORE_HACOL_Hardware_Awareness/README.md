# ORE HACOL — Point 21: Hardware-Aware Numerical Computing and Quantum Algorithms

Full implementation of the integrated algorithm specified in
"ORE Integrated Algorithm — Point 21: Hardware-Aware Numerical Computing and
Quantum Algorithms" (Most Intelligent Systems 01-24, Point 21, sections
21.1–21.8), integrated with the
[ORE-Neurocomputational](https://github.com/wizekin/ORE-Neurocomputational)
repository conventions.

This archive contains **only the algorithm's files** (not the whole repository).

## Contents

```
ore_hacol/                    The HACOL package (20 modules)
  __init__.py                 Public API
  __main__.py                 Demo CLI (python -m ore_hacol)
  contracts.py                Typed request/plan/result contracts ("ore_compute")
  registry.py                 The 48 integrated steps with phases and controls
  capabilities.py             Hardware discovery + measured bandwidth/GEMM calibration
  profiling.py                Seeded fixtures, latency stats, bottleneck classification
  planning.py                 Algorithm-selection policy table + candidate costing
  attention.py                Phase C: online softmax + FlashAttention-style tiling
  randomized_linalg.py        Phase D: randomized range/SVD/sketching/least squares
  structured.py               Phase E: butterfly + Toeplitz structured transforms
  precision.py                Phase E: mixed precision, BF16 emulation, refinement
  optimization.py             Phase F: classical baselines + QUBO/Ising + solvers
  quantum_research.py         Phase F: Grover/Shor/QEC research records + gate
  backends.py                 Backend registry (reference/optimized/optional)
  validation.py               Comparison, adversarial fixtures, regression records
  cros.py                     CROS admission gate (never bypassed)
  workspace.py                Blackboard publication of versioned envelopes
  logging_adapter.py          ore_logger SQLite integration
  biology.py                 Biological-reference network with honest evidence levels
  pipeline.py                 HardwareAwareOptimizer: full 48-step orchestration
tests/
  test_hacol.py               62 acceptance tests (spec section 10)
docs/
  HARDWARE_AWARE_COMPUTING_QUANTUM_POINT21.md   Full algorithm documentation
README.md                     This file
```

## Installation into the ORE-Neurocomputational repository

1. Copy this archive's `ore_hacol/`, `tests/test_hacol.py`, and `docs/` into
   the repository root (merging `tests/` and `docs/`).
2. Add `"ore_hacol*"` to the package include list in `pyproject.toml`
   (`[tool.setuptools.packages.find] include = [...]`). This archive's
   in-repo copy already did this; a fresh repository needs the one-line edit.
3. Verify:

```
python -m pip install -e ".[dev]"
python -m pytest tests/test_hacol.py
python -m ore_hacol --demo all
```

## Integration points

- **ore_logger (SQLite)** — `register_hacol_annotations()` indexes the package;
  every pipeline result records per-step events (module/class/function, task,
  dimensions, algorithm/backend, precision/device, duration, memory,
  transfers, error, status, provenance).
- **ore_workspace (Blackboard)** — `publish_result()` submits metadata-only
  versioned envelopes through `FindingPublisher` when a workspace is provided.
- **CROS** — every optimization passes through `LocalCROSGate` or an
  `ExternalCROSAdapter` wrapping the repository's coordinator; HACOL never
  bypasses it and always releases granted admission.

## Guarantees encoded in the implementation

- Every optimized path preserves a trusted reference implementation and
  compares outputs within the declared correctness contract.
- Approximation is disabled when exactness is required; approximate results
  carry reproducible metadata (rank, oversampling, sketch family, seed, power
  iterations, error estimate, validation status).
- Quantum algorithms (Shor, Grover, QEC) are research records only and are
  never executed or routed for ordinary ORE tasks.
- Feature flags gate promotion; rollback restores the baseline; cancellation
  is safe between steps; validation failure falls back to the reference.

Scientific status: a biologically motivated computational synthesis, not a
validated neural simulator; quantum content is not biologically grounded.
See `docs/HARDWARE_AWARE_COMPUTING_QUANTUM_POINT21.md` for the full
step-by-step traceability and the source spec crosswalk.
