ORE Point 14 - Specialized Learning Architectures and Foundation Models
Replaceable Learned-Model Subsystem (RLMS), "ore_learning"

Contents (repository-relative paths; drop into the ORE-Neurocomputational
repository root to install):
  ore_learning/            the algorithm package (14 subpackages + root modules)
    contracts/             manifests, task/capability/evidence schemas, lifecycle events
    registry/              model registry, lifecycle state machine, lineage, 56-step specs
    routing/               eligibility (S/C/P/R/L), utility ranking, MoE, ensembles, CROS
    adapters/              typed adapter boundary, modality gateway, fusion
    inference/             bounded executor (timeouts/cancellation), least-privilege sandbox
    adaptation/            transfer ladder, reproducible datasets, TTT, distillation
    generation/            GAN objective gate, synthetic validation, diffusion, gating
    explainability/        SHAP, LIME, counterfactuals, explanation governance
    rewards/               PRM screening, deterministic validators, RLVR, hacking checks
    federated/             FedAvg coordinator, privacy controls, poisoning screens
    optimization/          QUBO formulation, conventional baselines, verification
    monitoring/            calibration, drift, behavior, quarantine/rollback, canary
    audit/                 sanitized append-only SQLite audit events
    edge/                  edge decisions, quantization/pruning with rollback
    (root)                 config, functional, nn, pipeline (56-step engine),
                           registration (ore_contracts), annotations (ore_logger),
                           integration (ore_workspace), CLI demo
  tests/                   acceptance tests (94 tests) + shared helper
  docs/                    architecture, step mapping, acceptance mapping
  examples/                runnable end-to-end demonstration

Requirements: Python >= 3.10, numpy (the repository's existing dependencies).
Run the demo:      python -m ore_learning
Run the example:   python -m examples.point14_rlms
Run the tests:     python -m pytest tests/test_learning_*_point14.py -q
