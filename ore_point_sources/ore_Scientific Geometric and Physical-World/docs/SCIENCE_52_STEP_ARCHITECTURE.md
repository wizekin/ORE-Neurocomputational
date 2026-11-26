# ORE Scientific Reasoning Layer — 52-Step Architecture

**Source specification:** `Algorithms_Chats/ORE Integrated Algorithm — Point 12: Scientific, Geometric and Physical-World AI.txt` (source Point 12 of `Most Intelligent Systems 01-24.txt`, ORE integration priority 22 of 24)

**Package:** `ore_science` (module id `ore_science`, version 1.0.0, schema `ore.science.52_step.v1`)

**Status:** Implemented reference architecture with local reference models. This document records the mapping from the specification to the implementation, the package structure, and the validation evidence.

## Purpose

The scientific layer gives ORE the ability to reason about scientific and physical-world problems by selecting representations, equations, models, and solvers that respect the structure of each domain, validating their outputs independently, quantifying uncertainty, recording provenance, and updating knowledge only when evidence warrants it. Prediction is not proof; generation is not discovery. The central artifact is the closed scientific reasoning loop executed as a 52-step pipeline with a typed outcome, an explicit evidence-status ladder, and complete provenance.

## Architectural principles → implementation

| Principle | Implementation |
|---|---|
| Representation before computation | `RepresentationBuilder` selects among molecular graphs, crystal graphs, fields, geometric point sets, factor graphs, trajectory ensembles, and state spaces, with per-candidate decision records (Step 8) |
| Laws and constraints are explicit | `Constraint`, `EquationModel`, and `Assumption` records carry units, sources (USER/USG/INFERRED/MODEL_IMPOSED), and tolerances as data |
| Model selection is conditional | `ModelRegistry.find_compatible` runs hard domain, problem-class, representation, interaction-class, and availability gates before the Section 5.8 score ranks candidates |
| Prediction is not proof | Every prediction passes through `ValidationEngine` checks before any status promotion; a low PINN loss never certifies correctness |
| Generation is not discovery | Molecular and materials generators emit `CANDIDATE` status; plausibility filters gate downstream work; promotion is review-only |
| Uncertainty is first-class | `UncertaintyReport` carries intervals, aleatoric/epistemic separation where supported, alternative futures, and limitations |
| Independent validation | Numerical, analytic, ground-truth, and cross-method baselines are compared with recorded discrepancies and tolerances |
| Provenance is mandatory | `RunManifest` records datasets, models, solver settings, seeds, runtime, code revision, input/output hashes, validation outcomes, with secret redaction |
| Specialists remain modular | External services (AlphaFold-family, RFdiffusion, GNoME/MatterGen, GraphCast/GenCast, GTSAM bindings) are declared in the registry as `not_configured` and rejected explicitly until licensing, access, and budgets exist |
| Fail safely | Typed outcomes (`INVALID_INPUT`, `UNSUPPORTED_DOMAIN`, `RESOURCE_EXHAUSTED`, `VALIDATION_FAILURE`, `LIMITED_RESULT`) replace exceptions and silent guesses |

## Package structure (spec Section 6 mapping)

The spec's proposed `ore_science.*` subpackages are reconciled with the repository's flat-module convention:

| Spec boundary | Implementation |
|---|---|
| `ore_science.contracts` | `contracts.py` (task, variable, constraint, equation, status, result schemas) + `units.py` (SI dimensions, conversions) |
| `ore_science.interpretation` | `interpretation.py` (`DomainRouter`, `ScientificTaskParser`, Steps 1–7) |
| `ore_science.knowledge` | `knowledge.py` (read-only `ScienceKnowledgeBase`, `USGScienceBridge`) |
| `ore_science.representations` | `representations.py` (all seven representation families, Steps 8–15) |
| `ore_science.registry` | `model_registry.py` (capability specs, gates, 5.8 score) + `registry.py` (52-step operation/station registry, repo convention) |
| `ore_science.adapters.*` | `adapters/` subpackage: `numerical.py`, `pinn.py`, `operators.py`, `geometric.py`, `forecasting.py`, `factor_graphs.py`, `molecular.py`, `materials.py` |
| `ore_science.validation` | `validation.py` (Steps 35–38, 44) |
| `ore_science.provenance` | `provenance.py` (Step 50, redaction, reproducibility) |
| `ore_science.orchestration` | `orchestration.py` (decomposition, admission, status ladder, workspace publication, candidate knowledge) + `pipeline.py` (the 52-step `ScientificReasoner`) |

ORE integration (Section 6 table): `registration.py` registers the module with `ore_contracts` (contract `ORE.SCIENCE.RESULT`, manifest, backend with `describe()`/`execute()`); `orchestration.create_science_workspace` grants the layer its workspace fields; `annotations.py` glues the 52-step trace into the `ore_logger` SQLite schema.

## The 52 steps

Five phases execute in order; every step is recorded in `operation_trace` with an explicit status (`executed`, `not_applicable` with reason, `blocked`, `failed`, or `not_configured`):

- **Phase I (Steps 1–7, station A/B):** domain routing, objective extraction, variable registry with units, constraints and assumption tagging, dimensional homogeneity (fail-loud), problem class, evidence requirements.
- **Phase II (Steps 8–15, station C):** representation decision record, family-specific construction, construction provenance with payload hashes.
- **Phase III (Steps 16–34, station D):** specialized routing per domain — molecular structure/interaction/design, materials prediction/generation, governing-equation assembly, PINN objective with explicit weights, Fourier operator with retained modes and periodic convention, numerical reference solver, equivariance tests, ensemble forecasting, factor-graph construction and optimization through the GTSAM-compatible adapter. Step 49 resource admission gates dispatch.
- **Phase IV (Steps 35–44, station F):** dimensional and algebraic validation, Richardson convergence (explicit `converged`/`not_converged`), conservation invariants, independent baselines with discrepancy and tolerance, uncertainty (resolution ensemble, cross-method, ensemble members, sensitivity/identifiability), calibration (Brier, reliability), out-of-domain warnings, status ladder, and the domain review gate.
- **Phase V (Steps 45–52, stations G/H):** structured result, workspace publication (artifacts + hypotheses + questions, permission-gated), candidate knowledge update (USG never rewritten), multi-domain coordination preserving per-stage assumptions, provenance manifest, versioned prediction-error learning with rollback, calibrated explanation separating derivation, prediction, and observation.

## Mathematics (spec Section 5 → code)

- **5.1/5.2 PINN:** `L(φ) = λ_d L_data + λ_p L_physics + λ_b L_boundary + λ_i L_initial` with independently reported components (`pinn.py`); residuals by finite differences; training via scipy BFGS with finite-difference gradients (explicitly permitted by 5.7).
- **5.3 Neural operators:** diagonal spectral operator with responses interpolated in physical frequency, enabling genuine multi-resolution evaluation (`operators.py`).
- **5.4 Equivariance:** `ε_eq = ||f(g·x) − g·f(x)|| / (||g·f(x)|| + ε)` measured over representative rotations; invariance as the special case (`geometric.py`).
- **5.5 Factor graphs:** `x* = argmin Σ ρ_k(r_k^T Σ_k⁻¹ r_k)` solved by Gauss-Newton with Levenberg–Marquardt damping and Huber robust loss; marginal variances from the information matrix (`factor_graphs.py`).
- **5.6 Ensembles:** mean `(1/M)Σx^(m)`, variance with `(M−1)` denominator, event probabilities as member fractions (`forecasting.py`).
- **5.7 Sensitivity:** normalized `J_ij = ∂y_i/∂θ_j` by central differences, identifiability flags for insensitive, correlated, and ill-conditioned parameters (`uncertainty.py`).
- **5.8 Model selection:** `Score = w_q Q + w_c C + w_d D − w_r R − w_k K` applied only to gate-passing candidates (`model_registry.py`).

## Validation evidence (spec Section 8)

86 acceptance tests in `tests/test_science.py`, `tests/test_science_adapters.py`, and `tests/test_science_integration.py` cover the Section 8 criteria, including: dimensional failure on inconsistent equations and units; unit conversion exactness; initial/boundary separation; representation metadata preservation; factor-graph rejection of undefined variables and inconsistent frames; registry rejection of incompatible schemas, unsupported interaction classes, and unconfigured services; PINN residual against an analytic test case, independent loss reporting, wrong-PDE detection, and numerical comparison; multi-resolution operator evaluation; equivariance measurement and invariance verification; ensemble member identity and calibration with overconfidence detection; factor-graph ground-truth recovery and outlier robustness; conservation detection of perturbed solutions; explicit convergence reporting; uncertainty types with limitations; OOD warnings; sensitivity and identifiability; the non-conflatable status ladder; workspace publication with immutable artifacts; provenance-required candidate knowledge updates; per-stage assumption preservation; resource admission and cancellation; reproducibility metadata; backend failure isolation; secret redaction; calibrated explanations; and versioned, rollbackable policy updates.

## Scope and limitations

- External scientific services are declared but not configured: the registry rejects them with explicit reasons until access exists. Local reference models exercise the same interfaces and make no claims of domain-leading skill.
- The natural-language parser is rule-based; natural-language understanding is not claimed.
- The materials predictor is a transparent heuristic behind a GNoME-style interface; predicted stability is always reported as a model prediction, never a laboratory result.
- The Fourier operator is a linear diagonal spectral operator, disclosed as such; resolution transfer is evaluated, never assumed.
- GTSAM bindings are not required: the local Gauss-Newton backend implements the same adapter interface and the swap is transparent.
