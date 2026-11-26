# Specialized Learning Architectures and Foundation Models — Point 14 RLMS

**Replaceable Learned-Model Subsystem (RLMS) for the ORE architecture.**

This document describes the `ore_learning` package, a complete implementation of
"ORE Integrated Algorithm — Point 14: Specialized Learning Architectures and
Foundation Models". The subsystem lets ORE *discover, evaluate, select, isolate,
monitor, and replace* learned models without making any model the authority for
semantic truth, reasoning policy, or durable knowledge.

Status: reference implementation. All models in the package are deterministic,
seeded numpy reference implementations behind typed adapters; the lifecycle,
governance, and validation machinery is production-shaped.

---

## 1. Executive design

The authority chain is implemented end to end:

```
model proposal -> typed evidence record -> independent validation
              -> coordinator policy -> approved state transition
```

A learned model may propose interpretations, features, hypotheses, candidate
actions, generated data, rankings, or explanations. It must not silently rewrite
authoritative USG knowledge, turn confidence into proof, change coordinator
permissions, or install itself into production without validation and approval.
`ore_learning.contracts.AuthorityBoundary` enforces this one-way boundary:
raw model output is refused, unvalidated proposals are refused, and validated
proposals without coordinator approval are refused — every refusal is recorded.

Governing principle: **ORE owns task semantics, policy, evidence, verification, and
state transitions. Learned models supply replaceable capabilities under those
controls.**

## 2. Package boundaries (specification Section 5)

| Subpackage | Responsibility | Source steps |
| --- | --- | --- |
| `ore_learning/contracts/` | manifests, task/capability/evidence schemas, lifecycle events | Sections 2, 5.1 |
| `ore_learning/registry/` | immutable model identity, digests, lifecycle state, lineage, 56-step operation specs | 6, 51–56 |
| `ore_learning/routing/` | eligibility conjunction, utility ranking, MoE reference, bounded ensembles, CROS reservations | 6–11 |
| `ore_learning/adapters/` | typed adapter boundary, deterministic reference specialists, multimodal gateway, fusion | 18–22 |
| `ore_learning/inference/` | resource-aware execution, timeouts, cancellation, least-privilege sandbox | 11, 26 |
| `ore_learning/adaptation/` | transfer learning ladder, reproducible datasets, isolated runs, TTT, distillation | 12–17, 39 |
| `ore_learning/generation/` | GAN objective gate, synthetic-data validation, diffusion benchmark, output gating | 23–26 |
| `ore_learning/explainability/` | exact/sampled SHAP, LIME surrogates, counterfactuals, explanation governance | 27–31 |
| `ore_learning/rewards/` | typed reasoning steps, PRM screening, deterministic validators, RLVR, hacking detection | 32–37 |
| `ore_learning/federated/` | participant authentication, FedAvg, clipping/DP, secure aggregation, poisoning screens | 41–43 |
| `ore_learning/optimization/` | QUBO formulation with penalties, conventional baselines first, independent verification | 44–47 |
| `ore_learning/monitoring/` | calibration, PSI/KL + canary drift, behavioral regression, quarantine/rollback, canary | 48–52 |
| `ore_learning/audit/` | sanitized append-only audit events in SQLite | 54 |
| `ore_learning/edge/` | edge deployment decisions, quantization/pruning with rollback, split execution | 38–40 |
| root modules | `config` (governance parameters), `functional` (pure math), `nn` (reference models), `pipeline` (56-step engine), `registration`, `annotations`, `integration` | all |

## 3. The 56-step pipeline

`ore_learning.pipeline.LearnedModelEngine` executes the phases in order and records
exactly one operation-trace entry per source step (1–56):

* **Phase A (1–5)** — task normalization, learned-inference necessity check,
  capability request, verifiable-boundary decomposition
  (`build_default_stage_contracts`), evidence and authority policy.
* **Phase B (6–11)** — registry query, the eligibility conjunction
  `E = S ∧ C ∧ P ∧ R ∧ L`, measured-utility ranking
  `U = w_q Q + w_c C + w_r R − w_l L̂ − w_k K̂ − w_u Û`, MoE-as-routing-pattern
  note, single-specialist-or-bounded-ensemble selection, CROS reservation.
* **Phase C (12–17)** — frozen features first, smallest adaptation method,
  reproducible datasets with leakage audits, isolated training runs,
  base-vs-adapted comparison with promotion gates, disposable opt-in TTT.
* **Phase D (18–22)** — gateway validation per modality, perception specialists,
  USG-aligned semantic mapping with explicit unknowns, reliability-weighted fusion
  `s(c) = Σ α_j r_j s_j(c) / Σ α_j r_j` with retained conflicts.
* **Phase E (23–26)** — defined-objective gate for GANs, synthetic-data validation
  (PSI, diversity, collapse, memorization, downstream utility on real held-out
  data), diffusion-vs-autoregressive benchmark, untrusted-candidate gating with
  sandboxed code execution.
* **Phase F (27–31)** — two separate explanation channels; SHAP with recorded
  background and configuration; LIME with perturbation method and local fidelity;
  counterfactuals under domain constraints; limitations recorded on every report;
  diagnostics can never be filed as formal proof.
* **Phase G (32–37)** — typed reasoning transitions, PRM screening kept separate
  from deterministic validators, `V_i(s_i, a_i, s_{i+1}) = PASS` required for
  verified status, RLVR reward `R(τ) = Σ λ_i r_i + λ_f r_final − λ_v p_violation`
  computed only from validator outputs (replayable), hidden-test hacking
  detection.
* **Phase H (38–43)** — measured edge decisions, compression with retention
  thresholds and rollback reference, split execution with central authorization,
  federated rounds under authentication, clipping, DP noise, secure aggregation,
  cohort minimums, poisoning screens, independent validation before activation.
* **Phase I (44–47)** — combinatorial admissibility, penalized QUBO
  `E(x) = f(x) + ρ Σ g_j(x)²` verified exhaustively on small instances,
  conventional baselines recorded *before* Ising evaluation, independent
  feasibility checks, no backend reaches CROS or coordinator policy.
* **Phase J (48–56)** — calibration (Brier, ECE, reliability), PSI/KL + canary
  drift with false-alarm controls, behavioral regression attribution,
  deterministic quarantine and rollback, canary exposure with exact fraction
  control and rollback gates, lineage, SQLite audit events, approval gates,
  retirement preserving lineage.

Deferred and failed runs still produce a complete 56-entry trace with explicit
skip reasons — there are no silent gaps.

## 4. Core interfaces (Section 5.1)

`ModelManifest`, `CapabilityRequest`, `ModelCandidate`, `ModelSelection`,
`InferenceRequest`, `InferenceResult`, `ValidationReport`, and
`ModelLifecycleEvent` are all implemented as frozen, validated dataclasses in
`ore_learning/contracts/`, with `to_dict()` views for audit and publication.

Lifecycle states: `proposed → validated → canary → active → deprecated → retired`,
with `quarantined` and `revoked` reachable from most states and re-entry from
quarantine requiring re-evaluation **and** approval. Illegal transitions raise
`LifecycleTransitionError`; identity is immutable (a different artifact digest for a
registered identity is refused).

## 5. Mathematical reference (Section 4)

All formulas live as pure functions in `ore_learning/functional.py` and are covered
by tests: the eligibility conjunction and utility score, MoE gating
`y = Σ g_i(x) E_i(x)`, fusion weighting, GAN min-max objectives, exact Shapley
weights `|S|!(|F|−|S|−1)!/|F|!`, LIME weighted ridge surrogates, RLVR reward
composition, FedAvg weights `n_k/Σ n_j`, QUBO energy `x^⊤Qx` with exhaustive and
annealed solvers, PSI/KL drift statistics, Brier/ECE/reliability, and symmetric
quantization.

## 6. Integration with ORE

* **`ore_contracts`** — `register_ore_learning_module()` registers contracts
  `ORE.LEARNING.TASK / CAPABILITY_REQUEST / RESULT @1.0.0`, a validated
  `ModuleManifest`, and a `LearningModuleBackend` satisfying the module protocol
  (`describe` / `execute`).
* **CROS** — `routing.resources` defines the `CROSPort` protocol with a
  deterministic local coordinator (capacity tracking, safety margin, release
  semantics); deployments wire a real AdaptiveResourceCoordinator behind the same
  port.
* **`ore_workspace`** — `integration.WorkspaceBridge` publishes proposals as
  append-only findings behind the documented permission-grant handshake;
  `build_coordinator_notification` produces bounded coordinator messages.
* **`ore_logger`/SQLite** — `annotations.register_annotations` indexes the package
  (`symbols_written`), `record_result_to_sqlite` persists one operation event per
  pipeline step, and `audit.AuditStore` keeps the append-only lifecycle event table
  with a structural no-raw-payload guarantee.
* **`ore_usg`** — reachable only through `AuthorityBoundary` update requests;
  there is no write path from a model or a proposal.

## 7. Acceptance criteria → tests (Section 7)

| Acceptance criterion | Test |
| --- | --- |
| Registry rejects missing artifact digest | `test_learning_registry_point14.py::test_registry_rejects_missing_artifact_digest` |
| Registry rejects incompatible schemas | `test_learning_registry_point14.py::test_registry_rejects_incompatible_schemas` |
| Registry rejects revoked/quarantined models | `test_learning_registry_point14.py::test_registry_rejects_revoked_or_quarantined_models` |
| Registry records provenance and license | `test_learning_registry_point14.py::test_registry_records_provenance_and_license_metadata` |
| Capability resolver returns only eligible models | `test_learning_routing_point14.py::test_capability_resolver_returns_only_eligible_models` |
| Router ranking deterministic | `test_learning_routing_point14.py::test_router_ranking_is_deterministic_for_identical_state_and_inputs` |
| Hard policy overrides utility | `test_learning_routing_point14.py::test_hard_policy_constraints_override_a_higher_utility_score` |
| Resource failure → fallback/deferral | `test_learning_routing_point14.py::test_resource_reservation_failure_triggers_safe_fallback_or_deferral` |
| Inference timeout cancels and releases | `test_learning_inference_point14.py::test_inference_timeout_cancels_execution_and_releases_resources` |
| Adapter output validated against schema | `test_learning_inference_point14.py::test_adapter_output_is_validated_against_its_schema` |
| Raw output cannot write to USG | `test_learning_inference_point14.py::test_raw_model_output_cannot_write_directly_to_usg` |
| Results retain identity + provenance | `test_learning_inference_point14.py::test_every_result_retains_identity_and_input_provenance` |
| Held-out evaluation without leakage | `test_learning_adaptation_point14.py::test_transfer_evaluation_uses_held_out_data_without_leakage` |
| Promotion needs predefined improvement | `test_learning_adaptation_point14.py::test_adapted_model_promotion_requires_predefined_improvement` |
| TTT cannot persist into production | `test_learning_adaptation_point14.py::test_ttt_cannot_persist_into_the_production_artifact` |
| Gateway preserves references/timestamps | `test_learning_adaptation_point14.py::test_multimodal_gateway_preserves_references_and_timestamps` |
| Invalid modality data rejected | `test_learning_adaptation_point14.py::test_invalid_modality_dimensions_or_encodings_are_rejected` |
| Conflicting observations remain visible | `test_learning_adaptation_point14.py::test_conflicting_modality_observations_remain_visible` |
| GAN data evaluated for diversity/memorization | `test_learning_generation_point14.py::test_synthetic_data_is_evaluated_for_diversity_and_memorization` |
| Synthetic augmentation tested on real held-out | `test_learning_generation_point14.py::test_synthetic_augmentation_is_tested_against_real_held_out_data` |
| Diffusion benchmarked vs baseline | `test_learning_generation_point14.py::test_diffusion_generation_is_benchmarked_against_a_baseline` |
| Generated code only in sandbox | `test_learning_generation_point14.py::test_generated_code_executes_only_inside_the_sandbox` |
| SHAP report has configuration + background | `test_learning_explain_rewards_point14.py::test_shap_report_includes_explainer_configuration_and_background` |
| LIME report has method + fidelity | `test_learning_explain_rewards_point14.py::test_lime_report_includes_perturbation_method_and_local_fidelity` |
| Diagnostics never formal proof | `test_learning_explain_rewards_point14.py::test_explanation_diagnostics_are_never_recorded_as_formal_proof` |
| PRM separate from validator results | `test_learning_explain_rewards_point14.py::test_prm_scores_stay_separate_from_validator_results` |
| Failed validation blocks verified | `test_learning_explain_rewards_point14.py::test_failed_intermediate_validation_blocks_verified_status` |
| RLVR reproducible from validators | `test_learning_explain_rewards_point14.py::test_rlvr_reward_is_reproducible_from_validator_outputs` |
| Hacking detection | `test_learning_explain_rewards_point14.py::test_independent_checks_detect_simple_reward_hacking` |
| Quantized model meets thresholds | `test_learning_edge_federated_optimization_point14.py::test_quantized_edge_model_meets_memory_and_quality_thresholds` |
| FedAvg uses declared sample counts | `test_learning_edge_federated_optimization_point14.py::test_fedavg_weights_use_declared_participant_sample_counts` |
| Unauthenticated participants rejected | `test_learning_edge_federated_optimization_point14.py::test_federated_pipeline_rejects_unauthenticated_participants` |
| Secure aggregation/privacy enforced | `test_learning_edge_federated_optimization_point14.py::test_secure_aggregation_and_privacy_controls_are_enforced` |
| Aggregates need independent validation | `test_learning_edge_federated_optimization_point14.py::test_aggregated_model_cannot_activate_without_independent_validation` |
| QUBO reproduces exhaustive optima | `test_learning_edge_federated_optimization_point14.py::test_qubo_mapping_reproduces_known_optima_on_small_exhaustive_cases` |
| QUBO outputs independently checked | `test_learning_edge_federated_optimization_point14.py::test_qubo_outputs_are_independently_checked_for_feasibility` |
| Baseline before Ising | `test_learning_edge_federated_optimization_point14.py::test_conventional_solver_baseline_is_recorded_before_ising_evaluation` |
| Drift monitor detects degradation | `test_learning_lifecycle_point14.py::test_drift_monitor_detects_a_controlled_synthetic_degradation` |
| Drift alert → quarantine + fallback | `test_learning_lifecycle_point14.py::test_drift_alert_triggers_quarantine_and_known_good_fallback` |
| Canary limits exposure fraction | `test_learning_lifecycle_point14.py::test_canary_deployment_limits_candidate_exposure_to_its_fraction` |
| Candidate rolls back on regression | `test_learning_lifecycle_point14.py::test_candidate_rolls_back_when_regression_threshold_is_crossed` |
| Lifecycle events persist and link | `test_learning_registry_point14.py::test_lifecycle_events_persist_and_link_to_task_and_model` |
| Audit leaks no raw payloads | `test_learning_registry_point14.py::test_audit_records_do_not_leak_raw_sensitive_payloads` |
| Human approval for high-risk promotion | `test_learning_registry_point14.py::test_high_risk_promotion_requires_recorded_approval` |
| Retirement preserves lineage | `test_learning_registry_point14.py::test_retirement_preserves_lineage_and_evaluation_evidence` |

Additional integration tests cover the full 56-step engine, module registration,
ore_logger annotation persistence, and workspace publication
(`test_learning_lifecycle_point14.py`).

## 8. Failure modes and safeguards (Section 9)

| Failure mode | Safeguard implemented |
| --- | --- |
| Model authority leakage | `AuthorityBoundary` one-way chain; refusals recorded |
| Negative transfer | base-vs-adapted comparison; base artifact preserved |
| Data leakage | split lineage; near-duplicate audit; overlap refused |
| Synthetic memorization | nearest-neighbour privacy checks; flagged samples rejected |
| Misleading explanations | method/limitation records; proof filing refused |
| Reward hacking | hidden tests; independent validators; refusal to certify |
| MoE/router collapse | concentration monitor with fallback retained |
| Multimodal misalignment | provenance, timestamps, coordinate frames; retained conflicts |
| Federated poisoning | authentication, clipping, DP, cohort minimums, outlier screens |
| Quantization regressions | retention thresholds; uncompressed rollback reference |
| QUBO penalty errors | exhaustive small-instance verification; independent checks |
| Resource exhaustion | CROS reservations, deadlines, budgets, cancellation, release |
| Supply-chain compromise | artifact digests verified before execution |
| Silent drift | canaries, calibration, outcome monitoring; critical quarantine |
| Unreviewed deployment | approval gates; append-only immutable audit events |

## 9. Limitations

* All specialists are deterministic numpy reference models; the adapter boundary
  exists precisely so real frameworks can replace them without touching governance.
* The least-privilege sandbox uses AST screening plus restricted builtins; real
  deployments add OS-level isolation.
* Secure aggregation is simulated (the coordinator only observes protected
  updates); production deployments add cryptographic aggregation.
* Federated learning, TinyML, GANs, diffusion, and QUBO remain optional research
  adapters until an actual ORE workload demonstrates value (Section 10).
