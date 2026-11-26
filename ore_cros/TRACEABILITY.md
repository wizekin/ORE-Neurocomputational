# Traceability — Point 23 LDOS/CROS Implementation

Mapping from the Point 23 specification ("ORE Integrated Algorithm — Point 23:
Learning-Directed Operating Systems and Adaptive Resource Management") to code
modules and acceptance tests. Every one of the 52 steps and every acceptance
item in section 8 maps to at least one module and one test.

## Part 1 — The 52 steps

| Step | Spec requirement | Module(s) | Test(s) |
| --- | --- | --- | --- |
| 1 | Discover execution environment; unknown capacity ≠ zero | `telemetry/collectors.py` (`HostProfiler`), `resource_model.py` (`EnvironmentProfile`) | `test_cros_telemetry.py::test_discovery_reports_unknown_capacity_as_none_not_zero` |
| 2 | Resource identities and scopes (machine/process/worker/module/task/device) | `resource_model.py` (`ResourceScope`, `ScopeLevel`) | `test_cros_resource_model.py::test_scope_topology_paths`, `test_cros_telemetry.py::test_scopes_cannot_be_silently_mixed` |
| 3 | Typed telemetry schemas, unit-safe, canonical units | `telemetry/schemas.py` | `test_cros_telemetry.py::test_measurement_rejects_invalid_units`, `..._impossible_values`, `..._malformed_timestamps`, `..._restricted_to_queue_resource` |
| 4 | Baseline measurements with overhead and missing-data rates | `telemetry/collectors.py` (`record_baseline`, `BaselineProfile`) | `test_cros_telemetry.py::test_baseline_profile_records_overhead_and_missing_rate` |
| 5 | Attribute usage to tasks/modules; bounded estimates labelled | `accounting/task_costs.py` | `test_cros_accounting.py::test_directly_attributed_usage_is_exact`, `..._labelled_estimated`, `..._unattributed_kinds_are_explicit`, `..._module_totals` |
| 6 | Pressure and queueing; sustained vs spike; Little's Law | `mape/analyze.py` (`PressureDetector`, `QueueReading`) | `test_cros_mape.py::test_pressure_detector_separates_sustained_saturation_from_spikes`, `..._littles_law_check_in_queue_reading` |
| 7 | Time-windowed history with bounded retention and aggregates | `telemetry/history.py` | `test_cros_telemetry.py::test_history_windows_and_bounded_retention` |
| 8 | Measurement quality; confidence gates for high-impact actions | `telemetry/quality.py` | `test_cros_telemetry.py::test_stale_telemetry_blocks_high_impact_actions`, `..._tracks_missing_sources_and_series`, `..._counter_resets_are_detected` |
| 9 | Monitor: coherent, versioned observation snapshot | `mape/monitor.py` | `test_cros_mape.py::test_monitor_stage_emits_versioned_coherent_snapshots` |
| 10 | Analyze: bottleneck diagnosis ranked by contribution/persistence/impact/confidence | `mape/analyze.py` (`Analyzer`) | `test_cros_mape.py::test_analyze_stage_diagnoses_cpu_bound_with_evidence` |
| 11 | Plan: candidates with predicted effects, costs, rollback conditions | `mape/plan.py` (`Planner`) | `test_cros_mape.py::test_planner_generates_candidates_with_costs_and_rollback` |
| 12 | Knowledge: versioned, provenance-separated (measured vs learned) | `mape/plan.py` (`KnowledgeBase`) | `test_cros_mape.py::test_knowledge_base_distinguishes_measured_and_learned` |
| 13 | Validate hard constraints; reject violators regardless of reward | `safety/constraints.py` (`HardConstraintSet`) | `test_cros_mape.py::test_constraint_filter_rejects_violating_candidates` |
| 14 | Execute through controlled interface: typed params, bounds, audit, rollback | `mape/execute.py` (`ControlledExecutionManager`), `actions.py` | `test_cros_mape.py::test_policy_engine_cannot_bypass_execution_manager`, `test_cros_scheduling.py` executor paths |
| 15 | Measure outcomes; residual = observed − predicted | `mape/execute.py` (`OutcomeMeasurer`) | `test_cros_mape.py::test_outcome_measurer_computes_residuals`, `test_cros_cycle.py::test_cycle_versions_state_and_stores_decisions` |
| 16 | Deterministic baseline scheduler; admission; bounded workers | `policies/baseline.py` | `test_cros_scheduling.py::test_identical_traces_produce_identical_decisions`, `..._worker_pool_bounds_admitted_concurrency`, `..._dependency_and_capability_readiness` |
| 17 | Proportional fairness; share_i = w_i/Σw; service owed/received | `policies/cpu.py` (`WeightedFairScheduler`) | `test_cros_scheduling.py::test_weighted_fairness_matches_configured_weights` |
| 18 | CFS/EEVDF as references, not claimed semantics | `policies/cpu.py` (`FairnessReport`) | `test_cros_scheduling.py::test_weighted_fairness_matches_configured_weights` (Jain index + deviation report) |
| 19 | Deadline-aware EDF with feasibility checks | `policies/cpu.py` (`EarliestDeadlineFirstScheduler`) | `test_cros_scheduling.py::test_edf_orders_by_deadline_and_checks_feasibility` |
| 20 | Rate Monotonic for periodic tasks; validate periods/WCETs | `policies/cpu.py` (`RateMonotonicScheduler`) | `test_cros_scheduling.py::test_rate_monotonic_priorities_and_bound` |
| 21 | Multilevel feedback with aging; no indefinite demotion | `policies/cpu.py` (`MultilevelFeedbackQueue`) | `test_cros_scheduling.py::test_mlfq_aging_prevents_indefinite_demotion` |
| 22 | Gang scheduling only for co-dependent groups | `policies/cpu.py` (`GangScheduler`) | `test_cros_scheduling.py::test_gang_scheduler_dispatches_only_complete_gangs` |
| 23 | Module quotas and reservations; explicit borrowing | `policies/cpu.py` (`QuotaManager`), `safety/constraints.py` (`QuotaLedger`) | `test_cros_scheduling.py::test_quotas_isolate_reserved_budgets`, `test_cros_safety.py::test_quota_ledger_records_authorized_exceptions` |
| 24 | Prevent starvation and priority inversion; aging; wait monitoring | `policies/cpu.py` (`StarvationGuard`) | `test_cros_scheduling.py::test_starvation_guard_flags_and_ages_waiting_tasks` |
| 25 | Shadow-mode evaluation of learned policies | `policies/cpu.py` (`ShadowModeEvaluator`) | `test_cros_scheduling.py::test_shadow_mode_records_without_controlling` |
| 26 | Constrained adaptation with fallback | `policies/cpu.py` (`ConstrainedAdaptation`, `REVERSIBLE_ACTIONS`) | `test_cros_scheduling.py::test_constrained_adaptation_reverts_to_baseline` |
| 27 | Working-set estimation; sustained vs peaks; rate of change | `policies/memory.py` (`WorkingSetEstimator`) | `test_cros_memory.py::test_working_set_estimates_respond_to_access_patterns` |
| 28 | Hot/warm/cold cache tiers; semantics preserved | `policies/memory.py` (`TieredCache`) | `test_cros_memory.py::test_cache_tiering_preserves_values_and_provenance`, `..._eviction_only_from_cold_tier` |
| 29 | LRU/CLOCK baselines; hit rate, miss cost, scan resistance | `policies/memory.py` (`LRUReplacer`, `CLOCKReplacer`) | `test_cros_memory.py::test_lru_is_scan_sensitive_clock_resists` |
| 30 | MGLRU / OS-native reclaim evaluation | `policies/memory.py` (`MultiGenerationReplacer`) | `test_cros_memory.py::test_multi_generation_replacer_evicts_oldest_first` |
| 31 | DAMON access monitoring, read-only | `policies/memory.py` (`DamonAdapter`) | `test_cros_memory.py::test_damon_adapter_reads_without_mutating` |
| 32 | NUMA locality only when topology available | `policies/memory.py` (`NumaTopology`) | `test_cros_memory.py::test_numa_locality_only_when_topology_available`, `..._discovery_from_fake_sysfs` |
| 33 | Compression and tiering by access frequency vs cost | `policies/memory.py` (`CompressionTierPolicy`) | `test_cros_memory.py::test_compression_tiering_cost_model` |
| 34 | GPU memory admission, transfers, batching | `policies/memory.py` (`AcceleratorMemoryManager`) | `test_cros_memory.py::test_gpu_admission_rejects_or_defers_oversized_tasks` |
| 35 | CXL as future extension interface | `policies/memory.py` (`CxlExtensionPoint`) | `test_cros_memory.py::test_cxl_extension_point_stays_minimal` |
| 36 | Node advertisements with expiry; versioned cluster map | `distributed/placement.py` (`NodeAdvertisement`, `ClusterResourceMap`) | `test_cros_distributed.py::test_advertisements_expire_and_refresh` |
| 37 | Eligibility = capability AND budget AND policy AND data AND health | `distributed/placement.py` (`check_eligibility`, `filter_eligible_nodes`) | `test_cros_distributed.py::test_eligibility_filters_incapable_and_unhealthy_nodes`, `..._respects_policy_allowlist` |
| 38 | Placement scoring by cost and locality with explanations | `distributed/placement.py` (`PlacementScorer`) | `test_cros_distributed.py::test_placement_scores_rank_by_cost_with_explanations` |
| 39 | Borg / Kubernetes / Decima as documented references | `distributed/placement.py` (`CLUSTER_DESIGN_REFERENCES`) | `test_cros_distributed.py::test_cluster_design_references_documented` |
| 40 | Migration only when benefit > transfer + restart + risk; checkpoint required | `distributed/placement.py` (`decide_migration`) | `test_cros_distributed.py::test_migration_rejected_when_costs_exceed_benefit` |
| 41 | Node-loss handling; side-effect-aware recovery; timeout ≠ proof | `distributed/placement.py` (`FailureDetector`) | `test_cros_distributed.py::test_node_failure_recovery_is_side_effect_aware`, `..._timeout_does_not_prove_task_stopped` |
| 42 | I/O and network demand profiling | `policies/io_network.py` (`IoNetworkProfiler`) | `test_cros_io_network.py::test_io_profiler_identifies_bottlenecks_with_evidence` |
| 43 | Async I/O (io_uring) with portable fallback and benchmark gate | `policies/io_network.py` (`AsyncIoBackend`) | `test_cros_io_network.py::test_async_io_backend_falls_back_cleanly` |
| 44 | eBPF/XDP/DPDK/Demikernel require evidence | `policies/io_network.py` (`SpecializedNetworkGate`) | `test_cros_io_network.py::test_specialized_network_gate_requires_evidence` |
| 45 | Storage capability matrix, not one default fs | `policies/io_network.py` (`StorageCapabilityMatrix`) | `test_cros_io_network.py::test_storage_matrix_returns_candidates_not_defaults` |
| 46 | Durability contracts preserved under optimization | `policies/io_network.py` (`DurabilityContractRegistry`) | `test_cros_io_network.py::test_durability_contracts_block_silent_downgrades`, `..._validation` |
| 47 | Criticality classes; defer optional before critical | `safety/constraints.py` (`CriticalityMatrix`), `resource_model.py` | `test_cros_safety.py::test_criticality_matrix_governs_deferral_and_reservations`, `test_cros_cycle.py::test_pressure_defers_optional_before_integrity_critical`, `test_cros_resource_model.py::test_task_records_cannot_self_promote_criticality` |
| 48 | WCET analysis; no hard real-time claim without evidence | `safety/constraints.py` (`WcetAnalyzer`) | `test_cros_safety.py::test_wcet_analyzer_validates_rm_feasibility`, `..._hard_real_time_claim_requires_platform_evidence` |
| 49 | Predictive resource models with uncertainty and calibration | `prediction/demand.py` | `test_cros_cycle.py::test_prediction_error_and_uncertainty_measured_by_class`, `..._cycle_updates_demand_predictors_by_class` |
| 50 | Constrained contextual bandit; logging; offline replay | `learning/bandit.py` | `test_cros_learning.py::test_bandit_selects_only_from_approved_safe_actions`, `..._updates_value_estimates_from_rewards`, `..._logging_records_context_propensity_outcome`, `..._offline_replay_uses_inverse_propensity_weighting` |
| 51 | Constrained RL/MPC only as prerequisites-gated research path | `learning/bandit.py` (`ConstrainedRlResearchPath`) | `test_cros_learning.py::test_rl_research_path_refuses_training_without_prerequisites` |
| 52 | Deterministic fallback and rollback for every adaptive policy | `safety/fallback.py` | `test_cros_safety.py::test_rollback_manager_restores_known_good_baseline`, `..._safe_fallback_policy_disables_adaptation_after_trigger`, `..._watchdog_triggers_and_enforces_cooldown` |

End-to-end cycle (section 5): `cycle.py` — tested by `test_cros_cycle.py`
(full-cycle result, deferral ordering, versioning, predictor updates, long-replay
stability, shadow integration) and `test_cros_safety.py` (executor failure,
learning-component failure).

## Part 2 — Section 8 acceptance items

| Acceptance item | Test |
| --- | --- |
| Telemetry rejects invalid units, impossible values, malformed timestamps | `test_cros_telemetry.py::test_measurement_rejects_invalid_units`, `..._impossible_values`, `..._malformed_timestamps` |
| Machine/process/task/module metrics cannot be silently mixed | `test_cros_telemetry.py::test_scopes_cannot_be_silently_mixed` |
| Stale or missing telemetry cannot trigger unsafe high-impact actions | `test_cros_telemetry.py::test_stale_telemetry_blocks_high_impact_actions` |
| Identical traces → equivalent deterministic decisions | `test_cros_scheduling.py::test_identical_traces_produce_identical_decisions` |
| Usage attributed correctly or marked estimated | `test_cros_accounting.py::test_directly_attributed_usage_is_exact`, `..._labelled_estimated` |
| No budget overrun without recorded authorized exception | `test_cros_safety.py::test_quota_ledger_records_authorized_exceptions` |
| Integrity-critical retains reserved capacity under batch overload | `test_cros_scheduling.py::test_integrity_critical_retains_reserved_capacity_under_batch_overload` |
| Admission defers/rejects before ceilings exceeded | `test_cros_scheduling.py::test_admission_defers_optionals_before_ceilings_breach` |
| Service consistent with configured weights | `test_cros_scheduling.py::test_weighted_fairness_matches_configured_weights` |
| Aging/reservations prevent indefinite waiting | `test_cros_scheduling.py::test_mlfq_aging_prevents_indefinite_demotion`, `..._starvation_guard...` |
| EDF orders correctly and checks feasibility | `test_cros_scheduling.py::test_edf_orders_by_deadline_and_checks_feasibility` |
| Rate Monotonic validates periods and execution assumptions | `test_cros_scheduling.py::test_rate_monotonic_priorities_and_bound` |
| Noisy measurements do not cause rapid oscillation | `test_cros_safety.py::test_hysteresis_prevents_rapid_oscillation` |
| Saturated controller recovers without excessive overshoot | `test_cros_safety.py::test_pid_controller_anti_windup_bounds_overshoot` |
| Each MAPE-K stage independently testable | `test_cros_mape.py::test_each_mape_stage_is_independently_testable` |
| Constraint-violating candidate plans rejected | `test_cros_mape.py::test_constraint_filter_rejects_violating_candidates` |
| Policy engine cannot invoke arbitrary OS commands / bypass executor | `test_cros_mape.py::test_policy_engine_cannot_bypass_execution_manager` |
| Simulated policy failure restores known-good baseline | `test_cros_safety.py::test_rollback_manager_restores_known_good_baseline`, `..._cycle_survives_executor_failure_with_baseline` |
| Learned actions cannot change allocation in shadow mode | `test_cros_learning.py::test_shadow_mode_never_changes_allocation`, `test_cros_scheduling.py::test_shadow_mode_records_without_controlling` |
| Predictor error and uncertainty measured by class | `test_cros_cycle.py::test_prediction_error_and_uncertainty_measured_by_class` |
| Working-set estimates respond to access patterns | `test_cros_memory.py::test_working_set_estimates_respond_to_access_patterns` |
| Cache tiering never changes retrieval results or provenance | `test_cros_memory.py::test_cache_tiering_preserves_values_and_provenance` |
| NUMA locality only when topology available | `test_cros_memory.py::test_numa_locality_only_when_topology_available` |
| GPU tasks rejected/deferred when memory insufficient | `test_cros_memory.py::test_gpu_admission_rejects_or_defers_oversized_tasks` |
| Unsupported io_uring/platform interfaces fall back cleanly | `test_cros_io_network.py::test_async_io_backend_falls_back_cleanly` |
| Resource optimization cannot violate durability contracts | `test_cros_io_network.py::test_durability_contracts_block_silent_downgrades` |
| Placement never selects incapable or unhealthy nodes | `test_cros_distributed.py::test_eligibility_filters_incapable_and_unhealthy_nodes` |
| Migration rejected when costs exceed benefit | `test_cros_distributed.py::test_migration_rejected_when_costs_exceed_benefit` |
| Node failure triggers side-effect-aware recovery | `test_cros_distributed.py::test_node_failure_recovery_is_side_effect_aware` |
| Optional exploration/indexing deferred before integrity-critical | `test_cros_cycle.py::test_pressure_defers_optional_before_integrity_critical` |
| No hard real-time claim without bounded execution and platform evidence | `test_cros_safety.py::test_hard_real_time_claim_requires_platform_evidence` |
| Bandit selects only from approved safe action set | `test_cros_learning.py::test_bandit_selects_only_from_approved_safe_actions` |
| Learned control limited to a configured canary | `test_cros_learning.py::test_canary_fraction_limits_learned_control` |
| Adaptive control cannot exceed configured unfairness threshold | `test_cros_safety.py::test_fairness_floor_constrains_adaptive_actions` |
| Benchmarks report throughput, p50/p95/p99, queue delay, pressure, energy, overhead | `test_cros_cycle.py::test_benchmark_reports_required_metrics` |
| Every action records state version, policy version, constraints, rationale, predicted/measured outcome | `test_cros_mape.py::test_every_decision_records_full_provenance`, `test_cros_cycle.py::test_cycle_versions_state_and_stores_decisions` |
| One module cannot silently consume another's reserved budget | `test_cros_scheduling.py::test_quotas_isolate_reserved_budgets` |
| Long replay: no unbounded growth, oscillation or drift | `test_cros_cycle.py::test_long_replay_stability_no_drift_or_growth` |
| Missing telemetry / collector crash / executor failure trigger safe behavior | `test_cros_telemetry.py::test_collector_set_isolates_crashing_collectors`, `test_cros_safety.py::test_cycle_survives_executor_failure_with_baseline`, `..._deterministic_baseline_available_when_learning_fails` |
| Deterministic baseline available if predictors/learning fail | `test_cros_safety.py::test_deterministic_baseline_available_when_learning_fails` |

## Part 3 — Verification record

- `python -m pytest tests/ -k cros` → **127 passed**.
- Full repository suite (`python -m pytest tests/`) → **1314 passed, 1 failed**;
  the single failure is the pre-existing, timing-sensitive
  `tests/test_ore_logger_runtime.py::test_sitecustomize_automatically_logs_a_launched_repository_script`
  (passes in isolation; unrelated to `ore_cros`, which is purely additive).
