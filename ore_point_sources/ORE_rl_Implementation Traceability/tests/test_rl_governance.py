"""Governance: registry promotion, rollback, shadow mode, coordinator bridges (Steps 51-52)."""
import tempfile
from pathlib import Path

from ore_contracts import (
    ContractArtifact,
    ContractCatalog,
    CognitiveTask,
    ModuleRegistry,
)
from ore_contracts.adapters import make_envelope
from ore_contracts.resource_coordination import AdaptiveResourceCoordinator
from ore_contracts.resource_profiles import CapabilityDemand, TaskProfile

from ore_rl import OREEnvironment, TaskSampler, register_rl_example_modules
from ore_rl.controller_adapter import (
    DecisionBoundary,
    DecisionBoundaryPolicy,
    DecisionEvent,
    ModulePreferenceBridge,
    RLPolicyService,
    ShadowModeController,
)
from ore_rl.evaluation import (
    BenchmarkSuite,
    PromotionThresholds,
    evaluate_policy,
)
from ore_rl.policies.base import (
    DeterministicBaselinePolicy,
    ProgressBoundaryPolicy,
    RandomEligiblePolicy,
)
from ore_rl.policies.bandit import LinUCBPolicy
from ore_rl.registry import PolicyRegistry, PolicyStatus, PromotionApproval


def _env(seed: int = 2, n_tasks: int = 8) -> OREEnvironment:
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_rl_example_modules(catalog, registry)
    return OREEnvironment(
        registry=registry, catalog=catalog,
        sampler=TaskSampler(seed=seed, n_tasks=n_tasks), seed=seed,
    )


def _registry() -> PolicyRegistry:
    return PolicyRegistry(Path(tempfile.mkdtemp()) / "registry.sqlite3")


def test_shadow_mode_records_proposals_without_execution_authority():
    """Step 51: shadow decisions are recorded; nothing is adopted."""
    env = _env()
    linucb = LinUCBPolicy(seed=1, alpha=0.3)
    service = RLPolicyService(linucb)
    shadow = ShadowModeController(service)
    baseline = DeterministicBaselinePolicy()
    state, info = env.reset()
    eligible = info["feasible_action_ids"]
    for _ in range(5):
        coordinator_decision = baseline.propose_actions(state, eligible)
        shadow.consider(
            DecisionEvent(boundary=DecisionBoundary.AFTER_MODULE_COMPLETION, task_id="t"),
            state, eligible, coordinator_decision.chosen_action_id,
        )
        state, _reward, terminated, truncated, info = env.step(
            coordinator_decision.chosen_action_id
        )
        eligible = info["feasible_action_ids"]
        if terminated or truncated:
            break
    summary = shadow.summary()
    assert summary["records"] >= 4
    assert summary["authority"] == "SHADOW"
    assert all(not record.adopted for record in shadow.records)
    assert 0.0 <= summary["agreement_rate"] <= 1.0
    env.close()


def test_decision_boundaries_are_explicit_and_filterable():
    """Step 2: consultation happens only at declared boundaries."""
    policy = DecisionBoundaryPolicy(enabled=(DecisionBoundary.STOP_CONTINUE,))
    assert not policy.should_consult(
        DecisionEvent(boundary=DecisionBoundary.AFTER_MODULE_COMPLETION, task_id="t")
    )
    assert policy.should_consult(
        DecisionEvent(boundary=DecisionBoundary.STOP_CONTINUE, task_id="t")
    )
    full = DecisionBoundaryPolicy()
    assert full.should_consult(
        DecisionEvent(boundary=DecisionBoundary.EVIDENCE_INSUFFICIENT, task_id="t",
                      evidence_sufficient=False)
    )
    assert not full.should_consult(
        DecisionEvent(boundary=DecisionBoundary.EVIDENCE_INSUFFICIENT, task_id="t",
                      evidence_sufficient=True)
    )
    assert not full.should_consult(
        DecisionEvent(boundary=DecisionBoundary.DEPENDENCY_READY, task_id="t",
                      dependency_ready=False)
    )


def test_policy_crash_falls_back_to_the_deterministic_coordinator():
    """Acceptance 8.1: a crashing policy yields a deterministic fallback decision."""
    class Exploding:
        def __init__(self):
            from ore_rl import PolicyMetadata, PolicyAuthority

            self.metadata = PolicyMetadata(
                policy_id="exploding", policy_version="1.0.0", algorithm_family="test",
                execution_authority=PolicyAuthority.PROPOSAL,
            )

        def propose_actions(self, state, eligible, budget=None):
            raise RuntimeError("boom")

    env = _env()
    state, info = env.reset()
    service = RLPolicyService(Exploding())
    decision = service.propose_actions(state, info["feasible_action_ids"])
    assert decision.fallback_used
    assert decision.chosen_action_id in info["feasible_action_ids"]
    assert service.stats()["fallbacks"] == 1
    assert decision.rationale.startswith("policy crashed")
    env.close()


def test_promotion_is_blocked_without_meeting_predeclared_thresholds():
    """Acceptance 8.3: promotion is blocked unless thresholds are met."""
    env = _env()
    registry = _registry()
    registry.register(
        policy_id="weak", policy_version="1.0.0", algorithm_family="contextual_bandit/ucb",
        metadata={},
    )
    random_policy = RandomEligiblePolicy(seed=3)
    baseline = DeterministicBaselinePolicy()
    report = evaluate_policy(
        random_policy, env,
        suite=BenchmarkSuite(name="gate", n_episodes=6, seed=6),
        baseline=baseline,
        thresholds=PromotionThresholds(min_episodes=5, min_baseline_improvement=0.9),
    )
    approval = PromotionApproval(
        approver="reviewer", approved_at="2026-11-22", justification="try",
        evaluation_fingerprint="weak",
    )
    try:
        registry.promote("weak@1.0.0", report, approval, thresholds=report.thresholds)
        promoted = True
    except PermissionError:
        promoted = False
    assert not promoted
    assert registry.get("weak@1.0.0").status is PolicyStatus.CANDIDATE
    assert any(
        row["action"] == "PROMOTION_BLOCKED" for row in registry.audit_log()
    )
    env.close()


def test_promotion_and_rollback_restore_previous_validated_version():
    """Acceptance 8.1: rollback restores the previous validated policy."""
    env = _env()
    registry = _registry()
    linucb = LinUCBPolicy(seed=4, alpha=0.3)
    baseline = DeterministicBaselinePolicy()
    # brief supervised training so the candidate is competitive
    for _ in range(12):
        state, info = env.reset()
        eligible = info["feasible_action_ids"]
        done = False
        while not done:
            progress = [a for a in eligible if a.startswith("run.")]
            if "meta.stop_success" in eligible and not progress:
                action = "meta.stop_success"
            elif progress:
                action = linucb.propose_actions(state, progress).chosen_action_id
            else:
                action = "meta.stop_insufficient_evidence"
            previous = state
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if action.startswith("run."):
                linucb.update(previous, action, reward)
            eligible = info["feasible_action_ids"]
    for version, policy in (("1.0.0", linucb), ("2.0.0", linucb)):
        registry.register(
            policy_id="linucb", policy_version=version,
            algorithm_family="contextual_bandit/linucb", metadata={"seed": 4},
        )
        report = evaluate_policy(
            ProgressBoundaryPolicy(policy), env,
            suite=BenchmarkSuite(name=f"promo-{version}", n_episodes=6, seed=6),
            baseline=baseline,
            thresholds=PromotionThresholds(min_episodes=5, min_baseline_improvement=-0.2),
        )
        assert report.meets_promotion_thresholds
        from ore_rl.contracts import stable_hash

        approval = PromotionApproval(
            approver="reviewer", approved_at="2026-11-22", justification="good",
            evaluation_fingerprint=stable_hash(report.to_dict()),
        )
        registry.promote(f"linucb@{version}", report, approval, thresholds=report.thresholds)
    assert registry.active_policy("linucb").policy_version == "2.0.0"
    restored = registry.rollback("linucb", actor="sre")
    assert restored.policy_version == "1.0.0"
    assert registry.active_policy("linucb").policy_version == "1.0.0"
    statuses = {record.policy_version: record.status
                for record in registry.list_records()}
    assert statuses["2.0.0"] is PolicyStatus.RETIRED
    assert any(row["action"] == "ROLLBACK" for row in registry.audit_log())
    env.close()


def test_monitoring_triggers_retirement_on_drift_and_ood():
    """Step 52: reward drift, constraint rejections, and OOD states freeze the policy."""
    registry = _registry()
    registry.register(
        policy_id="drifty", policy_version="1.0.0", algorithm_family="test", metadata={},
    )
    key = "drifty@1.0.0"
    for reward in (1.0, 0.8, 0.3, 0.05):
        registry.record_monitoring_sample(
            key, mean_reward=reward, success_rate=reward,
            constraint_rejection_rate=0.05, out_of_distribution_rate=0.05,
        )
    summary = registry.monitoring_summary(key)
    assert summary["retirement_triggers"] == ["reward_drift"]
    registry.record_monitoring_sample(
        key, mean_reward=0.05, success_rate=0.05,
        constraint_rejection_rate=0.5, out_of_distribution_rate=0.6,
    )
    summary = registry.monitoring_summary(key)
    assert set(summary["retirement_triggers"]) >= {"reward_drift", "constraint_rejections",
                                                   "out_of_distribution"}
    frozen = registry.freeze(key, actor="monitor", reason="drift")
    assert frozen.status is PolicyStatus.FROZEN


def test_module_preference_bridge_drives_the_real_adaptive_coordinator():
    """Integration mapping (section 6): ranked proposals become coordinator preferences."""
    env = _env(seed=8, n_tasks=6)
    state, info = env.reset()
    linucb = LinUCBPolicy(seed=9, alpha=0.3)
    decision = linucb.propose_actions(state, info["feasible_action_ids"])
    bridge = ModulePreferenceBridge.from_environment(env)
    preferences = bridge.to_preferences(decision)
    assert preferences
    assert all("@" not in module for module in preferences.values())
    assert all(module.startswith("ore.rl.") for module in preferences.values())

    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_rl_example_modules(catalog, registry)
    adaptive = AdaptiveResourceCoordinator(registry, catalog, enable_logging=False, max_workers=2)
    profile = TaskProfile(
        task_id="task-adaptive-bridge", task_family="fact_verification",
        demands=(CapabilityDemand(capability_id="cognition.rl.input_analysis"),),
        stop_confidence=0.8, max_module_calls=4,
    )
    from ore_rl.modules import CONTRACT_INPUT_TEXT

    artifact = ContractArtifact(
        make_envelope(CONTRACT_INPUT_TEXT, "bridge-run", "ore.rl_environment"),
        "the gradient governs the response",
    )
    task = CognitiveTask(
        goal="bridge run", task_id="task-adaptive-bridge",
        required_capabilities=("cognition.rl.input_analysis",),
        input_artifacts={"source_text": artifact},
        expected_outputs=("ORE.RL.INPUT_ANALYSIS",),
    )
    result = adaptive.coordinate(profile, bridge.attach_to_task(task, decision))
    assert result.completed
    assert result.module_results["ore.rl.input_analyzer"].status.value == "success"
    env.close()
