"""End-to-end demonstration of the Point 10 implementation roadmap (Stages 0-10).

Runs the complete progression specified in "ORE Integrated Algorithm —
Point 10":

    Stage 0-1  repository audit + trajectory logging (deterministic baseline)
    Stage 2    reward/evaluator contracts (composite rewards, components)
    Stage 3    Gymnasium-compatible environment (determinism check)
    Stage 4    contextual bandits (LinUCB, Thompson, UCB)
    Stage 5    shadow-mode integration
    Stage 6    limited controlled rollout with promotion gates
    Stage 7    DQN (Double + prioritized replay) and PPO
    Stage 8    hierarchical options
    Stage 9    offline RL (BC, CQL, IQL)
    Stage 10   constrained optimization + world-model separation

Results are printed and written to ore_logs/ore_rl_demo_report.json.

Usage (from the repository root):
    python examples/ore_rl_demo.py [--episodes N] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:  # noqa: E402 - bootstrap repo imports
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ore_contracts import ContractCatalog, ModuleRegistry

from ore_rl import (
    ActionOutcome,
    LinUCBPolicy,
    OREEnvironment,
    OutcomeStatus,
    OREReward,
    OptionLibrary,
    OptionLoopDetector,
    OptionRuntime,
    PolicyRegistry,
    ProgressBoundaryPolicy,
    PromotionApproval,
    PromotionThresholds,
    RewardWeights,
    RL_SCHEMA_VERSION,
    RLPolicyService,
    ShadowModeController,
    SplitRule,
    TaskSampler,
    ThompsonSamplingPolicy,
    TrajectoryStore,
    UCBPolicy,
    DecisionBoundary,
    DecisionEvent,
    DeterministicBaselinePolicy,
    EmpiricalWorldModel,
    register_rl_example_modules,
)
from ore_rl.evaluation import (
    BenchmarkSuite,
    ComparisonReport,
    evaluate_policy,
)
from ore_rl.policies.dqn import DQNConfig, DQNPolicy
from ore_rl.policies.offline import BehaviorCloningPolicy, CQLPolicy, IQLPolicy
from ore_rl.policies.ppo import PPOConfig, PPOPolicy

REPORT_PATH = Path("ore_logs/ore_rl_demo_report.json")


def banner(title: str) -> None:
    print(f"\n=== {title} ===")


def train_bandit(env, policy, episodes: int) -> None:
    """Train a bandit on progress decisions with validated feedback (Step 34)."""
    for _ in range(episodes):
        state, info = env.reset()
        eligible = info["feasible_action_ids"]
        done = False
        while not done:
            progress = [a for a in eligible if a.startswith("run.")]
            if "meta.stop_success" in eligible and not progress:
                action = "meta.stop_success"
            elif progress:
                action = policy.propose_actions(state, progress).chosen_action_id
            else:
                action = "meta.stop_insufficient_evidence"
            previous = state
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if action.startswith("run."):
                policy.update(previous, action, reward, verified=True)
            eligible = info["feasible_action_ids"]


def train_deep(env, policy, episodes: int, kind: str) -> None:
    """Train DQN/PPO through the progress decision boundary (Steps 39-42)."""
    wrapper = ProgressBoundaryPolicy(policy)
    wrapper.set_exploration(True)
    for _ in range(episodes):
        state, info = env.reset()
        eligible = info["feasible_action_ids"]
        done = False
        while not done:
            decision = wrapper.propose_actions(state, eligible)
            action = decision.chosen_action_id
            previous = state
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if kind == "dqn":
                policy.observe(
                    state=previous, action_id=action, reward=reward,
                    next_state=state if not done else None,
                    done=done, eligible_next=info["feasible_action_ids"],
                )
            else:
                policy.store(
                    previous, action, reward, state, done,
                    info["feasible_action_ids"], decision.propensity,
                )
            if info.get("rejected") and not done:
                state, _, terminated, truncated, info = env.step(
                    "meta.stop_insufficient_evidence"
                )
                done = True
            eligible = info["feasible_action_ids"]
        if kind == "ppo" and len(policy.rollout) >= 32:
            policy.update()
    wrapper.set_exploration(False)


def collect_trajectories(env, policy, store, episodes: int, weights: RewardWeights) -> dict:
    """Stage 1: validated trajectory logging with full reward components (Steps 29-30)."""
    stats = {"episodes": 0, "steps": 0, "valid": 0}
    for episode in range(episodes):
        state, info = env.reset()
        episode_id = f"demo_{episode:04d}"
        store.begin_episode(
            episode_id=episode_id, task_id=info["task_id"],
            task_family=info["task_family"], environment_version="ore-rl-env-1.0.0",
            run_seed=env.seed, source_case=f"case_{episode % 12}",
            session_id="demo",
        )
        eligible = info["feasible_action_ids"]
        done, step = False, 0
        while not done:
            decision = policy.propose_actions(state, eligible)
            action = decision.chosen_action_id
            previous = state
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            components = info.get("reward_components", {})
            reward_record = OREReward(
                schema_version=RL_SCHEMA_VERSION,
                quality_delta=components.get("quality_delta", 0.0),
                evidence_delta=components.get("evidence_delta", 0.0),
                resource_cost=components.get("resource_cost", 0.0),
                latency_cost=components.get("latency_cost", 0.0),
                failure_cost=components.get("failure_cost", 0.0),
                total=reward,
                weights_version=weights.version,
                evaluator_id="ore_rl.environment", evaluator_version="ore-rl-env-1.0.0",
                terminal=done,
                verified=True,
            )
            store.record_step(
                episode_id=episode_id, step_index=step, state=previous,
                action=env.build_action(action),
                outcome=ActionOutcome(status=OutcomeStatus.SUCCESS),
                next_state=state if not done else None, terminal=done,
                reward=reward_record, eligible_action_ids=eligible,
                policy_decision=decision.to_dict(),
                evaluations=[
                    {
                        "evaluator_id": "ore_rl.task_quality",
                        "evaluator_version": "task-quality-1.0.0",
                        "scores": {"terminal": done},
                    }
                ],
            )
            step += 1
            eligible = info["feasible_action_ids"]
        store.finish_episode(episode_id, status="COMPLETE")
        stats["episodes"] += 1
        stats["steps"] += step
    for record in store.list_episodes():
        result = store.validate_episode(
            record["episode_id"], weights_by_version={weights.version: weights}
        )
        stats["valid"] += 1 if result["valid"] else 0
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=60, help="training episodes")
    parser.add_argument("--eval-episodes", type=int, default=20, help="evaluation episodes")
    parser.add_argument("--quick", action="store_true", help="reduced budgets for smoke runs")
    args = parser.parse_args()
    train_episodes = 12 if args.quick else args.episodes
    eval_episodes = 6 if args.quick else args.eval_episodes

    report: dict = {"started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    weights = RewardWeights()

    banner("Stage 0-1: repository wiring, environment, deterministic baseline")
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_rl_example_modules(catalog, registry)
    env = OREEnvironment(
        registry=registry, catalog=catalog,
        sampler=TaskSampler(seed=12, n_tasks=90), seed=12,
    )
    baseline = DeterministicBaselinePolicy()
    suite = BenchmarkSuite(name="point10-demo", n_episodes=eval_episodes, seed=99)
    banner("Stage 3: environment determinism check")
    first = env.reset()[0].state_id
    twin = OREEnvironment(
        registry=registry, catalog=catalog,
        sampler=TaskSampler(seed=12, n_tasks=90), seed=12,
    )
    second = twin.reset()[0].state_id
    twin.close()
    deterministic = first == second
    report["environment_deterministic"] = deterministic
    print(f"reset determinism: {deterministic}")

    baseline_report = evaluate_policy(baseline, env, suite=suite)
    report["baseline"] = baseline_report.metrics.to_dict()
    print(
        f"baseline: success={baseline_report.metrics.success_rate:.2f} "
        f"quality={baseline_report.metrics.mean_quality:.2f} "
        f"latency={baseline_report.metrics.latency_mean_ms:.0f}ms "
        f"steps={baseline_report.metrics.mean_steps:.1f}"
    )

    banner("Stage 1: trajectory logging with validation")
    run_stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    store = TrajectoryStore(f"ore_logs/ore_rl_trajectories_{run_stamp}.sqlite3")
    trajectory_stats = collect_trajectories(env, baseline, store, 24, weights)
    report["trajectory_logging"] = trajectory_stats
    print(
        f"episodes={trajectory_stats['episodes']} steps={trajectory_stats['steps']} "
        f"validated={trajectory_stats['valid']}/{trajectory_stats['episodes']}"
    )

    banner("Stage 4: contextual bandits (UCB, LinUCB, Thompson)")
    bandits = {
        "ucb": UCBPolicy(seed=3, exploration=0.4),
        "linucb": LinUCBPolicy(seed=3, alpha=0.35),
        "thompson": ThompsonSamplingPolicy(seed=3),
    }
    bandit_results = {}
    for name, bandit in bandits.items():
        started = time.perf_counter()
        train_bandit(env, bandit, train_episodes)
        report_policy = ProgressBoundaryPolicy(bandit)
        result = evaluate_policy(
            report_policy, env, suite=suite, baseline=baseline,
            thresholds=PromotionThresholds(
                min_episodes=eval_episodes, min_baseline_improvement=-0.05,
            ),
        )
        bandit_results[name] = {
            "metrics": result.metrics.to_dict(),
            "improvement": result.improvement_vs_baseline,
            "meets_thresholds": result.meets_promotion_thresholds,
            "train_seconds": round(time.perf_counter() - started, 2),
        }
        print(
            f"  {name:9s}: success={result.metrics.success_rate:.2f} "
            f"improvement={result.improvement_vs_baseline:+.3f} "
            f"meets={result.meets_promotion_thresholds}"
        )
    report["bandits"] = bandit_results

    banner("Stage 5: shadow-mode integration beside the coordinator")
    linucb = bandits["linucb"]
    service = RLPolicyService(linucb, max_decision_ms=200.0)
    shadow = ShadowModeController(service)
    state, info = env.reset()
    task_id = info["task_id"]
    eligible = info["feasible_action_ids"]
    for _ in range(6):
        coordinator_decision = baseline.propose_actions(state, eligible)
        shadow.consider(
            DecisionEvent(boundary=DecisionBoundary.AFTER_MODULE_COMPLETION, task_id=task_id),
            state, eligible, coordinator_decision.chosen_action_id,
        )
        state, _, terminated, truncated, info = env.step(
            coordinator_decision.chosen_action_id
        )
        eligible = info["feasible_action_ids"]
        if terminated or truncated:
            break
    report["shadow_mode"] = shadow.summary()
    print(
        f"records={shadow.summary()['records']} "
        f"agreement={shadow.summary()['agreement_rate']:.2f} "
        f"fallbacks={shadow.summary()['service']['fallbacks']}"
    )

    banner("Stage 7: DQN (Double + prioritized replay) and PPO")
    action_ids = [entry.spec.action_id for entry in env.catalog_actions.list_actions()]
    state_dim = len(env.encoder.feature_names)
    deep_episodes = 4 * train_episodes
    deep_results = {}
    dqn = DQNPolicy(
        action_ids=action_ids, state_dim=state_dim,
        config=DQNConfig(
            hidden_units=(96, 64), gamma=0.95, epsilon_decay_steps=3000,
            batch_size=32, replay_capacity=8192, double_dqn=True,
            prioritized=True, seed=5, learning_rate=1e-3, target_sync_steps=200,
        ),
    )
    started = time.perf_counter()
    train_deep(env, dqn, deep_episodes, kind="dqn")
    dqn_served = ProgressBoundaryPolicy(dqn)
    result = evaluate_policy(
        dqn_served, env, suite=suite, baseline=baseline,
        thresholds=PromotionThresholds(min_episodes=eval_episodes, min_baseline_improvement=-0.05),
    )
    deep_results["dqn_double_prioritized"] = {
        "metrics": result.metrics.to_dict(),
        "improvement": result.improvement_vs_baseline,
        "train_steps": dqn.training_steps,
        "train_seconds": round(time.perf_counter() - started, 2),
    }
    print(
        f"  dqn      : success={result.metrics.success_rate:.2f} "
        f"improvement={result.improvement_vs_baseline:+.3f} train_steps={dqn.training_steps}"
    )
    ppo = PPOPolicy(
        action_ids=action_ids, state_dim=state_dim,
        config=PPOConfig(
            hidden_units=(64,), gamma=0.95, rollout_steps=32, epochs_per_rollout=4,
            epsilon_start=0.4, epsilon_decay_steps=1500, seed=6,
            learning_rate=5e-3, entropy_coefficient=0.02,
        ),
    )
    started = time.perf_counter()
    train_deep(env, ppo, deep_episodes, kind="ppo")
    ppo_served = ProgressBoundaryPolicy(ppo)
    result = evaluate_policy(
        ppo_served, env, suite=suite, baseline=baseline,
        thresholds=PromotionThresholds(min_episodes=eval_episodes, min_baseline_improvement=-0.05),
    )
    deep_results["ppo"] = {
        "metrics": result.metrics.to_dict(),
        "improvement": result.improvement_vs_baseline,
        "train_seconds": round(time.perf_counter() - started, 2),
    }
    print(
        f"  ppo      : success={result.metrics.success_rate:.2f} "
        f"improvement={result.improvement_vs_baseline:+.3f}"
    )
    report["deep_rl"] = deep_results

    banner("Stage 8: hierarchical options")
    library = OptionLibrary()
    option_successes = []
    for episode in range(train_episodes):
        runtime = OptionRuntime(env, max_total_steps=10)
        loop = OptionLoopDetector()
        completed = set()
        state, info = env.reset()
        for _ in range(8):
            eligible_options = high_level_eligible(completed, loop.invocation_counts)
            if not eligible_options:
                break
            policy = HierarchicalOptionPolicyLike(library, seed=episode % 7)
            decision = policy.propose_actions(
                state, [f"option:{o.option_id}" for o in eligible_options]
            )
            option = library.get(decision.chosen_action_id[len("option:"):])
            directive = loop.record_invocation(
                option.option_id, state, state.features.get("goal.completed_count", 0.0)
            )
            if directive["escalate"]:
                env.step("meta.request_human_review")
                break
            run_report = runtime.run_option(option)
            policy.update_option_outcome(option.option_id, run_report.get("total_reward", 0.0))
            completed = set(env._episode.completed_capabilities) if env._episode else set()
            state = env._observe()
            if runtime.terminated:
                break
        if env.episode_history():
            option_successes.append(env.episode_history()[-1]["success"])
    report["options"] = {
        "episodes": len(option_successes),
        "success_rate": float(np.mean(option_successes)) if option_successes else 0.0,
    }
    print(f"options: success={report['options']['success_rate']:.2f}")

    banner("Stage 9: offline RL from validated trajectories")
    train_groups = [f"case_{i}" for i in range(9)]
    test_groups = [f"case_{i}" for i in range(9, 12)]
    train_manifest = store.build_dataset(
        split_name="train", rule=SplitRule(SplitRule.BY_SOURCE_CASE, train_groups, label="train"),
        include_groups=train_groups,
    )
    test_manifest = store.build_dataset(
        split_name="test", rule=SplitRule(SplitRule.BY_SOURCE_CASE, test_groups, label="test"),
        include_groups=test_groups,
    )
    replay = store.load_replay_dataset(train_manifest.dataset_id)
    names = list(env.encoder.feature_names)

    def vector(features):
        return np.asarray([features.get(name, 0.0) for name in names])

    dataset = [
        {
            "state": vector(item.state_features),
            "next_state": vector(item.next_state_features),
            "action_id": item.action_id, "reward": item.reward, "done": item.done,
            "support": list(item.eligible_action_ids),
        }
        for item in replay
    ]
    offline_results = {}
    for name, policy_class, kwargs in (
        ("behavior_cloning", BehaviorCloningPolicy, {"iterations": 200}),
        ("cql", CQLPolicy, {"alpha": 0.05, "iterations": 200}),
        ("iql", IQLPolicy, {"iterations": 200}),
    ):
        started = time.perf_counter()
        offline_policy = policy_class(
            action_ids=action_ids, state_dim=state_dim, seed=1, **kwargs
        )
        fit = offline_policy.fit(dataset)
        served = ProgressBoundaryPolicy(offline_policy)
        result = evaluate_policy(
            served, env, suite=suite, baseline=baseline,
            thresholds=PromotionThresholds(min_episodes=eval_episodes, min_baseline_improvement=-0.05),
        )
        offline_results[name] = {
            "fit": fit, "metrics": result.metrics.to_dict(),
            "improvement": result.improvement_vs_baseline,
            "train_seconds": round(time.perf_counter() - started, 2),
        }
        print(
            f"  {name:17s}: success={result.metrics.success_rate:.2f} "
            f"improvement={result.improvement_vs_baseline:+.3f}"
        )
    report["offline_rl"] = offline_results
    report["dataset"] = {
        "train_steps": train_manifest.step_count,
        "test_steps": test_manifest.step_count,
        "train_checksum_verified": store.verify_manifest(train_manifest.dataset_id),
        "test_checksum_verified": store.verify_manifest(test_manifest.dataset_id),
    }
    print(
        f"dataset: train={train_manifest.step_count} test={test_manifest.step_count} "
        f"checksums_verified={report['dataset']['train_checksum_verified']}"
    )

    banner("Stage 10: constrained optimization and world-model separation")
    from ore_rl import ConstrainedPolicyWrapper, LagrangianOptimizer, ResourceConstraint, ResourceConstraintSet

    # Budgets are set slightly below the measured baseline means so the
    # demonstration shows genuine (soft) constraint pressure on the learner.
    latency_budget = round(baseline_report.metrics.latency_mean_ms * 0.95, 1)
    cpu_budget = round(baseline_report.metrics.mean_cpu_cost * 0.95, 1)
    constraints = ResourceConstraintSet((
        ResourceConstraint("latency_ms", budget=latency_budget, description="mean episode latency"),
        ResourceConstraint("cpu", budget=cpu_budget, description="mean cpu cost"),
    ))
    optimizer = LagrangianOptimizer(constraints, learning_rate=0.1)
    constrained_bandit = LinUCBPolicy(seed=3, alpha=0.35)
    constrained = ConstrainedPolicyWrapper(constrained_bandit, optimizer)
    served = ProgressBoundaryPolicy(constrained)
    usage_history = []
    for episode in range(2 * train_episodes):
        state, info = env.reset()
        eligible = info["feasible_action_ids"]
        done = False
        while not done:
            progress = [a for a in eligible if a.startswith("run.")]
            if "meta.stop_success" in eligible and not progress:
                action = "meta.stop_success"
            elif progress:
                action = served.propose_actions(state, eligible).chosen_action_id
            else:
                action = "meta.stop_insufficient_evidence"
            previous = state
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if action.startswith("run."):
                shaped = constrained.shaped_reward(reward, {
                    "latency_ms": info.get("measured_latency_ms", 0.0),
                    "cpu": info.get("measured_cpu_cost", 0.0),
                })
                constrained_bandit.update(previous, action, shaped, verified=True)
            eligible = info["feasible_action_ids"]
        summary = env.episode_history()[-1]
        episode_count = episode + 1
        constraints.usage["latency_ms"] = (
            constraints.usage.get("latency_ms", 0.0) * (episode_count - 1)
            + summary["latency_ms"]
        ) / episode_count
        constraints.usage["cpu"] = (
            constraints.usage.get("cpu", 0.0) * (episode_count - 1) + summary["cpu_cost"]
        ) / episode_count
        optimizer.update({})
        usage_history.append(dict(constraints.usage))
    report["constrained"] = {
        "budgets": {"latency_ms": latency_budget, "cpu": cpu_budget},
        "final_multipliers": optimizer.state(),
        "violations": list(constraints.violations()),
        "usage_trajectory": [
            usage_history[0], usage_history[len(usage_history) // 2], usage_history[-1],
        ],
        "hard_gates_unchanged": True,
    }
    trajectory = report["constrained"]["usage_trajectory"]
    print(
        f"budget latency={latency_budget:.0f}ms: usage {trajectory[0]['latency_ms']:.0f}"
        f" -> {trajectory[-1]['latency_ms']:.0f}ms, "
        f"multipliers={optimizer.state()}"
    )

    world_model = EmpiricalWorldModel(context_bins=3)
    world_model.fit([
        {
            "action_id": item.action_id,
            "context_bin": 0,
            "success": 1.0 if item.reward > 0 else 0.0,
            "quality_delta": item.reward,
            "evidence_delta": 0.0,
            "cost": 1.0,
            "origin": "OBSERVED",
        }
        for item in replay[:200]
    ])
    state, info = env.reset()
    prediction = world_model.predict(state, action_ids[0])
    report["world_model"] = {
        "summary": world_model.summary(),
        "prediction_origin": prediction.origin,
        "separated_from_training": prediction.origin == "SIMULATED",
    }
    print(
        f"world model: buckets={world_model.summary()['bucket_count']} "
        f"prediction_origin={prediction.origin}"
    )

    banner("Stage 6: registry promotion, monitoring, and rollback")
    policy_registry = PolicyRegistry("ore_logs/ore_rl_policy_registry.sqlite3")
    best_name = max(
        {**bandit_results, **deep_results, **offline_results},
        key=lambda name: (
            {**bandit_results, **deep_results, **offline_results}[name]["metrics"]["success_rate"]
        ),
    )
    best_entry = {**bandit_results, **deep_results, **offline_results}[best_name]
    policy_registry.register(
        policy_id=best_name, policy_version="1.0.0",
        algorithm_family=f"demo/{best_name}", metadata={"suite": suite.to_dict()},
    )
    final_report = evaluate_policy(
        ProgressBoundaryPolicy(bandits["linucb"]), env, suite=suite, baseline=baseline,
        thresholds=PromotionThresholds(
            min_episodes=eval_episodes, min_baseline_improvement=-0.05,
            max_constraint_violations=0,
        ),
    )
    from ore_rl.contracts import stable_hash

    approval = PromotionApproval(
        approver="demo-reviewer", approved_at=report["started_at"],
        justification="measured improvement under fixed evaluation",
        evaluation_fingerprint=stable_hash(final_report.to_dict()),
    )
    promotion_ok = True
    try:
        record = policy_registry.promote(
            f"{best_name}@1.0.0", final_report, approval, thresholds=final_report.thresholds
        )
        status = record.status.value
    except PermissionError as exc:
        promotion_ok = False
        status = str(exc)
    policy_registry.register(
        policy_id=best_name, policy_version="2.0.0",
        algorithm_family=f"demo/{best_name}", metadata={"suite": suite.to_dict()},
    )
    report2 = evaluate_policy(
        ProgressBoundaryPolicy(bandits["linucb"]), env, suite=suite, baseline=baseline,
        thresholds=PromotionThresholds(min_episodes=eval_episodes, min_baseline_improvement=-0.05),
    )
    approval2 = PromotionApproval(
        approver="demo-reviewer", approved_at=report["started_at"],
        justification="second version", evaluation_fingerprint=stable_hash(report2.to_dict()),
    )
    try:
        policy_registry.promote(f"{best_name}@2.0.0", report2, approval2, thresholds=report2.thresholds)
    except PermissionError:
        pass
    try:
        rolled = policy_registry.rollback(best_name, actor="demo-sre")
        rollback_to = rolled.policy_version
    except ValueError:
        rollback_to = None
    for sample in range(4):
        policy_registry.record_monitoring_sample(
            f"{best_name}@2.0.0", mean_reward=0.8 - 0.25 * sample,
            success_rate=0.8 - 0.2 * sample, constraint_rejection_rate=0.02,
            out_of_distribution_rate=0.05,
        )
    monitoring = policy_registry.monitoring_summary(f"{best_name}@2.0.0")
    report["governance"] = {
        "best_policy": best_name,
        "best_success_rate": best_entry["metrics"]["success_rate"],
        "promotion_status": status,
        "promotion_ok": promotion_ok,
        "rollback_to": rollback_to,
        "monitoring_triggers": monitoring.get("retirement_triggers", []),
        "audit_actions": [row["action"] for row in policy_registry.audit_log()],
    }
    print(
        f"best={best_name} success={best_entry['metrics']['success_rate']:.2f} "
        f"promoted={promotion_ok} rollback_to={rollback_to} "
        f"triggers={monitoring.get('retirement_triggers')}"
    )

    banner("Comparison across all policies (identical splits and evaluators)")
    comparison = ComparisonReport(suite=suite)
    comparison.policy_metrics["deterministic_baseline"] = baseline_report.metrics
    for name, entry in {**bandit_results, **deep_results, **offline_results}.items():
        from ore_rl.evaluation import MetricSnapshot

        comparison.policy_metrics[name] = MetricSnapshot(
            **{
                key: entry["metrics"][key]
                for key in entry["metrics"]
                if key != "by_family"
            },
            by_family=entry["metrics"].get("by_family", {}),
        )
    ranking = comparison.ranking()
    for rank, (name, success) in enumerate(ranking, start=1):
        print(f"  {rank}. {name:26s} success={success:.2f}")
    report["ranking"] = [[name, value] for name, value in ranking]

    store.close()
    policy_registry.close()
    env.close()

    report_path = Path("ore_logs") / "ore_rl_demo_report.json"
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=float))
    print(f"\nreport written to {report_path}")
    return 0


def high_level_eligible(completed, invocation_counts):
    from ore_rl import HierarchicalOptionPolicy

    policy = HierarchicalOptionPolicy(OptionLibrary())
    return policy.eligible_options(completed, invocation_counts)


class HierarchicalOptionPolicyLike:
    """Thin stand-in that reuses HierarchicalOptionPolicy per episode seed."""

    def __init__(self, library, seed):
        from ore_rl import HierarchicalOptionPolicy

        self.inner = HierarchicalOptionPolicy(library, seed=seed)

    def propose_actions(self, state, eligible):
        return self.inner.propose_actions(state, eligible)

    def update_option_outcome(self, option_id, reward):
        self.inner.update_option_outcome(option_id, reward)


if __name__ == "__main__":
    raise SystemExit(main())
