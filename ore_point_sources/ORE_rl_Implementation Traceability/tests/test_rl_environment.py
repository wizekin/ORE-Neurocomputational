"""Deterministic Gymnasium-shaped ORE environment (roadmap Stage 3, 8.1)."""

from ore_contracts import ContractCatalog, ModuleRegistry

from ore_rl.environment import EnvironmentBudget, OREEnvironment, TaskSampler
from ore_rl.modules import register_rl_example_modules


def _env(seed: int = 5, **kwargs) -> OREEnvironment:
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_rl_example_modules(catalog, registry)
    defaults = dict(sampler=TaskSampler(seed=seed, n_tasks=6), seed=seed)
    defaults.update(kwargs)
    return OREEnvironment(registry=registry, catalog=catalog, **defaults)


def _run_plan(env: OREEnvironment, plan) -> tuple:
    state, info = env.reset()
    rewards = []
    terminal_kind = ""
    for action in plan:
        state, reward, terminated, truncated, info = env.step(action)
        rewards.append(round(reward, 6))
        if info.get("rejected"):
            state, reward, terminated, truncated, info = env.step("meta.stop_insufficient_evidence")
            rewards.append(round(reward, 6))
            break
        if terminated or truncated:
            break
    terminal_kind = info.get("terminal_kind", "")
    return tuple(rewards), terminal_kind, info


def _golden_plan() -> list:
    return [
        "run.ore.rl.input_analyzer@depth4",
        "run.ore.rl.knowledge_retriever@depth4",
        "run.ore.rl.hypothesis_generator@depth4",
        "run.ore.rl.verifier@depth4",
        "meta.stop_success",
    ]


def test_reset_and_step_follow_the_gymnasium_contract():
    """reset(seed) -> (obs, info); step -> (obs, reward, terminated, truncated, info)."""
    env = _env()
    state, info = env.reset()
    assert hasattr(state, "state_id") and "feasible_action_ids" in info
    result = env.step("run.ore.rl.input_analyzer@depth3")
    assert len(result) == 5
    observation, reward, terminated, truncated, info = result
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert info["outcome"]["status"] == "SUCCESS"
    env.close()


def test_environment_is_deterministic_for_a_fixed_seed():
    """Stage 3 exit: deterministic reset/step behavior under a fixed seed."""
    first = _run_plan(_env(seed=11), _golden_plan())
    second = _run_plan(_env(seed=11), _golden_plan())
    assert first[:2] == second[:2]
    env_a = _env(seed=11)
    env_b = _env(seed=11)
    state_a = env_a.reset()[0]
    state_b = env_b.reset()[0]
    assert state_a.state_id == state_b.state_id
    env_a.close()
    env_b.close()


def test_full_pipeline_through_the_real_coordinator_succeeds():
    """Module actions execute via CognitiveTaskCoordinator and verify the subject."""
    env = _env()
    rewards, terminal_kind, info = _run_plan(env, _golden_plan())
    assert terminal_kind == "STOP_SUCCESS"
    record = env.episode_history()[-1]
    assert record["success"]
    assert record["verified_term"] == record["ground_truth"]
    assert record["module_calls"] == 4
    env.close()


def test_insufficient_depth_fails_verification_and_stops_prematurely():
    """Depth 1 cannot resolve the subject: safe termination with premature label."""
    env = _env()
    plan = [
        "run.ore.rl.input_analyzer@depth1",
        "run.ore.rl.knowledge_retriever@depth1",
        "run.ore.rl.hypothesis_generator@depth1",
        "run.ore.rl.verifier@depth1",
        "meta.stop_success",
    ]
    rewards, terminal_kind, info = _run_plan(env, plan)
    record = env.episode_history()[-1]
    assert not record["success"]
    assert record["stop_label"] == "premature"
    assert info.get("stop_label") == "premature"
    env.close()


def test_outcome_statuses_are_distinct_and_measured():
    """Step 20: statuses are distinct; measured costs recorded, not just estimates."""
    env = _env()
    env.reset()
    _, _, _, _, info = env.step("run.ore.rl.input_analyzer@depth2")
    assert info["outcome"]["status"] == "SUCCESS"
    assert info["outcome"]["measured_latency_ms"] > 0
    assert info["outcome"]["measured_cpu_cost"] > 0
    # ineligible action -> deterministic rejection without execution
    _, _, _, _, info = env.step("run.ore.rl.verifier@depth2")
    assert info["outcome"]["status"] == "INVALID_OUTPUT"
    assert info["rejected"]
    env.close()


def test_budget_exhaustion_truncates_with_defer_semantics():
    """Hard budgets truncate the episode (resource exhaustion is its own outcome)."""
    env = _env(budget=EnvironmentBudget(
        latency_ms=500.0, cpu_cost=60.0, memory_mb=64.0, max_module_calls=2, max_steps=20,
    ))
    env.reset()
    _, _, terminated, truncated, info = env.step("run.ore.rl.input_analyzer@depth3")
    assert not terminated and not truncated
    result = env.step("run.ore.rl.knowledge_retriever@depth3")
    assert result[3]  # module-call budget exhausted -> truncated
    record = env.episode_history()[-1]
    assert record["terminal_kind"] == "TRUNCATED_BUDGET"
    env.close()


def test_injected_module_failure_is_reported_as_exception():
    """Failure injection surfaces as EXCEPTION, and downstream steps become invalid."""
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_rl_example_modules(
        catalog, registry, failure_injection={"ore.rl.knowledge_retriever": 1.1}
    )
    env = OREEnvironment(
        registry=registry, catalog=catalog,
        sampler=TaskSampler(seed=5, n_tasks=6), seed=5,
    )
    env.reset()
    env.step("run.ore.rl.input_analyzer@depth3")
    _, _, _, _, info = env.step("run.ore.rl.knowledge_retriever@depth3")
    assert info["outcome"]["status"] == "EXCEPTION"
    _, _, _, _, info = env.step("run.ore.rl.hypothesis_generator@depth3")
    assert info["outcome"]["status"] == "INVALID_OUTPUT"
    env.close()


def test_stop_and_defer_outcomes_are_distinguishable():
    """Acceptance 8.1: stop, defer, review, and truncation are distinct terminal kinds."""
    env = _env()
    env.reset()
    _, _, terminated, _, info = env.step("meta.stop_insufficient_evidence")
    assert terminated and info["terminal_kind"] == "STOP_INSUFFICIENT_EVIDENCE"
    env.reset()
    _, _, terminated, _, info = env.step("meta.defer_resource_limit")
    assert terminated and info["terminal_kind"] == "DEFER_RESOURCE_LIMIT"
    env.reset()
    _, _, terminated, _, info = env.step("meta.request_human_review")
    assert terminated and info["terminal_kind"] == "REQUEST_HUMAN_REVIEW"
    env.close()


def test_concurrent_batch_runs_independent_operations_in_one_step():
    """Step 15: independent eligible operations run concurrently (latency = max)."""
    env = _env()
    env.reset()
    env.step("run.ore.rl.input_analyzer@depth3")
    env.step("run.ore.rl.knowledge_retriever@depth3")
    env.step("run.ore.rl.hypothesis_generator@depth3")
    _, reward, _, _, info = env.step(("meta.concurrent_batch", {"members": (
        "run.ore.rl.causal_analyzer@depth3", "run.ore.rl.verifier@depth3",
    )}))
    assert info["outcome"]["status"] in ("SUCCESS", "PARTIAL")
    expected_latency = max(18.0 + 4.0 * 3, 30.0 + 7.0 * 3)
    assert abs(info["measured_latency_ms"] - expected_latency) < 1e-6
    assert "cognition.rl.verification" in env._episode.completed_capabilities
    env.close()


def test_adjust_depth_is_parameter_validated():
    """Step 16: contradictory or out-of-range depth parameters are rejected."""
    env = _env()
    env.reset()
    try:
        env.build_action("meta.adjust_depth", target_depth=3, depth_delta=1)
        accepted = True
    except ValueError:
        accepted = False
    assert not accepted
    try:
        env.build_action("meta.adjust_depth", target_depth=9)
        accepted = True
    except ValueError:
        accepted = False
    assert not accepted
    env.close()
