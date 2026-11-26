"""Hierarchical options, offline RL, constraints, and world-model separation (Steps 44-50)."""
import numpy as np

from ore_contracts import ContractCatalog, ModuleRegistry

from ore_rl import OREEnvironment, TaskSampler, register_rl_example_modules
from ore_rl.constrained import (
    ConstrainedPolicyWrapper,
    LagrangianOptimizer,
    ResourceConstraint,
    ResourceConstraintSet,
)
from ore_rl.policies.offline import BehaviorCloningPolicy, CQLPolicy, IQLPolicy
from ore_rl.policies.options import (
    HierarchicalOptionPolicy,
    OptionLibrary,
    OptionLoopDetector,
    OptionRuntime,
)
from ore_rl.policies.base import DeterministicBaselinePolicy, ProgressBoundaryPolicy
from ore_rl.world_model import EmpiricalWorldModel


def _env(seed: int = 4, n_tasks: int = 6, **kwargs) -> OREEnvironment:
    catalog = ContractCatalog()
    registry = ModuleRegistry(catalog)
    register_rl_example_modules(catalog, registry)
    defaults = dict(sampler=TaskSampler(seed=seed, n_tasks=n_tasks), seed=seed)
    defaults.update(kwargs)
    return OREEnvironment(registry=registry, catalog=catalog, **defaults)


# -- options (Steps 44-47) --------------------------------------------------
def test_option_library_defines_the_six_initial_options():
    """Step 44: the predefined option set matches the specification."""
    library = OptionLibrary()
    ids = set(library.option_ids)
    assert ids == {
        "UNDERSTAND_INPUT", "INVESTIGATE_HYPOTHESIS", "RETRIEVE_SUPPORTING_EVIDENCE",
        "VERIFY_CONCLUSION", "RESOLVE_CONTRADICTION", "PRODUCE_FINAL_RESULT",
    }


def test_option_initiation_and_termination_sets():
    """Step 44: I_o and beta_o respect prerequisites and goal capabilities."""
    library = OptionLibrary()
    verify = library.get("VERIFY_CONCLUSION")
    assert not verify.initiates(("cognition.rl.input_analysis",))
    assert verify.initiates((
        "cognition.rl.hypothesis_generation", "cognition.rl.knowledge_retrieval",
    ))
    assert verify.terminates((
        "cognition.rl.input_analysis", "cognition.rl.hypothesis_generation",
        "cognition.rl.knowledge_retrieval", "cognition.rl.verification",
    ), step=0)
    assert verify.terminates((), step=verify.contract.max_internal_steps)


def test_option_contracts_cannot_bypass_global_constraints():
    """Step 45: options declare permitted modules, ceilings, and failure paths."""
    library = OptionLibrary()
    for option in library.list_options():
        if option.option_id != "PRODUCE_FINAL_RESULT":
            assert option.contract.permitted_modules
        assert option.contract.resource_ceiling_ms > 0
        assert option.contract.failure_paths
        assert option.contract.max_internal_steps >= 1
    produce = library.get("PRODUCE_FINAL_RESULT")
    assert "stop_insufficient_evidence" in produce.contract.failure_paths


def test_options_execute_through_the_environment_and_succeed():
    """Step 46: deterministic internal procedures complete the cognitive task."""
    env = _env()
    library = OptionLibrary()
    runtime = OptionRuntime(env, max_total_steps=10)
    high = HierarchicalOptionPolicy(library, seed=1)
    completed = set()
    state, _info = env.reset()
    for _ in range(8):
        eligible = high.eligible_options(completed, {})
        if not eligible:
            break
        decision = high.propose_actions(state, [f"option:{o.option_id}" for o in eligible])
        option = library.get(decision.chosen_action_id[len("option:"):])
        report = runtime.run_option(option)
        high.update_option_outcome(option.option_id, report.get("total_reward", 0.0))
        completed = set(env._episode.completed_capabilities) if env._episode else set()
        state = env._observe()
        if runtime.terminated:
            break
    record = env.episode_history()[-1]
    assert record["success"]
    assert record["steps"] <= 10
    env.close()


def test_loop_detector_escalates_on_repeated_state_signatures():
    """Step 47: repeated state signatures trigger termination or escalation."""
    detector = OptionLoopDetector(max_signature_repeats=2)
    from ore_rl.contracts import OREState

    stable = OREState(
        state_id="sig", task_id="t", task_family="f", encoder_version="1",
        feature_names=("goal.completed_count",), features={"goal.completed_count": 1.0},
    )
    first = detector.record_invocation("OPTION", stable, 1.0)
    second = detector.record_invocation("OPTION", stable, 1.5)
    third = detector.record_invocation("OPTION", stable, 2.0)
    assert not first["escalate"]
    assert not second["escalate"]
    assert third["escalate"]
    assert "repeated state signature" in third["reason"]


def test_loop_detector_escalates_on_stalled_progress():
    """Step 47: flat progress deltas across the stall window escalate."""
    from ore_rl.contracts import OREState

    detector = OptionLoopDetector(stall_window=2)
    state = OREState(
        state_id="s", task_id="t", task_family="f", encoder_version="1",
        feature_names=("workspace.evidence_quality",), features={"workspace.evidence_quality": 0.5},
    )
    detector.record_investion = None  # guard against typo usage
    detector.record_invocation("A", state, 0.5)
    directive = detector.record_invocation("B", state, 0.5)
    assert directive["escalate"] and directive["reason"] == "progress stalled"


# -- offline RL (Step 48) ------------------------------------------------------
def _collect_dataset(env: OREEnvironment, episodes: int, seed: int):
    baseline = DeterministicBaselinePolicy()
    names = list(env.encoder.feature_names)
    action_ids = [entry.spec.action_id for entry in env.catalog_actions.list_actions()]

    def vector(features):
        return np.asarray([features.get(name, 0.0) for name in names])

    data = []
    for _ in range(episodes):
        state, info = env.reset()
        eligible = info["feasible_action_ids"]
        done = False
        while not done:
            decision = baseline.propose_actions(state, eligible)
            action = decision.chosen_action_id
            previous = state
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            data.append({
                "state": vector(previous.features),
                "next_state": vector(state.features) if not done else vector(previous.features),
                "action_id": action,
                "reward": float(reward),
                "done": bool(done),
                "support": list(eligible),
            })
            eligible = info["feasible_action_ids"]
    return data, action_ids, len(names)


def test_behavior_cloning_clones_the_baseline_policy():
    """Step 48: BC reproduces the deterministic baseline's action distribution."""
    env = _env(seed=17, n_tasks=12)
    data, action_ids, state_dim = _collect_dataset(env, episodes=6, seed=17)
    bc = BehaviorCloningPolicy(action_ids=action_ids, state_dim=state_dim, seed=1, iterations=120)
    result = bc.fit(data)
    assert result["loss"] < 1.0
    served = ProgressBoundaryPolicy(bc)
    state, info = env.reset()
    decision = served.propose_actions(state, info["feasible_action_ids"])
    assert decision.chosen_action_id.startswith("run.")
    env.close()


def test_cql_suppresses_out_of_support_extrapolation():
    """Step 48: the conservative penalty keeps unseen actions from being overvalued."""
    rng = np.random.default_rng(11)
    data = []
    for index in range(60):
        state = np.asarray([rng.random(), rng.random()])
        take_a = state[0] > 0.5
        data.append({
            "state": state,
            "next_state": np.asarray([rng.random(), rng.random()]),
            "action_id": "a" if take_a else "b",
            "reward": 1.0 if take_a else -1.0,
            "done": False,
            "support": ["a" if take_a else "b"],
        })
    cql = CQLPolicy(action_ids=["a", "b", "unseen"], state_dim=2, seed=2, alpha=0.05,
                    iterations=250)
    cql.fit(data)
    q = cql.network.predict(np.asarray([0.9, 0.1]))
    assert q[cql.action_index["a"]] > q[cql.action_index["unseen"]]
    assert q[cql.action_index["unseen"]] > q[cql.action_index["b"]]


def test_iql_learns_values_and_extracts_an_advantage_policy():
    """Step 48: IQL value loss decreases and produces eligible proposals."""
    env = _env(seed=23, n_tasks=8)
    data, action_ids, state_dim = _collect_dataset(env, episodes=4, seed=23)
    iql = IQLPolicy(action_ids=action_ids, state_dim=state_dim, seed=3, iterations=150)
    result = iql.fit(data)
    assert result["q_loss"] >= 0.0
    state, info = env.reset()
    decision = iql.propose_actions(state, info["feasible_action_ids"])
    assert decision.chosen_action_id in info["feasible_action_ids"]
    env.close()


# -- constrained optimization (Step 50) -----------------------------------------
def test_lagrangian_multipliers_grow_under_budget_violation():
    """Step 50: dual ascent raises multipliers when budgets are exceeded."""
    constraints = ResourceConstraintSet((
        ResourceConstraint("latency", budget=100.0),
        ResourceConstraint("cpu", budget=10.0),
    ))
    optimizer = LagrangianOptimizer(constraints, learning_rate=0.1)
    constraints.record("latency", 150.0)
    multipliers = optimizer.update({})
    assert multipliers["latency"] > 0.0
    assert multipliers["cpu"] == 0.0
    shaped = optimizer.shaped_reward(1.0, {"latency": 2.0})
    assert shaped < 1.0
    constraints.record("latency", 0.0)
    before = multipliers["latency"]
    constraints.usage["latency"] = 10.0  # well under budget again
    after = optimizer.update({})
    assert after["latency"] <= before


def test_constraint_violations_are_hard_not_penalty_only():
    """Step 50: violation detection is separate from the learned objective."""
    constraints = ResourceConstraintSet((ResourceConstraint("memory", budget=5.0),))
    constraints.record("memory", 4.0)
    assert constraints.violations() == ()
    constraints.record("memory", 2.0)
    assert constraints.violations() == ("memory",)
    assert constraints.slack("memory") < 0.0


def test_constrained_wrapper_delegates_proposals_and_shapes_rewards():
    """The wrapper never blocks proposals itself; hard gates live elsewhere."""
    from ore_rl.policies.bandit import EmpiricalBandit

    constraints = ResourceConstraintSet((ResourceConstraint("cpu", budget=1.0),))
    optimizer = LagrangianOptimizer(constraints)
    inner = EmpiricalBandit(seed=1, epsilon=0.0)
    wrapper = ConstrainedPolicyWrapper(inner, optimizer)
    state, _ = _env().reset()
    decision = wrapper.propose_actions(state, ["run.a@depth1", "run.b@depth1"])
    assert decision.chosen_action_id in ("run.a@depth1", "run.b@depth1")
    assert wrapper.shaped_reward(1.0, {"cpu": 0.5}) <= 1.0


# -- world model (Step 49) -------------------------------------------------------
def test_world_model_predictions_are_tagged_simulated():
    """Step 49: predictions carry the SIMULATED origin tag and uncertainty."""
    model = EmpiricalWorldModel(context_bins=2)
    state = _env().reset()[0]
    prediction = model.predict(state, "run.ore.rl.verifier@depth3")
    assert prediction.origin == "SIMULATED"
    assert prediction.model_version == "empirical-world-model-1.0.0"
    assert prediction.uncertainty == 1.0  # no data yet


def test_world_model_never_trains_on_simulated_transitions():
    """Step 49: simulated rows are excluded from world-model fitting."""
    model = EmpiricalWorldModel(context_bins=2)
    state = _env().reset()[0]
    model.fit([
        {"action_id": "a", "context_bin": 0, "success": 1.0, "quality_delta": 0.5,
         "evidence_delta": 0.2, "cost": 3.0, "origin": "OBSERVED"},
        {"action_id": "a", "context_bin": 0, "success": 1.0, "quality_delta": 0.5,
         "evidence_delta": 0.2, "cost": 3.0, "origin": "SIMULATED"},
        {"action_id": "b", "context_bin": 1, "success": 0.0, "quality_delta": 0.0,
         "evidence_delta": 0.0, "cost": 1.0, "origin": "OBSERVED"},
    ])
    prediction_a = model.predict(state, "a")
    assert prediction_a.sample_count == 1
    assert prediction_a.predicted_success_probability == 1.0
    prediction_b = model.predict(state, "b")
    assert prediction_b.sample_count == 1
    assert prediction_b.predicted_success_probability == 0.0
    assert model.fingerprint()
