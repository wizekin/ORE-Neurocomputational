"""Acceptance tests for constrained learning: bandit, shadow, canary (Steps 50-51)."""
import pytest

from ore_cros.config import AdaptationLimits, CROSConfig
from ore_cros.learning.bandit import (
    ConstrainedContextualBandit, ConstrainedRlResearchPath, replay_offline,
)
from ore_cros.cycle import CROSCoordinator
from ore_cros.resource_model import ResourceDemand, TaskRecord, WorkloadClass
from ore_cros.telemetry.collectors import CollectorSet, SimulatedCollector

APPROVED = ("keep-current", "defer-optional-work", "scale-workers-up")
FEATURES = ("cpu_pressure", "ram_pressure", "io_pressure", "net_pressure",
            "queue_depth", "worker_load")


def _bandit(**overrides):
    """Build a constrained contextual bandit with test defaults."""
    defaults = dict(
        approved_actions=APPROVED, feature_names=FEATURES,
        epsilon=0.0, seed=11, shadow=True,
    )
    defaults.update(overrides)
    return ConstrainedContextualBandit(**defaults)


def _context(cpu=0.9, queue=0.5):
    """Return one bandit context vector."""
    return {
        "cpu_pressure": cpu, "ram_pressure": 0.2, "io_pressure": 0.1,
        "net_pressure": 0.1, "queue_depth": queue, "worker_load": 0.5,
    }


def test_bandit_selects_only_from_approved_safe_actions():
    """Acceptance: the bandit selects only from the approved safe action set."""
    bandit = _bandit()
    context = _context()
    action, propensity = bandit.propose(context, currently_safe=APPROVED[:2])
    assert action in APPROVED[:2]
    assert 0.0 < propensity <= 1.0
    # The approved set minus the currently-safe subset is never chosen.
    for _ in range(20):
        action, _ = bandit.propose(context, currently_safe=["keep-current"])
        assert action == "keep-current"
    with pytest.raises(ValueError):
        bandit.propose(context, currently_safe=["not-a-real-action"])


def test_bandit_updates_value_estimates_from_rewards():
    """Rewards update the linear value estimates per action (Step 50)."""
    bandit = _bandit(shadow=False)
    context = _context()
    bandit.update("defer-optional-work", 1.0, context=context)
    bandit.update("defer-optional-work", 1.0, context=context)
    counts = bandit.counts()
    assert counts["defer-optional-work"] == 2
    # A constraint-violating outcome never reinforces the policy.
    bandit.update("defer-optional-work", 5.0, context=context, constraints_satisfied=False)
    assert bandit.counts()["defer-optional-work"] == 2
    with pytest.raises(ValueError):
        bandit.update("not-approved", 1.0, context=context)


def test_bandit_logging_records_context_propensity_outcome():
    """Context, action, propensity, outcome and constraints are logged (Step 50)."""
    bandit = _bandit(shadow=False)
    bandit.propose(_context(), currently_safe=APPROVED, timestamp=10.0)
    bandit.update("defer-optional-work", 0.7, context=_context())
    records = bandit.log_records()
    assert len(records) == 2
    assert records[0].shadow is False
    assert records[0].propensity > 0.0
    assert records[1].reward == 0.7
    assert records[1].constraints_satisfied


def test_shadow_mode_never_changes_allocation():
    """Acceptance: learned actions cannot change allocation in shadow mode."""
    bandit = _bandit(shadow=True)  # shadow mode
    collector_set = CollectorSet([SimulatedCollector(seed=7, load_profile=lambda s: 0.9)])
    coordinator = CROSCoordinator(
        config=CROSConfig(), collector_set=collector_set, bandit=bandit,
    )
    task = TaskRecord(
        task_id="t1", parent_goal_id=None, module_id="m1",
        workload_class=WorkloadClass.CORE_COGNITIVE,
        demand=ResourceDemand(cpu=0.1), remaining_work=1.0,
    )
    before = coordinator.control.to_dict()
    coordinator.adaptive_resource_cycle([task], now=1_000.0)
    after = coordinator.control.to_dict()
    # Whatever the shadow proposal was, only deterministic decisions applied.
    assert coordinator.shadow_evaluator.dataset()
    assert coordinator.bandit.log_records()
    applied = {
        key: (before[key], after[key]) for key in before
        if before[key] != after[key]
    }
    # Any applied change came from the deterministic selection, verified by
    # the decision trace's policy_origin never being LEARNED.
    assert all(
        record.policy_origin in ("DETERMINISTIC", "BASELINE")
        for record in coordinator.decisions
    )


def test_canary_fraction_limits_learned_control():
    """Acceptance: learned control can be limited to a configured canary."""
    import dataclasses

    collector_set = CollectorSet([SimulatedCollector(seed=7, load_profile=lambda s: 0.9)])
    # canary_fraction=0 (the default): learned proposals are never used.
    coordinator = CROSCoordinator(
        config=CROSConfig(), collector_set=collector_set,
        bandit=_bandit(shadow=False),
    )
    assert coordinator._canary_active() is False

    # canary_fraction=1: the learned proposal participates in every cycle.
    config_on = dataclasses.replace(
        CROSConfig(), limits=AdaptationLimits(canary_fraction=1.0),
    )
    coordinator_on = CROSCoordinator(
        config=config_on, collector_set=collector_set,
        bandit=_bandit(shadow=False), seed=3,
    )
    assert coordinator_on._canary_active() is True


def test_offline_replay_uses_inverse_propensity_weighting():
    """Offline replay evaluates counterfactual policies from logged data."""
    bandit = _bandit(shadow=False)
    for _ in range(5):
        bandit.propose(_context(), currently_safe=APPROVED)
        bandit.update("defer-optional-work", 1.0, context=_context())
    result = replay_offline(bandit.log_records(), target_action_id="defer-optional-work")
    assert result["estimate"] == pytest.approx(1.0)
    assert result["effective_samples"] > 0
    empty = replay_offline([], target_action_id="defer-optional-work")
    assert empty["estimate"] == 0.0


def test_rl_research_path_refuses_training_without_prerequisites():
    """Step 51: constrained RL stays gated behind attested prerequisites."""
    research = ConstrainedRlResearchPath()
    assert research.missing_prerequisites() == research.PREREQUISITES
    with pytest.raises(NotImplementedError):
        research.train()
    partial = ConstrainedRlResearchPath({"reliable_telemetry": True})
    assert "reliable_telemetry" not in partial.missing_prerequisites()
    with pytest.raises(NotImplementedError):
        partial.train()


def test_bandit_constructor_validation():
    """Bandit configuration rejects empty and duplicate action sets."""
    with pytest.raises(ValueError):
        ConstrainedContextualBandit(approved_actions=[], feature_names=FEATURES)
    with pytest.raises(ValueError):
        ConstrainedContextualBandit(
            approved_actions=["a", "a"], feature_names=FEATURES,
        )
    with pytest.raises(ValueError):
        ConstrainedContextualBandit(
            approved_actions=APPROVED, feature_names=(), epsilon=0.5,
        )
