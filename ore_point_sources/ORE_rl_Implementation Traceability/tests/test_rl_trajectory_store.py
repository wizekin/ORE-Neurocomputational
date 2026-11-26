"""Trajectory store: integrity, quarantine, splits, simulated separation (Steps 29-33, 49)."""

import pytest

from ore_rl.contracts import (
    ActionOutcome,
    OREAction,
    OREState,
    OREReward,
    OutcomeStatus,
    RL_SCHEMA_VERSION,
)
from ore_rl.reward import RewardWeights
from ore_rl.trajectory_store import SplitRule, TrajectoryStore


def _state(index: int) -> OREState:
    return OREState(
        state_id=f"s{index}",
        task_id=f"task_{index}",
        task_family="fact_verification",
        workspace_id="ws",
        workspace_revision=index,
        encoder_version="1.0.0",
        feature_names=("f",),
        features={"f": float(index)},
        episode_id="ep",
    )


def _action(index: int) -> OREAction:
    return OREAction(action_id=f"run.mod@depth{index % 5 + 1}")


def _reward(total: float, *, verified: bool = True) -> OREReward:
    return OREReward(
        schema_version=RL_SCHEMA_VERSION, total=total,
        weights_version="reward-weights-1.0.0", evaluator_id="e", evaluator_version="1",
        verified=verified,
    )


def _episode(store: TrajectoryStore, episode_id: str, steps: int, *, source: str = "case_a") -> None:
    store.begin_episode(
        episode_id=episode_id, task_id=f"task_{episode_id}", task_family="fact_verification",
        environment_version="test", run_seed=1, source_case=source, session_id="sess-1",
    )
    for step in range(steps):
        store.record_step(
            episode_id=episode_id, step_index=step, state=_state(step), action=_action(step),
            outcome=ActionOutcome(status=OutcomeStatus.SUCCESS),
            next_state=_state(step + 1) if step + 1 < steps else None,
            terminal=step + 1 == steps, reward=_reward(0.1 * step),
            eligible_action_ids=[_action(step).action_id, "meta.stop_insufficient_evidence"],
            policy_decision={"policy_id": "p", "policy_version": "1",
                             "chosen_action_id": _action(step).action_id, "ranked": []},
        )
    store.finish_episode(episode_id, status="COMPLETE")


def test_transitions_link_episode_state_action_outcome_and_next_state():
    """Acceptance 8.2: every transition links all record families (unless terminal)."""
    with TrajectoryStore(":memory:") as store:
        _episode(store, "ep1", 3)
        validation = store.validate_episode("ep1")
        assert validation["valid"], validation["problems"]
        assert store.counts()["steps"] == 3
        assert store.counts()["rewards"] == 3
        assert store.counts()["decisions"] == 3


def test_integrity_chain_detects_tampering():
    """Append-only hash chain: any edited row breaks validation."""
    with TrajectoryStore(":memory:") as store:
        _episode(store, "ep1", 3)
        store._connection.execute(
            "UPDATE rl_steps SET measured_latency_ms = 999.0 WHERE step_index = 1"
        )
        store._connection.commit()
        validation = store.validate_episode("ep1")
        assert not validation["valid"]
        assert any(
            "integrity" in problem or "measured cost columns diverge" in problem
            for problem in validation["problems"]
        )


def test_unverified_rewards_are_quarantined_not_stored():
    """Acceptance 8.1: unverified reward evidence cannot enter the trusted training set."""
    with TrajectoryStore(":memory:") as store:
        store.begin_episode(episode_id="ep_bad", task_id="t", task_family="f",
                            environment_version="test")
        with pytest.raises(ValueError):
            store.record_step(
                episode_id="ep_bad", step_index=0, state=_state(0), action=_action(0),
                outcome=ActionOutcome(status=OutcomeStatus.SUCCESS),
                next_state=_state(1), terminal=False, reward=_reward(0.5, verified=False),
                eligible_action_ids=[_action(0).action_id],
            )
        quarantine = store.list_quarantine()
        assert len(quarantine) == 1
        assert quarantine[0]["reason"] == "unverified_reward"
        assert store.counts()["steps"] == 0


def test_reward_recomputation_matches_within_tolerance():
    """Acceptance 8.2: stored components recompute to the recorded total."""
    weights = RewardWeights()
    with TrajectoryStore(":memory:") as store:
        store.begin_episode(episode_id="ep_r", task_id="t", task_family="f",
                            environment_version="test")
        from ore_rl.reward import recompute_total

        reward = OREReward(
            schema_version=RL_SCHEMA_VERSION,
            quality_delta=0.3, evidence_delta=0.1, resource_cost=0.05,
            latency_cost=0.02, failure_cost=0.0,
            weights_version=weights.version, evaluator_id="e", evaluator_version="1",
            verified=True,
        )
        reward = OREReward(
            schema_version=RL_SCHEMA_VERSION,
            quality_delta=0.3, evidence_delta=0.1, resource_cost=0.05,
            latency_cost=0.02, failure_cost=0.0,
            total=recompute_total(reward, weights),
            weights_version=weights.version, evaluator_id="e", evaluator_version="1",
            verified=True,
        )
        store.record_step(
            episode_id="ep_r", step_index=0, state=_state(0), action=_action(0),
            outcome=ActionOutcome(status=OutcomeStatus.SUCCESS),
            next_state=_state(1), terminal=False, reward=reward,
            eligible_action_ids=[_action(0).action_id],
        )
        validation = store.validate_episode(
            "ep_r", weights_by_version={weights.version: weights}
        )
        assert validation["valid"], validation["problems"]


def test_leakage_safe_splits_and_reproducible_manifests():
    """Acceptance 8.2: task/source-level splits; manifest checksums reproduce the split."""
    with TrajectoryStore(":memory:") as store:
        _episode(store, "ep_a1", 2, source="case_a")
        _episode(store, "ep_a2", 2, source="case_a")
        _episode(store, "ep_b1", 3, source="case_b")
        train = store.build_dataset(
            split_name="train",
            rule=SplitRule(SplitRule.BY_SOURCE_CASE, ["case_a"], label="train"),
            include_groups=["case_a"],
        )
        test = store.build_dataset(
            split_name="test",
            rule=SplitRule(SplitRule.BY_SOURCE_CASE, ["case_b"], label="test"),
            include_groups=["case_b"],
        )
        assert set(train.episode_ids) == {"ep_a1", "ep_a2"}
        assert set(test.episode_ids) == {"ep_b1"}
        assert not set(train.episode_ids) & set(test.episode_ids)
        assert store.verify_manifest(train.dataset_id)
        assert store.verify_manifest(test.dataset_id)
        replay = store.load_replay_dataset(train.dataset_id)
        assert len(replay) == 4
        assert all(item.done is False or item.terminal for item in replay)
        bandit = store.load_bandit_dataset(train.dataset_id)
        assert len(bandit) == 4
        assert bandit[0].action_id.startswith("run.")


def test_simulated_transitions_are_tagged_and_excluded_by_default():
    """Acceptance 8.1 / Step 49: predicted transitions never mix into trusted data."""
    with TrajectoryStore(":memory:") as store:
        _episode(store, "ep_obs", 2, source="case_a")
        store.begin_episode(episode_id="ep_sim", task_id="t2", task_family="f",
                            environment_version="test", source_case="case_sim")
        store.record_step(
            episode_id="ep_sim", step_index=0, state=_state(0), action=_action(0),
            outcome=ActionOutcome(status=OutcomeStatus.SUCCESS),
            next_state=_state(1), terminal=False, reward=_reward(0.4),
            eligible_action_ids=[_action(0).action_id], origin="SIMULATED",
        )
        store.finish_episode("ep_sim", status="COMPLETE")
        sim_manifest = store.build_dataset(
            split_name="all",
            rule=SplitRule(SplitRule.BY_SOURCE_CASE, ["case_a", "case_sim"], label="all"),
            include_groups=["case_a", "case_sim"],
        )
        default = store.load_replay_dataset(sim_manifest.dataset_id)
        assert all("ep_sim" not in item.action_id for item in default)
        rows = store._connection.execute(
            "SELECT origin FROM rl_steps WHERE episode_id = 'ep_sim'"
        ).fetchall()
        assert rows[0]["origin"] == "SIMULATED"
        with_sim = store.load_replay_dataset(sim_manifest.dataset_id, include_simulated=True)
        assert len(with_sim) == len(default) + 1


def test_recorded_action_outside_eligible_mask_is_flagged():
    """Acceptance 8.2: the action mask stored with each step is checked in validation."""
    with TrajectoryStore(":memory:") as store:
        store.begin_episode(episode_id="ep_mask", task_id="t", task_family="f",
                            environment_version="test")
        store.record_step(
            episode_id="ep_mask", step_index=0, state=_state(0), action=_action(7),
            outcome=ActionOutcome(status=OutcomeStatus.SUCCESS),
            next_state=_state(1), terminal=False, reward=_reward(0.2),
            eligible_action_ids=["meta.stop_insufficient_evidence"],
        )
        store.finish_episode("ep_mask", status="COMPLETE")
        validation = store.validate_episode("ep_mask")
        assert not validation["valid"]
        assert any("eligible mask" in problem for problem in validation["problems"])


def test_state_refs_are_immutable_and_deduplicated():
    """Large artifacts are stored once by immutable reference (Step 30)."""
    with TrajectoryStore(":memory:") as store:
        _episode(store, "ep_d1", 2)
        _episode(store, "ep_d2", 2)
        counts = store.counts()
        assert counts["state_refs"] < counts["steps"]
        restored = store.state_by_ref("s0")
        assert restored is not None and restored.task_id == "task_0"
        assert store.state_by_ref("missing") is None
