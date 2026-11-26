"""RLVR: rewards from verifiable outcomes, kept separate from correctness (Steps 35-36).

    R(tau) = sum_i lambda_i r_i^step + lambda_f r^final - lambda_v p^violation

Rewards are constructed from validator outputs (unit tests, exact solver agreement,
constraint satisfaction), are reproducible from those outputs, and remain training
signals -- never authoritative facts.  A learned reward model cannot approve its own
output: reward identity, validator versions, and environments are preserved with
every reward record.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Sequence

from ..config import LearningConfig
from ..contracts import ValidationReport
from .process import ReasoningTrace, StepStatus
from .validators import apply_validators

__all__ = ("RewardHackingError", "RewardRecord", "compute_rlvr_reward",
           "replay_reward", "reward_from_validators")


@dataclass(frozen=True, slots=True)
class RewardRecord:
    """Step 36 -- a reward with full provenance, separate from correctness records."""

    task_id: str
    reward: float
    step_rewards: tuple[float, ...]
    final_reward: float
    violation_penalty: float
    validator_identity: tuple[str, ...]
    test_versions: tuple[str, ...]
    environment: str
    digest: str
    note: str = "a reward is a training signal, not an authoritative fact"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"task_id": self.task_id, "reward": round(self.reward, 6),
                "step_rewards": [round(value, 6) for value in self.step_rewards],
                "final_reward": round(self.final_reward, 6),
                "violation_penalty": round(self.violation_penalty, 6),
                "validator_identity": list(self.validator_identity),
                "test_versions": list(self.test_versions),
                "environment": self.environment, "digest": self.digest,
                "note": self.note}


class RewardHackingError(ValueError):
    """Raised when a reward is constructed outside its verifiable provenance."""


def reward_from_validators(trace: ReasoningTrace, report: ValidationReport,
                           *, final_outcome: bool | None = None,
                           config: LearningConfig | None = None,
                           environment: str = "ore_learning.rewards@1.0.0") -> RewardRecord:
    """Compute R(tau) from validator outputs only (Step 35).

    Every step reward comes from a deterministic validator result; the final term is
    the held-out final-outcome check; violations subtract the configured penalty.
    PRM scores are deliberately ignored here -- they are screening signals, not
    verifiable outcomes.
    """
    effective = config or LearningConfig()
    if len(report.checks) != len(trace.steps):
        # checks and steps can diverge only through bugs; refuse silently-mixed rewards
        raise RewardHackingError("validation report does not align with the trace")
    step_rewards: list[float] = []
    validator_identity: list[str] = []
    versions: list[str] = []
    for step in trace.steps:
        if step.validator_result == "pass":
            step_rewards.append(1.0)
        elif step.validator_result == "fail":
            step_rewards.append(-1.0)
        else:
            step_rewards.append(0.0)
        validator_identity.append(step.validator_name or "none")
        versions.append(step.validator_result)
    if final_outcome is None:
        final_reward = 1.0 if trace.verified else 0.0
    else:
        final_reward = 1.0 if final_outcome else -1.0
    violations = sum(1 for step in trace.steps if step.status is StepStatus.FAILED)
    violation_penalty = float(violations)
    total = (effective.rlvr_step_weight * sum(step_rewards)
             + effective.rlvr_final_weight * final_reward
             - effective.rlvr_violation_weight * violation_penalty)
    digest = _reward_digest(trace.task_id, step_rewards, final_reward,
                            violation_penalty, validator_identity, versions)
    return RewardRecord(
        task_id=trace.task_id, reward=round(total, 9),
        step_rewards=tuple(step_rewards), final_reward=final_reward,
        violation_penalty=violation_penalty,
        validator_identity=tuple(validator_identity), test_versions=tuple(versions),
        environment=environment, digest=digest)


def compute_rlvr_reward(trace: ReasoningTrace,
                        payloads: dict[str, dict[str, Any]] | None = None,
                        *, config: LearningConfig | None = None,
                        final_outcome: bool | None = None) -> RewardRecord:
    """Validate the trace deterministically, then construct the reward (Step 35)."""
    report = apply_validators(trace, payloads)
    return reward_from_validators(trace, report, final_outcome=final_outcome,
                                  config=config)


def replay_reward(record: RewardRecord, trace: ReasoningTrace,
                  payloads: dict[str, dict[str, Any]] | None = None,
                  *, config: LearningConfig | None = None,
                  final_outcome: bool | None = None) -> bool:
    """Reproducibility check: recomputing the reward from validators matches (Step 35).

    Acceptance criterion: "RLVR reward is reproducible from validator outputs."
    """
    recomputed = compute_rlvr_reward(trace, payloads, config=config,
                                     final_outcome=final_outcome)
    return recomputed.digest == record.digest and recomputed.reward == record.reward


def _reward_digest(task_id: str, step_rewards: Sequence[float], final_reward: float,
                   violation_penalty: float, validator_identity: Sequence[str],
                   versions: Sequence[str]) -> str:
    """A stable digest over everything the reward depends on."""
    payload = json.dumps({
        "task_id": task_id, "step_rewards": list(step_rewards),
        "final_reward": final_reward, "violation_penalty": violation_penalty,
        "validator_identity": list(validator_identity), "versions": list(versions)},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
