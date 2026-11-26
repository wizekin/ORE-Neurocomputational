"""Federated Learning Coordinator (Point 14, Steps 41 and 43).

The coordinator distributes a base model, collects eligible local updates from
authenticated participants, and aggregates them with FedAvg weighting

    w_{t+1} = sum_k (n_k / sum_j n_j) w_{t+1}^{(k)}

Aggregated models are validated independently before any redistribution or
activation: participant updates can never directly alter production state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import fedavg_weights
from .privacy import (FederatedPolicyError, ParticipantCredentials, PoisoningScreen,
                      protect_update, screen_poisoning, threat_model_summary)

__all__ = ("FederatedCoordinator", "FederatedRound", "ParticipantUpdate")


@dataclass(slots=True)
class ParticipantUpdate:
    """One protected local update from an authenticated participant."""

    participant_id: str
    sample_count: int
    protected_update: np.ndarray
    raw_update_exposed: bool = False  # structurally never True

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view (the update vector itself is never serialized)."""
        return {"participant_id": self.participant_id,
                "sample_count": self.sample_count,
                "update_norm": round(float(np.linalg.norm(self.protected_update)), 6),
                "raw_update_exposed": self.raw_update_exposed}


@dataclass(slots=True)
class FederatedRound:
    """The record of one aggregated federated round (Steps 41-43)."""

    round_id: int
    participants: list[str]
    sample_counts: list[int]
    aggregation_weights: list[float]
    excluded_participants: list[str]
    aggregate_norm: float
    validated: bool
    activated: bool
    threat_model: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"round_id": self.round_id,
                "participants": list(self.participants),
                "sample_counts": list(self.sample_counts),
                "aggregation_weights": [round(value, 6)
                                        for value in self.aggregation_weights],
                "excluded_participants": list(self.excluded_participants),
                "aggregate_norm": round(self.aggregate_norm, 6),
                "validated": self.validated, "activated": self.activated,
                "threat_model": self.threat_model}


class FederatedCoordinator:
    """Steps 41-43 -- optional distributed update orchestration under controls."""

    def __init__(self, config: LearningConfig | None = None) -> None:
        """Configure cohort, clipping, DP, and secure-aggregation policy."""
        self.config = config or LearningConfig()
        self.registered: dict[str, ParticipantCredentials] = {}
        self.rounds: list[FederatedRound] = []
        self.refusals: list[dict[str, Any]] = []

    def enroll(self, credentials: ParticipantCredentials) -> None:
        """Register an authenticated participant (Step 41)."""
        self.registered[credentials.participant_id] = credentials

    def submit_update(self, participant_id: str, authentication_token_digest: str,
                      sample_count: int, raw_update: np.ndarray, *,
                      participant_seed: int = 0) -> ParticipantUpdate:
        """Authenticate, then clip and noise the update before it is accepted.

        Unauthenticated participants are rejected with a recorded refusal; the raw
        update never crosses the participant boundary.
        """
        credentials = self.registered.get(participant_id)
        if credentials is None or credentials.authentication_token_digest != \
                authentication_token_digest:
            refusal = {"participant_id": participant_id,
                       "reason": "authentication failed; unauthenticated participants "
                                 "cannot contribute updates"}
            self.refusals.append(refusal)
            raise FederatedPolicyError(refusal["reason"])
        if int(sample_count) <= 0:
            raise FederatedPolicyError("sample_count must be positive")
        protected = protect_update(np.asarray(raw_update, dtype=float),
                                   participant_seed, self.config)
        return ParticipantUpdate(participant_id=participant_id,
                                 sample_count=int(sample_count),
                                 protected_update=protected)

    def aggregate(self, updates: Sequence[ParticipantUpdate],
                  validation_callback: Any = None) -> FederatedRound:
        """FedAvg aggregation with cohort, poisoning, and validation gates.

        - the cohort must meet the configured minimum after poisoning screening;
        - weights are the declared participant sample counts n_k / sum_j n_j;
        - secure aggregation is enforced when configured: this coordinator computes
          over protected updates only and refuses rounds that would expose raw ones;
        - the aggregate is marked validated only through the independent validation
          callback, and never activates directly.
        """
        if self.config.federated_require_secure_aggregation and any(
                getattr(update, "raw_update_exposed", False) for update in updates):
            raise FederatedPolicyError(
                "secure aggregation is required; raw updates cannot enter a round")
        excluded: list[str] = []
        eligible = list(updates)
        if len(updates) >= 5:
            # outlier screening is meaningful only with a cohort to compare against
            screen = screen_poisoning([update.protected_update for update in updates],
                                      self.config)
            excluded = [updates[index].participant_id
                        for index in screen.flagged_indices]
            eligible = [update for index, update in enumerate(updates)
                        if index not in set(screen.flagged_indices)]
            if excluded and len(eligible) < self.config.federated_min_cohort:
                raise FederatedPolicyError(
                    f"poisoning indicators exclude {len(excluded)} participants and "
                    f"the remaining cohort of {len(eligible)} is below the minimum "
                    f"{self.config.federated_min_cohort}; the round is refused")
        if len(eligible) < self.config.federated_min_cohort:
            raise FederatedPolicyError(
                f"cohort of {len(eligible)} below the configured minimum "
                f"{self.config.federated_min_cohort} after screening")
        weights = fedavg_weights([update.sample_count for update in eligible])
        aggregate = np.zeros_like(eligible[0].protected_update)
        for weight, update in zip(weights, eligible):
            aggregate = aggregate + weight * update.protected_update
        round_id = len(self.rounds) + 1
        validated = False
        if validation_callback is not None:
            validated = bool(validation_callback(aggregate))
        record = FederatedRound(
            round_id=round_id,
            participants=[update.participant_id for update in eligible],
            sample_counts=[update.sample_count for update in eligible],
            aggregation_weights=[float(value) for value in weights],
            excluded_participants=excluded,
            aggregate_norm=float(np.linalg.norm(aggregate)),
            validated=validated, activated=False,
            threat_model=threat_model_summary())
        self.rounds.append(record)
        return record

    def activate_aggregate(self, record: FederatedRound,
                           validation_callback: Any = None) -> FederatedRound:
        """Step 43 -- activation requires independent validation, never direct."""
        if not record.validated:
            if validation_callback is None or not bool(validation_callback(record)):
                raise FederatedPolicyError(
                    "aggregated model cannot activate without independent validation")
            record.validated = True
        record.activated = True
        return record

    def describe(self) -> dict[str, Any]:
        """Summarize coordinator state."""
        return {"registered_participants": sorted(self.registered),
                "rounds": len(self.rounds), "refusals": len(self.refusals),
                "min_cohort": self.config.federated_min_cohort,
                "clip_norm": self.config.federated_clip_norm,
                "dp_noise_multiplier": self.config.federated_dp_noise_multiplier,
                "secure_aggregation_required":
                    self.config.federated_require_secure_aggregation}
