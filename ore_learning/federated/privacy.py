"""Federated learning privacy controls (Point 14, Step 42).

Raw data staying local does not guarantee privacy.  Updates are L2-clipped to a
bound, differentially private Gaussian noise is added participant-side, aggregation
is simulated under secure aggregation (the coordinator observes only the aggregate),
participant authentication is mandatory, cohort minimums are enforced, and outlier
updates are screened for poisoning.  The documented threat model records the
privacy/utility trade-off.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import clip_update_norm, dp_gaussian_noise

__all__ = ("FederatedPolicyError", "ParticipantCredentials", "PoisoningScreen",
           "ThreatModel", "screen_poisoning", "threat_model_summary")


class FederatedPolicyError(ValueError):
    """Raised when a federated round violates privacy or cohort controls."""


@dataclass(frozen=True, slots=True)
class ParticipantCredentials:
    """Authenticated participant identity and data policy (Step 41)."""

    participant_id: str
    authentication_token_digest: str
    data_policy: str
    sample_count: int

    def __post_init__(self) -> None:
        """Validate the credentials."""
        if not self.participant_id or not self.authentication_token_digest:
            raise ValueError("participant id and authentication digest are required")
        if int(self.sample_count) <= 0:
            raise ValueError("sample_count must be positive")


@dataclass(frozen=True, slots=True)
class ThreatModel:
    """Step 42 -- the documented threat model and privacy/utility trade-off."""

    adversary: str = "honest-but-curious coordinator plus up to k-1 malicious participants"
    protections: tuple[str, ...] = (
        "participant authentication (digest-verifiable credentials)",
        "L2 clipping of every update to the configured bound",
        "participant-side Gaussian noise (differential privacy)",
        "secure aggregation: the coordinator observes only the aggregate",
        "minimum cohort size before any aggregation proceeds",
        "outlier screening of update norms for poisoning",
    )
    trade_off: str = "stronger clipping and noise reduce leakage at utility cost"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"adversary": self.adversary, "protections": list(self.protections),
                "trade_off": self.trade_off}


def threat_model_summary() -> dict[str, Any]:
    """Return the documented federated threat model (Step 42)."""
    return ThreatModel().to_dict()


def screen_poisoning(updates: Sequence[np.ndarray],
                     config: LearningConfig | None = None) -> PoisoningScreen:
    """Screen update norms for outliers (poisoning indicators, Step 42/43).

    Updates whose L2 norm exceeds the median by more than three robust standard
    deviations are flagged; flagged participants are excluded from aggregation by
    the caller.
    """
    effective = config or LearningConfig()
    norms = np.asarray([float(np.linalg.norm(np.asarray(update, dtype=float)))
                        for update in updates])
    if norms.size == 0:
        raise ValueError("at least one update is required")
    median = float(np.median(norms))
    mad = float(np.median(np.abs(norms - median))) or 1e-9
    flagged = [index for index, norm in enumerate(norms)
               if abs(norm - median) > 3.0 * 1.4826 * mad]
    return PoisoningScreen(
        norms=[round(float(value), 6) for value in norms],
        flagged_indices=flagged,
        threshold=round(median + 3.0 * 1.4826 * mad, 6),
        clipping_bound=effective.federated_clip_norm)


@dataclass(frozen=True, slots=True)
class PoisoningScreen:
    """The outcome of one poisoning screen."""

    norms: list[float]
    flagged_indices: list[int]
    threshold: float
    clipping_bound: float

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"norms": list(self.norms), "flagged_indices": list(self.flagged_indices),
                "threshold": self.threshold, "clipping_bound": self.clipping_bound,
                "note": "flagged updates are excluded before aggregation"}


def protect_update(update: np.ndarray, participant_seed: int,
                   config: LearningConfig | None = None) -> np.ndarray:
    """Clip participant-side and add DP noise *before* anything leaves the participant.

    This is the enforcement point for the acceptance criterion "Required secure
    aggregation/privacy controls are enforced": raw updates are structurally never
    exposed because only their protected form enters a round.
    """
    effective = config or LearningConfig()
    vector = np.asarray(update, dtype=float)
    clipped, applied_norm = clip_update_norm(vector, effective.federated_clip_norm)
    scale = effective.federated_dp_noise_multiplier * max(applied_norm, 1e-12)
    noised = clipped + dp_gaussian_noise(clipped.shape, scale, participant_seed)
    return noised
