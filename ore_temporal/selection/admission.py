"""Resource admission before any weights are loaded (specification Step 6)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..contracts import AdmissionDecision
from ..models import family_descriptions

__all__ = ("LocalResourceCoordinator", "ResourceAdmission", "ResourceCoordinatorPort")


@runtime_checkable
class ResourceCoordinatorPort(Protocol):
    """The port through which TEDIL queries resource coordination (CROS-compatible)."""

    def estimate_cost(self, family: str, n_observations: int, n_channels: int) -> float: ...


@dataclass(frozen=True, slots=True)
class LocalResourceCoordinator:
    """A local, deterministic resource estimator used when CROS is not wired in.

    Estimated cost is expressed in normalized resource units: roughly the number of
    scalar operations the family needs per observation, scaled by state size.
    """

    cost_per_observation: dict[str, float] | None = None

    def __post_init__(self) -> None:
        """Default per-family costs when not provided."""
        object.__setattr__(self, "cost_per_observation", self.cost_per_observation or {
            "baseline": 8.0,
            "lnn": 64.0,
            "cfc": 32.0,
            "neural_ode": 128.0,
            "ssm": 24.0,
            "snn": 48.0,
        })

    def estimate_cost(self, family: str, n_observations: int, n_channels: int) -> float:
        """Estimate the normalized cost of running one family over one window."""
        table = self.cost_per_observation or {}
        per_observation = table.get(family, 64.0)
        return float(per_observation * max(n_observations, 1) * max(n_channels, 1))


class ResourceAdmission:
    """Step 6 -- admission, deferral, downsampling, and baseline fallback decisions."""

    def __init__(self, coordinator: ResourceCoordinatorPort | None = None) -> None:
        """Configure the coordinator port; a local estimator is used when absent."""
        self.coordinator = coordinator or LocalResourceCoordinator()

    def admit(self, family: str, n_observations: int, n_channels: int, *,
              resource_budget: float, max_ingest_observations: int,
              neural_enabled: bool = True) -> AdmissionDecision:
        """Decide admission before loading any model weights."""
        estimated = self.coordinator.estimate_cost(family, n_observations, n_channels)
        budget = max(float(resource_budget), 0.0)
        cost_cap = budget * max(max_ingest_observations, 1) * 128.0
        if family != "baseline" and not neural_enabled:
            return AdmissionDecision(
                admitted=False, reason="neural families disabled by feature flag",
                fallback_to_baseline=True, estimated_cost=estimated, budget=cost_cap)
        if estimated <= cost_cap:
            return AdmissionDecision(
                admitted=True, reason="estimated cost within the declared budget",
                estimated_cost=estimated, budget=cost_cap)
        downsample_factor = estimated / cost_cap if cost_cap > 0 else float("inf")
        if downsample_factor <= 4.0:
            return AdmissionDecision(
                admitted=True,
                reason=("estimated cost exceeds the budget; window downsampled by factor "
                        f"{downsample_factor:.2f} under the documented policy"),
                downsample=True, estimated_cost=estimated, budget=cost_cap)
        return AdmissionDecision(
            admitted=False,
            reason=("estimated cost exceeds the budget by more than the downsample policy "
                    "allows; deferring to the deterministic baseline"),
            deferred=True, fallback_to_baseline=True,
            estimated_cost=estimated, budget=cost_cap)

    def describe(self) -> dict[str, Any]:
        """Describe the admission configuration."""
        return {
            "coordinator": type(self.coordinator).__name__,
            "family_costs": dict(getattr(self.coordinator, "cost_per_observation", {})
                                 or {}),
            "family_descriptions": family_descriptions(),
        }
