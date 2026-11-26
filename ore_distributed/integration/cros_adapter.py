"""CROS resource telemetry and bounded adaptation of distributed work (Step 47).

CROS monitors network, CPU, memory, storage, and queue pressure and may adjust
replication batch sizes, gossip frequency, and task placement within safe
policy bounds; resource adaptation must never violate consensus or
data-integrity invariants, so consensus safety parameters stay outside the
adaptable set unless reconfigured through authorized configuration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ore_contracts.resource_profiles import CapabilityDemand


#: Adaptation knobs CROS may tune within safe policy bounds.
ADAPTABLE_PARAMETERS = frozenset(
    {
        "replication_batch_size",
        "gossip_frequency_ticks",
        "gossip_fanout",
        "task_placement_domain",
    }
)

#: Safety-critical consensus parameters CROS must never touch (Step 47).
PROTECTED_PARAMETERS = frozenset(
    {
        "quorum_size",
        "election_timeout",
        "commit_rule",
        "lease_fencing_policy",
        "membership_configuration",
    }
)

#: Bounds for every adaptable parameter.
PARAMETER_BOUNDS: Mapping[str, tuple[int, int]] = {
    "replication_batch_size": (1, 64),
    "gossip_frequency_ticks": (1, 64),
    "gossip_fanout": (1, 8),
    "task_placement_domain": (0, 4),
}


@dataclass(frozen=True, slots=True)
class ResourceTelemetry:
    """One sampled resource-pressure snapshot (Step 47)."""

    network_pressure: float
    cpu_pressure: float
    memory_pressure: float
    storage_pressure: float
    queue_depth: int

    def __post_init__(self) -> None:
        """Validate pressure samples."""
        for name in ("network_pressure", "cpu_pressure", "memory_pressure", "storage_pressure"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.queue_depth < 0:
            raise ValueError("queue_depth must be non-negative")

    @property
    def overall_pressure(self) -> float:
        """Return the maximum pressure across sampled resources."""
        return max(
            self.network_pressure,
            self.cpu_pressure,
            self.memory_pressure,
            self.storage_pressure,
        )

    def to_demand(self) -> CapabilityDemand:
        """Convert telemetry into the repo's CROS capability-demand record."""
        return CapabilityDemand(
            capability_id="ore.distributed.resource_pressure",
            priority=round(min(1.0, self.overall_pressure), 4),
            mandatory=False,
            target_confidence=round(1.0 - self.overall_pressure, 4),
        )


class CROSResourceAdapter:
    """Bounded, invariant-preserving resource adaptation (Step 47)."""

    def __init__(
        self,
        *,
        initial: Mapping[str, int] | None = None,
    ) -> None:
        """Create an adapter with an initial adaptable-parameter set."""
        self._parameters: dict[str, int] = {
            "replication_batch_size": 8,
            "gossip_frequency_ticks": 4,
            "gossip_fanout": 2,
            "task_placement_domain": 1,
        }
        for key, value in (initial or {}).items():
            self.set_parameter(key, value)
        self.telemetry_samples: list[ResourceTelemetry] = []
        self.adaptations: list[tuple[str, int, int]] = []
        self.violations_blocked = 0

    def parameters(self) -> dict[str, int]:
        """Return the current adaptable parameter values."""
        return dict(self._parameters)

    def set_parameter(self, name: str, value: int) -> None:
        """Set one adaptable parameter inside its safe bounds only."""
        if name in PROTECTED_PARAMETERS:
            self.violations_blocked += 1
            raise ValueError(
                f"{name!r} is a consensus safety parameter; CROS adaptation may "
                "not change it outside authorized configuration (Step 47)"
            )
        if name not in ADAPTABLE_PARAMETERS:
            raise ValueError(f"unknown parameter {name!r}")
        low, high = PARAMETER_BOUNDS[name]
        if not low <= int(value) <= high:
            raise ValueError(f"{name} must stay within [{low}, {high}]")
        self._parameters[name] = int(value)

    def record_telemetry(self, telemetry: ResourceTelemetry) -> None:
        """Record one telemetry sample."""
        self.telemetry_samples.append(telemetry)

    def adapt(self, telemetry: ResourceTelemetry) -> dict[str, int]:
        """Adjust distributed-work parameters within bounds (Step 47)."""
        self.record_telemetry(telemetry)
        pressure = telemetry.overall_pressure
        previous_batch = self._parameters["replication_batch_size"]
        previous_frequency = self._parameters["gossip_frequency_ticks"]
        if pressure > 0.75:
            # Under heavy pressure, shrink batches and gossip less often.
            new_batch = max(1, previous_batch // 2)
            new_frequency = min(
                PARAMETER_BOUNDS["gossip_frequency_ticks"][1], previous_frequency * 2
            )
        elif pressure < 0.25:
            # With headroom, restore defaults.
            new_batch = min(16, previous_batch * 2)
            new_frequency = max(1, previous_frequency // 2)
        else:
            new_batch = previous_batch
            new_frequency = previous_frequency
        for name, value in (
            ("replication_batch_size", new_batch),
            ("gossip_frequency_ticks", new_frequency),
        ):
            if self._parameters[name] != value:
                self.adaptations.append((name, self._parameters[name], value))
                self._parameters[name] = value
        return self.parameters()

    def try_violate_safety(self, name: str, value: int) -> bool:
        """Attempt to change a protected parameter; returns whether it was blocked."""
        try:
            self.set_parameter(name, value)
            return False
        except ValueError:
            return True

    def queue_pressure(self) -> float:
        """Return the latest queue pressure ratio for observability."""
        if not self.telemetry_samples:
            return 0.0
        return self.telemetry_samples[-1].overall_pressure


def cros_cannot_change_consensus_safety(adapter: CROSResourceAdapter) -> bool:
    """Run the acceptance check that CROS cannot alter consensus safety."""
    blocked = adapter.try_violate_safety("quorum_size", 1)
    blocked = adapter.try_violate_safety("election_timeout", 0) and blocked
    blocked = adapter.try_violate_safety("commit_rule", 0) and blocked
    return blocked


__all__ = [
    "ADAPTABLE_PARAMETERS",
    "PROTECTED_PARAMETERS",
    "PARAMETER_BOUNDS",
    "ResourceTelemetry",
    "CROSResourceAdapter",
    "cros_cannot_change_consensus_safety",
]
