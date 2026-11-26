"""Resource reservations through CROS (Point 14, Step 11).

The router requests CPU, memory, accelerator, storage, and concurrency budgets from
CROS/AdaptiveResourceCoordinator before anything executes.  When resources are
unavailable the task defers, downscales, selects a smaller model, or falls back to
a deterministic alternative -- never bypassing resource coordination.  The port is
CROS-compatible: deployments can wire a real coordinator behind the same protocol.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import time
from typing import Any, Protocol, runtime_checkable

from ..config import LearningConfig
from ..contracts import ModelManifest, ResourceProfile

__all__ = ("CROSPort", "LocalResourceCoordinator", "ReservationLease",
           "ResourceReservation", "DenyingResourceCoordinator", "reserve_for_model")


@dataclass(frozen=True, slots=True)
class ResourceReservation:
    """The outcome of one CROS reservation attempt."""

    reservation_id: str
    granted: bool
    reason: str
    units: float = 0.0
    profile: ResourceProfile | None = None


class ReservationLease:
    """An acquired reservation that must be released (timeouts release theirs)."""

    def __init__(self, reservation: ResourceReservation,
                 release_hook: Any = None) -> None:
        """Wrap a granted reservation."""
        self.reservation = reservation
        self._release_hook = release_hook
        self.released = False
        self.released_at: float | None = None

    def release(self, reason: str = "completed") -> bool:
        """Release the reservation; idempotent."""
        if self.released:
            return False
        self.released = True
        self.released_at = time.time()
        if self._release_hook is not None:
            self._release_hook(self.reservation.reservation_id, reason)
        return True

    def describe(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"reservation_id": self.reservation.reservation_id,
                "granted": self.reservation.granted, "units": self.reservation.units,
                "released": self.released, "released_at": self.released_at}


@runtime_checkable
class CROSPort(Protocol):
    """The port through which the RLMS requests resources from CROS."""

    def request(self, model_id: str, profile: ResourceProfile, *,
                margin: float) -> ResourceReservation: ...

    def release(self, reservation_id: str, reason: str = "completed") -> bool: ...


class LocalResourceCoordinator:
    """A local, deterministic CROS-compatible coordinator used when CROS is absent.

    Tracks a declared capacity and the set of active reservations; requests are
    granted only when the peak (plus safety margin) fits the remaining capacity.
    """

    def __init__(self, cpu_cores: float = 8.0, memory_mb: float = 8192.0,
                 accelerator_mb: float = 0.0, storage_mb: float = 4096.0,
                 max_concurrency: int = 4) -> None:
        """Declare the available capacity."""
        self.capacity = {"cpu_cores": float(cpu_cores), "memory_mb": float(memory_mb),
                         "accelerator_mb": float(accelerator_mb),
                         "storage_mb": float(storage_mb),
                         "concurrency": int(max_concurrency)}
        self._active: dict[str, ResourceProfile] = {}
        self._ids = itertools.count(1)
        self.release_log: list[tuple[str, str, float]] = []

    def _remaining(self) -> dict[str, float]:
        """Capacity remaining after active reservations."""
        remaining = dict(self.capacity)
        for profile in self._active.values():
            remaining["cpu_cores"] -= profile.cpu_cores
            remaining["memory_mb"] -= profile.memory_mb
            remaining["accelerator_mb"] -= profile.accelerator_mb
            remaining["storage_mb"] -= profile.storage_mb
            remaining["concurrency"] -= 1
        return remaining

    def request(self, model_id: str, profile: ResourceProfile, *,
                margin: float) -> ResourceReservation:
        """Grant when peak usage plus safety margin fits the remaining capacity."""
        reservation_id = f"cros-{next(self._ids):05d}"
        remaining = self._remaining()
        demand = {"cpu_cores": profile.cpu_cores * (1.0 + margin),
                  "memory_mb": profile.memory_mb * (1.0 + margin),
                  "accelerator_mb": profile.accelerator_mb,
                  "storage_mb": profile.storage_mb * (1.0 + margin)}
        for resource, required in demand.items():
            if required > remaining[resource] + 1e-9:
                return ResourceReservation(
                    reservation_id=reservation_id, granted=False,
                    reason=(f"insufficient {resource}: need {required:.2f}, "
                            f"have {remaining[resource]:.2f}"))
        if remaining["concurrency"] < 1:
            return ResourceReservation(
                reservation_id=reservation_id, granted=False,
                reason="concurrency budget exhausted")
        self._active[reservation_id] = profile
        return ResourceReservation(
            reservation_id=reservation_id, granted=True,
            reason="reserved through the local CROS-compatible coordinator",
            units=demand["cpu_cores"] + demand["memory_mb"] / 1024.0, profile=profile)

    def release(self, reservation_id: str, reason: str = "completed") -> bool:
        """Release an active reservation."""
        if reservation_id in self._active:
            del self._active[reservation_id]
            self.release_log.append((reservation_id, reason, time.time()))
            return True
        return False

    def describe(self) -> dict[str, Any]:
        """Summarize capacity and active reservations."""
        return {"capacity": dict(self.capacity), "remaining": self._remaining(),
                "active_reservations": len(self._active),
                "releases": list(self.release_log)}


class DenyingResourceCoordinator:
    """A coordinator that always denies: exercises deferral and fallback paths."""

    def __init__(self, reason: str = "CROS reports no capacity for this request") -> None:
        """Configure the denial reason."""
        self.reason = reason
        self.requests: list[str] = []

    def request(self, model_id: str, profile: ResourceProfile, *,
                margin: float) -> ResourceReservation:
        """Deny every request deterministically."""
        self.requests.append(model_id)
        return ResourceReservation(reservation_id="cros-denied", granted=False,
                                   reason=self.reason)

    def release(self, reservation_id: str, reason: str = "completed") -> bool:
        """Nothing to release."""
        return False


def reserve_for_model(coordinator: CROSPort, manifest: ModelManifest,
                      config: LearningConfig | None = None) -> ReservationLease:
    """Reserve resources for one manifest or raise with the refusal reason.

    Step 11 never bypasses resource coordination: when the request is refused the
    caller must defer, downscale, select a smaller model, or use a deterministic
    fallback.
    """
    effective_config = config or LearningConfig()
    reservation = coordinator.request(
        manifest.model_id, manifest.resource_profile,
        margin=effective_config.resource_safety_margin)
    if not reservation.granted:
        raise RuntimeError(f"resource reservation refused: {reservation.reason}")
    return ReservationLease(
        reservation, release_hook=lambda reservation_id, reason:
        coordinator.release(reservation_id, reason))
