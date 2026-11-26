"""Leases with explicit clock assumptions and fencing tokens (Step 21).

Leases reduce coordination overhead, but a lease timeout alone is not a safe
proof that an old owner has stopped. Every ownership-sensitive write therefore
carries a monotonically increasing fencing epoch that the resource checks:
a stale lease holder is rejected by fencing-token validation.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Mapping


class FencingRejected(RuntimeError):
    """Raised when a write carries a stale owner or fencing epoch."""


def _default_clock() -> float:
    """Return the monotonic clock used for lease expiry."""
    return time.monotonic()


@dataclass(frozen=True, slots=True)
class LeaseGrant:
    """One issued lease with owner, expiry, and fencing epoch."""

    resource: str
    owner: str
    fencing_epoch: int
    granted_at: float
    expires_at: float

    @property
    def valid_for_seconds(self) -> float:
        """Return remaining lease validity under the issuing clock."""
        return max(0.0, self.expires_at - self.granted_at)

    def to_dict(self) -> dict[str, object]:
        """Return the grant as a plain dictionary."""
        return {
            "resource": self.resource,
            "owner": self.owner,
            "fencing_epoch": self.fencing_epoch,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
        }


@dataclass(frozen=True, slots=True)
class LeasePolicy:
    """Explicit clock-drift assumptions and renewal rules for leases (Step 21)."""

    lease_duration: float = 5.0
    clock_drift_bound: float = 0.25
    renewal_margin: float = 0.5
    safety_multiplier: float = 2.0

    def __post_init__(self) -> None:
        """Validate that clock assumptions stay conservative."""
        for name in ("lease_duration", "clock_drift_bound", "renewal_margin", "safety_multiplier"):
            value = float(getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.clock_drift_bound >= self.lease_duration:
            raise ValueError("clock drift bound must be smaller than the lease duration")
        if self.renewal_margin >= self.lease_duration:
            raise ValueError("renewal margin must be smaller than the lease duration")

    def effective_duration(self) -> float:
        """Return the duration minus the drift safety margin."""
        return max(0.0, self.lease_duration - self.clock_drift_bound * self.safety_multiplier)

    def should_renew(self, grant: LeaseGrant, now: float) -> bool:
        """Return whether a lease is inside its renewal window."""
        return (grant.expires_at - now) <= self.renewal_margin


class LeaseManager:
    """Issue leases whose ownership-sensitive writes are fenced by epoch.

    The manager tracks the highest issued fencing epoch per resource. An old
    owner whose lease appears to have timed out may still be alive; the
    resource boundary therefore validates the fencing epoch on every write and
    rejects writes from stale holders.
    """

    def __init__(
        self,
        policy: LeasePolicy | None = None,
        clock: Callable[[], float] = _default_clock,
    ) -> None:
        """Create a manager with an injectable clock for deterministic tests."""
        self.policy = policy or LeasePolicy()
        self._clock = clock
        self._grants: dict[str, LeaseGrant] = {}
        self._fencing_epochs: dict[str, int] = {}
        self.stale_rejections = 0
        self.renewals = 0

    def current_epoch(self, resource: str) -> int:
        """Return the highest fencing epoch issued for a resource."""
        return self._fencing_epochs.get(resource, 0)

    def grant(
        self,
        resource: str,
        owner: str,
        duration: float | None = None,
    ) -> LeaseGrant:
        """Grant or renew a lease, always advancing the fencing epoch."""
        if not resource.strip() or not owner.strip():
            raise ValueError("resource and owner must be non-empty strings")
        now = self._clock()
        effective = self.policy.effective_duration() if duration is None else float(duration)
        if effective <= 0.0:
            raise ValueError("effective lease duration must be positive")
        epoch = self.current_epoch(resource) + 1
        grant = LeaseGrant(
            resource=resource,
            owner=owner,
            fencing_epoch=epoch,
            granted_at=now,
            expires_at=now + effective,
        )
        self._fencing_epochs[resource] = epoch
        self._grants[resource] = grant
        self.renewals += 1
        return grant

    def active_grant(self, resource: str) -> LeaseGrant | None:
        """Return the unexpired grant for a resource, if any."""
        grant = self._grants.get(resource)
        if grant is None:
            return None
        if self._clock() >= grant.expires_at:
            return None
        return grant

    def validate_write(
        self,
        resource: str,
        owner: str,
        fencing_epoch: int,
        *,
        now: float | None = None,
    ) -> bool:
        """Check an ownership-sensitive write's fencing epoch at the boundary."""
        now = self._clock() if now is None else now
        highest = self.current_epoch(resource)
        if fencing_epoch < highest:
            self.stale_rejections += 1
            raise FencingRejected(
                f"stale fencing epoch {fencing_epoch} for {resource!r}; current epoch is {highest}"
            )
        grant = self._grants.get(resource)
        if grant is None or grant.owner != owner:
            self.stale_rejections += 1
            raise FencingRejected(
                f"owner {owner!r} does not hold the lease for {resource!r}"
            )
        if now >= grant.expires_at:
            # Lease timeout alone never authorizes the old owner to write.
            self.stale_rejections += 1
            raise FencingRejected(
                f"lease for {resource!r} expired; renew before writing"
            )
        return True

    def reassign_after_expiry(self, resource: str, new_owner: str) -> LeaseGrant | None:
        """Reassign a resource only after its lease has truly expired.

        Reassignment issues a strictly higher fencing epoch, so a stale worker
        holding an old epoch cannot overwrite newer results.
        """
        grant = self._grants.get(resource)
        if grant is not None and self._clock() < grant.expires_at:
            return None
        return self.grant(resource, new_owner)


def lease_clock_drift_bound(drift_per_second: float, observed_seconds: float) -> float:
    """Return the explicit drift bound over an observation window."""
    if drift_per_second < 0.0 or observed_seconds < 0.0:
        raise ValueError("drift and observation window must be non-negative")
    return drift_per_second * observed_seconds


__all__ = [
    "FencingRejected",
    "LeaseGrant",
    "LeasePolicy",
    "LeaseManager",
    "lease_clock_drift_bound",
]
