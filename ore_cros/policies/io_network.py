"""I/O, networking and storage policies (ORE Point 23, Steps 42-46).

Covers I/O and network demand profiling, asynchronous I/O backend
selection with a portable fallback, authorization gates for specialized
kernel technologies (eBPF/XDP/DPDK/Demikernel), a storage capability
matrix and durability contracts that resource optimization may never
silently downgrade.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from ..actions import ActionType, ResourceAction
from ..resource_model import ResourceKind
from ..telemetry.schemas import Measurement, MetricKind


@dataclass(frozen=True)
class BottleneckProfile:
    """I/O and network bottleneck profile (spec Step 42)."""

    bottleneck: str
    evidence: tuple[str, ...]
    p99_latency_seconds: float | None = None
    queue_depth: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the profile."""
        return {
            "bottleneck": self.bottleneck, "evidence": list(self.evidence),
            "p99_latency_seconds": self.p99_latency_seconds,
            "queue_depth": self.queue_depth,
        }


class IoNetworkProfiler:
    """Profile I/O and network demand from telemetry (Step 42)."""

    @staticmethod
    def profile(measurements: Sequence[Measurement]) -> BottleneckProfile:
        """Identify the dominant bottleneck before any optimization."""
        by_key: dict[tuple[str, str], list[float]] = {}
        for item in measurements:
            by_key.setdefault((item.resource.value, item.metric.value), []).append(item.value)

        def latest(kind: ResourceKind, metric: MetricKind) -> float | None:
            values = by_key.get((kind.value, metric.value))
            return values[-1] if values else None

        evidence: list[str] = []
        io_util = latest(ResourceKind.IO, MetricKind.UTILIZATION)
        net_util = latest(ResourceKind.NET, MetricKind.UTILIZATION)
        queue_depth = latest(ResourceKind.QUEUE, MetricKind.QUEUE_DEPTH)
        latency = latest(ResourceKind.IO, MetricKind.LATENCY_SECONDS)
        if io_util is not None and io_util >= 0.9:
            evidence.append(f"IO utilization {io_util:.2f} sustained near saturation")
        if net_util is not None and net_util >= 0.9:
            evidence.append(f"NET utilization {net_util:.2f} sustained near saturation")
        if queue_depth is not None and queue_depth > 0:
            evidence.append(f"queue depth {queue_depth:.0f}")
        if latency is not None and latency > 0.05:
            evidence.append(f"I/O latency {latency:.3f}s above 50ms threshold")
        if io_util is not None and io_util >= (net_util or 0.0):
            bottleneck = "IO" if io_util is not None and io_util > 0.5 else "NONE"
        else:
            bottleneck = "NET"
        if not evidence:
            bottleneck = "NONE"
        return BottleneckProfile(
            bottleneck=bottleneck, evidence=tuple(evidence),
            p99_latency_seconds=latency, queue_depth=queue_depth,
        )


class AsyncIoBackend:
    """Asynchronous I/O backend selection with portable fallback (Step 43).

    io_uring is selected only when the platform supports it and a measured
    benchmark shows a real gain; otherwise the portable fallback is used.
    Unsupported interfaces fall back cleanly rather than failing.
    """

    PORTABLE_FALLBACK = "portable-blocking"
    IO_URING = "io_uring"

    def __init__(self, *, measured_gain: float | None = None, gain_threshold: float = 0.10) -> None:
        """Configure the measured benchmark gain and selection threshold."""
        if measured_gain is not None and (
            not math.isfinite(measured_gain) or measured_gain < 0.0
        ):
            raise ValueError("measured_gain must be finite and non-negative")
        self.measured_gain = measured_gain
        self.gain_threshold = gain_threshold

    def select_backend(self, *, io_uring_supported: bool) -> tuple[str, str]:
        """Return (backend, rationale) for this platform and workload."""
        if not io_uring_supported:
            return (
                self.PORTABLE_FALLBACK,
                "io_uring unsupported on this platform; portable fallback selected",
            )
        if self.measured_gain is None:
            return (
                self.PORTABLE_FALLBACK,
                "io_uring supported but no benchmark evidence; portable fallback kept "
                "until measured gains justify the complexity",
            )
        if self.measured_gain < self.gain_threshold:
            return (
                self.PORTABLE_FALLBACK,
                f"measured gain {self.measured_gain:.2f} below threshold "
                f"{self.gain_threshold:.2f}; complexity not justified",
            )
        return (
            self.IO_URING,
            f"measured gain {self.measured_gain:.2f} exceeds threshold; io_uring selected",
        )


class SpecializedNetworkGate:
    """Authorization gate for specialized kernel technologies (Step 44).

    eBPF, XDP, DPDK and Demikernel require explicit evidence: a real
    workload requirement, in-house expertise and benchmark data. Without
    all three the gate denies authorization with a rationale.
    """

    AUTHORIZED_TOOLS = frozenset({"eBPF", "XDP", "DPDK", "Demikernel"})

    def __init__(
        self,
        *,
        evidence: Mapping[str, bool] | None = None,
        authorized_tools: Mapping[str, bool] | None = None,
    ) -> None:
        """Configure the standing evidence flags per tool."""
        self.evidence = dict(evidence or {})
        self.authorized_tools = dict(authorized_tools or {})

    def authorize(self, tool: str) -> tuple[bool, str]:
        """Return (authorized, rationale) for one specialized tool."""
        if tool not in self.AUTHORIZED_TOOLS:
            return False, f"unknown specialized tool {tool!r}"
        if self.authorized_tools.get(tool):
            return True, f"{tool} previously authorized with evidence on file"
        requirements = {
            "workload_requirement": "a measured workload requirement",
            "expertise": "in-house expertise to operate it safely",
            "benchmark": "benchmark data showing ordinary paths are insufficient",
        }
        missing = [
            description for flag, description in requirements.items()
            if not self.evidence.get(flag)
        ]
        if missing:
            return False, f"{tool} denied: missing {missing}"
        return True, f"{tool} authorized: workload, expertise and benchmark evidence present"


#: Storage capability matrix rows (spec Step 45).
STORAGE_CANDIDATES: tuple[dict[str, Any], ...] = (
    {
        "filesystem": "Btrfs", "strengths": ("snapshots", "checksums", "copy-on-write"),
        "suitable_for": ("snapshot-heavy workspaces",),
    },
    {
        "filesystem": "XFS", "strengths": ("high-throughput parallel writes",),
        "suitable_for": ("large batch datasets",),
    },
    {
        "filesystem": "ZFS/OpenZFS", "strengths": ("integrity", "replication", "snapshots"),
        "suitable_for": ("integrity-critical stores",),
    },
    {
        "filesystem": "EROFS", "strengths": ("read-only", "compressed images"),
        "suitable_for": ("immutable model images",),
    },
    {
        "filesystem": "F2FS", "strengths": ("flash-friendly",),
        "suitable_for": ("flash-backed caches",),
    },
    {
        "filesystem": "zonefs", "strengths": ("zoned devices",),
        "suitable_for": ("zoned storage hardware",),
    },
    {
        "filesystem": "NOVA/Mnemosyne (research)", "strengths": ("persistent memory",),
        "suitable_for": ("research deployments only",),
    },
)


class StorageCapabilityMatrix:
    """Select storage architecture by workload requirement (Step 45).

    Returns a capability matrix, not a one-filesystem-fits-all assumption:
    callers state requirements and receive the ranked candidates.
    """

    def recommend(self, requirements: Sequence[str]) -> tuple[dict[str, Any], ...]:
        """Return candidates whose strengths match the stated requirements."""
        wanted = {item.lower() for item in requirements}
        if not wanted:
            return ()
        scored: list[tuple[int, dict[str, Any]]] = []
        for candidate in STORAGE_CANDIDATES:
            haystack = {
                *candidate["strengths"], *candidate["suitable_for"],
                candidate["filesystem"].lower(),
            }
            score = sum(1 for item in wanted if any(item in entry for entry in haystack))
            if score > 0:
                scored.append((score, candidate))
        scored.sort(key=lambda pair: (-pair[0], pair[1]["filesystem"]))
        return tuple(candidate for _, candidate in scored)


@dataclass(frozen=True)
class DurabilityContract:
    """Durability and recovery requirements of one data class (Step 46)."""

    data_class: str
    durability: str  # "none" | "fsync" | "replicated"
    consistency: str  # "best-effort" | "atomic" | "strict"
    backup: bool
    recovery_point_objective_seconds: float
    recovery_time_objective_seconds: float

    def __post_init__(self) -> None:
        """Validate the contract fields."""
        if not isinstance(self.data_class, str) or not self.data_class.strip():
            raise ValueError("data_class must be a non-empty string")
        if self.durability not in ("none", "fsync", "replicated"):
            raise ValueError("durability must be one of none/fsync/replicated")
        if self.consistency not in ("best-effort", "atomic", "strict"):
            raise ValueError("consistency must be best-effort/atomic/strict")
        for name in (
            "recovery_point_objective_seconds", "recovery_time_objective_seconds",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the contract."""
        return {
            "data_class": self.data_class, "durability": self.durability,
            "consistency": self.consistency, "backup": self.backup,
            "recovery_point_objective_seconds": self.recovery_point_objective_seconds,
            "recovery_time_objective_seconds": self.recovery_time_objective_seconds,
        }


#: Ordering used to detect silent durability downgrades.
_DURABILITY_RANK = {"none": 0, "fsync": 1, "replicated": 2}
_CONSISTENCY_RANK = {"best-effort": 0, "atomic": 1, "strict": 2}


class DurabilityContractRegistry:
    """Registry enforcing that optimization never downgrades durability (Step 46)."""

    def __init__(self, contracts: Sequence[DurabilityContract] | None = None) -> None:
        """Register one contract per data class."""
        self._contracts: dict[str, DurabilityContract] = {}
        for contract in contracts or ():
            if contract.data_class in self._contracts:
                raise ValueError(f"duplicate durability contract for {contract.data_class!r}")
            self._contracts[contract.data_class] = contract

    def contract(self, data_class: str) -> DurabilityContract | None:
        """Return the contract registered for one data class."""
        return self._contracts.get(data_class)

    def validate_transition(
        self, data_class: str, proposed: DurabilityContract,
    ) -> tuple[bool, str]:
        """Reject any transition that weakens durability or consistency."""
        current = self._contracts.get(data_class)
        if current is None:
            return True, "no prior contract; transition accepted"
        if _DURABILITY_RANK[proposed.durability] < _DURABILITY_RANK[current.durability]:
            return (
                False,
                f"resource optimization cannot downgrade durability of {data_class!r} "
                f"from {current.durability} to {proposed.durability}",
            )
        if _CONSISTENCY_RANK[proposed.consistency] < _CONSISTENCY_RANK[current.consistency]:
            return (
                False,
                f"resource optimization cannot downgrade consistency of {data_class!r} "
                f"from {current.consistency} to {proposed.consistency}",
            )
        if proposed.backup and not current.backup:
            return True, "backup strengthened; accepted"
        if current.backup and not proposed.backup:
            return False, f"backup cannot be removed from {data_class!r}"
        return True, "transition preserves durability contracts"

    def validate_action(self, action: ResourceAction) -> tuple[bool, str]:
        """Validate a resource action against every registered contract."""
        if action.action_type is not ActionType.CHANGE_CACHE_TIER:
            return True, "action does not touch storage durability"
        affected = action.parameters.get("affected_data_classes", ())
        for data_class in affected:
            contract = self._contracts.get(data_class)
            if contract is None:
                continue
            proposed = DurabilityContract(
                data_class=data_class, durability="none", consistency="best-effort",
                backup=False, recovery_point_objective_seconds=0.0,
                recovery_time_objective_seconds=0.0,
            )
            allowed, reason = self.validate_transition(data_class, proposed)
            if not allowed:
                return False, reason
        return True, "no durability contract violated"


__all__ = [
    "IoNetworkProfiler", "BottleneckProfile", "AsyncIoBackend", "SpecializedNetworkGate",
    "StorageCapabilityMatrix", "STORAGE_CANDIDATES", "DurabilityContract",
    "DurabilityContractRegistry",
]
