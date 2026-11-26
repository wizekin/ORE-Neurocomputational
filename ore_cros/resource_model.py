"""Typed resources, capacities, allocations and budgets (ORE Point 23, Steps 1-2 and 47).

This module defines the unified resource-state model of section 2 of the
specification: the resource vector R_t, task demand vectors, workload and
criticality classes, and the versioned resource state that every other CROS
component reads. Unknown capacities are represented as ``None`` (unknown),
never as zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence


class ResourceKind(str, Enum):
    """The seven dimensions of the CROS resource vector (spec section 2.1)."""

    CPU = "CPU"
    RAM = "RAM"
    IO = "IO"
    NET = "NET"
    GPU = "GPU"
    STORAGE = "STORAGE"
    QUEUE = "QUEUE"


#: Task demand vectors exclude QUEUE; queue depth is a system observation.
DEMAND_KINDS: tuple[ResourceKind, ...] = (
    ResourceKind.CPU, ResourceKind.RAM, ResourceKind.IO,
    ResourceKind.NET, ResourceKind.GPU, ResourceKind.STORAGE,
)


class WorkloadClass(str, Enum):
    """Policy-controlled workload classes (spec section 2.2)."""

    INTEGRITY_CRITICAL = "INTEGRITY_CRITICAL"
    INTERACTIVE = "INTERACTIVE"
    CORE_COGNITIVE = "CORE_COGNITIVE"
    BATCH = "BATCH"
    EXPLORATORY = "EXPLORATORY"
    MAINTENANCE = "MAINTENANCE"


class CriticalityLevel(int, Enum):
    """Consequence-of-failure ordering; smaller values are more critical (Step 47)."""

    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    OPTIONAL = 4


#: Criticality assignment is owned by policy, never by the task itself.
WORKLOAD_CRITICALITY: Mapping[WorkloadClass, CriticalityLevel] = {
    WorkloadClass.INTEGRITY_CRITICAL: CriticalityLevel.CRITICAL,
    WorkloadClass.INTERACTIVE: CriticalityLevel.HIGH,
    WorkloadClass.CORE_COGNITIVE: CriticalityLevel.HIGH,
    WorkloadClass.BATCH: CriticalityLevel.NORMAL,
    WorkloadClass.EXPLORATORY: CriticalityLevel.OPTIONAL,
    WorkloadClass.MAINTENANCE: CriticalityLevel.LOW,
}

#: Deterministic tie-break order used by the baseline scheduler (Step 16).
CLASS_ORDER: Mapping[WorkloadClass, int] = {
    WorkloadClass.INTEGRITY_CRITICAL: 0,
    WorkloadClass.INTERACTIVE: 1,
    WorkloadClass.CORE_COGNITIVE: 2,
    WorkloadClass.BATCH: 3,
    WorkloadClass.EXPLORATORY: 4,
    WorkloadClass.MAINTENANCE: 5,
}

DEFAULT_CLASS_WEIGHTS: Mapping[WorkloadClass, float] = {
    WorkloadClass.INTEGRITY_CRITICAL: 4.0,
    WorkloadClass.INTERACTIVE: 3.0,
    WorkloadClass.CORE_COGNITIVE: 2.0,
    WorkloadClass.BATCH: 1.0,
    WorkloadClass.EXPLORATORY: 0.5,
    WorkloadClass.MAINTENANCE: 0.5,
}


class TaskState(str, Enum):
    """Lifecycle states of a CROS-tracked task."""

    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    DEFERRED = "DEFERRED"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


class SideEffectPolicy(str, Enum):
    """Side-effect classification used by failure recovery (spec Steps 2.2 and 41)."""

    IDEMPOTENT = "IDEMPOTENT"
    NON_IDEMPOTENT_CHECKPOINTED = "NON_IDEMPOTENT_CHECKPOINTED"
    NON_IDEMPOTENT_UNPROTECTED = "NON_IDEMPOTENT_UNPROTECTED"


class CheckpointSupport(str, Enum):
    """Checkpoint and restart characteristics declared per task (Step 2.2)."""

    NONE = "NONE"
    SAFE_RESTART = "SAFE_RESTART"
    CHECKPOINT = "CHECKPOINT"


class ScopeLevel(str, Enum):
    """Resource identity scopes that must never be silently mixed (Steps 2-3)."""

    MACHINE = "MACHINE"
    PROCESS = "PROCESS"
    WORKER_POOL = "WORKER_POOL"
    MODULE = "MODULE"
    TASK = "TASK"
    DEVICE = "DEVICE"


@dataclass(frozen=True)
class ResourceScope:
    """A stable resource identity used to join metrics without confusing scope."""

    level: ScopeLevel
    scope_id: str
    parent: "ResourceScope | None" = None

    def __post_init__(self) -> None:
        """Reject blank scope identifiers and invalid parent links."""
        if not isinstance(self.scope_id, str) or not self.scope_id.strip():
            raise ValueError("scope_id must be a non-empty string")
        if self.parent is not None and self.parent == self:
            raise ValueError("a scope cannot be its own parent")

    def path(self) -> tuple[ResourceScope, ...]:
        """Return the scope chain from this scope up to the machine root."""
        chain: list[ResourceScope] = []
        current: ResourceScope | None = self
        while current is not None:
            chain.append(current)
            current = current.parent
        return tuple(reversed(chain))

    def is_descendant_of(self, other: "ResourceScope") -> bool:
        """Return True when this scope lies below ``other`` in the topology."""
        return other in self.path()


def _finite_non_negative(name: str, value: float) -> float:
    """Validate a finite, non-negative quantity and return it as float."""
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _finite_unit(name: str, value: float) -> float:
    """Validate a finite value in the unit interval."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return number


@dataclass(frozen=True)
class ResourceDemand:
    """A task demand vector D_i (spec section 2.1), excluding queue depth."""

    cpu: float = 0.0
    ram: float = 0.0
    io: float = 0.0
    net: float = 0.0
    gpu: float = 0.0
    storage: float = 0.0

    def __post_init__(self) -> None:
        """Reject negative or non-finite demands."""
        for name in ("cpu", "ram", "io", "net", "gpu", "storage"):
            _finite_non_negative(name, getattr(self, name))

    def get(self, kind: ResourceKind) -> float:
        """Return the demand component for one resource kind."""
        return float(getattr(self, _FIELD_BY_KIND[kind]))

    def scaled_by(self, factor: float) -> "ResourceDemand":
        """Return a uniformly scaled copy of this demand vector."""
        factor = float(factor)
        if not math.isfinite(factor) or factor < 0.0:
            raise ValueError("factor must be finite and non-negative")
        return ResourceDemand(
            cpu=self.cpu * factor, ram=self.ram * factor, io=self.io * factor,
            net=self.net * factor, gpu=self.gpu * factor, storage=self.storage * factor,
        )

    def dominates(self, other: "ResourceDemand") -> bool:
        """Return True when every component is at least the other's component."""
        return all(self.get(kind) >= other.get(kind) for kind in DEMAND_KINDS)

    def to_dict(self) -> dict[str, float]:
        """Return a detached mapping of demand components."""
        return {
            "cpu": self.cpu, "ram": self.ram, "io": self.io,
            "net": self.net, "gpu": self.gpu, "storage": self.storage,
        }


_FIELD_BY_KIND: Mapping[ResourceKind, str] = {
    ResourceKind.CPU: "cpu", ResourceKind.RAM: "ram", ResourceKind.IO: "io",
    ResourceKind.NET: "net", ResourceKind.GPU: "gpu", ResourceKind.STORAGE: "storage",
}


@dataclass(frozen=True)
class ResourceDimension:
    """Per-resource capacity, allocation, reservation and pressure (spec 2.1)."""

    kind: ResourceKind
    capacity: float | None
    allocated: float = 0.0
    reserved: float = 0.0
    utilization: float | None = None
    pressure: float = 0.0
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate the dimension; unknown capacity is None, never zero."""
        _finite_non_negative("allocated", self.allocated)
        _finite_non_negative("reserved", self.reserved)
        _finite_unit("pressure", self.pressure)
        _finite_unit("confidence", self.confidence)
        if self.utilization is not None:
            _finite_non_negative("utilization", self.utilization)
        if self.capacity is not None:
            _finite_non_negative("capacity", self.capacity)
            if self.reserved > self.capacity:
                raise ValueError(f"reserved capacity exceeds capacity for {self.kind.value}")
        if self.capacity is not None and self.utilization is not None:
            if self.utilization * self.capacity < -1e-9:
                raise ValueError("utilization cannot imply negative demand")

    @property
    def headroom(self) -> float | None:
        """Capacity minus reservation minus allocation; None when capacity is unknown."""
        if self.capacity is None:
            return None
        return max(0.0, self.capacity - self.reserved - self.allocated)

    @property
    def unknown(self) -> bool:
        """Return True when this dimension's capacity is unknown."""
        return self.capacity is None

    def with_allocation(self, extra: float) -> "ResourceDimension":
        """Return a copy with additional allocation applied."""
        return ResourceDimension(
            kind=self.kind, capacity=self.capacity,
            allocated=self.allocated + _finite_non_negative("extra", extra),
            reserved=self.reserved, utilization=self.utilization,
            pressure=self.pressure, confidence=self.confidence,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the dimension."""
        return {
            "kind": self.kind.value, "capacity": self.capacity,
            "allocated": self.allocated, "reserved": self.reserved,
            "utilization": self.utilization, "pressure": self.pressure,
            "confidence": self.confidence, "headroom": self.headroom,
        }


@dataclass(frozen=True)
class ResourceState:
    """A versioned snapshot of the full resource vector R_t (spec section 2.1)."""

    version: int
    timestamp: float
    scope: ResourceScope
    dimensions: Mapping[ResourceKind, ResourceDimension]

    def __post_init__(self) -> None:
        """Reject malformed versions, timestamps and incomplete vectors."""
        if self.version < 0:
            raise ValueError("state version must be non-negative")
        if not math.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        required = set(ResourceKind)
        if set(self.dimensions) != required:
            missing = sorted(kind.value for kind in required - set(self.dimensions))
            extra = sorted(kind.value for kind in set(self.dimensions) - required)
            raise ValueError(
                f"resource vector must cover all kinds; "
                f"missing={missing}, extra={extra}"
            )

    def dimension(self, kind: ResourceKind) -> ResourceDimension:
        """Return the dimension for one resource kind."""
        return self.dimensions[kind]

    def headroom(self, kind: ResourceKind) -> float | None:
        """Return headroom for one kind, or None when capacity is unknown."""
        return self.dimensions[kind].headroom

    def can_admit(self, demand: ResourceDemand) -> bool:
        """Return True only when every hard demand component fits current headroom."""
        for kind in DEMAND_KINDS:
            room = self.headroom(kind)
            if room is None or demand.get(kind) > room + 1e-9:
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the state."""
        return {
            "version": self.version, "timestamp": self.timestamp,
            "scope": {"level": self.scope.level.value, "scope_id": self.scope.scope_id},
            "dimensions": {kind.value: dim.to_dict() for kind, dim in self.dimensions.items()},
        }


@dataclass(frozen=True)
class TaskRecord:
    """One task and its resource-relevant declarations (spec section 2.2)."""

    task_id: str
    parent_goal_id: str | None
    module_id: str
    workload_class: WorkloadClass
    demand: ResourceDemand
    remaining_work: float
    ready_time: float = 0.0
    weight: float = 1.0
    deadline: float | None = None
    dependencies: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    input_data_locality: tuple[str, ...] = ()
    input_size_mib: float = 0.0
    checkpoint: CheckpointSupport = CheckpointSupport.NONE
    side_effects: SideEffectPolicy = SideEffectPolicy.IDEMPOTENT
    state: TaskState = TaskState.READY
    allocation: ResourceDemand = field(default_factory=ResourceDemand)
    policy_provenance: str = "policy"
    estimated: bool = False

    def __post_init__(self) -> None:
        """Validate the task record and enforce policy-owned class assignment."""
        for name, value in (("task_id", self.task_id), ("module_id", self.module_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.workload_class, WorkloadClass):
            raise ValueError("workload_class must come from the policy-controlled class set")
        _finite_non_negative("remaining_work", self.remaining_work)
        _finite_non_negative("ready_time", self.ready_time)
        if not math.isfinite(self.weight) or self.weight <= 0.0:
            raise ValueError("weight must be finite and positive")
        if self.deadline is not None and not math.isfinite(self.deadline):
            raise ValueError("deadline must be finite when provided")
        if self.deadline is not None and self.deadline < self.ready_time:
            raise ValueError("deadline cannot precede ready_time")
        object.__setattr__(self, "dependencies", tuple(str(item) for item in self.dependencies))
        object.__setattr__(self, "capabilities", tuple(str(item) for item in self.capabilities))
        object.__setattr__(
            self, "input_data_locality", tuple(str(item) for item in self.input_data_locality),
        )
        _finite_non_negative("input_size_mib", self.input_size_mib)
        if (
            self.side_effects is SideEffectPolicy.NON_IDEMPOTENT_CHECKPOINTED
            and self.checkpoint is CheckpointSupport.NONE
        ):
            raise ValueError(
                "NON_IDEMPOTENT_CHECKPOINTED tasks require checkpoint or "
                "safe-restart support"
            )

    @property
    def criticality(self) -> CriticalityLevel:
        """Return the policy-assigned criticality of this task's class (Step 47)."""
        return WORKLOAD_CRITICALITY[self.workload_class]

    def slack(self, now: float) -> float | None:
        """Return deadline slack: deadline minus now minus remaining work (spec 3.3)."""
        if self.deadline is None:
            return None
        return self.deadline - float(now) - self.remaining_work

    def reclassify(
        self, new_class: WorkloadClass, *, policy_authority: bool = False,
    ) -> "TaskRecord":
        """Return a reclassified copy; tasks may never promote their criticality."""
        from dataclasses import replace

        current = self.criticality
        proposed = WORKLOAD_CRITICALITY[new_class]
        if proposed.value < current.value and not policy_authority:
            raise ValueError(
                "a task cannot promote its own criticality; reclassification to a more "
                "critical class requires policy_authority=True"
            )
        return replace(self, workload_class=new_class, policy_provenance="policy-reclassify")

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the task record."""
        return {
            "task_id": self.task_id, "parent_goal_id": self.parent_goal_id,
            "module_id": self.module_id,
            "workload_class": self.workload_class.value,
            "criticality": self.criticality.name,
            "demand": self.demand.to_dict(), "remaining_work": self.remaining_work,
            "ready_time": self.ready_time, "weight": self.weight,
            "deadline": self.deadline, "dependencies": list(self.dependencies),
            "capabilities": list(self.capabilities),
            "input_data_locality": list(self.input_data_locality),
            "input_size_mib": self.input_size_mib,
            "checkpoint": self.checkpoint.value,
            "side_effects": self.side_effects.value, "state": self.state.value,
            "allocation": self.allocation.to_dict(),
            "policy_provenance": self.policy_provenance, "estimated": self.estimated,
        }


@dataclass(frozen=True)
class ModuleBudget:
    """Per-module reservation, cap and borrowing rule (spec Step 23)."""

    module_id: str
    reserved: ResourceDemand
    cap: ResourceDemand
    borrowable: bool = False

    def __post_init__(self) -> None:
        """Validate the budget; a cap below the reservation is contradictory."""
        if not isinstance(self.module_id, str) or not self.module_id.strip():
            raise ValueError("module_id must be a non-empty string")
        if not self.cap.dominates(self.reserved):
            raise ValueError(f"cap must dominate reserved capacity for module {self.module_id}")


@dataclass(frozen=True)
class EnvironmentProfile:
    """Versioned hardware and OS profile produced by Step 1 discovery."""

    profile_version: str
    host_os: str
    kernel_release: str
    capacity: Mapping[ResourceKind, float | None]
    cgroup_support: bool = False
    features: Mapping[str, bool] = field(default_factory=dict)
    telemetry_permissions: Mapping[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate profile shape; unavailable dimensions stay unknown (None)."""
        if not isinstance(self.profile_version, str) or not self.profile_version.strip():
            raise ValueError("profile_version must be a non-empty string")
        for kind, value in self.capacity.items():
            if value is None:
                continue
            _finite_non_negative(f"capacity[{kind.value}]", value)

    def dimension(self, kind: ResourceKind) -> ResourceDimension:
        """Return an unallocated dimension for one resource kind."""
        return ResourceDimension(kind=kind, capacity=self.capacity.get(kind))

    def to_dict(self) -> dict[str, Any]:
        """Return a detached, JSON-compatible snapshot of the profile."""
        return {
            "profile_version": self.profile_version, "host_os": self.host_os,
            "kernel_release": self.kernel_release,
            "capacity": {kind.value: value for kind, value in self.capacity.items()},
            "cgroup_support": self.cgroup_support,
            "features": dict(self.features),
            "telemetry_permissions": dict(self.telemetry_permissions),
        }


def machine_scope(machine_id: str = "local") -> ResourceScope:
    """Return the machine-root scope used by default deployments."""
    return ResourceScope(ScopeLevel.MACHINE, machine_id)


def build_resource_state(
    profile: EnvironmentProfile,
    scope: ResourceScope,
    *,
    version: int = 0,
    timestamp: float = 0.0,
    allocations: Mapping[ResourceKind, float] | None = None,
    reservations: Mapping[ResourceKind, float] | None = None,
    utilizations: Mapping[ResourceKind, float] | None = None,
    pressures: Mapping[ResourceKind, float] | None = None,
    confidences: Mapping[ResourceKind, float] | None = None,
) -> ResourceState:
    """Assemble a complete versioned resource state from a capacity profile."""
    allocations = allocations or {}
    reservations = reservations or {}
    utilizations = utilizations or {}
    pressures = pressures or {}
    confidences = confidences or {}
    dimensions: dict[ResourceKind, ResourceDimension] = {}
    for kind in ResourceKind:
        dimensions[kind] = ResourceDimension(
            kind=kind,
            capacity=profile.capacity.get(kind),
            allocated=_finite_non_negative("allocation", allocations.get(kind, 0.0)),
            reserved=_finite_non_negative("reservation", reservations.get(kind, 0.0)),
            utilization=utilizations.get(kind),
            pressure=_finite_unit("pressure", pressures.get(kind, 0.0)),
            confidence=_finite_unit("confidence", confidences.get(kind, 1.0)),
        )
    return ResourceState(
        version=version, timestamp=timestamp, scope=scope,
        dimensions=dimensions,
    )


def jain_fairness_index(shares: Sequence[float]) -> float:
    """Return the Jain fairness index for a sequence of service shares."""
    values = tuple(float(item) for item in shares)
    if not values or any(not math.isfinite(item) or item < 0.0 for item in values):
        raise ValueError("shares must be finite and non-negative")
    total = sum(values)
    if total <= 0.0:
        return 1.0 if not any(values) else 0.0
    return (total * total) / (len(values) * sum(item * item for item in values))


def as_mapping_dict(record: Any) -> dict[str, Any]:
    """Return a JSON-compatible dict for any CROS record exposing ``to_dict``."""
    return dict(record.to_dict()) if hasattr(record, "to_dict") else dict(asdict(record))
