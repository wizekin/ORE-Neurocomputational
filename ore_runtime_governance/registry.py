"""Component registration, discovery, leases, and dependency graph (steps 1, 4, 34).

``ComponentRegistry`` validates manifests against administrator bounds and
issues typed registration results; it never grants authority (identity and
authorization stay separate, step 19). ``ServiceLease`` implements expiring
health leases so crashed workers eventually leave the active routing table
(step 34). ``DependencyGraph`` builds the component graph G=(V,E), detects
startup cycles, and permits runtime cycles only when every edge on the cycle
is asynchronous and bounded (step 4). ``inventory_repository`` reconciles the
component inventory with the actual repository contents instead of
architectural names (step 1, "Stage 0 reality check").
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .contracts import (
    ManifestValidationError,
    ResourceVector,
    TrustClass,
    ConsequenceClass,
    validate_component_id,
)

_REGISTRY_COLUMNS = (
    "component_id",
    "version",
    "owner",
    "trust_class",
    "consequence",
    "purpose",
    "protocol_version",
    "capabilities",
    "dependencies",
)


@dataclass(frozen=True)
class RegistrationResult:
    """Typed outcome of ``ComponentRuntime.register`` (section 6)."""

    component_id: str
    registered: bool
    reason: str
    manifest_digest: str = ""
    generation: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "registered": self.registered,
            "reason": self.reason,
            "manifest_digest": self.manifest_digest,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class IdentityRecord:
    """Authenticated identity of a sender (identity, not authority, step 19)."""

    component_id: str
    known: bool
    protocol_version: int = 0
    capabilities: tuple[str, ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "known": self.known,
            "protocol_version": self.protocol_version,
            "capabilities": list(self.capabilities),
            "note": self.note,
        }


@dataclass(frozen=True)
class ServiceLease:
    """Expiring route lease: crashed workers eventually leave routing (step 34)."""

    lease_id: str
    component_id: str
    endpoint: str
    protocol_version: int
    capabilities: tuple[str, ...]
    issued_at_epoch: float
    expires_at_epoch: float
    generation: int

    def is_expired(self, now: float | None = None) -> bool:
        current = now if now is not None else time.time()
        return current >= self.expires_at_epoch

    def remaining_seconds(self, now: float | None = None) -> float:
        current = now if now is not None else time.time()
        return max(0.0, self.expires_at_epoch - current)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lease_id": self.lease_id,
            "component_id": self.component_id,
            "endpoint": self.endpoint,
            "protocol_version": self.protocol_version,
            "capabilities": list(self.capabilities),
            "issued_at_epoch": self.issued_at_epoch,
            "expires_at_epoch": self.expires_at_epoch,
            "generation": self.generation,
        }


class RegistryError(ValueError):
    """Raised for duplicate or missing component registrations."""


class ComponentRegistry:
    """Registration and discovery of component manifests (steps 3-4)."""

    def __init__(self, bounds: ResourceVector | None = None, clock=None) -> None:
        self._bounds = bounds
        self._clock = clock or time.time
        self._manifests: dict[str, Any] = {}
        self._digests: dict[str, str] = {}
        self._leases: dict[str, ServiceLease] = {}
        self._generations: dict[str, int] = {}

    # -- registration --------------------------------------------------------

    def register(self, manifest, *, replace: bool = False) -> RegistrationResult:
        """Validate and register one manifest; never grants authority here."""
        if self._bounds is not None:
            try:
                manifest.validate_resource_bounds(self._bounds)
            except ManifestValidationError as exc:
                return RegistrationResult(
                    component_id=manifest.component_id,
                    registered=False,
                    reason=f"registration rejected: {exc}",
                )
        component_id = manifest.component_id
        if component_id in self._manifests and not replace:
            return RegistrationResult(
                component_id=component_id,
                registered=False,
                reason=(
                    f"component {component_id} is already registered "
                    "(pass replace=True for an authorized replacement)"
                ),
            )
        digest = manifest.manifest_digest()
        self._manifests[component_id] = manifest
        self._digests[component_id] = digest
        generation = self._generations.get(component_id, 0) + 1
        self._generations[component_id] = generation
        return RegistrationResult(
            component_id=component_id,
            registered=True,
            reason="manifest validated and registered",
            manifest_digest=digest,
            generation=generation,
        )

    def unregister(self, component_id: str) -> bool:
        self._manifests.pop(component_id, None)
        self._digests.pop(component_id, None)
        self._leases.pop(component_id, None)
        return True

    # -- identity and lookup --------------------------------------------------

    def manifest(self, component_id: str):
        try:
            return self._manifests[component_id]
        except KeyError as exc:
            raise RegistryError(f"unknown component {component_id!r}") from exc

    def has(self, component_id: str) -> bool:
        return component_id in self._manifests

    def digest(self, component_id: str) -> str:
        try:
            return self._digests[component_id]
        except KeyError as exc:
            raise RegistryError(f"unknown component {component_id!r}") from exc

    def generation(self, component_id: str) -> int:
        return self._generations.get(component_id, 0)

    def list_components(self) -> tuple[str, ...]:
        return tuple(sorted(self._manifests))

    def describe(self, component_id: str) -> dict[str, Any]:
        return self.manifest(component_id).to_dict()

    def authenticate(self, sender: str) -> IdentityRecord:
        """Authenticate which component is acting (never grants authority)."""
        if not isinstance(sender, str) or not sender.strip():
            return IdentityRecord(
                component_id=sender if isinstance(sender, str) else "",
                known=False,
                note="sender id is empty or malformed",
            )
        if sender not in self._manifests:
            return IdentityRecord(
                component_id=sender, known=False, note="sender is not a registered component"
            )
        manifest = self._manifests[sender]
        return IdentityRecord(
            component_id=sender,
            known=True,
            protocol_version=manifest.protocol_version,
            capabilities=tuple(manifest.capabilities),
        )

    # -- discovery --------------------------------------------------------------

    def discover(
        self,
        *,
        capability: str | None = None,
        protocol_version: int | None = None,
    ) -> tuple[str, ...]:
        """Service discovery by capability and protocol compatibility (step 34)."""
        matches: list[str] = []
        for component_id, manifest in sorted(self._manifests.items()):
            if capability is not None and capability not in manifest.capabilities:
                continue
            if protocol_version is not None and manifest.protocol_version != protocol_version:
                continue
            matches.append(component_id)
        return tuple(matches)

    # -- health leases ----------------------------------------------------------

    def issue_lease(self, component_id: str, ttl_seconds: float) -> ServiceLease:
        """Issue an expiring route lease for a live component (step 34)."""
        manifest = self.manifest(component_id)
        if ttl_seconds <= 0:
            raise ValueError("lease ttl must be positive")
        now = self._clock()
        lease = ServiceLease(
            lease_id=f"lease-{uuid.uuid4().hex}",
            component_id=component_id,
            endpoint=f"ipc://{component_id}",
            protocol_version=manifest.protocol_version,
            capabilities=tuple(manifest.capabilities),
            issued_at_epoch=now,
            expires_at_epoch=now + ttl_seconds,
            generation=self._generations.get(component_id, 0),
        )
        self._leases[component_id] = lease
        return lease

    def renew_lease(self, component_id: str, ttl_seconds: float) -> ServiceLease:
        if component_id not in self._leases:
            return self.issue_lease(component_id, ttl_seconds)
        now = self._clock()
        previous = self._leases[component_id]
        renewed = ServiceLease(
            lease_id=previous.lease_id,
            component_id=component_id,
            endpoint=previous.endpoint,
            protocol_version=previous.protocol_version,
            capabilities=previous.capabilities,
            issued_at_epoch=previous.issued_at_epoch,
            expires_at_epoch=now + ttl_seconds,
            generation=previous.generation,
        )
        self._leases[component_id] = renewed
        return renewed

    def expire_leases(self, now: float | None = None) -> tuple[str, ...]:
        """Drop expired leases so crashed workers leave active routing (step 34)."""
        current = now if now is not None else self._clock()
        expired = [
            component_id
            for component_id, lease in self._leases.items()
            if lease.is_expired(current)
        ]
        for component_id in expired:
            self._leases.pop(component_id, None)
        return tuple(sorted(expired))

    def active_leases(self) -> tuple[ServiceLease, ...]:
        return tuple(self._leases[component_id] for component_id in sorted(self._leases))

    def routes(self) -> dict[str, str]:
        return {cid: lease.endpoint for cid, lease in sorted(self._leases.items())}

    def health(self, component_id: str, now: float | None = None) -> dict[str, Any]:
        """Distinguish process liveness, responsiveness, and capability (step 34)."""
        current = now if now is not None else self._clock()
        lease = self._leases.get(component_id)
        return {
            "component_id": component_id,
            "registered": component_id in self._manifests,
            "lease_active": bool(lease and not lease.is_expired(current)),
            "lease_remaining_seconds": (
                lease.remaining_seconds(current) if lease else 0.0
            ),
            "protocol_version": (
                self._manifests[component_id].protocol_version
                if component_id in self._manifests
                else None
            ),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "registered": len(self._manifests),
            "components": list(self.list_components()),
            "active_leases": len(self._leases),
            "admin_bounds": self._bounds.to_dict() if self._bounds else None,
        }


@dataclass(frozen=True)
class DependencyEdge:
    """Directed service edge with protocol and failure metadata (step 4)."""

    consumer: str
    provider: str
    protocol_version: int = 1
    criticality: float = 1.0
    timeout_seconds: float = 30.0
    failure_behavior: str = "fail_task"
    asynchronous: bool = False
    bounded_queue: bool = False

    _FAILURE_BEHAVIORS = ("fail_task", "retry", "ignore", "degrade")

    def __post_init__(self) -> None:
        validate_component_id(self.consumer, "edge consumer")
        validate_component_id(self.provider, "edge provider")
        if not 0.0 <= self.criticality <= 1.0:
            raise ValueError("criticality must be between 0 and 1")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.failure_behavior not in self._FAILURE_BEHAVIORS:
            raise ValueError(
                f"failure_behavior must be one of {self._FAILURE_BEHAVIORS}"
            )

    def permits_runtime_cycle(self) -> bool:
        return self.asynchronous and self.bounded_queue

    def to_dict(self) -> dict[str, Any]:
        return {
            "consumer": self.consumer,
            "provider": self.provider,
            "protocol_version": self.protocol_version,
            "criticality": self.criticality,
            "timeout_seconds": self.timeout_seconds,
            "failure_behavior": self.failure_behavior,
            "asynchronous": self.asynchronous,
            "bounded_queue": self.bounded_queue,
        }


class DependencyGraphError(ValueError):
    """Raised for startup cycles or missing providers (step 4)."""


class DependencyGraph:
    """Component dependency graph G=(V,E) with cycle policy (step 4)."""

    def __init__(self, manifests: Iterable[Any]) -> None:
        self._vertices: dict[str, Any] = {}
        for manifest in manifests:
            self._vertices[manifest.component_id] = manifest
        self._edges: dict[tuple[str, str], DependencyEdge] = {}
        for manifest in self._vertices.values():
            for provider in manifest.dependencies:
                if provider not in self._vertices:
                    raise DependencyGraphError(
                        f"component {manifest.component_id} depends on unknown "
                        f"provider {provider!r}"
                    )
                self._edges[(manifest.component_id, provider)] = DependencyEdge(
                    consumer=manifest.component_id,
                    provider=provider,
                    protocol_version=manifest.protocol_version,
                )

    def add_edge(self, edge: DependencyEdge) -> None:
        if edge.consumer not in self._vertices or edge.provider not in self._vertices:
            raise DependencyGraphError("edge endpoints must be registered components")
        self._edges[(edge.consumer, edge.provider)] = edge

    def vertices(self) -> tuple[str, ...]:
        return tuple(sorted(self._vertices))

    def edges(self) -> tuple[DependencyEdge, ...]:
        return tuple(self._edges.values())

    def _adjacency(self) -> dict[str, list[str]]:
        adjacency: dict[str, list[str]] = {v: [] for v in self._vertices}
        for (consumer, provider) in self._edges:
            adjacency[consumer].append(provider)
        return adjacency

    def find_cycle(self) -> list[str] | None:
        """Return one dependency cycle (providers reachable) or None."""
        adjacency = self._adjacency()
        state: dict[str, int] = {v: 0 for v in self._vertices}

        def dfs(node: str, stack: list[str]) -> list[str] | None:
            state[node] = 1
            stack.append(node)
            for neighbour in sorted(adjacency[node]):
                if state[neighbour] == 1:
                    cycle_start = stack.index(neighbour)
                    return stack[cycle_start:]
                if state[neighbour] == 0:
                    found = dfs(neighbour, stack)
                    if found:
                        return found
            stack.pop()
            state[node] = 2
            return None

        for vertex in sorted(self._vertices):
            if state[vertex] == 0:
                cycle = dfs(vertex, [])
                if cycle:
                    return cycle
        return None

    def cycle_is_permitted(self, cycle: list[str]) -> bool:
        """Runtime cycles are legal only when fully asynchronous and bounded."""
        ring = cycle + [cycle[0]]
        for consumer, provider in zip(ring, ring[1:]):
            edge = self._edges.get((consumer, provider))
            if edge is None or not edge.permits_runtime_cycle():
                return False
        return True

    def validate_startup_acyclic(self) -> None:
        cycle = self.find_cycle()
        if cycle and not self.cycle_is_permitted(cycle):
            raise DependencyGraphError(
                f"startup dependency cycle {' -> '.join(cycle + [cycle[0]])} is not "
                "asynchronous-and-bounded; resolve before startup"
            )

    def startup_order(self) -> tuple[str, ...]:
        """Startup sequence; providers start before their consumers.

        Runtime cycles (asynchronous-and-bounded on every edge) are permitted
        (integrated step 4): when the walk stalls on such a cycle, its members
        are started as a group in component-id order. Impermissible cycles
        raise :class:`DependencyGraphError`.
        """
        self.validate_startup_acyclic()
        # Startup direction: provider -> consumer (the provider must be up
        # before anything that depends on it can start).
        dependents: dict[str, list[str]] = {v: [] for v in self._vertices}
        indegree: dict[str, int] = {v: 0 for v in self._vertices}
        for (consumer, provider) in self._edges:
            dependents[provider].append(consumer)
            indegree[consumer] += 1
        remaining = set(self._vertices)
        ready = sorted([v for v, degree in indegree.items() if degree == 0])
        order: list[str] = []
        while remaining:
            if not ready:
                # The walk stalled on a cycle. validate_startup_acyclic already
                # proved it is asynchronous-and-bounded, so start its members.
                cycle = self._stalled_cycle(remaining, indegree)
                if cycle is None:
                    raise DependencyGraphError(
                        f"dependency cycle prevents startup ordering: {sorted(remaining)}"
                    )
                forced = sorted(cycle)
                for vertex in forced:
                    remaining.discard(vertex)
                    order.append(vertex)
                    for consumer in sorted(dependents[vertex]):
                        if consumer in remaining:
                            indegree[consumer] -= 1
                ready = sorted(
                    [v for v in remaining if indegree.get(v, 0) == 0]
                )
                continue
            vertex = ready.pop(0)
            remaining.discard(vertex)
            order.append(vertex)
            for consumer in sorted(dependents[vertex]):
                if consumer in remaining:
                    indegree[consumer] -= 1
                    if indegree[consumer] == 0:
                        ready.append(consumer)
                        ready.sort()
        return tuple(order)

    def _stalled_cycle(self, remaining: set[str], indegree: dict[str, int]) -> list[str] | None:
        """Return one cycle among the remaining (all-blocked) vertices."""
        adjacency: dict[str, list[str]] = {v: [] for v in remaining}
        for (consumer, provider) in self._edges:
            if consumer in remaining and provider in remaining:
                adjacency[consumer].append(provider)
        state: dict[str, int] = {v: 0 for v in remaining}

        def dfs(node: str, stack: list[str]) -> list[str] | None:
            state[node] = 1
            stack.append(node)
            for neighbour in sorted(adjacency[node]):
                if state[neighbour] == 1:
                    return stack[stack.index(neighbour):]
                if state[neighbour] == 0:
                    found = dfs(neighbour, stack)
                    if found:
                        return found
            stack.pop()
            state[node] = 2
            return None

        for vertex in sorted(remaining):
            if state[vertex] == 0:
                cycle = dfs(vertex, [])
                if cycle:
                    return cycle
        return None

    def dependencies_ready(self, component_id: str, completed: set[str]) -> bool:
        """Admission predicate: the component's providers are all ready (step 38)."""
        manifest = self._vertices.get(component_id)
        if manifest is None:
            return False
        return all(provider in completed for provider in manifest.dependencies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vertices": list(self.vertices()),
            "edges": [edge.to_dict() for edge in self.edges()],
        }


@dataclass(frozen=True)
class InventoryRecord:
    """One inventoried executable component with suggested classes (step 1)."""

    component_id: str
    suggested_trust_class: str
    consequence: str
    basis: str
    package: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "suggested_trust_class": self.suggested_trust_class,
            "consequence": self.consequence,
            "basis": self.basis,
            "package": self.package,
        }


_KNOWN_COORDINATORS: dict[str, tuple[str, str]] = {
    "AdaptiveResourceCoordinator": ("trusted_core", "reversible_write"),
    "CognitiveTaskCoordinator": ("trusted_core", "reversible_write"),
    "ModuleScheduler": ("trusted_core", "reversible_write"),
    "SharedWorkspace": ("trusted_core", "reversible_write"),
    "WorkspaceEventBus": ("trusted_service", "read_only"),
    "PermissionPolicy": ("trusted_core", "safety_critical"),
    "HardwareAwareOptimizer": ("trusted_service", "read_only"),
    "LocalCROSGate": ("trusted_core", "reversible_write"),
    "USGKnowledgeAdapter": ("trusted_service", "read_only"),
    "ComponentRuntime": ("trusted_core", "safety_critical"),
    "CapabilityAuthority": ("trusted_core", "safety_critical"),
    "LifecycleMachine": ("trusted_core", "safety_critical"),
}


def inventory_repository(
    repository_root: str | Path | None = None,
    *,
    include_runtime: bool = True,
) -> dict[str, Any]:
    """Inventory executable components from the actual repository (step 1).

    Scans ``ore_*`` packages for known coordinator/service classes rather than
    inferring components from architectural names, and reconciles the result
    with what actually exists on disk. ``generated`` records stand in for
    HIC-generated programs which have no repository footprint.
    """
    root = Path(repository_root).expanduser() if repository_root else Path(
        __file__
    ).resolve().parents[1]
    records: list[InventoryRecord] = []
    packages_scanned: list[str] = []
    for package_dir in sorted(root.glob("ore_*")):
        if not package_dir.is_dir() or not (package_dir / "__init__.py").exists():
            continue
        package = package_dir.name
        packages_scanned.append(package)
        for py_file in sorted(package_dir.glob("*.py")):
            try:
                text = py_file.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for class_name, (trust, consequence) in _KNOWN_COORDINATORS.items():
                needle = f"class {class_name}"
                if needle in text:
                    records.append(
                        InventoryRecord(
                            component_id=f"{package}.{class_name.lower()}",
                            suggested_trust_class=trust,
                            consequence=consequence,
                            basis=f"{class_name} defined in {package}/{py_file.name}",
                            package=package,
                        )
                    )
    if include_runtime:
        records.append(
            InventoryRecord(
                component_id="hic.generated_programs",
                suggested_trust_class=TrustClass.UNTRUSTED_GENERATED.value,
                consequence=ConsequenceClass.EXTERNAL_SIDE_EFFECT.value,
                basis=(
                    "generated code has no repository footprint and is treated "
                    "as untrusted by default (integrated step 21)"
                ),
                package="<generated>",
            )
        )
    return {
        "repository_root": str(root),
        "packages_scanned": packages_scanned,
        "component_count": len(records),
        "records": [record.to_dict() for record in records],
        "reconciled": True,
    }
