"""Shared contracts for the ORE runtime governance layer (Point 22).

Implements the contract foundation of "ORE Integrated Algorithm -- Point 22:
Advanced Kernel, Isolation and Operating-System Architectures": the versioned
component manifest (integrated step 3), the typed message envelope with
protocol versioning (step 10), the trust/consequence classifications used to
derive execution tiers (steps 2 and 5), and the outcome taxonomy that
distinguishes denied, unavailable, invalid, exhausted, timed out, cancelled,
and internal failure. This module is deliberately free of domain reasoning:
it only defines the vocabulary every other ``ore_runtime`` plane agrees on.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

MODULE_VERSION = "1.0.0"
ENVELOPE_SCHEMA_VERSION = 1
SPEC_SOURCE = (
    "ORE Integrated Algorithm -- Point 22: Advanced Kernel, Isolation and "
    "Operating-System Architectures"
)

DEFAULT_ENV_ALLOWLIST: tuple[str, ...] = (
    "PATH",
    "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE",
    "LC_ALL",
)

_COMPONENT_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{2,63}$")
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?$")
_RIGHT_RE = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)+$")

#: Rights vocabulary (integrated steps 3 and 15-16). A manifest that requests a
#: right outside this catalogue is rejected at registration as an undeclared
#: privilege; narrow rights beat broad ones ("least authority", step 16).
KNOWN_RIGHTS: dict[str, str] = {
    "ipc.send": "Send typed request messages on the IPC plane",
    "task.execute": "Execute dispatched tasks for the owning component",
    "workspace.read_snapshot": "Read governed workspace snapshots",
    "workspace.propose_fact": "Propose a fact into the shared workspace",
    "workspace.append_observation": "Append observations to an owned workspace field",
    "workspace.request_merge": "Request a merge of pending workspace proposals",
    "usg.read_snapshot": "Read USG semantic knowledge snapshots",
    "usg.propose_update": "Propose a USG knowledge update for external validation",
    "artifact.read": "Read artifacts by immutable reference",
    "artifact.publish": "Publish owned artifacts to the artifact store",
    "sandbox.execute": "Execute jobs through the sandbox execution backend",
    "host.network": "Access the network (explicit grant only)",
    "host.filesystem.write": "Write inside the declared task file area",
    "remote.execute": "Execute on optional remote workers (Stage 7)",
    "operator.override": "Request a scoped manual operator override",
}


def canonical_json(value: Any) -> str:
    """Serialize deterministically for hashing and size accounting."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def generate_message_id(prefix: str = "msg") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def validate_component_id(value: str, field_name: str = "component_id") -> str:
    if not isinstance(value, str) or not _COMPONENT_ID_RE.match(value):
        raise ValueError(
            f"{field_name} must match {_COMPONENT_ID_RE.pattern} (got {value!r})"
        )
    return value


def validate_semantic_version(value: str, field_name: str = "version") -> str:
    if not isinstance(value, str) or not _SEMVER_RE.match(value):
        raise ValueError(f"{field_name} must be a semantic version (got {value!r})")
    return value


def validate_right(value: str, field_name: str = "capabilities") -> str:
    if not isinstance(value, str) or value not in KNOWN_RIGHTS:
        raise ValueError(
            f"{field_name} contains undeclared privilege {value!r}; "
            f"known rights: {sorted(KNOWN_RIGHTS)}"
        )
    if not _RIGHT_RE.match(value):
        raise ValueError(f"{field_name} entry {value!r} is not a namespaced right")
    return value


class _Coercible(str, Enum):
    """String enum mixin matching the repository's ``coerce`` convention."""

    @classmethod
    def coerce(cls, value: Any, field_name: str = "value") -> Any:
        if isinstance(value, cls):
            return value
        raw = str(value).strip()
        candidates = (
            raw,
            raw.lower(),
            raw.upper(),
            raw.lower().replace(" ", "_").replace("-", "_"),
        )
        for candidate in candidates:
            try:
                return cls(candidate)
            except ValueError:
                continue
        raise ValueError(
            f"{field_name} must be one of {[member.value for member in cls]} "
            f"(got {value!r})"
        )


class OreRuntimeError(RuntimeError):
    """Base class for typed runtime failures."""


class ManifestValidationError(ValueError):
    """Raised when a component manifest violates the registration contract."""


class EnvelopeValidationError(ValueError):
    """Raised when an IPC message is malformed, oversized, or unsupported."""


class CapabilityDeniedError(OreRuntimeError):
    """Raised when an authorization check denies an operation."""

    def __init__(self, reason: str, *, subject: str = "", right: str = "") -> None:
        super().__init__(reason)
        self.reason = reason
        self.subject = subject
        self.right = right


class ImpersonationError(CapabilityDeniedError):
    """Raised when a sender claims an identity its capability does not cover."""


class AdmissionDeniedError(OreRuntimeError):
    """Raised when admission control cannot safely support a task."""


class QueueBackpressureError(OreRuntimeError):
    """Raised when a bounded queue rejects or defers work (integrated step 12)."""

    def __init__(
        self, reason: str, *, kind: str = "rejected", admission_factor: float = 0.0
    ) -> None:
        super().__init__(reason)
        self.kind = kind
        self.admission_factor = admission_factor


class OperationTimedOutError(OreRuntimeError):
    """Elapsed time exceeded the request deadline (distinct from failure)."""


class OperationCancelledError(OreRuntimeError):
    """The operation was cancelled before a confirmed outcome."""


class BudgetExhaustedError(OreRuntimeError):
    """A hard resource ceiling was reached (denied/exhausted, not invalid)."""


class IllegalTransitionError(OreRuntimeError):
    """Raised when a lifecycle transition is not authorized (step 8)."""

    def __init__(
        self, message: str, *, from_state: Any = None, to_state: Any = None
    ) -> None:
        super().__init__(message)
        self.from_state = from_state
        self.to_state = to_state


class ArtifactCorruptionError(OreRuntimeError):
    """Artifact bytes fail their content-hash verification (step 27)."""


class BackendUnavailableError(OreRuntimeError):
    """The requested isolation backend is not available on this host (step 40)."""


class OutcomeKind(_Coercible):
    """Typed outcomes distinguishing denial from unavailability from failure."""

    OK = "ok"
    DENIED = "denied"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    EXHAUSTED = "exhausted"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class TrustClass(_Coercible):
    """Component trust classification (integrated step 2)."""

    TRUSTED_CORE = "trusted_core"
    TRUSTED_SERVICE = "trusted_service"
    CONSTRAINED_PLUGIN = "constrained_plugin"
    UNTRUSTED_GENERATED = "untrusted_generated"


class ConsequenceClass(_Coercible):
    """Consequence classification, independent of trust (step 2)."""

    READ_ONLY = "read_only"
    REVERSIBLE_WRITE = "reversible_write"
    EXTERNAL_SIDE_EFFECT = "external_side_effect"
    SAFETY_CRITICAL = "safety_critical"


class ExecutionTier(_Coercible):
    """Execution tiers derived from trust and consequence (step 5)."""

    IN_PROCESS = "in_process"
    LOCAL_PROCESS = "local_process"
    CONTAINER_SANDBOX = "container_sandbox"
    MICROVM = "microvm"
    REMOTE_WORKER = "remote_worker"


class DeliverySemantics(_Coercible):
    """Honest delivery semantics per operation (step 33)."""

    AT_MOST_ONCE = "at_most_once"
    AT_LEAST_ONCE = "at_least_once"
    EFFECTIVELY_ONCE = "effectively_once"


class DataClassification(_Coercible):
    """Data classification attached to manifests and artifacts."""

    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


@dataclass(frozen=True)
class ResourceVector:
    """Normalized multi-resource demand (integrated steps 36-38).

    ``cpu_shares`` is a normalized share (1.0 == one full core), ``memory_bytes``
    and ``storage_bytes`` are byte budgets, ``network_bps`` is bandwidth, and
    ``worker_slots``/``queue_slots``/``accelerator_slots`` are indivisible slots.
    """

    cpu_shares: float = 0.0
    memory_bytes: int = 0
    storage_bytes: int = 0
    network_bps: float = 0.0
    accelerator_slots: int = 0
    worker_slots: int = 0
    queue_slots: int = 0

    def __post_init__(self) -> None:
        for kind in self.kinds():
            value = getattr(self, kind)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"resource kind {kind} must be numeric")
            if value < 0:
                raise ValueError(f"resource kind {kind} must be non-negative")

    @classmethod
    def kinds(cls) -> tuple[str, ...]:
        return (
            "cpu_shares",
            "memory_bytes",
            "storage_bytes",
            "network_bps",
            "accelerator_slots",
            "worker_slots",
            "queue_slots",
        )

    def amounts(self) -> dict[str, float]:
        return {kind: float(getattr(self, kind)) for kind in self.kinds()}

    def _combine(self, other: "ResourceVector", sign: int) -> "ResourceVector":
        if not isinstance(other, ResourceVector):
            raise TypeError("can only combine ResourceVector instances")
        return ResourceVector(
            **{
                kind: getattr(self, kind) + sign * getattr(other, kind)
                for kind in self.kinds()
            }
        )

    def __add__(self, other: "ResourceVector") -> "ResourceVector":
        return self._combine(other, 1)

    def __sub__(self, other: "ResourceVector") -> "ResourceVector":
        return self._combine(other, -1)

    def scaled(self, factor: float) -> "ResourceVector":
        if factor < 0:
            raise ValueError("scale factor must be non-negative")
        return ResourceVector(**{k: getattr(self, k) * factor for k in self.kinds()})

    def elementwise_le(self, other: "ResourceVector") -> bool:
        """Return True when every component is <= the other's (x_ij <= L_ij)."""
        return all(getattr(self, k) <= getattr(other, k) for k in self.kinds())

    def is_zero(self) -> bool:
        return all(getattr(self, k) == 0 for k in self.kinds())

    def to_dict(self) -> dict[str, float]:
        return self.amounts()

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "ResourceVector":
        if not isinstance(mapping, Mapping):
            raise TypeError("resource vector must be a mapping of kind to amount")
        unknown = set(mapping).difference(cls.kinds())
        if unknown:
            raise ValueError(f"unknown resource kinds: {sorted(unknown)}")
        return cls(**{kind: mapping.get(kind, 0) for kind in cls.kinds()})


@dataclass(frozen=True)
class ResourceProfile:
    """Requested demand, minimum requirement, hard ceilings, and deadline (step 37)."""

    requested: ResourceVector = field(default_factory=ResourceVector)
    minimum: ResourceVector = field(default_factory=ResourceVector)
    limits: ResourceVector = field(default_factory=ResourceVector)
    deadline_seconds: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.requested, ResourceVector) or not isinstance(
            self.minimum, ResourceVector
        ) or not isinstance(self.limits, ResourceVector):
            raise TypeError("resource profile members must be ResourceVector instances")
        if not self.minimum.elementwise_le(self.requested):
            raise ValueError("minimum requirement a_ij must not exceed the request d_ij")
        if not self.requested.elementwise_le(self.limits):
            raise ValueError("requested demand must stay within the hard ceilings L_ij")
        if not (self.deadline_seconds > 0):
            raise ValueError("deadline_seconds must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested.to_dict(),
            "minimum": self.minimum.to_dict(),
            "limits": self.limits.to_dict(),
            "deadline_seconds": self.deadline_seconds,
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "ResourceProfile":
        if not isinstance(mapping, Mapping):
            raise TypeError("resource profile must be a mapping")
        unknown = set(mapping).difference(
            {"requested", "minimum", "limits", "deadline_seconds"}
        )
        if unknown:
            raise ValueError(f"unknown resource profile fields: {sorted(unknown)}")
        return cls(
            requested=ResourceVector.from_dict(mapping.get("requested", {})),
            minimum=ResourceVector.from_dict(mapping.get("minimum", {})),
            limits=ResourceVector.from_dict(mapping.get("limits", {})),
            deadline_seconds=float(mapping.get("deadline_seconds", 30.0)),
        )


@dataclass(frozen=True)
class IsolationProfile:
    """Declared isolation profile (integrated steps 5, 21, and 39).

    ``filesystem_scope``: "none" (no filesystem), "task_temp" (temporary
    input/output directories only), or "owned_area" (a declared artifact area).
    ``ambient_credentials`` and ``host_home_access`` must stay False for
    untrusted generated code, and network access requires the explicit
    ``host.network`` right on the manifest.
    """

    tier: ExecutionTier = ExecutionTier.LOCAL_PROCESS
    filesystem_scope: str = "task_temp"
    network_enabled: bool = False
    host_home_access: bool = False
    ambient_credentials: bool = False
    env_allowlist: tuple[str, ...] = DEFAULT_ENV_ALLOWLIST

    def __post_init__(self) -> None:
        object.__setattr__(self, "tier", _coerce_enum_value(self.tier, ExecutionTier, "tier"))
        if self.filesystem_scope not in {"none", "task_temp", "owned_area"}:
            raise ValueError(
                "filesystem_scope must be none/task_temp/owned_area "
                f"(got {self.filesystem_scope!r})"
            )
        if not isinstance(self.env_allowlist, tuple):
            raise TypeError("env_allowlist must be a tuple of variable names")
        for name in self.env_allowlist:
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"env allowlist entry {name!r} is not a variable name")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier.value,
            "filesystem_scope": self.filesystem_scope,
            "network_enabled": self.network_enabled,
            "host_home_access": self.host_home_access,
            "ambient_credentials": self.ambient_credentials,
            "env_allowlist": list(self.env_allowlist),
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "IsolationProfile":
        if not isinstance(mapping, Mapping):
            raise TypeError("isolation profile must be a mapping")
        unknown = set(mapping).difference(
            {
                "tier",
                "filesystem_scope",
                "network_enabled",
                "host_home_access",
                "ambient_credentials",
                "env_allowlist",
            }
        )
        if unknown:
            raise ValueError(f"unknown isolation profile fields: {sorted(unknown)}")
        return cls(
            tier=ExecutionTier.coerce(mapping.get("tier", "local_process")),
            filesystem_scope=mapping.get("filesystem_scope", "task_temp"),
            network_enabled=bool(mapping.get("network_enabled", False)),
            host_home_access=bool(mapping.get("host_home_access", False)),
            ambient_credentials=bool(mapping.get("ambient_credentials", False)),
            env_allowlist=tuple(mapping.get("env_allowlist", DEFAULT_ENV_ALLOWLIST)),
        )


def _coerce_enum_value(value: Any, enum_cls: type, field_name: str) -> Any:
    return enum_cls.coerce(value, field_name)


@dataclass(frozen=True)
class HealthPolicy:
    """Health-check, lease, and restart policy (integrated steps 9, 34, 35)."""

    heartbeat_interval_seconds: float = 5.0
    lease_ttl_seconds: float = 15.0
    unresponsive_after_seconds: float = 10.0
    restart_limit: int = 3
    backoff_base_seconds: float = 0.25
    backoff_max_seconds: float = 30.0
    quarantine_after_restarts: int = 3

    _FIELDS = (
        "heartbeat_interval_seconds",
        "lease_ttl_seconds",
        "unresponsive_after_seconds",
        "restart_limit",
        "backoff_base_seconds",
        "backoff_max_seconds",
        "quarantine_after_restarts",
    )

    def __post_init__(self) -> None:
        for name in (
            "heartbeat_interval_seconds",
            "lease_ttl_seconds",
            "unresponsive_after_seconds",
            "backoff_base_seconds",
            "backoff_max_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("restart_limit", "quarantine_after_restarts"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.quarantine_after_restarts > self.restart_limit:
            raise ValueError("quarantine threshold cannot exceed the restart budget")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self._FIELDS}

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "HealthPolicy":
        if not isinstance(mapping, Mapping):
            raise TypeError("health policy must be a mapping")
        unknown = set(mapping).difference(cls._FIELDS)
        if unknown:
            raise ValueError(f"unknown health policy fields: {sorted(unknown)}")
        return cls(**{name: mapping[name] for name in cls._FIELDS if name in mapping})


@dataclass(frozen=True)
class TraceContext:
    """Correlation and causal tracing fields propagated across messages (step 32)."""

    trace_id: str
    parent_task_id: str | None = None
    causation_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.trace_id, str) or not self.trace_id.strip():
            raise ValueError("trace_id must be a non-empty string")
        for name in ("parent_task_id", "causation_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")

    @classmethod
    def new_trace(cls, parent_task_id: str | None = None) -> "TraceContext":
        return cls(trace_id=uuid.uuid4().hex, parent_task_id=parent_task_id)

    def child(self, causation_id: str) -> "TraceContext":
        """Derive a causally-linked trace context for a downstream message."""
        return TraceContext(
            trace_id=self.trace_id,
            parent_task_id=self.parent_task_id,
            causation_id=causation_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "parent_task_id": self.parent_task_id,
            "causation_id": self.causation_id,
        }


def _entry_point_to_text(entry_point: Any) -> str:
    if callable(entry_point):
        module = getattr(entry_point, "__module__", "")
        name = getattr(entry_point, "__qualname__", getattr(entry_point, "__name__", ""))
        if module and name:
            return f"{module}:{name}"
        return repr(entry_point)
    if isinstance(entry_point, str):
        return entry_point
    raise TypeError("entry_point must be a 'module:attr' string or a callable")


@dataclass(frozen=True)
class ComponentManifest:
    """Canonical component descriptor (integrated steps 1-3 and section 3).

    The manifest is the registration contract: identity and semantic version,
    owner, trust class, declared purpose, input/output schemas and protocol
    version, requested capabilities, resource profile with hard ceilings,
    isolation and health policies, data classification and retention, plus
    provenance and build identity. Unknown mandatory fields, invalid versions,
    undeclared privileges, and resource requests outside administrator bounds
    are rejected at validation time.
    """

    component_id: str
    version: str = "1.0.0"
    owner: str = "unassigned"
    trust_class: TrustClass = TrustClass.TRUSTED_SERVICE
    consequence: ConsequenceClass = ConsequenceClass.READ_ONLY
    purpose: str = ""
    entry_point: Any = "ore_runtime.testing:noop_worker"
    protocol_version: int = ENVELOPE_SCHEMA_VERSION
    input_schemas: tuple[str, ...] = ()
    output_schemas: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    resources: ResourceProfile = field(default_factory=ResourceProfile)
    isolation: IsolationProfile = field(default_factory=IsolationProfile)
    health: HealthPolicy = field(default_factory=HealthPolicy)
    data_classification: DataClassification = DataClassification.INTERNAL
    retention_policy: str = "session"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    build_identity: str = "unpinned"
    policy_compatibility: tuple[int, ...] = ()

    _MANIFEST_FIELDS = (
        "component_id",
        "version",
        "owner",
        "trust_class",
        "consequence",
        "purpose",
        "entry_point",
        "protocol_version",
        "input_schemas",
        "output_schemas",
        "capabilities",
        "dependencies",
        "resources",
        "isolation",
        "health",
        "data_classification",
        "retention_policy",
        "provenance",
        "build_identity",
        "policy_compatibility",
        "manifest_version",
    )

    def __post_init__(self) -> None:
        validate_component_id(self.component_id)
        validate_semantic_version(self.version)
        if not isinstance(self.owner, str) or not self.owner.strip():
            raise ValueError("owner must be a non-empty string")
        object.__setattr__(
            self, "trust_class", _coerce_enum_value(self.trust_class, TrustClass, "trust_class")
        )
        object.__setattr__(
            self, "consequence", _coerce_enum_value(self.consequence, ConsequenceClass, "consequence")
        )
        if not isinstance(self.protocol_version, int) or isinstance(self.protocol_version, bool):
            raise TypeError("protocol_version must be an int")
        if self.protocol_version != ENVELOPE_SCHEMA_VERSION:
            raise ValueError(
                f"protocol_version {self.protocol_version} is incompatible with "
                f"envelope schema {ENVELOPE_SCHEMA_VERSION}"
            )
        for name in ("input_schemas", "output_schemas", "capabilities", "dependencies"):
            if not isinstance(getattr(self, name), tuple):
                raise TypeError(f"{name} must be a tuple")
        for right in self.capabilities:
            validate_right(right, "capabilities")
        for dep in self.dependencies:
            if not isinstance(dep, str) or not _COMPONENT_ID_RE.match(dep):
                raise ValueError(f"dependency {dep!r} is not a valid component id")
        if not isinstance(self.resources, ResourceProfile):
            if isinstance(self.resources, Mapping):
                object.__setattr__(
                    self, "resources", ResourceProfile.from_dict(self.resources)
                )
            else:
                raise TypeError("resources must be a ResourceProfile")
        if not isinstance(self.isolation, IsolationProfile):
            if isinstance(self.isolation, Mapping):
                object.__setattr__(
                    self, "isolation", IsolationProfile.from_dict(self.isolation)
                )
            else:
                raise TypeError("isolation must be an IsolationProfile")
        if not isinstance(self.health, HealthPolicy):
            if isinstance(self.health, Mapping):
                object.__setattr__(self, "health", HealthPolicy.from_dict(self.health))
            else:
                raise TypeError("health must be a HealthPolicy")
        object.__setattr__(
            self,
            "data_classification",
            _coerce_enum_value(self.data_classification, DataClassification, "data_classification"),
        )
        if self.retention_policy not in {"session", "persistent"}:
            raise ValueError("retention_policy must be 'session' or 'persistent'")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("provenance must be a mapping")
        if not isinstance(self.build_identity, str) or not self.build_identity.strip():
            raise ValueError("build_identity must be a non-empty string")
        for compatible in self.policy_compatibility:
            if not isinstance(compatible, int) or isinstance(compatible, bool) or compatible < 1:
                raise ValueError("policy_compatibility entries must be positive integers")
        self._validate_isolation_constraints()
        # Trust-tier floor (integrated step 5): never place untrusted generated
        # code in-process merely for performance.
        if (
            self.trust_class == TrustClass.UNTRUSTED_GENERATED
            and self.isolation.tier == ExecutionTier.IN_PROCESS
        ):
            raise ManifestValidationError(
                "untrusted generated code cannot request the in-process tier"
            )

    def _validate_isolation_constraints(self) -> None:
        iso = self.isolation
        if self.trust_class == TrustClass.UNTRUSTED_GENERATED:
            if iso.ambient_credentials:
                raise ManifestValidationError(
                    "untrusted generated code must not receive ambient credentials"
                )
            if iso.host_home_access:
                raise ManifestValidationError(
                    "untrusted generated code must not access the host home directory"
                )
        if iso.network_enabled and "host.network" not in self.capabilities:
            raise ManifestValidationError(
                "isolation profile enables the network without the host.network right"
            )

    def validate_resource_bounds(
        self, bounds: ResourceVector, label: str = "administrator bounds"
    ) -> None:
        """Reject resource requests or ceilings outside administrator bounds (step 3)."""
        if not isinstance(bounds, ResourceVector):
            raise TypeError("bounds must be a ResourceVector")
        if not self.resources.requested.elementwise_le(bounds):
            exceeded = {
                kind: self.resources.requested.amounts()[kind]
                for kind in ResourceVector.kinds()
                if self.resources.requested.amounts()[kind] > bounds.amounts()[kind]
            }
            raise ManifestValidationError(
                f"resource request exceeds {label}: {exceeded}"
            )
        if not self.resources.limits.elementwise_le(bounds):
            exceeded = {
                kind: self.resources.limits.amounts()[kind]
                for kind in ResourceVector.kinds()
                if self.resources.limits.amounts()[kind] > bounds.amounts()[kind]
            }
            raise ManifestValidationError(f"hard ceilings exceed {label}: {exceeded}")

    def entry_callable(self) -> Any:
        """Import and return the entry point callable ('module:attr' form)."""
        if callable(self.entry_point):
            return self.entry_point
        entry = _entry_point_to_text(self.entry_point)
        module_name, sep, attr = entry.partition(":")
        if not sep or not module_name or not attr:
            raise ValueError(f"entry_point {entry!r} must use the 'module:attr' form")
        import importlib

        module = importlib.import_module(module_name)
        target: Any = module
        for part in attr.split("."):
            target = getattr(target, part)
        if not callable(target):
            raise TypeError(f"entry_point {entry!r} does not resolve to a callable")
        return target

    def manifest_digest(self) -> str:
        """Content digest of the manifest contract for registration evidence."""
        return sha256_hex(canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "version": self.version,
            "owner": self.owner,
            "trust_class": self.trust_class.value,
            "consequence": self.consequence.value,
            "purpose": self.purpose,
            "entry_point": _entry_point_to_text(self.entry_point),
            "protocol_version": self.protocol_version,
            "input_schemas": list(self.input_schemas),
            "output_schemas": list(self.output_schemas),
            "capabilities": list(self.capabilities),
            "dependencies": list(self.dependencies),
            "resources": self.resources.to_dict(),
            "isolation": self.isolation.to_dict(),
            "health": self.health.to_dict(),
            "data_classification": self.data_classification.value,
            "retention_policy": self.retention_policy,
            "provenance": dict(self.provenance),
            "build_identity": self.build_identity,
            "policy_compatibility": list(self.policy_compatibility),
            "manifest_version": MODULE_VERSION,
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "ComponentManifest":
        if not isinstance(mapping, Mapping):
            raise TypeError("manifest source must be a mapping")
        unknown = set(mapping).difference(cls._MANIFEST_FIELDS)
        if unknown:
            raise ManifestValidationError(
                f"unknown mandatory manifest fields rejected: {sorted(unknown)}"
            )
        if "component_id" not in mapping:
            raise ManifestValidationError("missing mandatory manifest field: component_id")
        resources = mapping.get("resources", {})
        isolation = mapping.get("isolation", {})
        health = mapping.get("health", {})
        return cls(
            component_id=mapping["component_id"],
            version=mapping.get("version", "1.0.0"),
            owner=mapping.get("owner", "unassigned"),
            trust_class=TrustClass.coerce(mapping.get("trust_class", "trusted_service")),
            consequence=ConsequenceClass.coerce(mapping.get("consequence", "read_only")),
            purpose=mapping.get("purpose", ""),
            entry_point=mapping.get("entry_point", "ore_runtime.testing:noop_worker"),
            protocol_version=int(mapping.get("protocol_version", ENVELOPE_SCHEMA_VERSION)),
            input_schemas=tuple(mapping.get("input_schemas", ())),
            output_schemas=tuple(mapping.get("output_schemas", ())),
            capabilities=tuple(mapping.get("capabilities", ())),
            dependencies=tuple(mapping.get("dependencies", ())),
            resources=ResourceProfile.from_dict(resources) if resources else ResourceProfile(),
            isolation=IsolationProfile.from_dict(isolation) if isolation else IsolationProfile(),
            health=HealthPolicy.from_dict(health) if health else HealthPolicy(),
            data_classification=DataClassification.coerce(
                mapping.get("data_classification", "internal")
            ),
            retention_policy=mapping.get("retention_policy", "session"),
            provenance=dict(mapping.get("provenance", {})),
            build_identity=mapping.get("build_identity", "unpinned"),
            policy_compatibility=tuple(mapping.get("policy_compatibility", ())),
        )


@dataclass(frozen=True)
class MessageEnvelope:
    """Typed, versioned request envelope (integrated step 10).

    Every cross-component request carries ``message_id``, ``correlation_id``,
    ``sender``, ``recipient``, ``schema_version``, ``deadline`` (monotonic
    seconds), ``trace`` context, ``payload``, and the authorization reference
    (capability id). Payloads are treated as immutable: receivers copy before
    mutating (step 24).
    """

    message_id: str = field(default_factory=lambda: generate_message_id())
    correlation_id: str | None = None
    sender: str = ""
    recipient: str = ""
    kind: str = "task.execute"
    schema_version: int = ENVELOPE_SCHEMA_VERSION
    deadline_monotonic_s: float | None = None
    priority: int = 5
    trace: TraceContext = field(default_factory=TraceContext.new_trace)
    payload: Mapping[str, Any] = field(default_factory=dict)
    auth_capability_id: str | None = None
    delivery: DeliverySemantics = DeliverySemantics.AT_LEAST_ONCE
    idempotency_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message_id, str) or not self.message_id.strip():
            raise ValueError("message_id must be non-empty")
        for name in ("sender", "recipient"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty component id")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("kind must be a non-empty message kind")
        if not isinstance(self.schema_version, int) or isinstance(self.schema_version, bool):
            raise TypeError("schema_version must be an int")
        if self.schema_version != ENVELOPE_SCHEMA_VERSION:
            raise EnvelopeValidationError(
                f"unsupported envelope schema version {self.schema_version} "
                f"(supported: {ENVELOPE_SCHEMA_VERSION})"
            )
        if isinstance(self.priority, bool) or not isinstance(self.priority, int):
            raise TypeError("priority must be an int between 0 and 10")
        if not 0 <= self.priority <= 10:
            raise ValueError("priority must be between 0 and 10")
        if not isinstance(self.trace, TraceContext):
            raise TypeError("trace must be a TraceContext")
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping")
        if self.deadline_monotonic_s is not None and self.deadline_monotonic_s <= 0:
            raise ValueError("deadline_monotonic_s must be positive when provided")
        object.__setattr__(
            self, "delivery", DeliverySemantics.coerce(self.delivery, "delivery")
        )

    def estimated_payload_bytes(self) -> int:
        """Approximate serialized size, used to reject oversized messages (step 10)."""
        return len(
            canonical_json({"kind": self.kind, "payload": dict(self.payload)}).encode("utf-8")
        )

    def deadline_remaining(self, now: float) -> float | None:
        """D_remaining on a monotonic clock (section 5.6); None means no deadline."""
        if self.deadline_monotonic_s is None:
            return None
        return self.deadline_monotonic_s - now

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "kind": self.kind,
            "schema_version": self.schema_version,
            "deadline_monotonic_s": self.deadline_monotonic_s,
            "priority": self.priority,
            "trace": self.trace.to_dict(),
            "payload": dict(self.payload),
            "auth_capability_id": self.auth_capability_id,
            "delivery": self.delivery.value,
            "idempotency_key": self.idempotency_key,
        }

    @classmethod
    def from_dict(cls, mapping: Mapping[str, Any]) -> "MessageEnvelope":
        if not isinstance(mapping, Mapping):
            raise TypeError("envelope source must be a mapping")
        trace = mapping.get("trace") or {}
        deadline = mapping.get("deadline_monotonic_s")
        return cls(
            message_id=mapping.get("message_id", generate_message_id()),
            correlation_id=mapping.get("correlation_id"),
            sender=mapping.get("sender", ""),
            recipient=mapping.get("recipient", ""),
            kind=mapping.get("kind", "task.execute"),
            schema_version=int(mapping.get("schema_version", ENVELOPE_SCHEMA_VERSION)),
            deadline_monotonic_s=float(deadline) if deadline is not None else None,
            priority=int(mapping.get("priority", 5)),
            trace=TraceContext(
                trace_id=trace.get("trace_id", uuid.uuid4().hex),
                parent_task_id=trace.get("parent_task_id"),
                causation_id=trace.get("causation_id"),
            ),
            payload=dict(mapping.get("payload", {})),
            auth_capability_id=mapping.get("auth_capability_id"),
            delivery=DeliverySemantics.coerce(mapping.get("delivery", "at_least_once")),
            idempotency_key=mapping.get("idempotency_key"),
        )


@dataclass(frozen=True)
class ResponseEnvelope:
    """Typed reply tied to its request through ``correlation_id`` (step 10)."""

    message_id: str = field(default_factory=lambda: generate_message_id("reply"))
    correlation_id: str = ""
    sender: str = ""
    recipient: str = ""
    status: OutcomeKind = OutcomeKind.OK
    result: Mapping[str, Any] | None = None
    reason: str = ""
    elapsed_monotonic_s: float = 0.0
    trace: TraceContext = field(default_factory=TraceContext.new_trace)

    def __post_init__(self) -> None:
        if not isinstance(self.correlation_id, str) or not self.correlation_id.strip():
            raise ValueError("correlation_id must reference the originating request")
        for name in ("sender", "recipient"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty component id")
        object.__setattr__(
            self, "status", _coerce_enum_value(self.status, OutcomeKind, "status")
        )
        if self.result is not None and not isinstance(self.result, Mapping):
            raise TypeError("result must be a mapping or None")
        if self.elapsed_monotonic_s < 0:
            raise ValueError("elapsed_monotonic_s must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "sender": self.sender,
            "recipient": self.recipient,
            "status": self.status.value,
            "result": dict(self.result) if self.result is not None else None,
            "reason": self.reason,
            "elapsed_monotonic_s": self.elapsed_monotonic_s,
            "trace": self.trace.to_dict(),
        }
