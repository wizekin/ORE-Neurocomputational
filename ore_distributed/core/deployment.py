"""Deployment topology detection and descriptor contracts (Step 1).

A single-process deployment must keep using local coordination primitives and must
not instantiate a consensus cluster merely because modules share a workspace;
stronger coordination activates only across process or machine boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any, Mapping


class DeploymentMode(str, Enum):
    """Classify how ORE is deployed for coordination purposes."""

    SINGLE_PROCESS = "single_process"
    MULTI_PROCESS_LOCAL = "multi_process_local"
    MULTI_MACHINE_CLUSTER = "multi_machine_cluster"

    @property
    def spans_process_boundary(self) -> bool:
        """Return whether this mode crosses a process boundary and needs transport."""
        return self is not DeploymentMode.SINGLE_PROCESS

    @property
    def spans_machine_boundary(self) -> bool:
        """Return whether this mode crosses a machine boundary and needs network placement."""
        return self is DeploymentMode.MULTI_MACHINE_CLUSTER


#: Transport endpoint kinds understood by the topology descriptor.
_TRANSPORT_KINDS = frozenset({"inproc", "local_socket", "tcp", "grpc", "custom"})


def _non_empty(name: str, value: str) -> str:
    """Return stripped non-empty text or raise a topology contract error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class TransportEndpoint:
    """Describe one advertised transport endpoint for a node."""

    kind: str
    address: str

    def __post_init__(self) -> None:
        """Validate endpoint kind and address text."""
        object.__setattr__(self, "kind", _non_empty("kind", self.kind))
        if self.kind not in _TRANSPORT_KINDS:
            raise ValueError(f"unsupported transport kind {self.kind!r}")
        object.__setattr__(self, "address", _non_empty("address", self.address))

    def to_dict(self) -> dict[str, str]:
        """Return the endpoint as a plain dictionary."""
        return {"kind": self.kind, "address": self.address}


@dataclass(frozen=True, slots=True)
class TopologyDescriptor:
    """Carry node identity, cluster identity, and deployment mode (Step 1).

    A topology descriptor is loaded at startup and decides which coordination
    primitives a node may instantiate; identity never comes from an IP address
    or a transient container identifier.
    """

    node_id: str
    cluster_id: str
    mode: DeploymentMode
    endpoints: tuple[TransportEndpoint, ...] = ()
    supported_protocol_versions: tuple[str, ...] = ("ORE.DIST/0.1.0",)
    roles: tuple[str, ...] = ("worker",)
    extra: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate identity fields, endpoints, and protocol versions."""
        object.__setattr__(self, "node_id", _non_empty("node_id", self.node_id))
        object.__setattr__(self, "cluster_id", _non_empty("cluster_id", self.cluster_id))
        if not isinstance(self.mode, DeploymentMode):
            raise ValueError("mode must be a DeploymentMode value")
        endpoints = tuple(
            item if isinstance(item, TransportEndpoint) else TransportEndpoint(**item)
            for item in self.endpoints
        )
        object.__setattr__(self, "endpoints", endpoints)
        versions = tuple(_non_empty("protocol version", v) for v in self.supported_protocol_versions)
        if not versions:
            raise ValueError("supported_protocol_versions must not be empty")
        object.__setattr__(self, "supported_protocol_versions", versions)
        roles = tuple(_non_empty("role", r) for r in self.roles) or ("worker",)
        object.__setattr__(self, "roles", roles)
        object.__setattr__(self, "extra", dict(self.extra))

    def endpoint_for(self, kind: str) -> TransportEndpoint | None:
        """Return the first endpoint of the requested kind, if advertised."""
        for endpoint in self.endpoints:
            if endpoint.kind == kind:
                return endpoint
        return None

    def to_dict(self) -> dict[str, Any]:
        """Return the descriptor as a JSON-compatible dictionary."""
        return {
            "node_id": self.node_id,
            "cluster_id": self.cluster_id,
            "mode": self.mode.value,
            "endpoints": [item.to_dict() for item in self.endpoints],
            "supported_protocol_versions": list(self.supported_protocol_versions),
            "roles": list(self.roles),
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TopologyDescriptor":
        """Rebuild a descriptor from its dictionary form."""
        mode = DeploymentMode(str(payload["mode"]))
        endpoints = tuple(
            TransportEndpoint(**item) for item in payload.get("endpoints", ())
        )
        return cls(
            node_id=str(payload["node_id"]),
            cluster_id=str(payload["cluster_id"]),
            mode=mode,
            endpoints=endpoints,
            supported_protocol_versions=tuple(payload.get("supported_protocol_versions", ("ORE.DIST/0.1.0",))),
            roles=tuple(payload.get("roles", ("worker",))),
            extra=dict(payload.get("extra", {})),
        )


def detect_deployment_mode(
    *,
    process_count: int,
    machine_count: int | None = None,
    cluster_id: str | None = None,
) -> DeploymentMode:
    """Classify the deployment from explicit process and machine counts (Step 1).

    The classifier only reads supplied counts; it never infers the mode from
    ambient process tables or hostnames, so results stay deterministic in tests.
    """

    if not isinstance(process_count, int) or process_count < 1:
        raise ValueError("process_count must be a positive integer")
    if machine_count is not None and (not isinstance(machine_count, int) or machine_count < 1):
        raise ValueError("machine_count must be a positive integer when supplied")
    if cluster_id is not None:
        _non_empty("cluster_id", cluster_id)
    if process_count == 1 and (machine_count is None or machine_count == 1):
        return DeploymentMode.SINGLE_PROCESS
    if machine_count is None or machine_count == 1:
        return DeploymentMode.MULTI_PROCESS_LOCAL
    return DeploymentMode.MULTI_MACHINE_CLUSTER


def deployment_allows_consensus(mode: DeploymentMode) -> bool:
    """Return whether a deployment mode may instantiate a consensus cluster."""
    return mode.spans_process_boundary


def load_topology(path: str | Path) -> TopologyDescriptor:
    """Load and validate a topology descriptor file from disk."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"topology descriptor not found: {file_path}")
    payload: dict[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
    return TopologyDescriptor.from_dict(payload)


__all__ = [
    "DeploymentMode",
    "TransportEndpoint",
    "TopologyDescriptor",
    "detect_deployment_mode",
    "deployment_allows_consensus",
    "load_topology",
]
