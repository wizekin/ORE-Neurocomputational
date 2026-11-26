"""Backend capability negotiation and safe fallback (specification Step 44)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..representations.portable_graph import SUPPORTED_GRAPH_OPERATIONS

__all__ = ("BackendCapabilities", "CapabilityNegotiation", "negotiate_capabilities")


@dataclass(frozen=True, slots=True)
class BackendCapabilities:
    """Declared capabilities of one backend or device (Step 44)."""

    backend_name: str
    supported_operations: tuple[str, ...]
    precision_bits: int
    memory_mb: float
    event_routing: bool
    runtime_version: str
    available: bool = True
    notes: str = ""

    def supports(self, operation: str) -> bool:
        """Report whether an operation is supported."""
        return operation in self.supported_operations

    def to_dict(self) -> dict[str, Any]:
        """Serialize the capability profile."""
        return {
            "backend_name": self.backend_name,
            "supported_operations": list(self.supported_operations),
            "precision_bits": self.precision_bits,
            "memory_mb": self.memory_mb,
            "event_routing": self.event_routing,
            "runtime_version": self.runtime_version,
            "available": self.available,
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class CapabilityNegotiation:
    """The outcome of capability negotiation for one workload."""

    selected_backend: str
    fallback_reason: str
    required_operations: tuple[str, ...] = field(default_factory=tuple)
    unsupported_operations: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the negotiation result."""
        return {
            "selected_backend": self.selected_backend,
            "fallback_reason": self.fallback_reason,
            "required_operations": list(self.required_operations),
            "unsupported_operations": list(self.unsupported_operations),
        }


def negotiate_capabilities(required_operations: tuple[str, ...],
                           candidates: tuple[BackendCapabilities, ...],
                           required_precision_bits: int = 32,
                           required_memory_mb: float = 1.0,
                           require_event_routing: bool = False) -> CapabilityNegotiation:
    """Select the first capable backend; capability mismatch triggers safe CPU fallback."""
    for candidate in candidates:
        if not candidate.available:
            continue
        unsupported = tuple(op for op in required_operations
                            if not candidate.supports(op))
        if unsupported:
            continue
        if candidate.precision_bits < required_precision_bits:
            continue
        if candidate.memory_mb < required_memory_mb:
            continue
        if require_event_routing and not candidate.event_routing:
            continue
        return CapabilityNegotiation(selected_backend=candidate.backend_name,
                                     fallback_reason="",
                                     required_operations=required_operations)
    return CapabilityNegotiation(
        selected_backend="cpu_reference",
        fallback_reason="no candidate backend satisfied the required operations, precision, "
                        "memory, or availability; falling back to the deterministic CPU "
                        "reference simulator",
        required_operations=required_operations,
        unsupported_operations=tuple(op for op in required_operations
                                     if op not in SUPPORTED_GRAPH_OPERATIONS))
