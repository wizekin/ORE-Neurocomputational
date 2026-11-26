"""Backend registry: CPU reference, optimized CPU, and optional accelerators.

Backends declare identity, supported dtypes, and operation eligibility
(integrated step 15 uses these declarations for candidate cost estimates).
No GPU or specialized backend is claimed unless the hardware profile actually
reports one; portable code always retains the CPU reference path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import HardwareProfile, OperationKind, OptimizationRequest

_BACKEND_OPERATIONS: dict[str, tuple[OperationKind, ...]] = {
    "cpu_reference": (
        OperationKind.ATTENTION,
        OperationKind.MATRIX_MULTIPLY,
        OperationKind.SIMILARITY,
        OperationKind.FACTORIZATION,
        OperationKind.LEAST_SQUARES,
        OperationKind.TRANSFORM,
        OperationKind.REDUCTION,
        OperationKind.OPTIMIZATION,
        OperationKind.OTHER,
    ),
    "cpu_optimized": (
        OperationKind.ATTENTION,
        OperationKind.MATRIX_MULTIPLY,
        OperationKind.FACTORIZATION,
        OperationKind.LEAST_SQUARES,
        OperationKind.TRANSFORM,
        OperationKind.OPTIMIZATION,
    ),
}


@dataclass(frozen=True)
class BackendSpec:
    """Declaration of one execution backend."""

    name: str
    kind: str
    supported_dtypes: tuple[str, ...]
    supported_operations: tuple[OperationKind, ...]
    available: bool
    notes: tuple[str, ...] = ()

    def supports(self, request: OptimizationRequest) -> bool:
        """Check dtype and operation eligibility for one request."""
        return (
            request.dtype in self.supported_dtypes
            and request.kind in self.supported_operations
            and self.available
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "supported_dtypes": list(self.supported_dtypes),
            "supported_operations": [item.value for item in self.supported_operations],
            "available": self.available,
            "notes": list(self.notes),
        }


def reference_cpu_backend() -> BackendSpec:
    """Trusted float64 CPU reference backend."""
    return BackendSpec(
        name="cpu_reference",
        kind="reference",
        supported_dtypes=("float64",),
        supported_operations=_BACKEND_OPERATIONS["cpu_reference"],
        available=True,
        notes=("trusted conventional implementations; always available",),
    )


def optimized_cpu_backend(profile: HardwareProfile | None = None) -> BackendSpec:
    """Optimized CPU backend (tiled/blocked/mixed-precision kernels)."""
    return BackendSpec(
        name="cpu_optimized",
        kind="optimized",
        supported_dtypes=("float64", "float32", "float16", "bfloat16"),
        supported_operations=_BACKEND_OPERATIONS["cpu_optimized"],
        available=True,
        notes=(
            "tiled attention, randomized linalg, structured and mixed-precision kernels",
            "every optimized path retains the reference fallback",
        ),
    )


def discover_backends(profile: HardwareProfile | None = None) -> tuple[BackendSpec, ...]:
    """List available backends from the hardware profile (no false claims)."""
    backends = [reference_cpu_backend(), optimized_cpu_backend(profile)]
    if profile is not None and profile.accelerator_types:
        for accelerator in profile.accelerator_types:
            backends.append(
                BackendSpec(
                    name=f"accelerator_{accelerator}",
                    kind="accelerator",
                    supported_dtypes=("float16", "bfloat16", "float32", "float64"),
                    supported_operations=(
                        OperationKind.ATTENTION,
                        OperationKind.MATRIX_MULTIPLY,
                    ),
                    available=True,
                    notes=("optional specialized backend; requires measured evidence",),
                )
            )
    return tuple(backends)


def select_backends(request: OptimizationRequest, backends: tuple[BackendSpec, ...]) -> tuple[BackendSpec, ...]:
    """Order backends eligible for a request; reference first for safety."""
    eligible = tuple(backend for backend in backends if backend.supports(request))
    reference = tuple(b for b in eligible if b.kind == "reference")
    others = tuple(b for b in eligible if b.kind != "reference")
    return reference + others
