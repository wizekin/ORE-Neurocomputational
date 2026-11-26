"""Optional Lava adapter kept lazy and out of core startup (specification Step 43).

The adapter never imports Lava at module import time; dependencies stay optional, and
core startup is unaffected whether or not neuromorphic frameworks are installed.
"""
from __future__ import annotations

import importlib
from typing import Any, Mapping

from ..representations.portable_graph import PortableGraph
from .capabilities import BackendCapabilities

__all__ = ("LavaBackendAdapter", "OptionalDependencyError")


class OptionalDependencyError(RuntimeError):
    """Raised when an optional backend framework is requested but unavailable."""


class LavaBackendAdapter:
    """Isolate Lava as an optional framework adapter for neuromorphic experiments."""

    def __init__(self, backend_hint: str = "loihi2") -> None:
        """Record the intended target device without importing anything."""
        self.backend_hint = backend_hint
        self._lava_module: Any = None

    @property
    def name(self) -> str:
        """Return the adapter name."""
        return f"lava_adapter[{self.backend_hint}]"

    def available(self) -> bool:
        """Attempt a lazy import of Lava; failure is reported, never raised."""
        if self._lava_module is not None:
            return True
        try:
            self._lava_module = importlib.import_module("lava")
            return True
        except ImportError:
            return False

    def capabilities(self) -> BackendCapabilities:
        """Return the adapter's declared capability profile."""
        available = self.available()
        return BackendCapabilities(
            backend_name=self.name,
            supported_operations=("lif_cell", "linear", "threshold") if available else (),
            precision_bits=8 if available else 0,
            memory_mb=512.0 if available else 0.0,
            event_routing=available,
            runtime_version=str(getattr(self._lava_module, "__version__", "unknown"))
            if available else "unavailable",
            available=available,
            notes="optional neuromorphic research adapter; never imported during core startup",
        )

    def execute(self, graph: PortableGraph, inputs: Mapping[str, Any],
                states: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """Execute a portable graph on Lava, or raise OptionalDependencyError."""
        if not self.available():
            raise OptionalDependencyError(
                "the optional Lava framework is not installed; the deterministic CPU "
                "reference backend remains the default development path")
        raise NotImplementedError(
            "a Lava process model for this graph requires a neuromorphic research "
            "environment; capability negotiation falls back to the CPU reference backend")
