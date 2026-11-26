"""Portability and hardware abstraction (specification Phase F, Steps 40-45)."""
from __future__ import annotations

from .capabilities import BackendCapabilities, CapabilityNegotiation, negotiate_capabilities
from .contract import TemporalBackend
from .cpu_reference import CPUReferenceBackend, GraphExecutionResult
from .export import export_model_graph, export_baseline_graph
from .lava_adapter import LavaBackendAdapter, OptionalDependencyError
from .benchmark import BackendBenchmark, BenchmarkEntry, benchmark_identical_workload

__all__ = (
    "BackendBenchmark",
    "BackendCapabilities",
    "BenchmarkEntry",
    "CPUReferenceBackend",
    "CapabilityNegotiation",
    "GraphExecutionResult",
    "LavaBackendAdapter",
    "OptionalDependencyError",
    "TemporalBackend",
    "benchmark_identical_workload",
    "export_baseline_graph",
    "export_model_graph",
    "negotiate_capabilities",
)
