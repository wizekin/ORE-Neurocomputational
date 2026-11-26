"""ORE Point 18: probabilistic data structures, streaming algorithms, storage and data integrity.

A layered data infrastructure for ORE-Neurocomputational that lets the system
process, search, store, compare, monitor and verify very large quantities of
information without every operation scanning the whole knowledge base:

- exact structures when correctness requires exact membership, counts,
  ordering or records;
- probabilistic structures (Bloom filters, Count-Min sketches, HyperLogLog)
  when bounded error is acceptable for filtering, estimation or candidate
  generation;
- streaming algorithms (KLL quantiles, Space-Saving heavy hitters) when
  observations arrive continuously and retaining the full stream is too
  expensive;
- durable storage structures (LSM write path, content-addressed object store)
  when write volume, retrieval patterns, recovery and historical
  traceability justify them;
- cryptographic integrity structures (Merkle commitments, Reed-Solomon
  erasure archives) when ORE must prove stored artifacts and historical
  records have not changed.

The central rule (source 1.2): approximate structures may accelerate or
prioritize work; they must never silently become authorities for semantic
truth, verified membership, exact counts or evidence correctness.
"""
from ore_data.registry import (
    ORE_DATA_CONTRACT,
    ORE_DATA_VERSION,
    OPERATION_REGISTRY,
    OperationSpec,
    operation_count,
    phases,
    registry_as_dicts,
)
from ore_data.pipeline import (
    DataInfrastructureCapabilities,
    DataInfrastructurePipeline,
    PipelineConfig,
    build_pipeline,
)
from ore_data.adapters import verify_protocol_implementations

__version__ = ORE_DATA_VERSION

__all__ = [
    "ORE_DATA_CONTRACT",
    "ORE_DATA_VERSION",
    "OPERATION_REGISTRY",
    "OperationSpec",
    "operation_count",
    "phases",
    "registry_as_dicts",
    "DataInfrastructureCapabilities",
    "DataInfrastructurePipeline",
    "PipelineConfig",
    "build_pipeline",
    "verify_protocol_implementations",
    "__version__",
]
