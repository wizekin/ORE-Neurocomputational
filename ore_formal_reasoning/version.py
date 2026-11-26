"""Version and schema identity for the ORE Formal Reasoning and Constraint Engine.

The engine implements ``ORE Integrated Algorithm — Point 5: Automated Reasoning,
Theorem Proving, SAT/SMT Solvers and Constraint-Based Intelligence`` as the
``ore_formal_reasoning`` package. Every externally visible artifact carries the
schema versions declared here so results stay replayable and auditable.
"""
from __future__ import annotations

FORMAL_REASONING_PACKAGE_VERSION = "1.0.0"
FORMAL_REASONING_SCHEMA_VERSION = "1.0.0"
FORMAL_REASONING_IR_VERSION = "FRCE-IR/1.0.0"
FORMAL_REASONING_RESULT_VERSION = "FRCE-RESULT/1.0.0"
FORMAL_REASONING_BENCHMARK_VERSION = "FRCE-BENCH/1.0.0"

FRCE_PRODUCER = "ore_formal_reasoning"

# Contract identifiers used when the engine registers with the ORE contract
# catalog and publishes typed artifacts to the shared cognitive workspace.
FRCE_REQUEST_CONTRACT = "ORE.FRCE.FORMALIZATION_REQUEST"
FRCE_RESULT_CONTRACT = "ORE.FRCE.RESULT"
FRCE_FINDING_BATCH_CONTRACT = "ORE.FRCE.FINDING_BATCH"
FRCE_CONTRACT_VERSION = "1.0.0"

__all__ = [
    "FORMAL_REASONING_PACKAGE_VERSION",
    "FORMAL_REASONING_SCHEMA_VERSION",
    "FORMAL_REASONING_IR_VERSION",
    "FORMAL_REASONING_RESULT_VERSION",
    "FORMAL_REASONING_BENCHMARK_VERSION",
    "FRCE_PRODUCER",
    "FRCE_REQUEST_CONTRACT",
    "FRCE_RESULT_CONTRACT",
    "FRCE_FINDING_BATCH_CONTRACT",
    "FRCE_CONTRACT_VERSION",
]
