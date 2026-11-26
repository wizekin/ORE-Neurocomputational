"""zkVM-style committed program execution and proof verification (Phase F)."""
from __future__ import annotations

from .executor import (
    ADAPTER_ID,
    PRIVACY_NOTE,
    PROVIDES_PRIVACY,
    PROOF_KIND,
    VERIFIER_VERSION,
    ExecutionVerifier,
    ExecutionWorker,
    benchmark_execution_verification,
)
from .minivm import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MEMORY_SIZE,
    DEFAULT_MODULUS,
    OPCODES,
    RUNTIME_VERSION,
    ExecutionState,
    Instruction,
    MiniVM,
    Program,
    program_commitment,
)

__all__ = [
    "ADAPTER_ID",
    "DEFAULT_MAX_STEPS",
    "DEFAULT_MEMORY_SIZE",
    "DEFAULT_MODULUS",
    "OPCODES",
    "PRIVACY_NOTE",
    "PROVIDES_PRIVACY",
    "PROOF_KIND",
    "RUNTIME_VERSION",
    "VERIFIER_VERSION",
    "ExecutionState",
    "ExecutionVerifier",
    "ExecutionWorker",
    "Instruction",
    "MiniVM",
    "Program",
    "benchmark_execution_verification",
    "program_commitment",
]
