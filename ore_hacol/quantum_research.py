"""Quantum algorithm research records and gating (integrated steps 46-48).

Shor, Grover, and quantum error correction are research awareness, not
general-purpose ORE accelerators. This module keeps them as records with
explicit assumptions, models quantum error-correction overhead with standard
literature approximations, and gates any production routing: no ordinary ORE
database search, factorization, or scheduling is routed to quantum algorithms
without a valid mapping and a suitable fault-tolerant device. Ideal query
complexity is never compared directly with end-to-end classical runtime.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .contracts import HardwareProfile

#: Textbook circuit bound for Shor's algorithm on n-bit integers.
SHOR_TEXTBOOK_LOGICAL_QUBITS_PER_BIT = 2.0


@dataclass(frozen=True)
class GroverRecord:
    """Research record for Grover's unstructured-search algorithm (step 47)."""

    search_space: int
    classical_queries_expected: int
    quantum_queries_ideal: int
    oracle_cost_model: str
    error_correction_overhead_factor: float
    data_loading_assumption: str
    status: str = "research_only"
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "grover",
            "search_space": self.search_space,
            "classical_queries_expected": self.classical_queries_expected,
            "quantum_queries_ideal": self.quantum_queries_ideal,
            "oracle_cost_model": self.oracle_cost_model,
            "error_correction_overhead_factor": self.error_correction_overhead_factor,
            "data_loading_assumption": self.data_loading_assumption,
            "status": self.status,
            "assumptions": list(self.assumptions),
        }


def grover_query_bound(search_space: int) -> tuple[int, int]:
    """Return (classical expected queries, ideal quantum oracle queries).

    Grover uses O(sqrt(N)) oracle queries in the ideal query model versus O(N)
    classical exhaustive queries; oracle construction, data loading, error
    correction, and hardware overhead are reported separately and never
    folded silently into the query count.
    """
    if search_space <= 0:
        raise ValueError("search_space must be positive")
    classical = int(search_space)
    quantum = int(math.ceil((math.pi / 4.0) * math.sqrt(search_space)))
    return classical, quantum


def grover_record(search_space: int, *, qec_distance: int = 25) -> GroverRecord:
    """Build the research record for a Grover search of the given size."""
    classical, quantum = grover_query_bound(search_space)
    physical = surface_code_physical_qubits(quantum, qec_distance)
    return GroverRecord(
        search_space=search_space,
        classical_queries_expected=classical,
        quantum_queries_ideal=quantum,
        oracle_cost_model="oracle construction cost not included; requires a valid reversible embedding",
        error_correction_overhead_factor=float(physical / max(1, quantum)),
        data_loading_assumption="QRAM-style data loading assumed in the ideal model; not available on current hardware",
        status="research_only",
        assumptions=(
            "ideal unit-query model",
            "fault-tolerant logical qubits required",
            "end-to-end classical runtime comparison requires oracle, queue, embedding, and correction overhead",
        ),
    )


@dataclass(frozen=True)
class ShorRecord:
    """Research record for Shor's factoring algorithm (step 47)."""

    bit_length: int
    logical_qubits_estimate: int
    gate_count_model: str
    status: str = "research_awareness_only"
    assumptions: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": "shor",
            "bit_length": self.bit_length,
            "logical_qubits_estimate": self.logical_qubits_estimate,
            "gate_count_model": self.gate_count_model,
            "status": self.status,
            "assumptions": list(self.assumptions),
        }


def shor_record(bit_length: int) -> ShorRecord:
    """Record Shor's scope for an n-bit integer with explicit assumptions.

    The 2n + 3 logical-qubit figure is the textbook circuit bound; modern
    compiled resource estimates differ substantially and neither figure
    implies a practically available machine.
    """
    logical = int(2 * bit_length + 3)
    return ShorRecord(
        bit_length=bit_length,
        logical_qubits_estimate=logical,
        gate_count_model=(
            "polynomial-time modular-exponentiation circuits; T-gate counts are "
            "research estimates and depend on compilation"
        ),
        status="research_awareness_only",
        assumptions=(
            "large fault-tolerant quantum computer required",
            "textbook circuit bound used for logical qubits",
            "no practical impact on ORE workloads today",
        ),
    )


def surface_code_physical_qubits(logical_qubits: int, code_distance: int) -> int:
    """Physical qubits for logical qubits under a surface-code layout.

    Standard literature approximation: approximately 2 d^2 physical qubits per
    logical qubit (data plus ancilla patches) for a rotationally symmetric
    surface-code layout. This is a planning heuristic, not a hardware claim.
    """
    if code_distance < 3 or code_distance % 2 == 0:
        raise ValueError("code_distance must be an odd integer >= 3")
    if logical_qubits < 0:
        raise ValueError("logical_qubits must be non-negative")
    return int(2 * code_distance**2 * logical_qubits)


def logical_error_rate_per_cycle(physical_error_rate: float, code_distance: int) -> float:
    """Standard surface-code logical error approximation (step 48).

    p_L ~ 0.1 * (100 p)^((d + 1) / 2) for physical error p below the ~1%
    threshold; Fowler et al. (2012)-style scaling used as a planning estimate
    with the constant labelled as a literature approximation.
    """
    if not 0.0 < physical_error_rate < 0.01:
        raise ValueError("physical_error_rate must be in (0, 0.01)")
    if code_distance < 3 or code_distance % 2 == 0:
        raise ValueError("code_distance must be an odd integer >= 3")
    exponent = (code_distance + 1) / 2.0
    return float(0.1 * (100.0 * physical_error_rate) ** exponent)


@dataclass(frozen=True)
class QECOverheadRecord:
    """Quantum error-correction overhead record (integrated step 48)."""

    code: str
    code_distance: int
    physical_error_rate: float
    logical_qubits: int
    physical_qubits: int
    logical_error_rate_per_cycle: float
    magic_state_distillation: str
    magic_state_overhead_factor: int
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "code_distance": self.code_distance,
            "physical_error_rate": self.physical_error_rate,
            "logical_qubits": self.logical_qubits,
            "physical_qubits": self.physical_qubits,
            "logical_error_rate_per_cycle": self.logical_error_rate_per_cycle,
            "magic_state_distillation": self.magic_state_distillation,
            "magic_state_overhead_factor": self.magic_state_overhead_factor,
            "notes": list(self.notes),
        }


def qec_overhead_record(
    *,
    logical_qubits: int = 1,
    code_distance: int = 25,
    physical_error_rate: float = 1e-3,
) -> QECOverheadRecord:
    """Assemble the QEC overhead record for planning and comparisons."""
    return QECOverheadRecord(
        code="surface_code",
        code_distance=code_distance,
        physical_error_rate=physical_error_rate,
        logical_qubits=logical_qubits,
        physical_qubits=surface_code_physical_qubits(logical_qubits, code_distance),
        logical_error_rate_per_cycle=logical_error_rate_per_cycle(
            physical_error_rate, code_distance
        ),
        magic_state_distillation="15_to_1 (Bravyi-Kitaev-style) distillation",
        magic_state_overhead_factor=15,
        notes=(
            "2 d^2 physical qubits per logical qubit is a literature planning approximation",
            "syndrome extraction, decoding, and magic-state costs dominate runtime",
            "never compare ideal query complexity directly with end-to-end classical runtime",
        ),
    )


#: Production task kinds that must never be routed to quantum algorithms.
_FORBIDDEN_PRODUCTION_KINDS = frozenset(
    {"database_search", "factorization", "scheduling", "ordinary_search"}
)


def quantum_research_gate(
    task_kind: str,
    hardware: HardwareProfile | None = None,
    *,
    fault_tolerant_qpu_available: bool = False,
) -> dict[str, Any]:
    """Decide whether any quantum path may run (integrated step 47 gate).

    Without a valid mapping and a suitable fault-tolerant device, every
    quantum algorithm remains a research record and ordinary ORE tasks are
    never routed to quantum backends.
    """
    kind = str(task_kind).strip().lower()
    reason = (
        "no fault-tolerant quantum capability is available; quantum paths remain "
        "research records and never run in the ORE core"
    )
    selected = False
    if fault_tolerant_qpu_available and kind not in _FORBIDDEN_PRODUCTION_KINDS:
        selected = True
        reason = (
            "fault-tolerant capability declared and mapping validated; still "
            "requires end-to-end overhead accounting before any speedup claim"
        )
    if kind in _FORBIDDEN_PRODUCTION_KINDS:
        reason = (
            f"ordinary ORE '{kind}' tasks are never routed to quantum algorithms "
            "without a valid mapping and a suitable fault-tolerant device"
        )
        selected = False
    return {
        "task_kind": kind,
        "selected": selected,
        "decision": "research_only" if not selected else "experimental_admission",
        "reason": reason,
        "quantum_runtime_in_core": False,
    }


def research_catalog(search_space: int = 1_000_000, bit_length: int = 2048) -> dict[str, Any]:
    """Full research record set for documentation and the demo CLI."""
    grover = grover_record(search_space)
    shor = shor_record(bit_length)
    qec = qec_overhead_record()
    return {
        "grover": grover.to_dict(),
        "shor": shor.to_dict(),
        "quantum_error_correction": qec.to_dict(),
        "gate": quantum_research_gate("database_search"),
        "status": (
            "Shor, Grover, and quantum error correction are research awareness; "
            "no quantum runtime exists in the ORE core and no production task is routed"
        ),
    }
