"""CROS adapter: bounded resource allocation for HIC compilation stages.

Thin, policy-driven adapter that allocates compute, memory and time budgets to
HIC stages under explicit bounds. It is the integration seam for the full CROS
resource-coordination subsystem: when the repository-wide CROS contracts are
available they can replace this adapter without changing HIC call sites.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import Capability, GateViolation


@dataclass(frozen=True, slots=True)
class StageBudget:
    """One granted resource budget for a pipeline stage."""

    stage: str
    max_cpu_seconds: float
    max_memory_mb: int
    max_wall_seconds: float
    capabilities: tuple[Capability, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_memory_mb": self.max_memory_mb,
            "max_wall_seconds": self.max_wall_seconds,
            "capabilities": [c.value for c in self.capabilities],
        }


_DEFAULT_STAGE_POLICY: dict[str, tuple[float, int, float]] = {
    # stage: (cpu_seconds, memory_mb, wall_seconds)
    "A": (1.0, 64, 2.0),
    "B": (2.0, 128, 5.0),
    "C": (2.0, 128, 5.0),
    "D": (5.0, 256, 15.0),
    "E": (5.0, 256, 15.0),
    "F": (60.0, 512, 120.0),
    "G": (120.0, 512, 300.0),
    "H": (30.0, 256, 60.0),
}


class CROSAdapter:
    """Allocate bounded execution budgets per HIC stage."""

    def __init__(
        self,
        stage_policy: Mapping[str, tuple[float, int, float]] | None = None,
        global_capabilities: Iterable[Capability] = (Capability.COMPUTE, Capability.MEMORY),
    ) -> None:
        self.stage_policy = dict(stage_policy or _DEFAULT_STAGE_POLICY)
        self.global_capabilities = tuple(global_capabilities)
        self._granted: list[StageBudget] = []

    def allocate(self, stage: str, *, requested_cpu: float | None = None) -> StageBudget:
        """Grant a stage budget, clamped to the policy bound (never above it)."""
        if stage not in self.stage_policy:
            raise GateViolation(f"no resource policy for stage '{stage}'")
        cpu, memory, wall = self.stage_policy[stage]
        granted_cpu = min(requested_cpu, cpu) if requested_cpu is not None else cpu
        budget = StageBudget(stage, granted_cpu, memory, wall, self.global_capabilities)
        self._granted.append(budget)
        return budget

    @property
    def grants(self) -> tuple[StageBudget, ...]:
        return tuple(self._granted)

    def describe(self) -> dict[str, Any]:
        return {
            "stage_policy": {k: list(v) for k, v in sorted(self.stage_policy.items())},
            "global_capabilities": [c.value for c in self.global_capabilities],
            "grants": [g.to_dict() for g in self._granted],
        }
