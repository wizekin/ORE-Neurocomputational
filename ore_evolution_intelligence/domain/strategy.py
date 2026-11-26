"""Strategy candidates, lifecycle states and lineage (sections 2.1, 11).

Candidate representation (section 2.1): ``s = (g, theta, pi, C, v, p)`` — genotype /
program structure, numerical parameters, decision policy, allowed capabilities and
constraints, version, and provenance/lineage. Genotypes are serializable, schema-valid
and restricted to the explicit action vocabulary (step 8); unrestricted Python
execution is structurally impossible because a genome is *data interpreted by the
sandbox*, never executed source code.

Strict lifecycle states (section 11): GENERATED, SANDBOX_TESTED, VALIDATED,
APPROVED, CANARY, ACTIVE, DEPRECATED, REJECTED. The production coordinator only ever
selects APPROVED / CANARY / ACTIVE strategies.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Mapping

from .genome import ParameterGenome, PolicyTableGenome, canonical_json, content_digest

#: Strict lifecycle states (section 11).
LIFECYCLE_STATES: tuple[str, ...] = (
    "GENERATED", "SANDBOX_TESTED", "VALIDATED", "APPROVED",
    "CANARY", "ACTIVE", "DEPRECATED", "REJECTED",
)

#: States from which the production coordinator may select a strategy.
PRODUCTION_ELIGIBLE_STATES: frozenset[str] = frozenset({"APPROVED", "CANARY", "ACTIVE"})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


@dataclass(frozen=True, slots=True)
class LineageRecord:
    """Provenance for one candidate: parents, operation, generation, run (step 27)."""

    parents: tuple[str, ...] = ()
    operation: str = "initial"  # initial | mutation | crossover | archive_seed | import
    generation: int = 0
    run_id: str | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "parents": list(self.parents), "operation": self.operation,
            "generation": self.generation, "run_id": self.run_id, "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """One candidate strategy: genome + capabilities + version + provenance (2.1).

    Immutable; state transitions happen only through the promotion policy which
    returns a new candidate instance.
    """

    genome: Any
    capabilities: frozenset[str]
    version: int = 1
    lineage: LineageRecord = field(default_factory=LineageRecord)
    state: str = "GENERATED"
    run_id: str | None = None
    created_at_utc: str = field(default_factory=_utc_now)
    content_hash: str = ""
    candidate_id: str = ""

    def __post_init__(self) -> None:
        if self.state not in LIFECYCLE_STATES:
            raise ValueError(f"unknown lifecycle state {self.state!r}")
        if not isinstance(self.version, int) or self.version < 1:
            raise ValueError("version must be a positive integer")
        errors = self.genome.validate()
        if errors:
            raise ValueError(f"candidate genome is not schema-valid: {errors[0]}")
        if not self.content_hash:
            object.__setattr__(
                self, "content_hash", self.compute_content_hash(self.genome, self.capabilities, self.version),
            )
        if not self.candidate_id:
            object.__setattr__(self, "candidate_id", f"cand_{self.content_hash[:12]}")

    @staticmethod
    def compute_content_hash(genome: Any, capabilities: frozenset[str], version: int) -> str:
        payload = {
            "genome": genome.content_payload(),
            "capabilities": sorted(capabilities),
            "version": version,
        }
        return sha256(canonical_json(payload).encode("utf-8")).hexdigest()

    # -- construction helpers ----------------------------------------------
    @classmethod
    def from_genome(
        cls,
        genome: Any,
        *,
        lineage: LineageRecord | None = None,
        run_id: str | None = None,
        version: int = 1,
        state: str = "GENERATED",
    ) -> "StrategyCandidate":
        return cls(
            genome=genome,
            capabilities=genome.capabilities(),
            version=version,
            lineage=lineage or LineageRecord(),
            state=state,
            run_id=run_id,
        )

    def with_state(self, new_state: str) -> "StrategyCandidate":
        """Return a copy in ``new_state`` (used by the promotion policy only)."""
        return StrategyCandidate(
            genome=self.genome,
            capabilities=self.capabilities,
            version=self.version,
            lineage=self.lineage,
            state=new_state,
            run_id=self.run_id,
            created_at_utc=self.created_at_utc,
            content_hash=self.content_hash,
            candidate_id=self.candidate_id,
        )

    # -- serialisation -------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "content_hash": self.content_hash,
            "state": self.state,
            "version": self.version,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "lineage": self.lineage.as_dict(),
            "capabilities": sorted(self.capabilities),
            "genome": self.genome.as_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "StrategyCandidate":
        genome_data = data["genome"]
        kind = genome_data.get("kind")
        if kind == ParameterGenome.kind:
            genome = ParameterGenome.from_dict(genome_data)
        elif kind == PolicyTableGenome.kind:
            genome = PolicyTableGenome.from_dict(genome_data)
        else:
            from ..search.genetic_programming import PolicyIRGenome
            genome = PolicyIRGenome.from_dict(genome_data)
        return cls(
            genome=genome,
            capabilities=frozenset(data.get("capabilities", genome.capabilities())),
            version=int(data.get("version", 1)),
            lineage=LineageRecord(**{k: v for k, v in data.get("lineage", {}).items() if k in LineageRecord.__dataclass_fields__}),
            state=data.get("state", "GENERATED"),
            run_id=data.get("run_id"),
            created_at_utc=data.get("created_at_utc", _utc_now()),
            content_hash=data.get("content_hash", ""),
            candidate_id=data.get("candidate_id", ""),
        )

    def is_production_eligible(self) -> bool:
        """Only APPROVED / CANARY / ACTIVE strategies are production-eligible."""
        return self.state in PRODUCTION_ELIGIBLE_STATES

    def __repr__(self) -> str:
        return f"StrategyCandidate({self.candidate_id}, state={self.state}, v{self.version})"


__all__ = [
    "LIFECYCLE_STATES", "LineageRecord", "PRODUCTION_ELIGIBLE_STATES", "StrategyCandidate",
]
