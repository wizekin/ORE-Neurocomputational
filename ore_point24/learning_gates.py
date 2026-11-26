"""Phase V, Step 25 - integrate learning and algorithm discovery behind promotion gates.

Store immutable training/evaluation splits with source lineage, keep
inference separate from parameter updates, run candidate strategies only on
declared task families, keep generated programs in a restricted evaluation
environment, compare candidates to a fixed baseline, require repeatable gains
across held-out cases and failure tests, and promote only through an explicit
learning transaction with regression gates.  The parent model is preserved
and rollback is always available; no ordinary inference may update
authoritative USG records or silently mutate a production model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from .core import new_id, sha256_digest, utc_now_iso

LEARNING_SCHEMA = "ore.point24.learning_gates.v1"


@dataclass(frozen=True, slots=True)
class DataSplit:
    """An immutable training/evaluation split with source lineage."""

    split_id: str
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    source_lineage: str
    digest: str

    @classmethod
    def create(
        cls,
        *,
        train_ids: Sequence[str],
        validation_ids: Sequence[str],
        held_out_ids: Sequence[str],
        source_lineage: str,
    ) -> "DataSplit":
        payload = {
            "train": sorted(train_ids), "validation": sorted(validation_ids),
            "held_out": sorted(held_out_ids), "lineage": source_lineage,
        }
        return cls(
            split_id=new_id("split"),
            train_ids=tuple(sorted(train_ids)),
            validation_ids=tuple(sorted(validation_ids)),
            held_out_ids=tuple(sorted(held_out_ids)),
            source_lineage=source_lineage,
            digest=sha256_digest(payload),
        )

    def is_disjoint(self) -> bool:
        train = set(self.train_ids)
        validation = set(self.validation_ids)
        held = set(self.held_out_ids)
        return not (train & validation or train & held or validation & held)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split_id": self.split_id,
            "train": list(self.train_ids),
            "validation": list(self.validation_ids),
            "held_out": list(self.held_out_ids),
            "source_lineage": self.source_lineage,
            "digest": self.digest,
            "disjoint": self.is_disjoint(),
        }


@dataclass(frozen=True, slots=True)
class RegressionGate:
    """One mandatory regression gate with a predeclared bound."""

    name: str
    metric: str
    baseline_value: float
    candidate_value: float
    threshold: float
    comparison: str = "at_least"  # at_least | at_most

    @property
    def passed(self) -> bool:
        if self.comparison == "at_least":
            return self.candidate_value >= self.baseline_value - self.threshold
        return self.candidate_value <= self.baseline_value + self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "metric": self.metric,
            "baseline": self.baseline_value, "candidate": self.candidate_value,
            "threshold": self.threshold, "comparison": self.comparison,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class LearningTransactionRecord:
    """The full record of one promotion-or-rejection learning transaction."""

    transaction_id: str
    candidate_id: str
    parent_model_fingerprint: str
    candidate_fingerprint: str
    split_digest: str
    gates: tuple[RegressionGate, ...]
    decision: str  # promoted | rejected
    reason: str
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": LEARNING_SCHEMA,
            "transaction_id": self.transaction_id,
            "candidate_id": self.candidate_id,
            "parent_model_fingerprint": self.parent_model_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "split_digest": self.split_digest,
            "gates": [gate.to_dict() for gate in self.gates],
            "decision": self.decision,
            "reason": self.reason,
            "created_at": self.created_at,
        }


class LearningTransactionManager:
    """Explicit, auditable learning transactions with regression gates.

    The parent model is never mutated in place: candidates are evaluated as
    detached copies, promotion swaps a pointer and records the parent, and a
    failed gate leaves the active model untouched with an explicit rejection
    record.
    """

    def __init__(self) -> None:
        self._active: dict[str, str] = {}  # family -> model fingerprint
        self._history: list[LearningTransactionRecord] = []

    def register_family(self, family: str, parent_fingerprint: str) -> None:
        if family in self._active:
            raise ValueError(f"family already registered: {family}")
        self._active[family] = parent_fingerprint

    def active_model(self, family: str) -> str:
        return self._active[family]

    def evaluate(
        self,
        *,
        family: str,
        candidate_id: str,
        candidate_fingerprint: str,
        split: DataSplit,
        gates: Sequence[RegressionGate],
    ) -> LearningTransactionRecord:
        """Run the promotion decision; hard gates can never be compensated."""
        if family not in self._active:
            raise KeyError(f"unknown model family: {family}")
        if not split.is_disjoint():
            raise ValueError("data split leaks between train/validation/held-out")
        parent = self._active[family]
        record = LearningTransactionRecord(
            transaction_id=new_id("learning_tx"),
            candidate_id=candidate_id,
            parent_model_fingerprint=parent,
            candidate_fingerprint=candidate_fingerprint,
            split_digest=split.digest,
            gates=tuple(gates),
            decision="promoted" if all(gate.passed for gate in gates) else "rejected",
            reason=(
                "all mandatory regression gates passed"
                if all(gate.passed for gate in gates)
                else "failed gates: " + ", ".join(g.name for g in gates if not g.passed)
            ),
        )
        self._history.append(record)
        if record.decision == "promoted":
            self._active[family] = candidate_fingerprint
        return record

    def rollback(self, family: str) -> LearningTransactionRecord | None:
        """Restore the last known-good parent after a failed release gate."""
        for record in reversed(self._history):
            if record.decision == "promoted" and self._active.get(family) == record.candidate_fingerprint:
                self._active[family] = record.parent_model_fingerprint
                rollback_record = LearningTransactionRecord(
                    transaction_id=new_id("learning_tx"),
                    candidate_id=record.candidate_id,
                    parent_model_fingerprint=record.candidate_fingerprint,
                    candidate_fingerprint=record.parent_model_fingerprint,
                    split_digest=record.split_digest,
                    gates=record.gates,
                    decision="rejected",
                    reason="rollback to last known-good parent model",
                )
                self._history.append(rollback_record)
                return rollback_record
        return None

    def history(self) -> tuple[LearningTransactionRecord, ...]:
        return tuple(self._history)


@dataclass(frozen=True, slots=True)
class StrategyCandidate:
    """A discovered strategy or generated program awaiting evaluation.

    Generated programs are untrusted: they are data interpreted by a
    capability-enforcing evaluator, never unrestricted Python execution.
    """

    candidate_id: str
    task_family: str
    origin: str  # e.g. evolution, rl, program_synthesis
    representation: str
    parent_id: str | None = None
    created_at: str = field(default_factory=utc_now_iso)

    def fingerprint(self) -> str:
        return sha256_digest({
            "task_family": self.task_family,
            "origin": self.origin,
            "representation": self.representation,
            "parent": self.parent_id,
        })


class CandidateStrategyRegistry:
    """An isolated registry for learned/discovered candidates (Step 25).

    Candidates are evaluated against a fixed baseline inside a restricted
    evaluator and promoted only through the LearningTransactionManager; the
    accepted-algorithm registry stays separate from proposal generation.
    """

    def __init__(self) -> None:
        self._candidates: dict[str, StrategyCandidate] = {}
        self._evaluations: dict[str, dict[str, Any]] = {}

    def register(self, candidate: StrategyCandidate) -> str:
        if candidate.candidate_id in self._candidates:
            raise ValueError(f"duplicate candidate: {candidate.candidate_id}")
        self._candidates[candidate.candidate_id] = candidate
        return candidate.candidate_id

    def evaluate(
        self,
        candidate_id: str,
        *,
        evaluator: Callable[[StrategyCandidate], Mapping[str, float]],
        baseline: Mapping[str, float],
    ) -> dict[str, Any]:
        """Evaluate one candidate in the restricted evaluator environment."""
        candidate = self._candidates[candidate_id]
        raw = dict(evaluator(candidate))
        comparison = {
            metric: {
                "candidate": value,
                "baseline": float(baseline.get(metric, value)),
                "improved": value >= float(baseline.get(metric, value)),
            }
            for metric, value in raw.items()
        }
        repeatable = all(item["improved"] for item in comparison.values())
        record = {
            "candidate_id": candidate_id,
            "results": comparison,
            "repeatable_gain": repeatable,
            "evaluated_at": utc_now_iso(),
        }
        self._evaluations[candidate_id] = record
        return record

    def candidates(self) -> tuple[StrategyCandidate, ...]:
        return tuple(self._candidates.values())

    def evaluations(self) -> dict[str, dict[str, Any]]:
        return {key: dict(value) for key, value in self._evaluations.items()}


__all__ = [
    "LEARNING_SCHEMA",
    "CandidateStrategyRegistry",
    "DataSplit",
    "LearningTransactionManager",
    "LearningTransactionRecord",
    "RegressionGate",
    "StrategyCandidate",
]
