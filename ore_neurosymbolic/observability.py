"""Observability, verified-outcome comparison and checkpointing (Phase H, Steps H1-H5).

Phase H keeps the reasoning layer measurable and recoverable:

- H1: rule fires, model calls, proof verification, constraint decisions,
  conflicts, resource use and publication status are recorded through the
  existing ``ore_logger`` SQLite infrastructure (never a parallel logger);
- H2: quality/cost metrics (proof validity, contradiction rate, ambiguity,
  candidate precision/recall where labels exist, latency, rule fires, memory,
  verification overhead) are computed from measured run data;
- H3: prediction error is recorded *only* when genuine ground truth or
  independently verified feedback exists (never self-labeled);
- H4: approved parameter updates are recorded as versioned transactions with
  their verification sources;
- H5: safe run state is checkpointed; timeouts return explicit partial status
  with resume information and never publish incomplete work as complete.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import make_id


@dataclass(slots=True)
class RunMetrics:
    """Measured quality/cost metrics for one run (Step H2)."""

    proof_validity_rate: float = 0.0
    contradiction_rate: float = 0.0
    ambiguity_rate: float = 0.0
    candidate_precision: float | None = None
    candidate_recall: float | None = None
    latency_ms: float = 0.0
    rule_fires: int = 0
    memory_estimate_mb: float = 0.0
    verification_overhead_ms: float = 0.0
    model_calls: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "proof_validity_rate": round(self.proof_validity_rate, 6),
            "contradiction_rate": round(self.contradiction_rate, 6),
            "ambiguity_rate": round(self.ambiguity_rate, 6),
            "candidate_precision": (
                round(self.candidate_precision, 6)
                if self.candidate_precision is not None else None
            ),
            "candidate_recall": (
                round(self.candidate_recall, 6) if self.candidate_recall is not None else None
            ),
            "latency_ms": round(self.latency_ms, 3),
            "rule_fires": self.rule_fires,
            "memory_estimate_mb": round(self.memory_estimate_mb, 3),
            "verification_overhead_ms": round(self.verification_overhead_ms, 3),
            "model_calls": self.model_calls,
        }


@dataclass(slots=True)
class CheckpointRecord:
    """One safe checkpoint with resume information (Step H5)."""

    checkpoint_id: str
    run_id: str
    phase: str
    completed_steps: tuple[int, ...]
    pending_step: int | None
    state_digest: str
    partial: bool
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "run_id": self.run_id,
            "phase": self.phase,
            "completed_steps": list(self.completed_steps),
            "pending_step": self.pending_step,
            "state_digest": self.state_digest,
            "partial": self.partial,
            "created_at": self.created_at,
        }


class NSRLObservability:
    """Record events, measure quality/cost, compare outcomes, checkpoint (Phase H)."""

    def __init__(
        self,
        run_id: str,
        *,
        logger_database: Any = None,
        enable_logging: bool = False,
    ) -> None:
        """Bind the optional ``LoggerDatabase`` for event correlation (Step H1)."""
        self.run_id = run_id
        self._database = logger_database
        self._logging_enabled = bool(enable_logging and logger_database is not None)
        self._run_row_id: str | None = None
        self._sequence: int = 0
        self._events: list[dict[str, Any]] = []
        if self._logging_enabled:
            self._run_row_id = self._database.create_run(
                script_path="ore_neurosymbolic.coordinator",
                module_name="ore_neurosymbolic",
                entry_kind="neurosymbolic_reasoning",
            )

    # ------------------------------------------------------------------ Step H1

    def record_event(
        self,
        event_kind: str,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Record one structured event; correlates run, proof and finding IDs (H1)."""
        self._sequence += 1
        event = {
            "sequence_no": self._sequence,
            "event_kind": event_kind,
            "run_id": self.run_id,
            "metadata": dict(metadata),
        }
        self._events.append(event)
        if self._logging_enabled:
            try:
                self._database.record_event(
                    run_id=self._run_row_id,
                    sequence_no=self._sequence,
                    event_kind=event_kind,
                    metadata={
                        "run_id": self.run_id,
                        **dict(metadata),
                    },
                )
            except Exception:  # pragma: no cover - logger failures never break inference
                pass
        return event

    @property
    def events(self) -> tuple[dict[str, Any], ...]:
        """Return the recorded in-memory events."""
        return tuple(self._events)

    def finish(self, status: str, error_summary: str | None = None) -> None:
        """Close the logger run row when logging is enabled."""
        if self._logging_enabled and self._run_row_id:
            try:
                self._database.finish_run(
                    self._run_row_id, status=status, error_summary=error_summary,
                )
            except Exception:  # pragma: no cover
                pass

    # ------------------------------------------------------------------ Step H2

    @staticmethod
    def measure(
        *,
        verified_node_count: int,
        strict_node_count: int,
        conflict_count: int,
        conclusion_count: int,
        ambiguity_group_count: int,
        candidate_count: int,
        rule_fires: int,
        latency_ms: float,
        verification_overhead_ms: float,
        memory_estimate_mb: float,
        model_calls: int,
        labeled_candidates: Mapping[str, bool] | None = None,
        predicted_relevant: tuple[str, ...] = (),
    ) -> RunMetrics:
        """Compute the Step H2 metrics from measured run data."""
        proof_validity = (
            verified_node_count / strict_node_count if strict_node_count else (
                1.0 if verified_node_count == 0 and strict_node_count == 0 else 0.0
            )
        )
        contradiction_rate = (
            conflict_count / conclusion_count if conclusion_count else 0.0
        )
        ambiguity_rate = (
            ambiguity_group_count / candidate_count if candidate_count else 0.0
        )
        precision: float | None = None
        recall: float | None = None
        if labeled_candidates:
            relevant = {
                key for key, value in labeled_candidates.items() if value
            }
            predicted_set = set(predicted_relevant)
            true_positive = len(predicted_set & relevant)
            precision = (
                true_positive / len(predicted_set) if predicted_set else None
            )
            recall = true_positive / len(relevant) if relevant else None
        return RunMetrics(
            proof_validity_rate=min(1.0, max(0.0, proof_validity)),
            contradiction_rate=min(1.0, max(0.0, contradiction_rate)),
            ambiguity_rate=min(1.0, max(0.0, ambiguity_rate)),
            candidate_precision=precision,
            candidate_recall=recall,
            latency_ms=latency_ms,
            rule_fires=rule_fires,
            memory_estimate_mb=memory_estimate_mb,
            verification_overhead_ms=verification_overhead_ms,
            model_calls=model_calls,
        )

    # ------------------------------------------------------------------ Step H3

    @staticmethod
    def compare_with_verified_outcomes(
        predictions: Mapping[str, Any],
        verified_outcomes: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Record prediction error only against genuine ground truth (Step H3).

        When ``verified_outcomes`` is None or empty, the comparison is
        explicitly reported as unavailable instead of inventing an error rate
        from self-generated labels (Section 10; Stage 7 gate).
        """
        if not verified_outcomes:
            return {
                "comparison_available": False,
                "reason": "no genuine ground truth or independently verified feedback",
            }
        scored: list[dict[str, Any]] = []
        errors: list[float] = []
        for key, prediction in predictions.items():
            if key not in verified_outcomes:
                continue
            outcome = verified_outcomes[key]
            numeric_prediction = (
                float(prediction) if isinstance(prediction, (int, float)) else None
            )
            numeric_outcome = (
                float(outcome) if isinstance(outcome, (int, float)) else None
            )
            if numeric_prediction is not None and numeric_outcome is not None:
                error = abs(numeric_prediction - numeric_outcome)
                errors.append(error)
            scored.append({
                "key": key,
                "prediction": prediction,
                "verified_outcome": outcome,
                "error": (
                    round(error, 6)
                    if numeric_prediction is not None and numeric_outcome is not None
                    else None
                ),
            })
        mean_error = sum(errors) / len(errors) if errors else None
        return {
            "comparison_available": bool(scored),
            "compared_count": len(scored),
            "mean_absolute_error": round(mean_error, 6) if mean_error is not None else None,
            "records": scored,
        }

    # ------------------------------------------------------------------ Step H4

    def record_parameter_transaction(
        self,
        ledger_snapshot: Mapping[str, Any],
        *,
        verification_source: str,
    ) -> dict[str, Any]:
        """Record one versioned, gated parameter-update transaction (Step H4)."""
        transaction = {
            "transaction_id": make_id("nsr_param"),
            "run_id": self.run_id,
            "verification_source": verification_source,
            "ledger_version": ledger_snapshot.get("version"),
            "accepted_updates": len(ledger_snapshot.get("updates", ())),
            "rejected_updates": len(ledger_snapshot.get("rejected_updates", ())),
        }
        self.record_event("parameter_update_transaction", transaction)
        return transaction

    # ------------------------------------------------------------------ Step H5

    def checkpoint(
        self,
        *,
        phase: str,
        completed_steps: tuple[int, ...],
        pending_step: int | None,
        state: Mapping[str, Any],
        partial: bool = False,
    ) -> CheckpointRecord:
        """Persist one safe checkpoint with resume information (Step H5)."""
        from .contracts import sha256_json

        record = CheckpointRecord(
            checkpoint_id=make_id("nsr_checkpoint"),
            run_id=self.run_id,
            phase=phase,
            completed_steps=tuple(sorted(completed_steps)),
            pending_step=pending_step,
            state_digest=sha256_json(dict(state)),
            partial=partial,
        )
        self.record_event("checkpoint", record.to_dict())
        return record


__all__ = ["CheckpointRecord", "NSRLObservability", "RunMetrics"]
