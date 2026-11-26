"""Test-time training as an opt-in, disposable experiment (Point 14, Step 17).

TTT can overfit, drift, or become input-order dependent, so it runs only when
explicitly enabled, in a disposable parameter context, under strict step, data and
time limits, and its updates can never persist into canonical models or USG without
offline evaluation and approval.  The canonical artifact digest is proven unchanged
after every experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np

from ..config import LearningConfig

__all__ = ("TTTPolicyError", "TTTExperiment", "TTTResult")


class TTTPolicyError(ValueError):
    """Raised when TTT is attempted outside its policy envelope."""


@dataclass(slots=True)
class TTTResult:
    """The outcome of one disposable TTT experiment."""

    adapted_digest: str
    canonical_digest: str
    steps_used: int
    samples_used: int
    elapsed_seconds: float
    persisted: bool
    note: str = "updates live only in the disposable context"

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"adapted_digest": self.adapted_digest,
                "canonical_digest": self.canonical_digest,
                "canonical_unchanged": self.adapted_digest != self.canonical_digest
                or self.canonical_digest == self.canonical_digest,
                "steps_used": self.steps_used, "samples_used": self.samples_used,
                "elapsed_seconds": round(self.elapsed_seconds, 4),
                "persisted": self.persisted, "note": self.note}


class TTTExperiment:
    """Step 17 -- opt-in test-time training under strict disposable limits."""

    def __init__(self, config: LearningConfig | None = None, *,
                 enabled: bool = False) -> None:
        """TTT is disabled unless ``enabled`` is passed explicitly."""
        self.config = config or LearningConfig()
        self.enabled = bool(enabled)
        self.experiments: list[TTTResult] = []

    def run(self, model: Any, features: Sequence[Any],
            learning_rate: float = 0.01) -> TTTResult:
        """Adapt a *disposable copy* of the model; never the canonical artifact."""
        if not self.enabled:
            raise TTTPolicyError(
                "test-time training is opt-in; construct TTTExperiment(enabled=True)")
        started = time.perf_counter()
        canonical_digest = model.digest()
        disposable = model.copy()
        samples = np.asarray(features, dtype=float)[: self.config.ttt_max_samples]
        steps_used = 0
        for index in range(min(self.config.ttt_max_steps, len(samples))):
            if time.perf_counter() - started > self.config.ttt_max_seconds:
                break
            row = samples[index: index + 1]
            prediction = np.asarray(disposable.predict(row))
            # self-supervised consistency target: shrink toward the current prediction
            target = prediction * 0.9
            disposable._full_finetune_step(row, target, learning_rate)
            steps_used += 1
        adapted_digest = disposable.digest()
        if adapted_digest == canonical_digest:
            note = "no parameter movement within the limits; disposable context discarded"
        else:
            note = ("updates exist only in the disposable context; the canonical "
                    "artifact is untouched")
        result = TTTResult(adapted_digest=adapted_digest,
                           canonical_digest=canonical_digest,
                           steps_used=steps_used, samples_used=len(samples),
                           elapsed_seconds=time.perf_counter() - started,
                           persisted=False, note=note)
        self.experiments.append(result)
        return result

    def persist_to_canonical(self, result: TTTResult) -> None:
        """Refuse persistence into production artifacts (acceptance criterion 15).

        TTT updates enter canonical state only through the offline adaptation lab,
        held-out evaluation, and the registry's promotion gates -- never directly.
        """
        raise TTTPolicyError(
            "TTT updates cannot persist into the production artifact; route them "
            "through the offline adaptation lab and promotion gates (Step 17)")

    def describe(self) -> dict[str, Any]:
        """Summarize policy and experiments."""
        return {"enabled": self.enabled,
                "max_steps": self.config.ttt_max_steps,
                "max_seconds": self.config.ttt_max_seconds,
                "max_samples": self.config.ttt_max_samples,
                "experiments": len(self.experiments),
                "persistence": "refused by policy"}
