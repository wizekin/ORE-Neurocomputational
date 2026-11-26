"""Teacher-student distillation for edge specialists (Point 14, Steps 39-40).

The student trains on the teacher's soft targets under a temperature, and the
distillation report compares both on held-out tasks: parameter counts, memory,
latency proxy, quality retention, and the compression ratio.  The uncompressed
teacher remains the rollback reference (Step 39).
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable, Sequence

import numpy as np

from ..functional import softmax

__all__ = ("DistillationReport", "distill", "distillation_target")


def distillation_target(teacher_logits: np.ndarray, temperature: float) -> np.ndarray:
    """Soft targets: softmax(teacher logits / T) with the configured temperature."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    return softmax(np.asarray(teacher_logits, dtype=float) / temperature)


@dataclass(slots=True)
class DistillationReport:
    """Step 39 -- measured trade-offs of one distillation run."""

    teacher_params: int
    student_params: int
    teacher_quality: float
    student_quality: float
    teacher_memory_bytes: int
    student_memory_bytes: int
    temperature: float
    epochs: int
    held_out: bool = True

    @property
    def retention(self) -> float:
        """Quality retention Q_student / Q_teacher (higher-is-better qualities)."""
        if self.teacher_quality == 0.0:
            return 0.0
        return self.student_quality / self.teacher_quality

    @property
    def compression_ratio(self) -> float:
        """Parameter compression teacher/student."""
        if self.student_params == 0:
            return float("inf")
        return self.teacher_params / self.student_params

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"teacher_params": self.teacher_params,
                "student_params": self.student_params,
                "teacher_quality": round(self.teacher_quality, 6),
                "student_quality": round(self.student_quality, 6),
                "retention": round(self.retention, 6),
                "teacher_memory_bytes": self.teacher_memory_bytes,
                "student_memory_bytes": self.student_memory_bytes,
                "compression_ratio": round(self.compression_ratio, 4),
                "temperature": self.temperature, "epochs": self.epochs,
                "held_out": self.held_out,
                "note": "the uncompressed teacher is preserved as the rollback reference"}


def distill(teacher: Any, student: Any, features: np.ndarray, *,
            quality_fn: Callable[[Any, np.ndarray], float],
            held_out_features: np.ndarray,
            temperature: float = 2.0, epochs: int = 60,
            learning_rate: float = 0.05) -> DistillationReport:
    """Train ``student`` on ``teacher`` soft targets and compare both on held-out data.

    ``quality_fn(model, features)`` returns a higher-is-better quality in [0, 1] so
    retention is meaningful.  The teacher is never modified.
    """
    matrix = np.asarray(features, dtype=float)
    held_out = np.asarray(held_out_features, dtype=float)
    if matrix.size == 0 or held_out.size == 0:
        raise ValueError("training and held-out features must be non-empty")
    teacher_soft = distillation_target(teacher.predict(matrix), temperature)
    for _ in range(epochs):
        student_output = np.asarray(student.predict(matrix))
        # squared loss against the teacher's soft targets (deterministic full batch)
        error = student_output - teacher_soft
        student._full_finetune_step(matrix, teacher_soft, learning_rate)
        _ = error  # retained for clarity; the step already uses teacher_soft
    return DistillationReport(
        teacher_params=teacher.num_parameters(),
        student_params=student.num_parameters(),
        teacher_quality=float(quality_fn(teacher, held_out)),
        student_quality=float(quality_fn(student, held_out)),
        teacher_memory_bytes=teacher.memory_bytes(),
        student_memory_bytes=student.memory_bytes(),
        temperature=temperature, epochs=epochs)
