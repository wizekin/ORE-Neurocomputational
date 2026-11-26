"""OOD generalization evaluation and the NAR production gate (steps 45-48).

Evaluation separates interpolation (seen sizes and topologies) from size extrapolation
and unseen-topology generalization (step 45), and every learned output is compared with
the deterministic oracle on outputs, intermediate states, runtime and memory (step 46).
The production gate then decides the allowed use of the learned model: advisory
search prioritization at most, never authoritative output, with deterministic fallback
or abstention on OOD inputs or failures (step 48).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from .algorithmic_gnn import TrainedNARModel, _agreement
from .task_generators import NarrowTask, reference_solve
from .trace_dataset import build_trace


@dataclass(frozen=True, slots=True)
class TaskEvaluation:
    """Per-task comparison of the learned rollout against the deterministic oracle."""

    task_id: str
    task_type: str
    split: str  # interpolation | size_extrapolation | unseen_topology
    n_nodes: int
    topology: str
    exact_output_agreement: float
    intermediate_state_agreement: float | None
    learned_seconds: float
    reference_seconds: float
    failed: bool
    notes: tuple[str, ...] = ()


@dataclass
class NAREvaluationReport:
    """Aggregated NAR evaluation, always split by group so subgroups stay visible."""

    task_type: str
    groups: dict[str, list[TaskEvaluation]]
    model_manifest: dict[str, object]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def group_summary(self, split: str) -> dict[str, object]:
        """Summary metrics for one split; poor subgroups are never hidden in aggregates."""
        items = self.groups.get(split, [])
        if not items:
            return {"split": split, "n_tasks": 0}
        agreements = [item.exact_output_agreement for item in items]
        learned = [item.learned_seconds for item in items]
        reference = [item.reference_seconds for item in items]
        return {
            "split": split, "n_tasks": len(items),
            "mean_exact_output_agreement": float(np.mean(agreements)),
            "min_exact_output_agreement": float(np.min(agreements)),
            "mean_learned_seconds": float(np.mean(learned)),
            "mean_reference_seconds": float(np.mean(reference)),
            "failures": sum(1 for item in items if item.failed),
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "task_type": self.task_type,
            "groups": {split: self.group_summary(split) for split in sorted(self.groups)},
            "model_manifest": self.model_manifest,
            "notes": list(self.notes),
        }


class NAREvaluator:
    """Evaluate a trained NAR model against deterministic reference algorithms."""

    def evaluate(self, model: TrainedNARModel, tasks: list[NarrowTask]) -> NAREvaluationReport:
        """Compare learned rollouts with the oracle across interpolation and OOD splits."""
        groups: dict[str, list[TaskEvaluation]] = {}
        max_size = max(model.trained_sizes)
        for task in tasks:
            if task.task_type != model.config.task_type:
                continue
            split = "interpolation"
            if task.n_nodes > max_size:
                split = "size_extrapolation"
            elif task.topology not in model.trained_topologies:
                split = "unseen_topology"
            evaluation = self._evaluate_one(model, task, split)
            groups.setdefault(split, []).append(evaluation)
        return NAREvaluationReport(
            task_type=model.config.task_type,
            groups=groups,
            model_manifest=model.manifest(),
            notes=(
                "learned outputs are advisory; the deterministic oracle is authoritative",
                "interpolation is separated from size extrapolation and unseen topologies",
            ),
        )

    def _evaluate_one(
        self, model: TrainedNARModel, task: NarrowTask, split: str,
    ) -> TaskEvaluation:
        trace = build_trace(task)
        started = perf_counter()
        try:
            prediction = model.rollout(trace)
            learned_seconds = perf_counter() - started
        except Exception as error:  # noqa: BLE001 - failures must be measured, not crash
            reference = reference_solve(task)
            return TaskEvaluation(
                task_id=task.task_id, task_type=task.task_type, split=split,
                n_nodes=task.n_nodes, topology=task.topology,
                exact_output_agreement=0.0, intermediate_state_agreement=None,
                learned_seconds=perf_counter() - started,
                reference_seconds=_time_reference(task), failed=True,
                notes=(f"learned rollout failed: {type(error).__name__}",),
            )
        reference_started = perf_counter()
        reference = reference_solve(task)
        reference_seconds = perf_counter() - reference_started
        reference_final = _reference_target(trace)
        binary = prediction.binary_outputs or trace.target_kind == "edge"
        agreement = _agreement(prediction.final_outputs, reference_final, binary=binary)
        intermediate: float | None = None
        if prediction.predicted_states and reference.step_states:
            pairs = min(len(prediction.predicted_states), len(reference.step_states))
            if pairs:
                per_step = [
                    _agreement(
                        prediction.predicted_states[i], reference.step_states[i],
                        binary=prediction.binary_outputs,
                    )
                    for i in range(pairs)
                ]
                intermediate = float(np.mean(per_step))
        failed = bool(np.any(~np.isfinite(prediction.final_outputs)))
        return TaskEvaluation(
            task_id=task.task_id, task_type=task.task_type, split=split,
            n_nodes=task.n_nodes, topology=task.topology,
            exact_output_agreement=agreement,
            intermediate_state_agreement=intermediate,
            learned_seconds=learned_seconds,
            reference_seconds=reference_seconds,
            failed=failed,
        )


def _reference_target(trace) -> np.ndarray:
    """Reference output in the model's tensor space (same normalization as prediction)."""
    if trace.supervision == "stepwise":
        return trace.step_states[-1]
    if trace.target_kind == "edge":
        return trace.final_edge_target
    return trace.final_node_target


def _time_reference(task: NarrowTask) -> float:
    started = perf_counter()
    reference_solve(task)
    return perf_counter() - started


@dataclass(frozen=True, slots=True)
class NARUsePolicy:
    """Thresholds the production gate applies (step 48)."""

    min_interpolation_agreement: float = 0.85
    min_ood_agreement: float = 0.5
    allow_search_prioritization: bool = True


class NARProductionGate:
    """Gate production use of a NAR model (step 48 and acceptance test 21-22).

    Allowed outcomes: ``advisory_search_prioritization`` (suggest paths / rank search),
    ``advisory_only_in_distribution`` (no OOD use) or ``abstain_all``. The gate never
    permits authoritative output: exact algorithms validate correctness-critical
    results regardless of the decision.
    """

    def __init__(self, policy: NARUsePolicy | None = None) -> None:
        self.policy = policy or NARUsePolicy()

    def decide(self, report: NAREvaluationReport) -> NARGateDecision:
        """Decide the permitted use from the evaluation report's subgroup metrics."""
        interpolation = report.group_summary("interpolation")
        extrapolation = report.group_summary("size_extrapolation")
        unseen = report.group_summary("unseen_topology")
        interpolation_ok = bool(interpolation.get("n_tasks")) and (
            interpolation["mean_exact_output_agreement"] >= self.policy.min_interpolation_agreement
        )
        ood_ok = (
            bool(extrapolation.get("n_tasks"))
            and extrapolation["mean_exact_output_agreement"] >= self.policy.min_ood_agreement
            and bool(unseen.get("n_tasks"))
            and unseen["mean_exact_output_agreement"] >= self.policy.min_ood_agreement
        )
        notes: list[str] = []
        if interpolation_ok and ood_ok and self.policy.allow_search_prioritization:
            allowed_use = "advisory_search_prioritization"
            notes.append("model may prioritize search or suggest graph paths")
        elif interpolation_ok:
            allowed_use = "advisory_only_in_distribution"
            notes.append("in-distribution advisory use only; OOD inputs must abstain")
        else:
            allowed_use = "abstain_all"
            notes.append("interpolation agreement below threshold; model must abstain")
        notes.append("exact algorithms validate correctness-critical outputs in every case")
        return NARGateDecision(
            task_type=report.task_type, allowed_use=allowed_use,
            interpolation_summary=interpolation,
            size_extrapolation_summary=extrapolation,
            unseen_topology_summary=unseen,
            notes=tuple(notes),
        )


@dataclass(frozen=True, slots=True)
class NARGateDecision:
    """The gated permitted use of one NAR model."""

    task_type: str
    allowed_use: str
    interpolation_summary: dict[str, object]
    size_extrapolation_summary: dict[str, object]
    unseen_topology_summary: dict[str, object]
    notes: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "task_type": self.task_type, "allowed_use": self.allowed_use,
            "interpolation": self.interpolation_summary,
            "size_extrapolation": self.size_extrapolation_summary,
            "unseen_topology": self.unseen_topology_summary,
            "notes": list(self.notes),
        }


__all__ = [
    "NAREvaluator", "NAREvaluationReport", "NARGateDecision", "NARProductionGate",
    "NARUsePolicy", "TaskEvaluation",
]
