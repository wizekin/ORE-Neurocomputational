"""Point 12: methodological evaluation and model selection for language neuroscience.

Researcher-facing workflow: statistical operations analyze supplied studies and are not
presented as computations performed by the brain.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


_NAMES = (
    "Define observable phenomenon", "Generate competing hypotheses", "Audit causal structure",
    "Evaluate contrast identifiability", "Preserve participant and item records",
    "Separate spatial temporal and functional claims", "Interpret ERP patterns conservatively",
    "Test language selectivity", "Evaluate network boundaries", "Separate association from causation",
    "Constrain neural interpretation with behavior", "Audit population generalization",
    "Prevent overfitting and selective reporting", "Evaluate out-of-sample prediction",
    "Compare candidate models", "Test boundary conditions", "Inspect individual heterogeneity",
    "Audit measurement quality and missingness", "Evaluate evidence dependence", "Check model adequacy",
    "Calibrate claim strength", "Preserve uncertainty and alternatives",
    "Choose discriminating experiments", "Construct calibrated mechanistic account",
    "Return auditable research report",
)
POINT12_METHODS = ("behavior", "fmri", "eeg", "meg", "connectivity", "lesion", "perturbation")
POINT12_STEP_SPECS = tuple({
    "operation_id": f"LC-P12-{i:02d}", "name": name,
    "objective": name + " using only supplied observations and design metadata.",
    "mathematical_operations": ("P(H|D) ∝ P(D|H)P(H)", "MSE=mean((y-yhat)^2)", "G=(V,E,W)"),
    "biological_reference": "Researcher-side analysis; not a literal neuronal calculation.",
} for i, name in enumerate(_NAMES, 1))


def _plain(value: Any) -> Any:
    """Convert arrays, NumPy scalars, dataclasses, and nested containers to plain Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dataclass_fields__"):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    """Validate a mapping input and return a detached dictionary snapshot."""
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return deepcopy(dict(value))


def _finite(value: Any) -> np.ndarray:
    """Return finite numeric observations as a one-dimensional array."""
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.asarray([], dtype=float)
    return arr[np.isfinite(arr)]


def _mean(value: Any) -> float | None:
    """Calculate a finite-value mean or return None when observations are absent."""
    arr = _finite(value)
    return float(np.mean(arr)) if arr.size else None


def _dag_audit(edges: Sequence[Any]) -> dict[str, Any]:
    """Detect cycles in a supplied directed graph using Kahn's topological-sort algorithm."""
    graph: dict[str, set[str]] = {}
    degree: dict[str, int] = {}
    clean = []
    for edge in edges:
        if isinstance(edge, Mapping):
            a, b = str(edge.get("source", "")).strip(), str(edge.get("target", "")).strip()
        elif isinstance(edge, (list, tuple)) and len(edge) == 2:
            a, b = str(edge[0]).strip(), str(edge[1]).strip()
        else:
            continue
        if not a or not b:
            continue
        graph.setdefault(a, set())
        graph.setdefault(b, set())
        degree.setdefault(a, 0)
        degree.setdefault(b, 0)
        if b not in graph[a]:
            graph[a].add(b)
            degree[b] += 1
        clean.append({"source": a, "target": b})
    queue = [n for n, d in degree.items() if d == 0]
    order = []
    while queue:
        node = queue.pop(0)
        order.append(node)
        for target in sorted(graph[node]):
            degree[target] -= 1
            if degree[target] == 0:
                queue.append(target)
    cyclic = len(order) != len(degree)
    return {"edges": clean, "node_count": len(degree), "acyclic": not cyclic,
            "cycle_detected": cyclic, "topological_order": [] if cyclic else order,
            "direction_not_inferred_from_correlation": True}


@dataclass(frozen=True, slots=True)
class Point12Request:
    """Immutable snapshot of a research-methodology evaluation request."""
    target_phenomenon: str
    observable_target: Mapping[str, Any]
    candidate_hypotheses: Sequence[Mapping[str, Any]]
    conditions: Mapping[str, Any] = field(default_factory=dict)
    behavioral_data: Mapping[str, Any] = field(default_factory=dict)
    neural_data: Mapping[str, Any] = field(default_factory=dict)
    participant_metadata: Sequence[Mapping[str, Any]] = field(default_factory=tuple)
    analysis_plan: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate required inputs and deep-copy nested structures to prevent external mutation."""
        if not str(self.target_phenomenon).strip():
            raise ValueError("target_phenomenon must be non-empty")
        if not isinstance(self.observable_target, Mapping):
            raise TypeError("observable_target must be a mapping")
        if not isinstance(self.candidate_hypotheses, Sequence) or isinstance(self.candidate_hypotheses, (str, bytes)):
            raise TypeError("candidate_hypotheses must be a sequence of mappings")
        hypotheses = tuple(_mapping(h, f"candidate_hypotheses[{i}]")
                           for i, h in enumerate(self.candidate_hypotheses))
        if not hypotheses:
            raise ValueError("at least one candidate hypothesis is required")
        for key in ("conditions", "behavioral_data", "neural_data", "analysis_plan", "metadata"):
            object.__setattr__(self, key, _mapping(getattr(self, key), key))
        participants = tuple(_mapping(p, f"participant_metadata[{i}]")
                             for i, p in enumerate(self.participant_metadata))
        object.__setattr__(self, "observable_target", deepcopy(dict(self.observable_target)))
        object.__setattr__(self, "candidate_hypotheses", hypotheses)
        object.__setattr__(self, "participant_metadata", participants)


@dataclass(slots=True)
class Point12Result:
    """Auditable 25-step result with separate evidence, confounds, uncertainty, and hypotheses."""
    target_phenomenon: str
    hypotheses: list[dict[str, Any]]
    ranked_hypotheses: list[dict[str, Any]]
    evidence_by_method: dict[str, Any]
    confound_audit: dict[str, Any]
    design_audit: dict[str, Any]
    generalization_report: dict[str, Any]
    uncertainty_report: dict[str, Any]
    next_experiments: list[dict[str, Any]]
    mechanistic_account: dict[str, Any]
    trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible result."""
        return _plain(asdict(self))


class Point12ModelEvaluator:
    """Evaluate competing language-neuroscience hypotheses against supplied study records."""

    def __init__(self, minimum_temporal_trials: int = 20) -> None:
        """Set the descriptive minimum EEG/MEG trial count; reject non-positive thresholds."""
        if minimum_temporal_trials < 1:
            raise ValueError("minimum_temporal_trials must be positive")
        self.minimum_temporal_trials = int(minimum_temporal_trials)

    def _record(self, trace: list[dict[str, Any]], step: int, result: Mapping[str, Any]) -> None:
        """Append a detached result using the stable source-ordered operation identifier."""
        spec = POINT12_STEP_SPECS[step - 1]
        trace.append({"step": step, "operation_id": spec["operation_id"],
                      "name": spec["name"], "result": _plain(dict(result))})

    def _hypotheses(self, items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Normalize hypotheses while preserving predictions, falsifiers, boundaries, and supplied likelihoods."""
        out = []
        for i, item in enumerate(items):
            predictions = item.get("predictions", [])
            falsifiers = item.get("falsifiers", item.get("disconfirming_predictions", []))
            out.append({"hypothesis_id": str(item.get("hypothesis_id", item.get("name", f"H{i+1}"))),
                "operation": str(item.get("operation", "")),
                "predictions": _plain(list(predictions) if isinstance(predictions, (list, tuple)) else []),
                "falsifiers": _plain(list(falsifiers) if isinstance(falsifiers, (list, tuple)) else []),
                "boundary_conditions": _plain(item.get("boundary_conditions", [])),
                "prior_probability": item.get("prior_probability"),
                "likelihoods": _plain(item.get("likelihoods", {})),
                "falsifiable": bool(predictions and falsifiers)})
        return out

    def _design(self, conditions: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
        """Compare target/control feature summaries and identify unmatched non-manipulated dimensions."""
        target, control = conditions.get("target", {}), conditions.get("control", {})
        target = target if isinstance(target, Mapping) else {}
        control = control if isinstance(control, Mapping) else {}
        contrasts = {}
        for key in sorted(set(target) | set(control)):
            a, b = _mean(target.get(key, [])), _mean(control.get(key, []))
            contrasts[str(key)] = {"target_mean": a, "control_mean": b,
                                   "difference": a-b if a is not None and b is not None else None}
        manipulated, matched = list(plan.get("manipulated_variables", [])), list(plan.get("matched_variables", []))
        unmatched = [k for k, v in contrasts.items()
                     if v["difference"] not in (None, 0.0) and k not in manipulated]
        confounds = [k for k in unmatched if k not in matched]
        return {"feature_contrasts": contrasts, "manipulated_variables": manipulated,
            "matched_variables": matched, "unmatched_nonmanipulated_features": unmatched,
            "potentially_confounded_features": confounds,
            "contrast_identifiability": "limited" if confounds else "no_numeric_mismatch_detected",
            "factorial_design": bool(plan.get("factorial_design", False)),
            "participant_item_effects_planned": bool(plan.get("participant_and_item_effects", False))}
