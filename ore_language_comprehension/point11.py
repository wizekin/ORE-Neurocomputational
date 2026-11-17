"""Point 11: functional separation and cooperation of language and other brain systems.

The module keeps linguistic, sensory, conceptual, social, executive, and memory
contributions distinct while permitting task-dependent information exchange. It is a
computational reconstruction, not a claim of literal neural routing or localization.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


POINT11_SYSTEMS = ("perceptual", "language", "conceptual_knowledge", "mentalizing", "multiple_demand", "memory")
POINT11_DEMAND_KEYS = ("sensory", "linguistic", "conceptual", "social", "working_memory", "executive", "response")
POINT11_DEFAULT_CONNECTIONS: dict[tuple[str, str], float] = {
    ("perceptual", "language"): 0.65,
    ("language", "conceptual_knowledge"): 0.40,
    ("conceptual_knowledge", "language"): 0.30,
    ("language", "mentalizing"): 0.28,
    ("mentalizing", "language"): 0.22,
    ("language", "memory"): 0.32,
    ("memory", "language"): 0.25,
    ("multiple_demand", "language"): 0.20,
    ("multiple_demand", "memory"): 0.35,
    ("memory", "multiple_demand"): 0.22,
    ("conceptual_knowledge", "memory"): 0.30,
    ("memory", "conceptual_knowledge"): 0.20,
    ("conceptual_knowledge", "mentalizing"): 0.22,
    ("mentalizing", "conceptual_knowledge"): 0.20,
}
POINT11_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {"operation_id": "LC-P11-01", "name": "Decompose task demands", "objective": "Separate sensory, linguistic, conceptual, social, memory, executive, and response requirements before assigning system roles.", "mathematical_operations": ("d=[d_sens,d_ling,d_concept,d_social,d_WM,d_exec,d_resp]", "feature-wise demand estimation"), "biological_reference": "Functional analysis, not a central homunculus or literal demand vector."},
    {"operation_id": "LC-P11-02", "name": "Encode sensory form", "objective": "Keep modality-specific acoustic/visual form separate from linguistic meaning and retain signal quality.", "mathematical_operations": ("p=phi(W_s*x+b_s)", "reliability-weighted cue combination"), "biological_reference": "Auditory/visual populations encode signal features; the equation is an abstraction."},
    {"operation_id": "LC-P11-03", "name": "Identify individual language-selective network", "objective": "Use participant-specific localizer evidence when supplied and label atlas/literature priors as approximate.", "mathematical_operations": ("SI=mean(BOLD_language)-mean(BOLD_control)", "d-prime and within-person reliability"), "biological_reference": "Language network is distributed and individually variable; no universal coordinate is assumed."},
    {"operation_id": "LC-P11-04", "name": "Route lexical and grammatical operations", "objective": "Use language-selective computations to access words/constructions and build linguistic structure.", "mathematical_operations": ("candidate competition", "incremental structured binding", "recurrent linguistic state update"), "biological_reference": "Core frontotemporal language network supports linguistic knowledge access and composition."},
    {"operation_id": "LC-P11-05", "name": "Separate composition from conceptual enrichment", "objective": "Retrieve relevant world knowledge while preserving the distinction between literal linguistic content and inferred features.", "mathematical_operations": ("e=F(l,k,c)", "source-tagged feature union"), "biological_reference": "Conceptual knowledge is distributed; enrichment does not imply a single semantic centre."},
    {"operation_id": "LC-P11-06", "name": "Conditionally recruit mentalizing", "objective": "Represent beliefs, knowledge, intentions, and perspectives only when they are relevant to interpretation.", "mathematical_operations": ("B_a(q,t)=P(belief|evidence)", "conditional gate sigmoid(score-threshold)"), "biological_reference": "mPFC/TPJ-related systems can support social inference but are not required for every literal sentence."},
    {"operation_id": "LC-P11-07", "name": "Estimate executive demand independently", "objective": "Separate linguistic complexity from working-memory, conflict, time-pressure, and response-selection requirements.", "mathematical_operations": ("D_exec=sum(w_i*d_i)", "independent demand channels"), "biological_reference": "Multiple-demand systems provide flexible control when task demands warrant it."},
    {"operation_id": "LC-P11-08", "name": "Exchange information while preserving specialization", "objective": "Permit sparse, gated communication and retain the source of each contribution.", "mathematical_operations": ("h_i(t+1)=phi(A_i*h_i+sum(g_ij*C_ij*h_j)+B_i*u_i)", "directed weighted functional graph"), "biological_reference": "Functional interaction does not imply identical function or direct anatomical wiring."},
    {"operation_id": "LC-P11-09", "name": "Evaluate task-appropriate selectivity", "objective": "Compare supplied response profiles across language, non-language, social, executive, and perceptual tasks.", "mathematical_operations": ("multi-task response vector", "within-person contrast and selectivity profile"), "biological_reference": "Selectivity is a profile across contrasts, not absolute exclusivity."},
    {"operation_id": "LC-P11-10", "name": "Separate co-activation from contribution", "objective": "Distinguish association, prediction, lesion support, and perturbational evidence.", "mathematical_operations": ("conditional response association", "active-control perturbation contrast"), "biological_reference": "Co-activation alone does not establish necessity or causal direction."},
    {"operation_id": "LC-P11-11", "name": "Preserve separate language and thought measures", "objective": "Keep linguistic ability, non-linguistic reasoning, social cognition, and executive measures distinct.", "mathematical_operations": ("delta_L=L_control-L_patient", "delta_T=T_control-T_patient", "profile dissociation"), "biological_reference": "Language and thought can be functionally separable without being wholly independent."},
    {"operation_id": "LC-P11-12", "name": "Incorporate individual network organization", "objective": "Retain individual localization, laterality, stability, and uncertainty instead of imposing a group atlas.", "mathematical_operations": ("SI_i,r=mean_i,L-mean_i,C", "test-retest reliability"), "biological_reference": "Individual functional topography varies and may be stable within a person."},
    {"operation_id": "LC-P11-13", "name": "Evaluate right-hemisphere task contributions", "objective": "Represent narrative, discourse, semantic-control, and prosodic demands without assuming right-hemisphere irrelevance.", "mathematical_operations": ("task-dependent right-system demand score", "contrast across word/sentence/narrative conditions"), "biological_reference": "Right homologous regions and control systems can contribute depending on task demands."},
    {"operation_id": "LC-P11-14", "name": "Update allocation as input changes", "objective": "Revise language, conceptual, social, and executive contributions incrementally as new words or context arrive.", "mathematical_operations": ("state_t+1=update(state_t,new_input)", "recurrent demand re-estimation"), "biological_reference": "Comprehension is dynamic and recurrent, not a fixed one-time routing decision."},
    {"operation_id": "LC-P11-15", "name": "Integrate outputs with provenance", "objective": "Combine literal propositions and supported enrichments while preserving source, confidence, and uncertainty.", "mathematical_operations": ("z=(value,source,confidence,time)", "provenance-preserving integration"), "biological_reference": "Explicit provenance fields are a computational design choice, not a known neural data structure."},
    {"operation_id": "LC-P11-16", "name": "Reduce unneeded additional recruitment", "objective": "Stop adding speculative inferences or control demands when they no longer improve the task-relevant interpretation.", "mathematical_operations": ("marginal_utility=delta(task_fit)-cost(additional_inference)", "demand decay"), "biological_reference": "Conditional recruitment changes with task needs; no single global switch is assumed."},
)


def _plain(value: Any) -> Any:
    """Convert NumPy and dataclass values into detached ordinary Python data."""
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


def _bounded(value: Any, default: float = 0.0) -> float:
    """Convert a numeric demand/activation into a finite value bounded to [0, 1]."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, number)) if math.isfinite(number) else default


def _sigmoid(value: float) -> float:
    """Compute a numerically stable logistic gate for conditional system recruitment."""
    value = max(-40.0, min(40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-value))


def _sequence(value: Any) -> list[Any]:
    """Normalize scalar, sequence, or missing values into a list without splitting strings."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _mean(values: Any) -> float | None:
    """Return a finite numeric mean or None when no usable observations were supplied."""
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else None


@dataclass(frozen=True, slots=True)
class Point11Request:
    """One incremental task update with task demands, prior stages, and individual profile."""

    sequence_id: str
    sensory_input: Any = None
    stage_outputs: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    task: Mapping[str, Any] = field(default_factory=dict)
    individual_profile: Mapping[str, Any] = field(default_factory=dict)
    conceptual_candidates: Sequence[Mapping[str, Any]] = ()
    agent_candidates: Sequence[Mapping[str, Any]] = ()
    task_response_profile: Mapping[str, Any] = field(default_factory=dict)
    network_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate inputs and snapshot nested mappings and candidate records."""
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        for name in ("stage_outputs", "context", "task", "individual_profile",
                     "task_response_profile", "network_weights"):
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise TypeError(f"{name} must be a mapping")
            object.__setattr__(self, name, deepcopy(dict(value)))
        for name in ("conceptual_candidates", "agent_candidates"):
            values = getattr(self, name)
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError(f"{name} must be a sequence")
            if not all(isinstance(item, Mapping) for item in values):
                raise TypeError(f"every {name} item must be a mapping")
            object.__setattr__(self, name, tuple(deepcopy(dict(item)) for item in values))


@dataclass(slots=True)
class Point11State:
    """Current multi-system activations and source-tagged features for one sequence."""

    time_index: int = 0
    systems: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        name: {"activation": 0.0, "representation": [], "source": "not_updated",
               "confidence": 0.0, "status": "inactive"} for name in POINT11_SYSTEMS
    })
    demand: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in POINT11_DEMAND_KEYS})
    localization: dict[str, Any] = field(default_factory=dict)
    communication_graph: dict[str, Any] = field(default_factory=dict)
    sourced_features: list[dict[str, Any]] = field(default_factory=list)
    alternatives: list[dict[str, Any]] = field(default_factory=list)
    integrated_representation: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached plain-data snapshot of the current multi-system state."""
        return _plain(asdict(self))


@dataclass(slots=True)
class Point11Result:
    """Auditable output separating language, conceptual, social, control, and sensory roles."""

    sequence_id: str
    time_index: int
    demand_profile: dict[str, float]
    localization: dict[str, Any]
    system_states: dict[str, Any]
    communication_graph: dict[str, Any]
    selectivity: dict[str, Any]
    contribution_evidence: dict[str, Any]
    provenance: list[dict[str, Any]]
    interpretation: dict[str, Any]
    functional_trace: list[dict[str, Any]]
    diagnostics: dict[str, Any]
    biological_summary: dict[str, Any] = field(default_factory=dict)
    biological_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result while converting NumPy values to ordinary Python values."""
        return _plain(asdict(self))


class Point11LIFReference:
    """Six-population leaky integrate-and-fire reference for distinct functional-system classes."""

    def __init__(self, population_size: int = 8, dt_ms: float = 1.0, tau_ms: float = 20.0,
                 threshold: float = 1.0, reset: float = 0.0) -> None:
        """Validate population and membrane parameters for the optional reference simulation."""
        if population_size < 1 or dt_ms <= 0 or tau_ms <= 0 or threshold <= reset:
            raise ValueError("population_size, dt_ms, tau_ms must be positive and threshold > reset")
        self.population_size = int(population_size)
        self.dt_ms = float(dt_ms)
        self.tau_ms = float(tau_ms)
        self.threshold = float(threshold)
        self.reset = float(reset)

    def run(self, drives: Mapping[str, float], steps: int = 16) -> dict[str, Any]:
        """Run deterministic population activity; this is not measured human neural data."""
        unknown = set(drives) - set(POINT11_SYSTEMS)
        if unknown:
            raise KeyError(f"unknown systems: {sorted(unknown)}")
        if steps < 1 or any(not math.isfinite(float(v)) for v in drives.values()):
            raise ValueError("steps must be positive and drives finite")
        names = list(POINT11_SYSTEMS)
        voltage = np.full((len(names), self.population_size), self.reset, dtype=float)
        alpha = self.dt_ms / self.tau_ms
        trace = []
        for t in range(steps):
            row = {}
            for i, name in enumerate(names):
                drive = _bounded(drives.get(name, 0.0))
                voltage[i] += alpha * (drive - voltage[i])
                spikes = voltage[i] >= self.threshold
                voltage[i, spikes] = self.reset
                row[name] = {"mean_voltage": float(np.mean(voltage[i])),
                             "spike_fraction": float(np.mean(spikes))}
            trace.append({"step": t + 1, "populations": row})
        return {"population_names": names, "trace": trace, "operation_count": steps,
                "biological_reference_is_measured_human_data": False}


class Point11FunctionalSeparationCoordinator:
    """Coordinate specialized comprehension systems without merging their representations."""

    def __init__(self, social_threshold: float = 0.55, executive_threshold: float = 0.55,
                 recurrent_gain: float = 0.35, simulate_biology: bool = False,
                 lif_population_size: int = 8) -> None:
        """Configure conditional recruitment, functional coupling, and optional LIF reference."""
        if not 0 <= social_threshold <= 1 or not 0 <= executive_threshold <= 1:
            raise ValueError("recruitment thresholds must lie in [0, 1]")
        if not 0 <= recurrent_gain <= 1:
            raise ValueError("recurrent_gain must lie in [0, 1]")
        self.social_threshold = float(social_threshold)
        self.executive_threshold = float(executive_threshold)
        self.recurrent_gain = float(recurrent_gain)
        self.simulate_biology = bool(simulate_biology)
        self.lif = Point11LIFReference(population_size=lif_population_size) if simulate_biology else None
        self.state = Point11State()

    def reset(self) -> None:
        """Clear sequence-specific state while preserving coordinator configuration."""
        self.state = Point11State()

    def _record(self, trace: list[dict[str, Any]], step: int, result: Mapping[str, Any]) -> None:
        """Append a detached, ordered operation result."""
        trace.append({"step": step, "operation_id": POINT11_STEP_SPECS[step - 1]["operation_id"],
                      "name": POINT11_STEP_SPECS[step - 1]["name"], "result": _plain(dict(result))})

    def _demand_profile(self, request: Point11Request) -> tuple[dict[str, float], dict[str, Any]]:
        """Estimate separate sensory, linguistic, conceptual, social, memory, executive, and response demands."""
        task = request.task
        supplied = task.get("demand_profile", {})
        supplied = supplied if isinstance(supplied, Mapping) else {}
        keys = {
            "sensory": ("sensory_demand", "signal_noise", "modality_complexity"),
            "linguistic": ("linguistic_demand", "syntactic_complexity", "lexical_ambiguity"),
            "conceptual": ("conceptual_demand", "world_knowledge_demand", "event_inference"),
            "social": ("social_demand", "perspective_dependence", "intention_relevance"),
            "working_memory": ("working_memory_load", "maintenance_load", "dependency_load"),
            "executive": ("executive_task_demand", "conflict", "time_pressure"),
            "response": ("response_demand", "response_selection_complexity", "explicit_judgment"),
        }
        profile: dict[str, float] = {}
        components: dict[str, dict[str, float]] = {}
        for dimension, fields in keys.items():
            values = [_bounded(supplied.get(dimension, 0.0))]
            values.extend(_bounded(task.get(key, 0.0)) for key in fields)
            profile[dimension] = max(values)
            components[dimension] = {key: _bounded(task.get(key, supplied.get(dimension, 0.0))) for key in fields}
        cues = {"belief", "believe", "intention", "intend", "perspective", "sarcasm", "irony", "deceive", "thinks", "knows"}
        context_text = " ".join(str(request.context.get(k, "")) for k in ("text", "discourse", "speaker", "situation")).casefold()
        cue_matches = sorted(word for word in cues if word in context_text)
        if task.get("mentalizing_required", False):
            profile["social"] = 1.0
        elif cue_matches:
            profile["social"] = max(profile["social"], min(1.0, len(cue_matches) / 3.0))
        return profile, {"components": components, "social_cues": cue_matches,
                         "mentalizing_required": bool(task.get("mentalizing_required", False))}

    def _localize_language(self, request: Point11Request) -> dict[str, Any]:
        """Prefer individual language-localizer data and never fabricate a personal anatomical map."""
        localizer = request.individual_profile.get("language_localizer", {})
        if isinstance(localizer, Mapping) and localizer:
            language = _mean(localizer.get("language", []))
            control = _mean(localizer.get("control", []))
            if language is not None and control is not None:
                return {"source": "individual_functional_localizer", "language_mean": language,
                        "control_mean": control, "selectivity_contrast": language - control,
                        "region_labels": _plain(localizer.get("region_labels", [])),
                        "individual_measurement_supplied": True}
        prior = request.individual_profile.get("literature_prior", {})
        if isinstance(prior, Mapping) and prior:
            return {"source": "approximate_literature_prior_only", "prior": _plain(prior),
                    "individual_measurement_supplied": False}
        return {"source": "not_localized", "individual_measurement_supplied": False,
                "warning": "no individual localizer or declared prior supplied"}

    def _stage_representation(self, outputs: Mapping[str, Any], names: Sequence[str]) -> list[Any]:
        """Extract the first supplied representation from an upstream stage alias set."""
        for name in names:
            if name in outputs:
                value = outputs[name]
                if isinstance(value, Mapping):
                    for key in ("candidates", "items", "propositions", "representation", "output"):
                        if key in value:
                            return _sequence(value[key])
                return _sequence(value)
        return []

    def _communication_graph(self, overrides: Mapping[str, float]) -> dict[str, Any]:
        """Build a sparse functional-information graph distinct from an anatomical tract map."""
        weights = dict(POINT11_DEFAULT_CONNECTIONS)
        for edge, value in overrides.items():
            if isinstance(edge, str) and "->" in edge:
                source, target = edge.split("->", 1)
                key = (source.strip(), target.strip())
            elif isinstance(edge, tuple) and len(edge) == 2:
                key = (str(edge[0]), str(edge[1]))
            else:
                raise ValueError(f"invalid functional edge: {edge!r}")
            if key[0] not in POINT11_SYSTEMS or key[1] not in POINT11_SYSTEMS:
                raise ValueError(f"unknown functional system in edge {key!r}")
            weight = float(value)
            if not math.isfinite(weight) or not 0 <= weight <= 1:
                raise ValueError("functional edge weights must lie in [0, 1]")
            weights[key] = weight
        return {"systems": list(POINT11_SYSTEMS),
                "edges": [{"source": a, "target": b, "weight": w,
                           "edge_type": "functional_information_dependency_not_anatomical_wiring"}
                          for (a, b), w in sorted(weights.items()) if w > 0],
                "sparsity": 1.0 - len([w for w in weights.values() if w > 0]) / float(len(POINT11_SYSTEMS) ** 2)}

    def _update_systems(self, drives: Mapping[str, float], representations: Mapping[str, Any],
                        gates: Mapping[str, float], graph: Mapping[str, Any]) -> dict[str, Any]:
        """Update system activations using gated messages while preserving system-specific records."""
        previous = {name: _bounded(self.state.systems[name]["activation"]) for name in POINT11_SYSTEMS}
        incoming: dict[str, list[dict[str, Any]]] = {name: [] for name in POINT11_SYSTEMS}
        for edge in graph.get("edges", []):
            source, target = edge["source"], edge["target"]
            gate = _bounded(gates.get(source, 1.0))
            message = float(edge["weight"]) * gate * previous[source]
            incoming[target].append({"source": source, "message": message, "gate": gate,
                                     "weight": float(edge["weight"])})
        updated = {}
        for name in POINT11_SYSTEMS:
            drive = _bounded(drives.get(name, 0.0))
            recurrent = sum(item["message"] for item in incoming[name])
            activation = math.tanh(drive + 0.10 * previous[name] + self.recurrent_gain * recurrent)
            if name == "mentalizing" and gates.get("mentalizing", 0.0) < 0.5:
                activation = 0.0
            if name == "multiple_demand" and gates.get("multiple_demand", 0.0) < 0.5:
                activation = 0.0
            updated[name] = _bounded(activation)
            self.state.systems[name] = {
                "activation": updated[name], "representation": deepcopy(list(representations.get(name, []))),
                "source": "point11_gated_functional_coordinator", "confidence": updated[name],
                "status": "recruited" if updated[name] >= 0.5 else "supporting_or_low_activity",
            }
        return {"prior_activity": previous, "new_activity": updated, "incoming_messages": incoming,
                "update_rule": "tanh(drive + 0.10*prior + recurrent_gain*sum(gated_messages))"}

    def _selectivity(self, profile: Mapping[str, Any]) -> dict[str, Any]:
        """Summarize supplied multi-task responses without inferring absent task responses."""
        if not profile:
            return {"available": False, "reason": "no task response profile supplied"}
        return {"available": True, "responses": _plain(profile),
                "language_vs_control": (_mean(profile.get("language", [])) - _mean(profile.get("non_language_control", [])))
                if _mean(profile.get("language", [])) is not None and _mean(profile.get("non_language_control", [])) is not None else None,
                "selectivity_is_profile_not_absolute_exclusivity": True}

    def process(self, request: Point11Request) -> Point11Result:
        """Run the 16-step functional-separation algorithm for one input increment."""
        trace: list[dict[str, Any]] = []
        self.state.time_index += 1
        demand, demand_detail = self._demand_profile(request)
        self.state.demand = demand
        self._record(trace, 1, {"demand_profile": demand, **demand_detail})

        sensory = {"modality": request.task.get("modality", "unspecified"),
                   "form": _plain(request.sensory_input), "quality": _bounded(request.task.get("signal_quality", 1.0)),
                   "meaning_assigned_by_sensory_encoder": False}
        self._update_placeholder("perceptual", sensory, demand["sensory"])
        self._record(trace, 2, sensory)

        localization = self._localize_language(request)
        self.state.localization = deepcopy(localization)
        self._record(trace, 3, localization)

        lexical = self._stage_representation(request.stage_outputs, ("lexical", "point2", "point02"))
        syntax = self._stage_representation(request.stage_outputs, ("syntactic", "point3", "point03"))
        semantic = self._stage_representation(request.stage_outputs, ("semantic", "propositions", "point4", "point04"))
        language_rep = {"lexical_candidates": lexical, "syntactic_structure": syntax,
                        "literal_semantic_structure": semantic, "linguistic_source_preserved": True}
        language_drive = max(demand["linguistic"], min(1.0, 0.2 * len(lexical) + 0.2 * len(syntax) + 0.25 * len(semantic)))
        self._update_placeholder("language", language_rep, language_drive)
        self._record(trace, 4, {"representation": language_rep, "drive": language_drive,
                                "linguistic_computation_not_assigned_to_executive_system": True})

        conceptual = list(request.conceptual_candidates)
        if not conceptual:
            conceptual = self._stage_representation(request.stage_outputs, ("conceptual_knowledge", "knowledge", "world_knowledge"))
        sourced = [{"value": _plain(item), "source": "conceptual_knowledge",
                    "confidence": _bounded(item.get("confidence", 0.5)), "time_index": self.state.time_index}
                   for item in conceptual]
        self.state.sourced_features = deepcopy(sourced)
        self._update_placeholder("conceptual_knowledge", conceptual, max(demand["conceptual"], min(1.0, len(conceptual) / 4.0)))
        self._record(trace, 5, {"conceptual_features": sourced,
                                "literal_content_kept_separate": True})

        social_score = demand["social"]
        social_gate = _sigmoid(8.0 * (social_score - self.social_threshold))
        if request.task.get("mentalizing_required", False):
            social_gate = 1.0
        agents = list(request.agent_candidates)
        if social_gate >= 0.5 and not agents:
            agents = self._stage_representation(request.stage_outputs, ("mentalizing", "agent_beliefs", "speaker_model"))
        agent_features = [{"value": _plain(item), "source": "speaker_model_or_agent_perspective",
                           "confidence": _bounded(item.get("confidence", 0.5)), "time_index": self.state.time_index}
                          for item in agents] if social_gate >= 0.5 else []
        self.state.sourced_features.extend(deepcopy(agent_features))
        self._update_placeholder("mentalizing", agents if social_gate >= 0.5 else [], social_gate if social_gate >= 0.5 else 0.0)
        self._record(trace, 6, {"score": social_score, "gate": social_gate,
                                "recruited": social_gate >= 0.5, "agent_hypotheses": agent_features,
                                "literal_proposition_overwritten": False})

        executive_score = max(demand["working_memory"], demand["executive"], demand["response"])
        executive_gate = _sigmoid(8.0 * (executive_score - self.executive_threshold))
        self._update_placeholder("multiple_demand", request.task.get("control_requirements", []),
                                 executive_gate if executive_gate >= 0.5 else 0.0)
        self._record(trace, 7, {"demand": executive_score, "gate": executive_gate,
                                "recruited": executive_gate >= 0.5,
                                "language_computation_assigned_to_multiple_demand": False})

        weights = dict(request.individual_profile.get("communication_weights", {}))
        weights.update(request.network_weights)
        graph = self._communication_graph(weights)
        self.state.communication_graph = deepcopy(graph)
        gates = {"mentalizing": social_gate if social_gate >= 0.5 else 0.0,
                 "multiple_demand": executive_gate if executive_gate >= 0.5 else 0.0}
        drives = {"perceptual": demand["sensory"], "language": language_drive,
                  "conceptual_knowledge": max(demand["conceptual"], min(1.0, len(conceptual) / 4.0)),
                  "mentalizing": social_gate if social_gate >= 0.5 else 0.0,
                  "multiple_demand": executive_gate if executive_gate >= 0.5 else 0.0,
                  "memory": max(demand["working_memory"], _bounded(request.task.get("episodic_memory_demand", 0.0)))}
        representations = {"perceptual": [sensory], "language": [language_rep],
                           "conceptual_knowledge": conceptual, "mentalizing": agents,
                           "multiple_demand": [request.task.get("control_requirements", {})],
                           "memory": self._stage_representation(request.stage_outputs, ("working_memory", "memory", "point6", "point7"))}
        recurrent = self._update_systems(drives, representations, gates, graph)
        self._record(trace, 8, {"graph": graph, "recurrent_update": recurrent,
                                "system_representations_remain_distinct": True})

        selectivity = self._selectivity(request.task_response_profile)
        self._record(trace, 9, selectivity)

        contribution = {
            "association": _plain(request.individual_profile.get("association_evidence", {})),
            "lesion": _plain(request.individual_profile.get("lesion_evidence", {})),
            "perturbation": _plain(request.individual_profile.get("perturbation_evidence", {})),
            "coactivation_is_not_causation": True,
            "evidence_labels_not_collapsed": True,
        }
        self._record(trace, 10, contribution)

        dissociation = _plain(request.individual_profile.get("language_thought_measures", {}))
        self._record(trace, 11, {"separate_measures": dissociation,
                                 "language_equals_all_thought": False,
                                 "complete_independence_claimed": False})

        individual = {"localization_source": localization["source"],
                      "laterality": _plain(request.individual_profile.get("laterality", "not_supplied")),
                      "stability": _plain(request.individual_profile.get("within_person_stability", {})),
                      "between_person_variation": _plain(request.individual_profile.get("between_person_variation", {}))}
        self._record(trace, 12, individual)

        right_demand = max(_bounded(request.task.get("narrative_level", 0.0)),
                           _bounded(request.task.get("semantic_control_demand", 0.0)),
                           _bounded(request.task.get("prosody_demand", 0.0)),
                           _bounded(request.task.get("discourse_coherence_demand", 0.0)))
        right_evidence = request.individual_profile.get("right_hemisphere_evidence", {})
        right_report = {"task_demand_score": right_demand,
                        "measured_evidence": _plain(right_evidence),
                        "activation_claim_made": bool(right_evidence),
                        "not_assumed_irrelevant": True}
        self._record(trace, 13, right_report)

        self._record(trace, 14, {"incremental_update": True, "time_index": self.state.time_index,
                                 "demand_profile_reestimated": True})

        all_features = deepcopy(self.state.sourced_features)
        literal = semantic if semantic else self._stage_representation(request.stage_outputs, ("literal_propositions",))
        integrated = {"literal_propositions": _plain(literal),
                      "source_tagged_enrichments": all_features,
                      "alternatives": _plain(request.task.get("interpretation_alternatives", [])),
                      "uncertainty": _bounded(request.task.get("interpretation_uncertainty", 0.0)),
                      "provenance_preserved": True}
        self.state.integrated_representation = deepcopy(integrated)
        self._record(trace, 15, integrated)

        marginal = _bounded(request.task.get("additional_inference_utility", 0.0))
        extra_cost = _bounded(request.task.get("additional_inference_cost", 0.0))
        continue_inference = marginal > extra_cost
        stopping = {"marginal_utility": marginal, "additional_inference_cost": extra_cost,
                    "continue_additional_inference": continue_inference,
                    "speculative_detail_not_added_when_cost_exceeds_utility": not continue_inference}
        self._record(trace, 16, stopping)

        self.state.history.append({"time_index": self.state.time_index, "demand": deepcopy(demand),
                                   "activations": {k: v["activation"] for k, v in self.state.systems.items()}})
        biological_summary, biological_trace = {}, []
        if self.lif is not None:
            biological = self.lif.run(drives, steps=16)
            biological_summary = {k: v for k, v in biological.items() if k != "trace"}
            biological_trace = biological["trace"]
        return Point11Result(request.sequence_id, self.state.time_index, demand, localization,
                             deepcopy(self.state.systems), graph, selectivity, contribution, all_features,
                             integrated, trace,
                             {"valid": True, "step_count": len(trace), "system_count": len(POINT11_SYSTEMS),
                              "functional_connectivity_is_not_anatomical_wiring": True,
                              "biological_reference_is_measured_human_data": False},
                             biological_summary, biological_trace)

    def _update_placeholder(self, system: str, representation: Any, drive: float) -> None:
        """Set an initial system record before the recurrent cross-system update."""
        self.state.systems[system] = {"activation": _bounded(drive), "representation": _sequence(representation),
                                      "source": "specialized_system_local_operation",
                                      "confidence": _bounded(drive),
                                      "status": "locally_active" if drive >= 0.5 else "low_local_demand"}


def register_point11_annotations(repository_root: str | None = None,
                                 database_path: str | None = None) -> dict[str, Any]:
    """Index Point 11 classes and functions in the shared ORE SQLite annotation catalogue."""
    import os
    from pathlib import Path
    from ore_logger import LoggerDatabase, sync_repository

    root = Path(repository_root).expanduser().resolve() if repository_root is not None else Path(__file__).resolve().parents[1]
    target = Path(database_path or os.environ.get("ORE_LOGGER_DB") or root / "ore_logs" / "ore_logger.sqlite3").expanduser().resolve()
    database = LoggerDatabase(target)
    modules = ["ore_language_comprehension"]
    report = sync_repository(database, root, include_paths=set(modules))
    return {**report, "database_path": str(target), "modules": modules,
            "annotation_source": "python_docstring_and_ast_signature",
            "storage": "ore_logger.annotations SQLite table"}
