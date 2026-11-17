"""Point 8: recurrent integration across the seven language-comprehension stages.

The processor coordinates partial sensory, lexical, syntactic, semantic, contextual,
working-memory, and memory/decision outputs. It is a functional orchestration model:
it does not claim that the brain executes a serial Python pipeline or a single
global Bayesian calculation. The optional LIF track is a transparent reference
simulation, not measured human neural data.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np


POINT8_STAGE_NAMES = (
    "sensory",
    "lexical",
    "syntactic",
    "semantic",
    "discourse",
    "prediction",
    "executive_control",
    "working_memory",
    "social_inference",
    "memory",
)

POINT8_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "operation_id": "LC-P08-01",
        "name": "Receive a new increment of sensory evidence",
        "objective": "Add time-indexed acoustic or visual evidence while retaining uncertainty and candidate alternatives.",
        "mathematical_operations": ("x_t=f_theta(s_(t-k:t))", "P(z|s) proportional to P(s|z)P(z)", "preserve time_index and signal_quality"),
        "biological_reference": "Auditory or visual pathways encode modality-specific features and interact recurrently with language-sensitive regions; the software record is not a neural packet.",
    },
    {
        "operation_id": "LC-P08-02",
        "name": "Activate lexical candidates before the stream is complete",
        "objective": "Maintain word-form alternatives and update their activation from sensory fit, lexical knowledge, morphology, and context.",
        "mathematical_operations": ("A_i=w_s*S_i+w_l*L_i+w_c*C_i+w_m*M_i-w_i*I_i", "p_i=softmax(A_i/tau)", "retain top-B without erasing consequential alternatives"),
        "biological_reference": "Distributed temporal and frontal language populations support lexical access; contextual feedback can bias candidates while bottom-up evidence remains influential.",
    },
    {
        "operation_id": "LC-P08-03",
        "name": "Update grammatical hypotheses in parallel",
        "objective": "Extend partial parses using lexical categories, dependencies, agreement, and open argument expectations.",
        "mathematical_operations": ("Score(G_j,t)=Score(G_j,t-1)+log P(w_t,m_t|G_j)", "G_t=TopB(G_(t-1) union extensions)", "preserve unresolved attachments"),
        "biological_reference": "Frontal and posterior temporal language-sensitive regions interact during structure building; no single region owns a whole parse.",
    },
    {
        "operation_id": "LC-P08-04",
        "name": "Retrieve context-sensitive meanings while structure develops",
        "objective": "Keep plausible senses and semantic features available, constrained by lexical identity, syntax, and context.",
        "mathematical_operations": ("P(s|w,C,G) proportional to P(w|s)P(s|C,G)", "v_sense=sum_i p_i*v_i", "retain ambiguity when evidence is insufficient"),
        "biological_reference": "Temporal semantic systems and language-selective regions exchange graded information; a feature-vector mixture is one computational option, not a unique neural code.",
    },
    {
        "operation_id": "LC-P08-05",
        "name": "Compose partial semantic structures",
        "objective": "Bind only supported predicates, arguments, modifiers, and operators; keep unfilled roles open.",
        "mathematical_operations": ("Compat(e,r)=v_e^T W_r v_predicate", "P(roles|words,syntax,context) proportional to P(words|roles)P(roles|syntax,context)", "preserve open slots"),
        "biological_reference": "Inferior frontal and temporal language systems contribute to composition; conceptual knowledge constrains interpretation without replacing grammatical evidence.",
    },
    {
        "operation_id": "LC-P08-06",
        "name": "Update the discourse and situation model",
        "objective": "Link current propositions to existing entities, events, time, space, causal relations, and common ground with provenance.",
        "mathematical_operations": ("R(e_i)=alpha*recency+beta*salience+gamma*agreement+delta*semantic_fit+eta*context_fit", "D_t=D_(t-1) union {(u,r,v,source,confidence)}", "check contradictions before state replacement"),
        "biological_reference": "Language and conceptual-memory systems interact with event and situation representations; an explicit symbolic graph is a model representation, not a claim about neural storage format.",
    },
    {
        "operation_id": "LC-P08-07",
        "name": "Generate graded expectations at multiple linguistic levels",
        "objective": "Use the current parse, meaning, context, and task to predict possible continuations without treating the strongest expectation as certain.",
        "mathematical_operations": ("P(X_(t+1)|X_(1:t),C_t,G_t,M_t)", "Surprisal(x_t)=-log P(x_t|X_(1:t-1),C_(t-1))", "keep a distribution over alternatives"),
        "biological_reference": "Predictive processing can involve multiple levels and distributed networks; surprisal is not identical to any one ERP component.",
    },
    {
        "operation_id": "LC-P08-08",
        "name": "Compare actual evidence with current predictions",
        "objective": "Compute separate sensory, lexical, syntactic, semantic, referential, discourse, and pragmatic mismatch signals.",
        "mathematical_operations": ("epsilon_t=x_t-x_hat_t", "epsilon=(epsilon_sensory,...,epsilon_pragmatic)", "surprisal=-log P(observed|context)"),
        "biological_reference": "Prediction errors may influence recurrent and control processes; different mismatch types should not be assigned to one universal neural error signal.",
    },
    {
        "operation_id": "LC-P08-09",
        "name": "Allocate attention and executive-control resources",
        "objective": "Prioritize unresolved issues by uncertainty, importance, task relevance, load, and expected value of further processing.",
        "mathematical_operations": ("Priority_i=Uncertainty_i*Impact_i*Relevance_i", "sum_i r_i <= B", "attention_i=B*Priority_i/sum_j Priority_j"),
        "biological_reference": "Frontoparietal control and attention networks can contribute under high demand; the budget is an engineering abstraction, not a literal central counter.",
    },
    {
        "operation_id": "LC-P08-10",
        "name": "Trigger targeted reanalysis when evidence warrants it",
        "objective": "Reopen only affected lexical, syntactic, semantic, referential, or contextual hypotheses and retain revision history.",
        "mathematical_operations": ("Score(H_i)=log P(X|H_i)+log P(H_i|C)-lambda*Cost_revision(H_i)", "replace only when new_score-old_score >= margin", "recompute dependent nodes"),
        "biological_reference": "Reanalysis plausibly recruits recurrent language and control interactions; local software revision is an approximation of distributed neural dynamics.",
    },
    {
        "operation_id": "LC-P08-11",
        "name": "Generate pragmatic or social inferences conditionally",
        "objective": "Generate a bounded set of candidate intentions or bridging links only when literal meaning leaves a relevant explanatory gap.",
        "mathematical_operations": ("P(I_j|U,C,S,K) proportional to P(U|I_j,S,C)P(I_j|C,K,S)", "Coherence=aL+bD+cW+dG-eConflict", "keep literal and inferred fields separate"),
        "biological_reference": "Medial prefrontal and temporoparietal social-cognition systems may contribute when beliefs, intentions, or perspectives matter; they are not required for every sentence.",
    },
    {
        "operation_id": "LC-P08-12",
        "name": "Maintain the evolving proposition and unresolved dependencies",
        "objective": "Prioritize task-relevant propositions, open roles, referents, and relations while tracking interference and deprioritized pointers.",
        "mathematical_operations": ("A_i(t+1)=rho*A_i(t)+Input_i(t)-kappa*Interference_i(t)", "WM=TopK(activation_i)", "preserve pointers for deprioritized items"),
        "biological_reference": "Working memory may involve active, latent, and reactivated population states across frontal and parietal networks; one decay equation is not a full biological account.",
    },
    {
        "operation_id": "LC-P08-13",
        "name": "Test whether interpretation quality is sufficient to continue",
        "objective": "Check parse support, proposition completeness, referential fit, context integration, and task-specific unresolved conflicts.",
        "mathematical_operations": ("Q_t=sum_i w_i*q_i", "continue if Q_t >= theta_task", "defer when a critical dependency remains unresolved"),
        "biological_reference": "Comprehension monitoring and control depend on interacting language and domain-general systems; a scalar threshold is a computational decision rule.",
    },
    {
        "operation_id": "LC-P08-14",
        "name": "Finalize a revisable sentence-level proposition",
        "objective": "Package supported literal content, contextually supported additions, explicit provenance, unresolved fields, and remaining alternatives.",
        "mathematical_operations": ("P_final=(predicate,roles,operators,relations,source,confidence,status)", "select argmax_H Q(H) subject to uncertainty constraints", "do not collapse inferred into explicit"),
        "biological_reference": "Distributed language and conceptual systems support an integrated interpretation; the proposition object is a software output, not a literal neural data structure.",
    },
    {
        "operation_id": "LC-P08-15",
        "name": "Route the interpretation to the appropriate downstream process",
        "objective": "Keep context active, prepare question-answer or decision support, and construct episodic or semantic memory handoffs only when requested.",
        "mathematical_operations": ("routes subset {discourse,answer,decision,episodic,semantic}", "P(route|goal,content,context)", "allow multiple simultaneous routes"),
        "biological_reference": "Language representations can guide responses, decisions, and memory systems; understanding, action execution, and durable memory encoding are distinct.",
    },
    {
        "operation_id": "LC-P08-16",
        "name": "Reopen earlier processing when later discourse changes the interpretation",
        "objective": "Apply new evidence to the affected prior proposition, revise dependent conclusions, and preserve what changed and why.",
        "mathematical_operations": ("M_t=Update(M_(t-1),E_t,Delta_t)", "revision_delta=affected_nodes(new_evidence,old_model)", "retain prior snapshot and source"),
        "biological_reference": "Later input can cause reactivation and revision of earlier representations through recurrent interactions; software revision logs make the dependency explicit.",
    },
    {
        "operation_id": "LC-P08-17",
        "name": "Prepare appropriate long-term memory consolidation",
        "objective": "Select only supported, useful information for episodic or semantic memory handoff, preserving source and uncertainty.",
        "mathematical_operations": ("retention=w_n*N+w_r*R+w_g*G+w_s*S", "episodic=(event,context,source,confidence)", "semantic_candidate requires generality and evidence"),
        "biological_reference": "Medial temporal and neocortical systems contribute to memory across different timescales; this step packages a handoff and does not assert immediate permanent storage.",
    },
    {
        "operation_id": "LC-P08-18",
        "name": "Monitor outcomes and adapt future comprehension strategy",
        "objective": "Use task feedback to identify recurring error types and adjust attention or revision policy without turning one correction into a universal rule.",
        "mathematical_operations": ("prediction_error=observed_outcome-predicted_outcome", "strategy_(t+1)=bounded_update(strategy_t,error_history)", "track calibration separately from activation frequency"),
        "biological_reference": "Learning and strategy adaptation can involve synaptic plasticity and neuromodulatory/control systems over multiple timescales; the model's parameter update is not a measured neural rule.",
    },
)


def _mapping(value: Any, field_name: str = "value") -> dict[str, Any]:
    """Return a mutable mapping from a mapping or a result object exposing to_dict()."""
    if isinstance(value, Mapping):
        return dict(value)
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return dict(converted)
    if value is None:
        return {}
    raise TypeError(f"{field_name} must be a mapping or expose to_dict()")


def _bounded_score(value: Any, default: float = 0.5) -> float:
    """Convert a value to a finite score in the closed interval [0, 1]."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(score):
        return default
    return min(1.0, max(0.0, score))


def _tokens(value: Any) -> set[str]:
    """Extract case-folded lexical cues from a value for transparent baseline matching."""
    if value is None:
        return set()
    if isinstance(value, Mapping):
        value = " ".join(f"{key} {item}" for key, item in value.items())
    elif isinstance(value, (list, tuple, set)):
        value = " ".join(str(item) for item in value)
    return {
        token.casefold().strip(".,!?;:()[]{}\"'")
        for token in str(value).split()
        if token.strip(".,!?;:()[]{}\"'")
    }


def _similarity(left: Any, right: Any) -> float:
    """Compute Jaccard cue overlap; this is not a learned semantic representation."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _sequence(value: Any) -> list[Any]:
    """Convert supported scalar, mapping, and sequence forms into a candidate list."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        for key in ("candidates", "alternatives", "hypotheses", "items", "propositions", "structures"):
            nested = value.get(key)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                return list(nested)
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _candidate(value: Any, stage: str, index: int) -> dict[str, Any]:
    """Normalize one stage candidate without discarding source, confidence, or metadata."""
    if isinstance(value, Mapping):
        raw = dict(value)
        content = raw.get("content", raw.get("text", raw.get("value", raw.get("interpretation", raw))))
        source = raw.get("source", raw.get("provenance", stage))
        confidence = _bounded_score(raw.get("confidence", raw.get("score", 0.5)))
        status = str(raw.get("status", "provisional"))
        candidate_id = str(raw.get("candidate_id", raw.get("id", f"{stage}-{index}")))
        metadata = dict(raw.get("metadata", {})) if isinstance(raw.get("metadata", {}), Mapping) else {}
        for key in ("sensory_score", "lexical_score", "context_score", "structural_score",
                    "semantic_score", "discourse_score", "goal_fit", "impact", "relevance",
                    "uncertainty", "expected_next", "expected_category", "senses", "roles",
                    "predicate", "semantic_roles", "relations", "operator_scope"):
            if key in raw and key not in metadata:
                metadata[key] = raw[key]
    else:
        content = value
        source = stage
        confidence = 0.5
        status = "provisional"
        candidate_id = f"{stage}-{index}"
        metadata = {}
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "content": deepcopy(content),
        "source": str(source),
        "confidence": confidence,
        "status": status,
        "metadata": metadata,
    }


def _extract_candidates(value: Any, stage: str, keys: Sequence[str]) -> list[dict[str, Any]]:
    """Extract candidates from a standard stage result or from a direct candidate sequence."""
    mapped = _mapping(value, f"{stage}_output") if hasattr(value, "to_dict") else value
    selected = None
    if isinstance(mapped, Mapping):
        for key in keys:
            if key in mapped:
                selected = mapped[key]
                break
    if selected is None:
        selected = mapped
    return [_candidate(item, stage, index) for index, item in enumerate(_sequence(selected))]


def _candidate_text(item: Mapping[str, Any]) -> str:
    """Return a stable text view used only for lexical cue overlap."""
    return str(item.get("content", ""))


def _stage_value(stage_outputs: Mapping[str, Any], names: Sequence[str]) -> Any:
    """Find the first present alias for a stage output without inventing missing data."""
    for name in names:
        if name in stage_outputs:
            return stage_outputs[name]
    return None


@dataclass(frozen=True, slots=True)
class Point8Request:
    """One new comprehension increment and any available partial outputs from Points 1–7."""

    sequence_id: str
    increment: Any = None
    stage_outputs: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    speaker_model: Mapping[str, Any] = field(default_factory=dict)
    task: Mapping[str, Any] = field(default_factory=dict)
    predictions: Mapping[str, Any] = field(default_factory=dict)
    observed_evidence: Mapping[str, Any] = field(default_factory=dict)
    unresolved: Sequence[Mapping[str, Any]] = ()
    revision_candidates: Sequence[Mapping[str, Any]] = ()
    inference_candidates: Sequence[Mapping[str, Any]] = ()
    later_evidence: Mapping[str, Any] = field(default_factory=dict)
    feedback: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate request containers and snapshot nested stage-result mappings."""
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        for name in ("stage_outputs", "context", "speaker_model", "task",
                     "predictions", "observed_evidence", "later_evidence", "feedback"):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping")
        for name in ("unresolved", "revision_candidates", "inference_candidates"):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{name} must be a sequence")
            if not all(isinstance(item, Mapping) for item in value):
                raise TypeError(f"every {name} item must be a mapping")
        object.__setattr__(self, "stage_outputs", deepcopy(dict(self.stage_outputs)))
        object.__setattr__(self, "context", deepcopy(dict(self.context)))
        object.__setattr__(self, "speaker_model", deepcopy(dict(self.speaker_model)))
        object.__setattr__(self, "task", deepcopy(dict(self.task)))
        object.__setattr__(self, "predictions", deepcopy(dict(self.predictions)))
        object.__setattr__(self, "observed_evidence", deepcopy(dict(self.observed_evidence)))
        object.__setattr__(self, "unresolved", tuple(deepcopy(dict(x)) for x in self.unresolved))
        object.__setattr__(self, "revision_candidates", tuple(deepcopy(dict(x)) for x in self.revision_candidates))
        object.__setattr__(self, "inference_candidates", tuple(deepcopy(dict(x)) for x in self.inference_candidates))
        object.__setattr__(self, "later_evidence", deepcopy(dict(self.later_evidence)))
        object.__setattr__(self, "feedback", deepcopy(dict(self.feedback)))

    @classmethod
    def from_stage_results(
        cls,
        sequence_id: str,
        stage_results: Mapping[str, Any],
        **overrides: Any,
    ) -> "Point8Request":
        """Build a request from any subset of the prior seven stage result objects."""
        if not isinstance(stage_results, Mapping):
            raise TypeError("stage_results must be a mapping")
        aliases = {
            "point1": "sensory", "point01": "sensory", "sensory": "sensory",
            "point2": "lexical", "point02": "lexical", "lexical": "lexical",
            "point3": "syntactic", "point03": "syntactic", "syntax": "syntactic",
            "syntactic": "syntactic",
            "point4": "semantic", "point04": "semantic", "semantic": "semantic",
            "point5": "discourse", "point05": "discourse", "contextual": "discourse",
            "point6": "working_memory", "point06": "working_memory",
            "point7": "memory", "point07": "memory",
        }
        normalized = {}
        for name, value in stage_results.items():
            key = aliases.get(str(name).casefold(), str(name))
            normalized[key] = _mapping(value, f"{name}_result")
        return cls(sequence_id=sequence_id, stage_outputs=normalized, **overrides)


@dataclass(slots=True)
class Point8State:
    """Mutable cross-increment state with separate channels and provenance-aware candidates."""

    time_index: int = 0
    sensory_candidates: list[dict[str, Any]] = field(default_factory=list)
    lexical_candidates: list[dict[str, Any]] = field(default_factory=list)
    syntactic_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    semantic_candidates: list[dict[str, Any]] = field(default_factory=list)
    propositions: list[dict[str, Any]] = field(default_factory=list)
    inferred_propositions: list[dict[str, Any]] = field(default_factory=list)
    discourse_model: dict[str, Any] = field(default_factory=dict)
    speaker_model: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)
    prediction_errors: dict[str, float] = field(default_factory=dict)
    unresolved: list[dict[str, Any]] = field(default_factory=list)
    attention_allocation: list[dict[str, Any]] = field(default_factory=list)
    working_memory: list[dict[str, Any]] = field(default_factory=list)
    deprioritized_ids: list[str] = field(default_factory=list)
    revision_log: list[dict[str, Any]] = field(default_factory=list)
    feedback_history: list[dict[str, Any]] = field(default_factory=list)
    strategy: dict[str, Any] = field(default_factory=lambda: {
        "attention_gain": 1.0,
        "revision_margin": 0.08,
        "candidate_beam": 8,
        "error_counts": {},
    })

    def to_dict(self) -> dict[str, Any]:
        """Return a detached JSON-compatible snapshot of the current recurrent state."""
        return deepcopy(asdict(self))


@dataclass(slots=True)
class Point8Result:
    """Outputs for one increment, including trace, loops, and downstream handoffs."""

    sequence_id: str
    time_index: int
    state: dict[str, Any]
    functional_trace: list[dict[str, Any]]
    feedback_loops: dict[str, dict[str, Any]]
    interpretation_quality: dict[str, Any]
    finalized_interpretation: dict[str, Any] | None
    routing: dict[str, Any]
    diagnostics: dict[str, Any]
    biological_summary: dict[str, Any] = field(default_factory=dict)
    biological_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result for logging, inspection, or the next processing stage."""
        return deepcopy(asdict(self))


class Point8LIFReference:
    """Small current-based LIF reference for ten interacting functional populations."""

    def __init__(
        self,
        population_size: int = 12,
        steps_per_operation: int = 2,
        dt_ms: float = 1.0,
        tau_ms: float = 20.0,
        threshold: float = 1.0,
        refractory_steps: int = 2,
    ) -> None:
        """Validate simulation parameters and prepare deterministic population dynamics."""
        if population_size < 1 or steps_per_operation < 1 or dt_ms <= 0 or tau_ms <= 0:
            raise ValueError("population_size, steps_per_operation, dt_ms, and tau_ms must be positive")
        if threshold <= 0 or refractory_steps < 0:
            raise ValueError("threshold must be positive and refractory_steps cannot be negative")
        self.population_size = int(population_size)
        self.steps_per_operation = int(steps_per_operation)
        self.dt_ms = float(dt_ms)
        self.tau_ms = float(tau_ms)
        self.threshold = float(threshold)
        self.refractory_steps = int(refractory_steps)

    def run(self, operation_count: int = 18) -> dict[str, Any]:
        """Run one deterministic reference pass and report population rates, not human data."""
        if operation_count < 1:
            raise ValueError("operation_count must be positive")
        names = list(POINT8_STAGE_NAMES)
        count = len(names)
        decay = math.exp(-self.dt_ms / self.tau_ms)
        membrane = np.zeros((count, self.population_size), dtype=np.float64)
        refractory = np.zeros((count, self.population_size), dtype=np.int32)
        spikes = np.zeros_like(membrane)
        traces: list[dict[str, Any]] = []
        for operation in range(operation_count):
            active = operation % count
            drive = np.full((count, self.population_size), 0.03, dtype=np.float64)
            drive[active, :] += 0.58
            if operation in (5, 7, 9, 15):
                drive[(active + 1) % count, :] += 0.18
            operation_spikes = 0
            for _ in range(self.steps_per_operation):
                available = refractory == 0
                membrane = np.where(available, decay * membrane + drive, membrane)
                fired = (membrane >= self.threshold) & available
                operation_spikes += int(fired.sum())
                membrane = np.where(fired, 0.0, membrane)
                refractory = np.where(fired, self.refractory_steps, np.maximum(0, refractory - 1))
                spikes += fired.astype(np.float64)
            traces.append({
                "operation": operation + 1,
                "active_population": names[active],
                "spikes": operation_spikes,
                "status": "simulated_reference_not_measured_human_data",
            })
        rates = (spikes / (self.steps_per_operation * operation_count)).mean(axis=1)
        return {
            "model": "current_based_leaky_integrate_and_fire_reference",
            "population_names": names,
            "population_rates": {name: float(rate) for name, rate in zip(names, rates)},
            "operation_count": operation_count,
            "steps_per_operation": self.steps_per_operation,
            "biological_reference_is_measured_human_data": False,
            "trace": traces,
        }


class Point8CrossStageIntegrator:
    """Coordinate recurrent, provenance-aware updates across language comprehension stages."""

    def __init__(
        self,
        max_candidates: int = 8,
        working_memory_capacity: int = 12,
        quality_threshold: float = 0.55,
        revision_margin: float = 0.08,
        simulate_biology: bool = False,
        population_size: int = 12,
        steps_per_operation: int = 2,
    ) -> None:
        """Configure bounded hypothesis retention, working memory, and optional LIF reference."""
        if max_candidates < 1 or working_memory_capacity < 1:
            raise ValueError("candidate and working-memory capacities must be positive")
        if not 0.0 <= quality_threshold <= 1.0 or revision_margin < 0:
            raise ValueError("quality_threshold must be in [0, 1] and revision_margin cannot be negative")
        self.max_candidates = int(max_candidates)
        self.working_memory_capacity = int(working_memory_capacity)
        self.quality_threshold = float(quality_threshold)
        self.revision_margin = float(revision_margin)
        self.state = Point8State()
        self.state.strategy["revision_margin"] = self.revision_margin
        self.lif = Point8LIFReference(population_size, steps_per_operation) if simulate_biology else None

    def reset(self) -> None:
        """Start a new independent stream while retaining processor configuration."""
        self.state = Point8State()

    def _record(self, trace: list[dict[str, Any]], step: int, output: Any) -> None:
        """Append one source-level operation with a detached intermediate output."""
        spec = POINT8_STEP_SPECS[step - 1]
        trace.append({
            "step": step,
            "operation_id": spec["operation_id"],
            "name": spec["name"],
            "output": deepcopy(output),
        })

    def _merge_candidates(
        self,
        existing: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge candidates by stable identity and content, retaining the strongest evidence."""
        merged = {item["candidate_id"]: deepcopy(item) for item in existing}
        for item in incoming:
            candidate_id = item["candidate_id"]
            prior = merged.get(candidate_id)
            if prior is None:
                merged[candidate_id] = deepcopy(item)
                continue
            old_score = _bounded_score(prior.get("confidence"))
            new_score = _bounded_score(item.get("confidence"))
            # Repeated activation of the same candidate is not new evidence.
            # Keep the strongest supplied confidence without inflating it by repetition.
            if new_score >= old_score:
                prior.update(deepcopy(item))
            prior["confidence"] = max(old_score, new_score)
        ranked = sorted(
            merged.values(),
            key=lambda item: (_bounded_score(item.get("confidence")), item["candidate_id"]),
            reverse=True,
        )
        return ranked[: self.max_candidates]

    def _stage_candidates(self, request: Point8Request, stage: str) -> list[dict[str, Any]]:
        """Extract one stage's candidates from explicit aliases in the supplied results."""
        aliases = {
            "sensory": ("sensory", "perceptual", "form_hypotheses"),
            "lexical": ("lexical", "lexical_candidates", "word_candidates"),
            "syntactic": ("syntactic", "syntax", "parse_hypotheses", "syntactic_hypotheses"),
            "semantic": ("semantic", "semantic_candidates", "propositions", "semantic_frames"),
        }
        value = _stage_value(request.stage_outputs, aliases.get(stage, (stage,)))
        keys = {
            "sensory": ("candidates", "form_hypotheses", "alternatives", "items"),
            "lexical": ("lexical_candidates", "candidates", "alternatives", "items"),
            "syntactic": ("parse_hypotheses", "syntactic_hypotheses", "candidates", "alternatives", "structures"),
            "semantic": ("propositions", "semantic_candidates", "semantic_frames", "candidates", "alternatives"),
        }
        if value is None:
            return []
        return _extract_candidates(value, stage, keys.get(stage, ("candidates", "alternatives", "items")))

    def _update_situation_model(self, context: Mapping[str, Any], propositions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Merge supported context fields and new proposition links without blind overwrites."""
        model = deepcopy(self.state.discourse_model)
        for key, value in context.items():
            if key not in model or model[key] in (None, [], {}, ""):
                model[key] = deepcopy(value)
            elif isinstance(model[key], list) and isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                known = {repr(item) for item in model[key]}
                model[key].extend(deepcopy(item) for item in value if repr(item) not in known)
            elif isinstance(model[key], Mapping) and isinstance(value, Mapping):
                merged = dict(model[key])
                for subkey, subvalue in value.items():
                    if subkey not in merged or merged[subkey] in (None, "", [], {}):
                        merged[subkey] = deepcopy(subvalue)
                    elif merged[subkey] != subvalue:
                        conflicts = model.setdefault("context_conflicts", [])
                        conflicts.append({
                            "field": f"{key}.{subkey}",
                            "prior_value": deepcopy(merged[subkey]),
                            "new_value": deepcopy(subvalue),
                            "status": "both_preserved_conflict",
                        })
                model[key] = merged
        records = model.setdefault("propositions", [])
        if not isinstance(records, list):
            records = [records]
            model["propositions"] = records
        known_ids = {str(item.get("candidate_id")) for item in records if isinstance(item, Mapping)}
        for proposition in propositions:
            if proposition["candidate_id"] not in known_ids:
                records.append(deepcopy(proposition))
                known_ids.add(proposition["candidate_id"])
        return model

    def _prediction_errors(
        self,
        predictions: Mapping[str, Any],
        observed: Mapping[str, Any],
    ) -> dict[str, float]:
        """Compute typed mismatches only where both an expected and observed value exist."""
        errors: dict[str, float] = {}
        for channel in ("sensory", "lexical", "syntactic", "semantic", "discourse", "pragmatic"):
            if channel not in predictions or channel not in observed:
                continue
            expected = predictions[channel]
            actual = observed[channel]
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                error = min(1.0, abs(float(actual) - float(expected)))
            else:
                error = 0.0 if expected == actual else 1.0
            errors[channel] = error
        for channel, value in observed.items():
            if channel.endswith("_error"):
                errors[channel.removesuffix("_error")] = _bounded_score(value, 0.0)
        return errors

    def _prioritize(
        self,
        unresolved: Sequence[Mapping[str, Any]],
        errors: Mapping[str, float],
        task: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Rank unresolved issues and distribute a bounded attention budget."""
        issues = [dict(item) for item in unresolved]
        for channel, error in errors.items():
            if error > 0:
                issues.append({
                    "issue_id": f"mismatch-{channel}",
                    "kind": channel,
                    "uncertainty": error,
                    "impact": task.get("mismatch_impact", 0.8),
                    "relevance": task.get("task_relevance", 0.8),
                    "source": "prediction_error",
                })
        weighted = []
        for index, issue in enumerate(issues):
            uncertainty = _bounded_score(issue.get("uncertainty", issue.get("confidence", 0.5)))
            impact = _bounded_score(issue.get("impact", 0.5))
            relevance = _bounded_score(issue.get("relevance", 0.5))
            priority = uncertainty * impact * relevance * _bounded_score(
                self.state.strategy.get("attention_gain", 1.0), 1.0
            )
            weighted.append({
                **issue,
                "issue_id": str(issue.get("issue_id", f"unresolved-{index}")),
                "priority": priority,
            })
        total = sum(item["priority"] for item in weighted)
        budget = max(0.0, float(task.get("control_budget", 1.0)))
        for item in weighted:
            item["allocated_attention"] = budget * item["priority"] / total if total > 0 else 0.0
        return sorted(weighted, key=lambda item: item["priority"], reverse=True)

    def _apply_revisions(
        self,
        candidates: Sequence[Mapping[str, Any]],
        errors: Mapping[str, float],
        reason: str,
    ) -> list[dict[str, Any]]:
        """Apply evidence-supported candidate revisions while retaining the displaced snapshot."""
        applied = []
        channel_map = {
            "sensory": "sensory_candidates",
            "lexical": "lexical_candidates",
            "syntactic": "syntactic_hypotheses",
            "semantic": "semantic_candidates",
            "proposition": "propositions",
        }
        for index, raw in enumerate(candidates):
            stage = str(raw.get("stage", "proposition"))
            target_name = channel_map.get(stage)
            if target_name is None:
                continue
            content = raw.get("content", raw.get("replacement", raw.get("value")))
            confidence = _bounded_score(raw.get("confidence", raw.get("score", 0.5)))
            evidence = _bounded_score(raw.get("evidence_fit", 0.5))
            cost = max(0.0, float(raw.get("revision_cost", 0.0)))
            new_score = 0.7 * confidence + 0.3 * evidence - 0.1 * cost
            current_items = getattr(self.state, target_name)
            current = current_items[0] if current_items else None
            old_score = _bounded_score(current.get("confidence")) if current else 0.0
            margin = new_score - old_score
            required = max(
                self.revision_margin,
                _bounded_score(self.state.strategy.get("revision_margin", self.revision_margin)),
            )
            if current is not None and margin < required:
                applied.append({"stage": stage, "status": "retained_current", "margin": margin})
                continue
            previous = deepcopy(current) if current else None
            replacement = _candidate(raw, stage, index)
            replacement["content"] = deepcopy(content)
            replacement["confidence"] = _bounded_score(new_score)
            replacement["status"] = "revised"
            replacement["metadata"]["revision_reason"] = reason
            if current is not None:
                current_items[0] = replacement
            else:
                current_items.append(replacement)
            event = {
                "stage": stage,
                "previous": previous,
                "replacement": deepcopy(replacement),
                "reason": reason,
                "margin": margin,
            }
            self.state.revision_log.append(event)
            applied.append({"stage": stage, "status": "revised", "margin": margin})
        return applied

    def _pragmatic_inferences(
        self,
        request: Point8Request,
        errors: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        """Evaluate caller-supplied pragmatic candidates only when social inference is relevant."""
        task = request.task
        demand = _bounded_score(task.get("social_inference_demand", 0.0))
        required = bool(task.get("pragmatic_inference_required", False))
        mismatch = errors.get("pragmatic", 0.0) >= _bounded_score(
            task.get("pragmatic_mismatch_threshold", 0.35), 0.35
        )
        if not (required or demand >= _bounded_score(task.get("social_inference_threshold", 0.5)) or mismatch):
            return []
        accepted = []
        threshold = _bounded_score(task.get("inference_acceptance_threshold", 0.6), 0.6)
        for index, raw in enumerate(request.inference_candidates):
            linguistic = _bounded_score(raw.get("linguistic_fit", raw.get("confidence", 0.5)))
            context_fit = _bounded_score(raw.get("context_fit", _similarity(raw.get("content"), request.context)))
            goal_fit = _bounded_score(raw.get("goal_fit", _similarity(raw.get("content"), request.speaker_model)))
            contradiction = _bounded_score(raw.get("contradiction", 0.0))
            score = 0.35 * linguistic + 0.3 * context_fit + 0.25 * goal_fit + 0.1 * _bounded_score(raw.get("confidence", 0.5)) - 0.5 * contradiction
            item = _candidate(raw, "pragmatic_inference", index)
            item["confidence"] = _bounded_score(score, 0.0)
            item["source"] = str(raw.get("source", "contextual_or_speaker_inference"))
            item["status"] = "inferred" if score >= threshold else "unresolved"
            item["metadata"]["literal_content_is_separate"] = True
            item["metadata"]["inference_components"] = {
                "linguistic_fit": linguistic,
                "context_fit": context_fit,
                "goal_fit": goal_fit,
                "contradiction": contradiction,
            }
            if score >= threshold:
                accepted.append(item)
        self.state.inferred_propositions = self._merge_candidates(
            self.state.inferred_propositions, accepted
        )
        return accepted

    def _update_working_memory(
        self,
        unresolved: Sequence[Mapping[str, Any]],
        task: Mapping[str, Any],
        errors: Mapping[str, float],
        supplied_items: Sequence[Any] = (),
    ) -> list[dict[str, Any]]:
        """Refresh active proposition and dependency accessibility under a configurable capacity."""
        sources = self.state.propositions + self.state.inferred_propositions + [
            _candidate(item, "unresolved", index) for index, item in enumerate(unresolved)
        ]
        sources.extend(
            _candidate(item, "working_memory", index)
            for index, item in enumerate(supplied_items)
        )
        previous = {item["candidate_id"]: item for item in self.state.working_memory}
        interference = sum(errors.values()) / max(1, len(errors))
        refreshed = []
        rho = min(0.99, max(0.0, float(task.get("wm_persistence", 0.82))))
        kappa = max(0.0, float(task.get("wm_interference_weight", 0.2)))
        for item in sources:
            prior = _bounded_score(previous.get(item["candidate_id"], {}).get("activation", 0.0), 0.0)
            input_drive = _bounded_score(item.get("confidence", 0.5))
            activation = min(1.0, max(0.0, rho * prior + input_drive - kappa * interference))
            refreshed.append({
                **deepcopy(item),
                "activation": activation,
                "accessibility_model": "rho*previous+input-kappa*mean_prediction_error",
            })
        refreshed.sort(key=lambda item: item["activation"], reverse=True)
        capacity = int(task.get("working_memory_capacity", self.working_memory_capacity))
        if capacity < 1:
            capacity = 1
        retained = refreshed[:capacity]
        retained_ids = {item["candidate_id"] for item in retained}
        self.state.deprioritized_ids = [
            item["candidate_id"] for item in refreshed if item["candidate_id"] not in retained_ids
        ]
        return retained

    def _quality(self, task: Mapping[str, Any], errors: Mapping[str, float]) -> dict[str, Any]:
        """Estimate adequacy from available channel evidence and task-critical open issues."""
        values: dict[str, float] = {}
        if self.state.sensory_candidates:
            values["sensory"] = _bounded_score(self.state.sensory_candidates[0]["confidence"])
        if self.state.lexical_candidates:
            values["lexical"] = _bounded_score(self.state.lexical_candidates[0]["confidence"])
        if self.state.syntactic_hypotheses:
            values["syntactic"] = _bounded_score(self.state.syntactic_hypotheses[0]["confidence"])
        if self.state.semantic_candidates:
            values["semantic"] = _bounded_score(self.state.semantic_candidates[0]["confidence"])
        if self.state.propositions:
            values["proposition"] = _bounded_score(self.state.propositions[0]["confidence"])
        if self.state.discourse_model:
            values["discourse"] = _bounded_score(task.get("discourse_support", 0.7))
        if not values:
            quality = 0.0
        else:
            weights = task.get("quality_weights", {})
            if isinstance(weights, Mapping):
                selected = {key: max(0.0, float(weights.get(key, 1.0))) for key in values}
            else:
                selected = {key: 1.0 for key in values}
            weight_total = sum(selected.values())
            quality = (
                sum(values[key] * selected[key] for key in values) / weight_total
                if weight_total > 0 else sum(values.values()) / len(values)
            )
        mismatch_penalty = min(0.5, sum(errors.values()) / max(1, len(errors)) * 0.5)
        quality = max(0.0, quality - mismatch_penalty)
        threshold = _bounded_score(task.get("quality_threshold", self.quality_threshold))
        critical = [
            item for item in self.state.unresolved
            if item.get("critical", False) and _bounded_score(item.get("uncertainty", 0.5)) >= 0.5
        ]
        adequate = quality >= threshold and not critical
        return {
            "score": quality,
            "threshold": threshold,
            "channel_scores": values,
            "mismatch_penalty": mismatch_penalty,
            "critical_unresolved": [item.get("issue_id", item.get("content")) for item in critical],
            "adequate_to_finalize": adequate,
            "status": "adequate" if adequate else "continue_or_reanalyse",
        }

    def _finalize(self, quality: Mapping[str, Any]) -> dict[str, Any] | None:
        """Package supported literal propositions separately from inferences and unresolved fields."""
        if not quality.get("adequate_to_finalize") or not self.state.propositions:
            return None
        ranked = sorted(
            self.state.propositions,
            key=lambda item: _bounded_score(item.get("confidence")),
            reverse=True,
        )
        best = deepcopy(ranked[0])
        return {
            "literal_propositions": ranked,
            "selected_literal_proposition": best,
            "inferred_propositions": deepcopy(self.state.inferred_propositions),
            "unresolved": deepcopy(self.state.unresolved),
            "remaining_alternatives": [
                item for item in ranked[1:] if item["candidate_id"] != best["candidate_id"]
            ],
            "provenance_policy": "literal_and_inferred_content_remain_separate",
        }

    def _route(
        self,
        request: Point8Request,
        finalized: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Prepare independent handoffs for discourse, response, action, and memory use."""
        task = request.task
        routes = ["discourse_continuation"]
        if task.get("requires_answer", False):
            routes.append("question_answering")
        if task.get("requires_action_plan", False):
            routes.append("decision_support")
        if task.get("encode_episode", False):
            routes.append("episodic_memory")
        if task.get("generalizable_knowledge", False):
            routes.append("semantic_memory_candidate")
        selected = finalized.get("selected_literal_proposition") if finalized else None
        memory_handoff = None
        if "episodic_memory" in routes or "semantic_memory_candidate" in routes:
            memory_handoff = {
                "sequence_id": request.sequence_id,
                "proposition": deepcopy(selected.get("content", {})) if selected else {},
                "components": [
                    {
                        "content": item.get("content"),
                        "provenance": item.get("source", item.get("status", "unresolved")),
                        "confidence": item.get("confidence", 0.0),
                        "status": item.get("status", "provisional"),
                    }
                    for item in (finalized or {}).get("literal_propositions", [])
                ],
                "context": deepcopy(self.state.discourse_model),
                "task": {
                    "encode_episode": "episodic_memory" in routes,
                    "generalizable_knowledge": "semantic_memory_candidate" in routes,
                    "semantic_verified": False,
                },
                "target": "Point7MemoryDecisionProcessor",
                "durable_storage_performed": False,
            }
        return {
            "routes": routes,
            "ready_for_downstream_use": finalized is not None,
            "question_answering_context": deepcopy(selected) if "question_answering" in routes else None,
            "decision_support_context": {
                "proposition": deepcopy(selected),
                "task_constraints": deepcopy(task.get("constraints", {})),
                "options": deepcopy(task.get("options", [])),
            } if "decision_support" in routes else None,
            "memory_handoff": memory_handoff,
            "note": "Routing prepares downstream inputs; it does not execute an answer, action, or durable write.",
        }

    def process(self, request: Point8Request) -> Point8Result:
        """Process one increment, update recurrent state, and return all 18 source-level steps."""
        if not isinstance(request, Point8Request):
            raise TypeError("request must be a Point8Request")
        state = self.state
        state.time_index += 1
        trace: list[dict[str, Any]] = []
        stage = request.stage_outputs

        # Step 1: receive bottom-up evidence and keep the original increment.
        sensory = []
        if request.increment is not None:
            sensory = [_candidate(item, "sensory", index) for index, item in enumerate(_sequence(request.increment))]
        sensory.extend(self._stage_candidates(request, "sensory"))
        state.sensory_candidates = self._merge_candidates(state.sensory_candidates, sensory)
        self._record(trace, 1, {
            "time_index": state.time_index,
            "new_increment_received": request.increment is not None,
            "sensory_candidate_count": len(state.sensory_candidates),
        })

        # Step 2: context and sensory evidence adjust explicit lexical candidates.
        lexical = self._stage_candidates(request, "lexical")
        for item in lexical:
            sensory_fit = _bounded_score(item["metadata"].get("sensory_score", item["confidence"]))
            context_fit = _bounded_score(
                item["metadata"].get("context_score", _similarity(item["content"], request.context))
            )
            lexical_fit = _bounded_score(item["metadata"].get("lexical_score", item["confidence"]))
            item["confidence"] = _bounded_score(
                0.4 * sensory_fit + 0.35 * lexical_fit + 0.25 * context_fit
            )
            item["metadata"]["activation_components"] = {
                "sensory_fit": sensory_fit,
                "lexical_fit": lexical_fit,
                "context_fit": context_fit,
            }
        state.lexical_candidates = self._merge_candidates(state.lexical_candidates, lexical)
        self._record(trace, 2, {
            "candidate_count": len(state.lexical_candidates),
            "alternatives_preserved": len(state.lexical_candidates) > 1,
            "context_bias_is_not_sensory_override": True,
        })

        # Step 3: update syntactic hypotheses independently from semantic plausibility.
        syntactic = self._stage_candidates(request, "syntactic")
        for item in syntactic:
            item["metadata"]["syntax_evidence_kept_separate"] = True
        state.syntactic_hypotheses = self._merge_candidates(state.syntactic_hypotheses, syntactic)
        self._record(trace, 3, {
            "hypothesis_count": len(state.syntactic_hypotheses),
            "unresolved_attachments": sum(
                bool(item.get("metadata", {}).get("unresolved", False))
                for item in state.syntactic_hypotheses
            ),
        })

        # Step 4: retrieve only senses actually supplied by the lexical/semantic stage.
        semantic = self._stage_candidates(request, "semantic")
        for item in semantic:
            lexical_support = max(
                (_similarity(item["content"], candidate["content"]) for candidate in state.lexical_candidates),
                default=0.0,
            )
            syntax_support = max(
                (_bounded_score(candidate["confidence"]) for candidate in state.syntactic_hypotheses),
                default=0.5,
            )
            item["metadata"]["lexical_support"] = lexical_support
            item["metadata"]["syntax_support"] = syntax_support
            item["confidence"] = _bounded_score(
                0.5 * item["confidence"] + 0.25 * lexical_support + 0.25 * syntax_support
            )
        state.semantic_candidates = self._merge_candidates(state.semantic_candidates, semantic)
        self._record(trace, 4, {
            "semantic_candidate_count": len(state.semantic_candidates),
            "meaning_sources": sorted({item["source"] for item in state.semantic_candidates}),
        })

        # Step 5: compose explicit proposition records; do not invent missing arguments.
        proposition_values = _stage_value(stage, ("propositions", "literal_propositions", "semantic_composition"))
        proposition_candidates = _extract_candidates(
            proposition_values, "proposition", ("propositions", "candidates", "alternatives", "events")
        ) if proposition_values is not None else []
        if not proposition_candidates and state.semantic_candidates:
            for item in state.semantic_candidates:
                content = item["content"]
                if isinstance(content, Mapping) and any(key in content for key in ("predicate", "semantic_roles", "arguments")):
                    candidate = deepcopy(item)
                    candidate["stage"] = "proposition"
                    proposition_candidates.append(candidate)
        state.propositions = self._merge_candidates(state.propositions, proposition_candidates)
        self._record(trace, 5, {
            "proposition_count": len(state.propositions),
            "open_roles_preserved": sum(
                bool(item.get("metadata", {}).get("unresolved_roles", item.get("metadata", {}).get("open_slots", [])))
                for item in state.propositions
            ),
        })

        # Step 6: merge discourse/situation evidence and current propositions.
        context_update = dict(request.context)
        discourse_input = _stage_value(stage, ("discourse", "contextual", "situation_model", "discourse_model"))
        if isinstance(discourse_input, Mapping):
            nested = discourse_input.get("enriched_interpretation", discourse_input)
            if isinstance(nested, Mapping):
                context_update.update({key: deepcopy(value) for key, value in nested.items()
                                       if key not in context_update})
        memory_input = _stage_value(stage, ("memory", "retrieved_memories", "memory_cues"))
        if memory_input is not None:
            context_update["memory_cues"] = deepcopy(_sequence(memory_input))
        state.discourse_model = self._update_situation_model(context_update, state.propositions)
        state.speaker_model.update({
            key: deepcopy(value) for key, value in request.speaker_model.items()
            if key not in state.speaker_model or state.speaker_model[key] in (None, "", [], {})
        })
        self._record(trace, 6, {
            "context_fields": sorted(state.discourse_model.keys()),
            "proposition_links": len(state.discourse_model.get("propositions", [])),
            "context_conflicts_preserved": len(state.discourse_model.get("context_conflicts", [])),
        })

        # Step 7: retain explicit predictions; only derive a transparent expectation if supplied.
        prediction_source = dict(request.predictions)
        if not prediction_source:
            value = _stage_value(stage, ("predictions", "expectations", "next_input_predictions"))
            if isinstance(value, Mapping):
                prediction_source = dict(value)
        if prediction_source:
            state.predictions.update(deepcopy(prediction_source))
        self._record(trace, 7, {
            "prediction_channels": sorted(state.predictions.keys()),
            "predictions_are_graded_caller_or_model_outputs": True,
        })

        # Step 8: compute independent errors for channels with paired evidence.
        state.prediction_errors = self._prediction_errors(state.predictions, request.observed_evidence)
        self._record(trace, 8, {
            "prediction_errors": deepcopy(state.prediction_errors),
            "error_channels_kept_separate": True,
        })

        # Step 9: allocate attention to explicit unresolved items and mismatch signals.
        incoming_unresolved = list(request.unresolved)
        state.unresolved = incoming_unresolved if incoming_unresolved else state.unresolved
        state.attention_allocation = self._prioritize(
            state.unresolved, state.prediction_errors, request.task
        )
        self._record(trace, 9, {
            "priority_order": [item["issue_id"] for item in state.attention_allocation],
            "allocated_attention_total": sum(item["allocated_attention"] for item in state.attention_allocation),
            "control_budget": max(0.0, float(request.task.get("control_budget", 1.0))),
        })

        # Step 10: trigger reanalysis only with supplied alternatives and adequate evidence.
        revision_reason = "typed_prediction_error_or_unresolved_high_priority_issue"
        revisions = self._apply_revisions(
            request.revision_candidates,
            state.prediction_errors,
            revision_reason,
        ) if request.revision_candidates else []
        self._record(trace, 10, {
            "revision_attempts": len(request.revision_candidates),
            "revision_outcomes": revisions,
            "revision_history_count": len(state.revision_log),
        })

        # Step 11: engage social/pragmatic inference only when task or mismatch demands it.
        inferences = self._pragmatic_inferences(request, state.prediction_errors)
        self._record(trace, 11, {
            "social_inference_engaged": bool(request.task.get("pragmatic_inference_required", False)
                or _bounded_score(request.task.get("social_inference_demand", 0.0)) >= 0.5
                or state.prediction_errors.get("pragmatic", 0.0) >= 0.35),
            "accepted_inference_count": len(inferences),
            "literal_inferred_separation": True,
        })

        # Step 12: refresh active propositions while preserving pointers to less-active items.
        supplied_wm = _stage_value(stage, ("working_memory", "working_memory_state", "active_items"))
        state.working_memory = self._update_working_memory(
            state.unresolved,
            request.task,
            state.prediction_errors,
            _sequence(supplied_wm),
        )
        self._record(trace, 12, {
            "active_item_count": len(state.working_memory),
            "capacity": int(request.task.get("working_memory_capacity", self.working_memory_capacity)),
            "deprioritized_ids": list(state.deprioritized_ids),
        })

        # Step 13: estimate adequacy and keep the process open when evidence is insufficient.
        quality = self._quality(request.task, state.prediction_errors)
        self._record(trace, 13, quality)

        # Step 14: finalize a coherent but revisable proposition when the task threshold is met.
        finalized = self._finalize(quality)
        self._record(trace, 14, {
            "finalized": finalized is not None,
            "selected_candidate_id": (
                finalized["selected_literal_proposition"]["candidate_id"] if finalized else None
            ),
            "unresolved_count": len(state.unresolved),
        })

        # Step 15: prepare parallel downstream routes without executing external side effects.
        routing = self._route(request, finalized)
        self._record(trace, 15, routing)

        # Step 16: incorporate later evidence through explicit, auditable revision candidates.
        later_revisions = request.later_evidence.get("revision_candidates", ())
        later_updates = self._apply_revisions(
            later_revisions,
            state.prediction_errors,
            "later_discourse_or_context_evidence",
        ) if isinstance(later_revisions, Sequence) and not isinstance(later_revisions, (str, bytes)) else []
        if request.later_evidence.get("context"):
            state.discourse_model = self._update_situation_model(
                _mapping(request.later_evidence["context"], "later_evidence.context"),
                state.propositions,
            )
        if later_updates:
            quality = self._quality(request.task, state.prediction_errors)
            finalized = self._finalize(quality)
            routing = self._route(request, finalized)
        self._record(trace, 16, {
            "later_evidence_received": bool(request.later_evidence),
            "revision_outcomes": later_updates,
            "revision_history_count": len(state.revision_log),
            "interpretation_rechecked": bool(later_updates),
        })

        # Step 17: construct memory routing data; Point 7 remains responsible for storage policy.
        if routing.get("memory_handoff") is not None and finalized is not None:
            selected_content = finalized["selected_literal_proposition"]["content"]
            routing["memory_handoff"]["proposition"] = (
                deepcopy(dict(selected_content))
                if isinstance(selected_content, Mapping)
                else {"content": deepcopy(selected_content)}
            )
        self._record(trace, 17, {
            "episodic_handoff_prepared": "episodic_memory" in routing["routes"],
            "semantic_handoff_prepared": "semantic_memory_candidate" in routing["routes"],
            "durable_memory_write_performed": False,
            "target": "Point7MemoryDecisionProcessor",
        })

        # Step 18: monitor feedback and make bounded strategy adjustments.
        feedback = dict(request.feedback)
        observed_outcome = feedback.get("observed_outcome")
        predicted_outcome = feedback.get("predicted_outcome")
        prediction_error = (
            float(observed_outcome) - float(predicted_outcome)
            if isinstance(observed_outcome, (int, float)) and isinstance(predicted_outcome, (int, float))
            else None
        )
        feedback_record = {
            **feedback,
            "prediction_error": prediction_error,
            "time_index": state.time_index,
            "error_channels": deepcopy(state.prediction_errors),
        }
        if feedback:
            state.feedback_history.append(feedback_record)
            counts = dict(state.strategy.get("error_counts", {}))
            for channel, error in state.prediction_errors.items():
                if error >= 0.5:
                    counts[channel] = int(counts.get(channel, 0)) + 1
            state.strategy["error_counts"] = counts
            recurrent_errors = sum(counts.values())
            state.strategy["attention_gain"] = min(2.0, 1.0 + 0.05 * recurrent_errors)
            if prediction_error is not None and abs(prediction_error) > 0.25:
                state.strategy["revision_margin"] = min(0.25, self.revision_margin + 0.02)
        self._record(trace, 18, {
            "feedback_received": bool(feedback),
            "prediction_error": prediction_error,
            "strategy": deepcopy(state.strategy),
        })

        biological = self.lif.run(len(POINT8_STEP_SPECS)) if self.lif else {}
        biological_trace = biological.get("trace", []) if biological else []
        diagnostics = {
            "valid": len(trace) == len(POINT8_STEP_SPECS) and all(
                item["operation_id"] == POINT8_STEP_SPECS[index]["operation_id"]
                for index, item in enumerate(trace)
            ),
            "step_count": len(trace),
            "operation_count": len(POINT8_STEP_SPECS),
            "time_index": state.time_index,
            "candidate_counts": {
                "sensory": len(state.sensory_candidates),
                "lexical": len(state.lexical_candidates),
                "syntactic": len(state.syntactic_hypotheses),
                "semantic": len(state.semantic_candidates),
                "propositions": len(state.propositions),
                "inferences": len(state.inferred_propositions),
            },
            "revision_count": len(state.revision_log),
            "working_memory_count": len(state.working_memory),
            "durable_memory_write_performed": False,
            "biological_reference_is_measured_human_data": False,
            "warnings": [
                "The integrator coordinates supplied stage outputs; it does not replace the specialized Points 1–7 processors.",
                "Candidate scores and Jaccard cue overlap are transparent engineering heuristics, not calibrated human probabilities.",
                "The optional LIF activity is a deterministic functional reference, not measured or anatomically fitted neural data.",
            ],
        }
        return Point8Result(
            sequence_id=request.sequence_id,
            time_index=state.time_index,
            state=state.to_dict(),
            functional_trace=trace,
            feedback_loops=self._feedback_loop_report(),
            interpretation_quality=quality,
            finalized_interpretation=finalized,
            routing=routing,
            diagnostics=diagnostics,
            biological_summary=biological,
            biological_trace=biological_trace,
        )

    def _feedback_loop_report(self) -> dict[str, dict[str, Any]]:
        """Describe the eight cross-stage feedback loops and whether this state engages them."""
        state = self.state
        return {
            "A_context_to_perception_and_lexicon": {
                "engaged": bool(state.discourse_model and state.lexical_candidates),
                "effect": "context fit contributes to lexical activation; sensory evidence remains separately represented",
            },
            "B_syntax_to_lexicon_and_back": {
                "engaged": bool(state.syntactic_hypotheses and state.lexical_candidates),
                "effect": "lexical alternatives and grammatical hypotheses remain jointly accessible",
            },
            "C_syntax_semantics_coupling": {
                "engaged": bool(state.syntactic_hypotheses and state.semantic_candidates),
                "effect": "semantic candidate confidence records a separate syntax-support component",
            },
            "D_proposition_context_exchange": {
                "engaged": bool(state.propositions and state.discourse_model),
                "effect": "propositions update the situation model and context remains available to later increments",
            },
            "E_prediction_error_reanalysis": {
                "engaged": bool(state.prediction_errors),
                "effect": "typed mismatches prioritize issues and can trigger evidence-supported revisions",
            },
            "F_working_memory_control": {
                "engaged": bool(state.working_memory or state.attention_allocation),
                "effect": "attention priorities and interference adjust active working-memory accessibility",
            },
            "G_pragmatic_to_literal_review": {
                "engaged": bool(state.inferred_propositions),
                "effect": "inferred intent and literal propositions remain in separate state fields",
            },
            "H_long_term_memory_to_current_comprehension": {
                "engaged": bool(state.discourse_model.get("memory_cues") or state.discourse_model.get("retrieved_memories")),
                "effect": "supplied memory cues remain contextual evidence and are not treated as guaranteed truth",
            },
        }


# A concise alias supports pipeline composition while keeping the class name descriptive.
Point8Processor = Point8CrossStageIntegrator
