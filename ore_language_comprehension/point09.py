"""Point 9: neural-system architecture and functional division of labour.

This module coordinates seven distinguishable system-level representations:
perceptual input, the core language network, conceptual/world knowledge,
mentalizing, multiple-demand control, working memory, and medial-temporal memory.
It preserves their roles while allowing task-dependent communication. Localization
and coupling values are model estimates, not anatomical diagnoses or direct neural
measurements. The optional LIF track is a population-level reference simulation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from ore_brain_networks.language_network import LanguageNetworkProcessor


POINT9_SYSTEM_NAMES = (
    "perceptual",
    "language",
    "conceptual_knowledge",
    "mentalizing",
    "multiple_demand",
    "working_memory",
    "medial_temporal",
)

POINT9_DEFAULT_CONNECTIONS: dict[tuple[str, str], float] = {
    ("perceptual", "language"): 0.65,
    ("language", "conceptual_knowledge"): 0.42,
    ("conceptual_knowledge", "language"): 0.28,
    ("language", "mentalizing"): 0.30,
    ("mentalizing", "language"): 0.22,
    ("language", "working_memory"): 0.38,
    ("working_memory", "language"): 0.32,
    ("multiple_demand", "language"): 0.22,
    ("multiple_demand", "working_memory"): 0.36,
    ("working_memory", "multiple_demand"): 0.24,
    ("conceptual_knowledge", "medial_temporal"): 0.32,
    ("medial_temporal", "conceptual_knowledge"): 0.24,
    ("working_memory", "medial_temporal"): 0.18,
    ("language", "multiple_demand"): 0.20,
    ("mentalizing", "conceptual_knowledge"): 0.20,
    ("conceptual_knowledge", "mentalizing"): 0.22,
}

POINT9_STEP_SPECS: tuple[dict[str, Any], ...] = (
    {
        "operation_id": "LC-P09-01",
        "name": "Initialize distinct system states",
        "objective": "Maintain separate perceptual, language, conceptual, mentalizing, executive, working-memory, and medial-temporal records with source, confidence, timing, dependency, and uncertainty.",
        "mathematical_operations": ("H(t)=[h_P,h_L,h_K,h_M,h_MD,h_WM,h_MTL]", "typed_state_graph=(V,E,system_type,provenance)", "preserve system-specific representations"),
        "biological_reference": "The architecture distinguishes large-scale functional systems without claiming that each is an isolated anatomical module or has a single exclusive operation.",
    },
    {
        "operation_id": "LC-P09-02",
        "name": "Estimate individual language-network localization",
        "objective": "Use participant-specific language-versus-matched-control localizer evidence when supplied; otherwise use an explicitly approximate literature prior.",
        "mathematical_operations": ("L_(s,r)=mean(BOLD_language)-mean(BOLD_control)", "d_prime=(mu_L-mu_C)/sqrt((var_L+var_C)/2)", "retain reliability and evidence_source"),
        "biological_reference": "Language-responsive regions are often left-dominant but vary across individuals; functional localizer contrasts are research estimates rather than fixed anatomical borders.",
    },
    {
        "operation_id": "LC-P09-03",
        "name": "Separate sensory-form evidence from linguistic computation",
        "objective": "Pass acoustic or orthographic evidence to language processing while keeping perceptual uncertainty independently accessible.",
        "mathematical_operations": ("p(t)=feature_population(sensory_input)", "u_L(t)=W_LP*p(t)", "retain modality and signal_quality"),
        "biological_reference": "Auditory and visual pathways provide structured evidence; sensory responses alone do not establish that a linguistic meaning has been constructed.",
    },
    {
        "operation_id": "LC-P09-04",
        "name": "Access words and constructions in the core language network",
        "objective": "Retrieve supplied lexical and grammatical candidates and update a distributed language-network state without premature selection.",
        "mathematical_operations": ("h_L(t+1)=f_L(W_LL*h_L+W_LP*p+W_LK*k_L+b_L)", "candidate_score=weighted(form,lexicon,structure,context)", "retain viable alternatives"),
        "biological_reference": "The lateral frontal and temporal language network supports language-selective access and combination; its components have distributed, task-dependent contributions.",
    },
    {
        "operation_id": "LC-P09-05",
        "name": "Build linguistic structure and literal compositional meaning",
        "objective": "Maintain syntactic hypotheses and a literal proposition with explicit role bindings and unresolved argument slots, separate from hidden-intention inference.",
        "mathematical_operations": ("P=R(a_1,...,a_k,□,...), k<n", "role_binding=constrained_graph_assignment", "preserve literal_content and open_slots"),
        "biological_reference": "Language-selective circuitry contributes to structure building and linguistic combination; the proposition object is a software representation, not a literal neural record.",
    },
    {
        "operation_id": "LC-P09-06",
        "name": "Retrieve relevant conceptual and world knowledge in parallel",
        "objective": "Activate only relevant object, event, property, physical, social, and causal knowledge while preserving its source and uncertainty.",
        "mathematical_operations": ("a_i=sim(q,m_i)", "r_i=exp(beta*a_i)/sum_j exp(beta*a_j)", "bounded_retrieval=TopK(r_i) subject to relevance"),
        "biological_reference": "Conceptual knowledge is distributed across temporal, parietal, and other association systems; it is not stored in one world-knowledge centre.",
    },
    {
        "operation_id": "LC-P09-07",
        "name": "Gate mentalizing according to social-inference demand",
        "objective": "Recruit belief, goal, perspective, or intention hypotheses when explicit task demands or meaningful discourse cues require them, not for every sentence.",
        "mathematical_operations": ("T_M=w_a*A+w_p*P+w_i*I+w_c*C", "g_M=sigmoid(k*(T_M-theta_M))", "mentalizing_route=task_override or g_M>=threshold"),
        "biological_reference": "mPFC and TPJ are candidate contributors to belief and intention reasoning; task and subregion matter, and the network is not mandatory for ordinary literal comprehension.",
    },
    {
        "operation_id": "LC-P09-08",
        "name": "Estimate multiple-demand and executive-control requirements",
        "objective": "Estimate control demand from unresolved ambiguity, memory load, conflict, time pressure, and task goals.",
        "mathematical_operations": ("D=alpha*U+beta*L_WM+gamma*C+delta*T", "g_MD=sigmoid(kappa*(D-theta_MD))", "control_gate in [0,1]"),
        "biological_reference": "Bilateral frontoparietal multiple-demand systems can support high-load selection and control; they are not the primary repository of grammar.",
    },
    {
        "operation_id": "LC-P09-09",
        "name": "Establish sparse inter-system communication pathways",
        "objective": "Create directed, weighted information dependencies only where the architecture requires exchange; keep functional edges distinct from anatomical claims.",
        "mathematical_operations": ("G=(V,E,W)", "W_ij=effective_message_weight", "sparse_routing avoids all-to-all coupling"),
        "biological_reference": "Information exchange can use recurrent cortical, subcortical, and white-matter pathways; an algorithmic graph edge does not mean a direct monosynaptic connection.",
    },
    {
        "operation_id": "LC-P09-10",
        "name": "Coordinate overlapping activity while preserving specialization",
        "objective": "Update system states from their own prior activity, external evidence, and gated messages from connected systems.",
        "mathematical_operations": ("h_i(t+1)=f_i(W_ii*h_i+sum_j g_ji*W_ij*h_j+B_i*x)", "synchronous_state_update from prior_t", "keep each system representation separately addressable"),
        "biological_reference": "Partially parallel recurrent interactions are plausible network-level abstractions; the update equation is not a measured equation for any named brain system.",
    },
    {
        "operation_id": "LC-P09-11",
        "name": "Preserve the functional distinction between language and mentalizing",
        "objective": "Keep linguistic structure and agent belief/goal representations separate even when both contribute to the same story interpretation.",
        "mathematical_operations": ("z_L=language_representation", "z_M=agent_belief_goal_representation", "combine_outputs without merging source labels"),
        "biological_reference": "Language and theory-of-mind networks can coordinate while remaining functionally distinguishable; co-activation does not imply identical computation.",
    },
    {
        "operation_id": "LC-P09-12",
        "name": "Recruit multiple-demand resources only when warranted",
        "objective": "Use graded control for competing parses, open dependencies, complex questions, or heavy maintenance, and keep basic composition separate from executive support.",
        "mathematical_operations": ("g_MD=sigmoid(k*(D-theta))", "attention_bias=g_MD*discriminative_cue", "control_gain=baseline+g_MD*delta_gain"),
        "biological_reference": "Bilateral dorsolateral prefrontal and intraparietal systems can assist difficult tasks; their engagement is conditional, not a requirement for every sentence.",
    },
    {
        "operation_id": "LC-P09-13",
        "name": "Estimate coordination without equating synchrony with shared function",
        "objective": "Estimate statistical association from supplied system time series when enough observations exist and report missing or non-estimable pairs honestly.",
        "mathematical_operations": ("FC_ij=corr(x_i(t),x_j(t))", "coherence or partial_correlation may be alternatives", "correlation does not establish direction or anatomy"),
        "biological_reference": "Functional connectivity measures statistical dependence; it does not independently establish direct anatomical wiring, causal flow, or identical computations.",
    },
    {
        "operation_id": "LC-P09-14",
        "name": "Route typed prediction errors to the challenged subsystem",
        "objective": "Map sensory, lexical, syntactic, semantic, social, and memory mismatch to the relevant system and then notify dependent systems.",
        "mathematical_operations": ("e_k(t)=y_k(t)-y_hat_k(t)", "target_system=error_channel_map[k]", "propagate_revision_notice along dependent_edges"),
        "biological_reference": "Distinct errors can recruit different perceptual, language, conceptual, social, and control processes; they should not trigger indiscriminate whole-network recruitment.",
    },
    {
        "operation_id": "LC-P09-15",
        "name": "Integrate candidate outputs while preserving provenance",
        "objective": "Combine linguistic, conceptual, and mental-state evidence only when relevant, retaining literal statements, knowledge-based inferences, and inferred intentions as different sources.",
        "mathematical_operations": ("logScore(H)=lambda_L*logP(E_L|H)+lambda_K*logP(E_K|H)+lambda_M*logP(E_M|H)+logP(H)", "retain evidence_channels and provenance", "abstain when evidence is insufficient"),
        "biological_reference": "Distributed systems can contribute complementary constraints; this weighted evidence model is a computational hypothesis, not a directly observed neural calculation.",
    },
    {
        "operation_id": "LC-P09-16",
        "name": "Check sufficiency and stop unnecessary additional recruitment",
        "objective": "Assess confidence, unresolved conflict, and task-specific consequences before finalizing or returning for another system update.",
        "mathematical_operations": ("Stop=1 when C(H)>=theta_C and U(H)<=theta_U", "thresholds depend on task consequence", "reduce support_system_gates after sufficiency"),
        "biological_reference": "Comprehension monitoring and effort allocation are distributed and task-sensitive; a deterministic threshold is an engineering rule.",
    },
    {
        "operation_id": "LC-P09-17",
        "name": "Route the integrated representation to memory, working memory, or action",
        "objective": "Maintain the interpretation temporarily, prepare memory handoffs when retention is useful, and prepare response/action support only when the task requests it.",
        "mathematical_operations": ("routes subset {working_memory,episodic,semantic,answer,action}", "P(route|task,content,context)", "keep processing, storage, and action execution distinct"),
        "biological_reference": "Working-memory, medial-temporal memory, and response systems are different destinations; durable encoding and action are not automatic consequences of comprehension.",
    },
    {
        "operation_id": "LC-P09-18",
        "name": "Update system-demand estimates as the context changes",
        "objective": "Recalculate mentalizing and executive demand after new speakers, topics, contradictions, questions, or ambiguities arrive.",
        "mathematical_operations": ("d(t+1)=F(new_cues,unresolved,task)", "g_j(t+1)=sigmoid(w_j^T*d-theta_j)", "decrease unused recruitment"),
        "biological_reference": "Network recruitment changes with discourse and task; the same system pattern should not be imposed on literal, sarcastic, and high-load sentences.",
    },
    {
        "operation_id": "LC-P09-19",
        "name": "Adapt to individual-specific network variation",
        "objective": "Preserve participant-specific localizer scores, lateralization, and supplied profile information without inferring anatomy from demographics or group averages.",
        "mathematical_operations": ("theta_s,r=calibrate(localizer_s,r)", "W_s=personalize(W_prior,validated_individual_data)", "group_prior is not an individual diagnosis"),
        "biological_reference": "Language dominance, functional topography, development, bilingual experience, and plasticity vary; participant-level evidence is needed for individual claims.",
    },
    {
        "operation_id": "LC-P09-20",
        "name": "Return an auditable network-level interpretation",
        "objective": "Return literal meaning, contextual knowledge, mental-state hypotheses, system contributions, uncertainty, localization status, and downstream routes with evidence sources.",
        "mathematical_operations": ("audit=(literal,knowledge,beliefs,systems,uncertainty,provenance,routes)", "serialize all system states and message paths", "mark simulated and inferred fields"),
        "biological_reference": "The final record is an inspectable computational summary of a distributed hypothesis, not a scan or measurement of a person's brain.",
    },
)


def _plain(value: Any) -> Any:
    """Convert nested Python and NumPy values into detached JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    return repr(value)


def _mapping(value: Any, field_name: str = "value") -> dict[str, Any]:
    """Return a detached mapping from a mapping or a result object's to_dict method."""
    if isinstance(value, Mapping):
        return deepcopy(dict(value))
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return deepcopy(dict(converted))
    if value is None:
        return {}
    raise TypeError(f"{field_name} must be a mapping or expose to_dict()")


def _bounded(value: Any, default: float = 0.0) -> float:
    """Convert a number into a finite value between zero and one."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(result):
        return default
    return min(1.0, max(0.0, result))


def _tokens(value: Any) -> set[str]:
    """Extract case-folded lexical cues from nested text without semantic guessing."""
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
    """Return an inspectable Jaccard cue overlap, not a trained semantic similarity."""
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _sequence(value: Any) -> list[Any]:
    """Normalize supported scalar, mapping, and sequence values to a list."""
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
    """Normalize a candidate while preserving source, status, confidence, and metadata."""
    if isinstance(value, Mapping):
        raw = dict(value)
        content = raw.get("content", raw.get("text", raw.get("value", raw.get("interpretation", raw))))
        metadata = dict(raw.get("metadata", {})) if isinstance(raw.get("metadata", {}), Mapping) else {}
        for key in ("beliefs", "goals", "intentions", "perspective", "semantic_roles",
                    "predicate", "expected_category", "unresolved", "open_slots",
                    "evidence_channels", "explicitness"):
            if key in raw and key not in metadata:
                metadata[key] = raw[key]
        return {
            "candidate_id": str(raw.get("candidate_id", raw.get("id", f"{stage}-{index}"))),
            "stage": stage,
            "content": deepcopy(content),
            "source": str(raw.get("source", raw.get("provenance", stage))),
            "confidence": _bounded(raw.get("confidence", raw.get("score", 0.5)), 0.5),
            "status": str(raw.get("status", "provisional")),
            "metadata": metadata,
        }
    return {
        "candidate_id": f"{stage}-{index}",
        "stage": stage,
        "content": deepcopy(value),
        "source": stage,
        "confidence": 0.5,
        "status": "provisional",
        "metadata": {},
    }


def _stage_value(outputs: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    """Return the first stage result under a known alias."""
    for name in aliases:
        if name in outputs:
            return outputs[name]
    return None


def _stage_candidates(
    outputs: Mapping[str, Any],
    aliases: Sequence[str],
    candidate_keys: Sequence[str],
    stage: str,
) -> list[dict[str, Any]]:
    """Extract normalized candidates from one optional upstream result."""
    value = _stage_value(outputs, aliases)
    if value is None:
        return []
    mapped = _mapping(value, f"{stage}_output") if hasattr(value, "to_dict") else value
    selected = mapped
    if isinstance(mapped, Mapping):
        for key in candidate_keys:
            if key in mapped:
                selected = mapped[key]
                break
    return [_candidate(item, stage, index) for index, item in enumerate(_sequence(selected))]


def _numeric_features(value: Any, prefix: str = "") -> dict[str, float]:
    """Extract finite numeric leaves for a feature-hash language-network input."""
    features: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            features.update(_numeric_features(item, f"{prefix}{key}."))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            features.update(_numeric_features(item, f"{prefix}{index}."))
    elif isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            features[prefix.rstrip(".")] = number
    return features


def _localizer_summary(person_profile: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate subject-specific language selectivity from supplied regional localizer data."""
    localizer = person_profile.get("functional_localizer")
    if not isinstance(localizer, Mapping) or not localizer:
        return {
            "status": "approximate_prior_only",
            "source": "literature_based_group_prior",
            "region_scores": {},
            "note": "No participant-level localizer data were supplied; no individual anatomy is inferred.",
        }
    scores: dict[str, Any] = {}
    for region, evidence in localizer.items():
        if not isinstance(evidence, Mapping):
            continue
        language_values = evidence.get("language_values")
        control_values = evidence.get("control_values")
        if language_values is None or control_values is None:
            if "language_mean" in evidence and "control_mean" in evidence:
                language_values = [evidence["language_mean"]]
                control_values = [evidence["control_mean"]]
            else:
                continue
        language = np.asarray(language_values, dtype=np.float64).reshape(-1)
        control = np.asarray(control_values, dtype=np.float64).reshape(-1)
        if language.size == 0 or control.size == 0 or not np.all(np.isfinite(language)) or not np.all(np.isfinite(control)):
            raise ValueError(f"localizer values for {region!r} must be non-empty and finite")
        language_mean, control_mean = float(language.mean()), float(control.mean())
        language_var = float(language.var(ddof=1)) if language.size > 1 else 0.0
        control_var = float(control.var(ddof=1)) if control.size > 1 else 0.0
        pooled = math.sqrt(max(1e-12, 0.5 * (language_var + control_var)))
        contrast = language_mean - control_mean
        scores[str(region)] = {
            "language_mean": language_mean,
            "control_mean": control_mean,
            "contrast": contrast,
            "standardized_selectivity": contrast / pooled,
            "language_trial_count": int(language.size),
            "control_trial_count": int(control.size),
            "evidence_source": str(evidence.get("source", "participant_supplied_localizer")),
            "status": "candidate_language_responsive" if contrast > 0 else "not_selective_in_supplied_contrast",
        }
    if not scores:
        return {
            "status": "localizer_supplied_but_not_estimable",
            "source": "participant_supplied_localizer",
            "region_scores": {},
            "note": "Regional data lacked valid language/control contrasts.",
        }
    return {
        "status": "participant_specific_estimate",
        "source": "participant_supplied_localizer",
        "region_scores": scores,
        "note": "Contrast estimates are participant-specific measurements supplied by the caller, not fixed network boundaries.",
    }


def _sigmoid(value: float) -> float:
    """Compute a numerically stable logistic recruitment gate."""
    clipped = min(40.0, max(-40.0, float(value)))
    return 1.0 / (1.0 + math.exp(-clipped))


def _system_series_connectivity(
    series: Mapping[str, Any],
    minimum_observations: int = 3,
) -> dict[str, Any]:
    """Estimate pairwise correlations only from explicitly supplied system time series."""
    normalized: dict[str, np.ndarray] = {}
    for name, raw in series.items():
        if name not in POINT9_SYSTEM_NAMES:
            continue
        values = np.asarray(raw, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"time series for {name!r} must contain only finite values")
        normalized[name] = values
    if len(normalized) < 2:
        return {
            "status": "not_computed_insufficient_system_series",
            "system_order": list(normalized),
            "correlation": {},
            "n_observations": {},
        }
    lengths = {values.size for values in normalized.values()}
    if len(lengths) != 1:
        raise ValueError("all supplied system time series must have equal lengths")
    count = next(iter(lengths))
    if count < minimum_observations:
        return {
            "status": "not_computed_too_few_observations",
            "system_order": list(normalized),
            "correlation": {},
            "n_observations": {name: int(values.size) for name, values in normalized.items()},
        }
    names = list(normalized)
    matrix = np.stack([normalized[name] for name in names], axis=1)
    corr = np.corrcoef(matrix, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "status": "estimated_from_caller_supplied_time_series",
        "system_order": names,
        "correlation": {
            left: {right: float(corr[i, j]) for j, right in enumerate(names)}
            for i, left in enumerate(names)
        },
        "n_observations": {name: count for name in names},
        "interpretation": "statistical_dependence_only_not_directional_or_anatomical_connectivity",
    }


@dataclass(frozen=True, slots=True)
class Point9Request:
    """One network-coordination update with optional Points 1–8 outputs and individual profile."""

    sequence_id: str
    increment: Any = None
    stage_outputs: Mapping[str, Any] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)
    task: Mapping[str, Any] = field(default_factory=dict)
    person_profile: Mapping[str, Any] = field(default_factory=dict)
    predictions: Mapping[str, Any] = field(default_factory=dict)
    observed_evidence: Mapping[str, Any] = field(default_factory=dict)
    unresolved: Sequence[Mapping[str, Any]] = ()
    knowledge_candidates: Sequence[Mapping[str, Any]] = ()
    mental_state_candidates: Sequence[Mapping[str, Any]] = ()
    candidate_interpretations: Sequence[Mapping[str, Any]] = ()
    system_time_series: Mapping[str, Any] = field(default_factory=dict)
    communication_weights: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate independent inputs and snapshot nested mutable containers."""
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a non-empty string")
        for name in (
            "stage_outputs", "context", "task", "person_profile", "predictions",
            "observed_evidence", "system_time_series", "communication_weights",
        ):
            if not isinstance(getattr(self, name), Mapping):
                raise TypeError(f"{name} must be a mapping")
        for name in (
            "unresolved", "knowledge_candidates", "mental_state_candidates",
            "candidate_interpretations",
        ):
            value = getattr(self, name)
            if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
                raise TypeError(f"{name} must be a sequence")
            if not all(isinstance(item, Mapping) for item in value):
                raise TypeError(f"every {name} item must be a mapping")
        object.__setattr__(self, "stage_outputs", deepcopy(dict(self.stage_outputs)))
        object.__setattr__(self, "context", deepcopy(dict(self.context)))
        object.__setattr__(self, "task", deepcopy(dict(self.task)))
        object.__setattr__(self, "person_profile", deepcopy(dict(self.person_profile)))
        object.__setattr__(self, "predictions", deepcopy(dict(self.predictions)))
        object.__setattr__(self, "observed_evidence", deepcopy(dict(self.observed_evidence)))
        object.__setattr__(self, "system_time_series", deepcopy(dict(self.system_time_series)))
        object.__setattr__(self, "communication_weights", deepcopy(dict(self.communication_weights)))
        for name in (
            "unresolved", "knowledge_candidates", "mental_state_candidates",
            "candidate_interpretations",
        ):
            object.__setattr__(self, name, tuple(deepcopy(dict(item)) for item in getattr(self, name)))

    @classmethod
    def from_stage_results(
        cls,
        sequence_id: str,
        stage_results: Mapping[str, Any],
        **overrides: Any,
    ) -> "Point9Request":
        """Build a request from any subset of prior comprehension result objects."""
        if not isinstance(stage_results, Mapping):
            raise TypeError("stage_results must be a mapping")
        aliases = {
            "point1": "sensory", "point01": "sensory", "point2": "lexical",
            "point02": "lexical", "point3": "syntactic", "point03": "syntactic",
            "point4": "semantic", "point04": "semantic", "point5": "discourse",
            "point05": "discourse", "point6": "working_memory", "point06": "working_memory",
            "point7": "memory", "point07": "memory", "point8": "integrated",
            "point08": "integrated",
        }
        normalized = {}
        for name, value in stage_results.items():
            key = aliases.get(str(name).casefold(), str(name))
            normalized[key] = _mapping(value, f"{name}_result")
        return cls(sequence_id=sequence_id, stage_outputs=normalized, **overrides)


@dataclass(slots=True)
class Point9State:
    """Persistent, typed system activity and dependencies across successive increments."""

    time_index: int = 0
    systems: dict[str, dict[str, Any]] = field(default_factory=lambda: {
        name: {
            "activation": 0.0,
            "representation": {},
            "source": "not_yet_updated",
            "confidence": 0.0,
            "last_updated": 0,
            "status": "inactive",
        } for name in POINT9_SYSTEM_NAMES
    })
    localization: dict[str, Any] = field(default_factory=dict)
    communication_graph: dict[str, Any] = field(default_factory=dict)
    executive_demand: dict[str, Any] = field(default_factory=dict)
    mentalizing_demand: dict[str, Any] = field(default_factory=dict)
    prediction_errors: dict[str, float] = field(default_factory=dict)
    revision_notices: list[dict[str, Any]] = field(default_factory=list)
    activity_history: list[dict[str, float]] = field(default_factory=list)
    knowledge_candidates: list[dict[str, Any]] = field(default_factory=list)
    mental_state_candidates: list[dict[str, Any]] = field(default_factory=list)
    literal_propositions: list[dict[str, Any]] = field(default_factory=list)
    demand_history: list[dict[str, Any]] = field(default_factory=list)
    individual_profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a detached plain-data snapshot of the recurrent state."""
        return _plain(asdict(self))


@dataclass(slots=True)
class Point9Result:
    """Auditable output separating functional, individual-localizer, and LIF reference tracks."""

    sequence_id: str
    time_index: int
    state: dict[str, Any]
    functional_trace: list[dict[str, Any]]
    localization: dict[str, Any]
    communication_graph: dict[str, Any]
    functional_connectivity: dict[str, Any]
    interpretation: dict[str, Any]
    routing: dict[str, Any]
    diagnostics: dict[str, Any]
    biological_summary: dict[str, Any] = field(default_factory=dict)
    biological_trace: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the result while converting NumPy values to ordinary Python values."""
        return _plain(asdict(self))


class Point9LIFReference:
    """Seven-population current-based LIF reference for interacting functional-system classes."""

    def __init__(
        self,
        population_size: int = 12,
        steps_per_operation: int = 3,
        dt_ms: float = 1.0,
        tau_ms: float = 20.0,
        threshold_mv: float = -50.0,
        rest_mv: float = -70.0,
        reset_mv: float = -68.0,
        refractory_steps: int = 2,
    ) -> None:
        """Validate numerical parameters and allocate independent membrane states."""
        if population_size < 1 or steps_per_operation < 1 or dt_ms <= 0 or tau_ms <= 0:
            raise ValueError("population_size, steps_per_operation, dt_ms, and tau_ms must be positive")
        if not rest_mv <= reset_mv < threshold_mv or refractory_steps < 0:
            raise ValueError("require rest_mv <= reset_mv < threshold and non-negative refractory_steps")
        self.population_size = int(population_size)
        self.steps_per_operation = int(steps_per_operation)
        self.dt_ms = float(dt_ms)
        self.tau_ms = float(tau_ms)
        self.threshold_mv = float(threshold_mv)
        self.rest_mv = float(rest_mv)
        self.reset_mv = float(reset_mv)
        self.refractory_steps = int(refractory_steps)

    def run(
        self,
        system_drives: Mapping[str, float] | None = None,
        operation_count: int = 20,
    ) -> dict[str, Any]:
        """Run a deterministic population reference pass; outputs are not measured human activity."""
        if operation_count < 1:
            raise ValueError("operation_count must be positive")
        drives = dict(system_drives or {})
        unknown = set(drives) - set(POINT9_SYSTEM_NAMES)
        if unknown:
            raise KeyError(f"unknown system drives: {sorted(unknown)}")
        for value in drives.values():
            if not math.isfinite(float(value)):
                raise ValueError("system drives must be finite")
        names = list(POINT9_SYSTEM_NAMES)
        count = len(names)
        weights = np.zeros((count, count), dtype=np.float64)
        for (source, target), weight in POINT9_DEFAULT_CONNECTIONS.items():
            weights[names.index(target), names.index(source)] = weight
        voltages = np.full((count, self.population_size), self.rest_mv, dtype=np.float64)
        refractory = np.zeros((count, self.population_size), dtype=np.int32)
        spike_counts = np.zeros((count, self.population_size), dtype=np.int64)
        activity = np.zeros(count, dtype=np.float64)
        decay = math.exp(-self.dt_ms / self.tau_ms)
        trace: list[dict[str, Any]] = []
        for operation in range(operation_count):
            drive_vector = np.asarray([
                _bounded(drives.get(name, 0.0)) for name in names
            ], dtype=np.float64)
            if not np.any(drive_vector):
                drive_vector[operation % count] = 0.15
            operation_spikes = 0
            for _ in range(self.steps_per_operation):
                recurrent = weights @ activity
                global_inhibition = 0.08 * float(np.mean(activity))
                scalar = np.clip(0.12 + 0.70 * drive_vector + recurrent - global_inhibition, 0.0, 1.0)
                available = refractory == 0
                voltage_delta = (self.dt_ms / self.tau_ms) * (
                    self.rest_mv - voltages + 25.0 * scalar[:, None]
                )
                voltages = np.where(available, voltages + voltage_delta, voltages)
                fired = (voltages >= self.threshold_mv) & available
                operation_spikes += int(fired.sum())
                spike_counts += fired.astype(np.int64)
                voltages = np.where(fired, self.reset_mv, voltages)
                refractory = np.where(
                    fired, self.refractory_steps, np.maximum(0, refractory - 1)
                )
                rate = fired.mean(axis=1)
                activity = decay * activity + (1.0 - decay) * rate
            trace.append({
                "operation": operation + 1,
                "spikes": operation_spikes,
                "population_activity": {name: float(activity[i]) for i, name in enumerate(names)},
                "status": "simulated_functional_population_reference",
            })
        elapsed_seconds = operation_count * self.steps_per_operation * self.dt_ms / 1000.0
        return {
            "model": "seven_system_current_based_LIF_reference",
            "population_names": names,
            "mean_firing_rates_hz": {
                name: float(spike_counts[i].sum() / self.population_size / max(elapsed_seconds, 1e-9))
                for i, name in enumerate(names)
            },
            "final_population_activity": {
                name: float(activity[i]) for i, name in enumerate(names)
            },
            "operation_count": operation_count,
            "steps_per_operation": self.steps_per_operation,
            "biological_reference_is_measured_human_data": False,
            "scope": "Simplified functional population model; not a fitted anatomical connectome or participant measurement.",
            "trace": trace,
        }


class Point9NeuralArchitectureCoordinator:
    """Coordinate specialized language, conceptual, mentalizing, and control system states."""

    def __init__(
        self,
        *,
        candidate_capacity: int = 8,
        knowledge_capacity: int = 8,
        mentalizing_threshold: float = 0.55,
        executive_threshold: float = 0.45,
        interpretation_threshold: float = 0.55,
        recurrent_gain: float = 0.35,
        simulate_biology: bool = False,
        lif_population_size: int = 12,
        lif_steps_per_operation: int = 2,
    ) -> None:
        """Configure bounded retrieval, conditional support gates, recurrent coupling, and optional LIF reference."""
        if candidate_capacity < 1 or knowledge_capacity < 1:
            raise ValueError("candidate and knowledge capacities must be positive")
        if not 0.0 <= mentalizing_threshold <= 1.0 or not 0.0 <= executive_threshold <= 1.0:
            raise ValueError("support thresholds must lie in [0, 1]")
        if not 0.0 <= interpretation_threshold <= 1.0 or recurrent_gain < 0.0:
            raise ValueError("interpretation_threshold must lie in [0, 1] and recurrent_gain be non-negative")
        self.candidate_capacity = int(candidate_capacity)
        self.knowledge_capacity = int(knowledge_capacity)
        self.mentalizing_threshold = float(mentalizing_threshold)
        self.executive_threshold = float(executive_threshold)
        self.interpretation_threshold = float(interpretation_threshold)
        self.recurrent_gain = float(recurrent_gain)
        self.state = Point9State()
        self.language_network = LanguageNetworkProcessor()
        self.lif = (
            Point9LIFReference(lif_population_size, lif_steps_per_operation)
            if simulate_biology else None
        )

    def reset(self) -> None:
        """Start a new independent stream while preserving coordinator configuration."""
        self.state = Point9State()

    def _record(self, trace: list[dict[str, Any]], step: int, output: Any) -> None:
        """Append a source-aligned step with a detached intermediate result."""
        spec = POINT9_STEP_SPECS[step - 1]
        trace.append({
            "step": step,
            "operation_id": spec["operation_id"],
            "name": spec["name"],
            "output": _plain(deepcopy(output)),
        })

    def _localization(self, profile: Mapping[str, Any]) -> dict[str, Any]:
        """Update participant-specific region selectivity estimates where localizer data exist."""
        summary = _localizer_summary(profile)
        self.state.localization = deepcopy(summary)
        self.state.individual_profile = {
            key: deepcopy(value) for key, value in profile.items()
            if key != "functional_localizer"
        }
        return summary

    def _system_drive(self, stage: str, candidates: Sequence[Mapping[str, Any]]) -> float:
        """Estimate one bounded system drive from the confidence of supplied candidates."""
        if not candidates:
            return 0.0
        values = [_bounded(item.get("confidence", 0.5), 0.5) for item in candidates]
        return float(np.mean(values))

    def _update_system(
        self,
        name: str,
        activation: float,
        representation: Any,
        source: str,
        confidence: float,
    ) -> None:
        """Replace one typed system state while retaining its identity and time index."""
        self.state.systems[name] = {
            "activation": _bounded(activation),
            "representation": _plain(deepcopy(representation)),
            "source": str(source),
            "confidence": _bounded(confidence),
            "last_updated": self.state.time_index,
            "status": "active" if activation > 0.0 else "available_but_not_recruited",
        }

    def _communication_graph(self, overrides: Mapping[str, float]) -> dict[str, Any]:
        """Build the default sparse functional graph and apply validated edge-weight overrides."""
        weights = dict(POINT9_DEFAULT_CONNECTIONS)
        edge_report = []
        for raw_key, raw_weight in overrides.items():
            if "->" not in str(raw_key):
                raise ValueError(f"communication edge {raw_key!r} must use 'source->target'")
            source, target = [part.strip() for part in str(raw_key).split("->", 1)]
            if source not in POINT9_SYSTEM_NAMES or target not in POINT9_SYSTEM_NAMES:
                raise ValueError(f"unknown communication systems in edge {raw_key!r}")
            weight = float(raw_weight)
            if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
                raise ValueError("communication weights must be finite and lie in [0, 1]")
            weights[(source, target)] = weight
        for (source, target), weight in sorted(weights.items()):
            edge_report.append({
                "source": source,
                "target": target,
                "weight": weight,
                "edge_type": "functional_information_dependency_not_anatomical_wiring",
            })
        return {
            "systems": list(POINT9_SYSTEM_NAMES),
            "edges": edge_report,
            "sparsity": 1.0 - len(edge_report) / float(len(POINT9_SYSTEM_NAMES) ** 2),
        }

    def _demand_estimates(
        self,
        request: Point9Request,
        syntactic: Sequence[Mapping[str, Any]],
        predictions: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Estimate social-inference and executive demand from explicit cues and task structure."""
        task = request.task
        context = request.context
        proposition = _stage_value(request.stage_outputs, ("propositions", "literal_propositions", "semantic_composition"))
        belief_terms = {"believe", "believed", "think", "thought", "know", "knew", "intend", "intended",
                        "want", "wanted", "pretend", "deceive", "deceived", "sarcastic", "sarcasm",
                        "irony", "ironic", "request", "perspective", "belief", "intention"}
        text_tokens = _tokens({
            "context": context,
            "proposition": proposition,
            "task": {key: value for key, value in task.items() if key not in {"options", "constraints"}},
        })
        cue_tokens = set(task.get("mentalizing_cues", ())) if isinstance(task.get("mentalizing_cues", ()), (list, tuple, set)) else set()
        cue_tokens = {str(item).casefold() for item in cue_tokens}
        matched = sorted((text_tokens & belief_terms) | cue_tokens)
        explicit_task = bool(task.get("mentalizing_required", False) or task.get("asks_belief_or_intention", False))
        agent_information = _bounded(task.get("agent_reference_strength", 0.0))
        perspective = _bounded(task.get("perspective_dependence", 0.0))
        intention = _bounded(task.get("intention_relevance", 0.0))
        literal_gap = _bounded(task.get("literal_context_mismatch", 0.0))
        cue_strength = min(1.0, len(matched) / 3.0)
        mental_score = min(1.0, 0.25 * cue_strength + 0.20 * agent_information
                           + 0.25 * perspective + 0.20 * intention + 0.10 * literal_gap)
        mental_threshold = _bounded(request.person_profile.get("mentalizing_threshold", self.mentalizing_threshold), self.mentalizing_threshold)
        executive_threshold = _bounded(request.person_profile.get("executive_threshold", self.executive_threshold), self.executive_threshold)
        mental_gate = 1.0 if explicit_task else _sigmoid(8.0 * (mental_score - mental_threshold))
        mental = {
            "score": mental_score,
            "gate": mental_gate,
            "engaged": explicit_task or mental_gate >= 0.5,
            "explicit_task_override": explicit_task,
            "matched_cues": matched,
            "threshold_used": mental_threshold,
            "status": "conditionally_recruited" if explicit_task or mental_gate >= 0.5 else "not_forced_for_literal_task",
        }

        unresolved = list(request.unresolved)
        ambiguous = max(0.0, min(1.0, (len(syntactic) - 1) / max(1, self.candidate_capacity - 1)))
        open_dependencies = min(1.0, len(unresolved) / max(1, self.candidate_capacity))
        memory_load = _bounded(task.get("working_memory_load", (
            len(syntactic) + len(unresolved) + len(_sequence(proposition))
        ) / float(self.candidate_capacity * 2)))
        conflict = max(
            [_bounded(item.get("conflict", 0.0)) for item in unresolved]
            + [_bounded(item.get("conflict", 0.0)) for item in syntactic]
            + [_bounded(v) for v in predictions.values() if isinstance(v, (int, float))]
            + [0.0]
        )
        time_pressure = _bounded(task.get("time_pressure", 0.0))
        task_control = _bounded(task.get("executive_task_demand", 0.0))
        demand = min(1.0, 0.25 * ambiguous + 0.20 * open_dependencies
                     + 0.25 * memory_load + 0.15 * conflict
                     + 0.10 * time_pressure + 0.05 * task_control)
        executive_gate = _sigmoid(8.0 * (demand - executive_threshold))
        executive = {
            "demand": demand,
            "threshold_used": executive_threshold,
            "gate": executive_gate,
            "engaged": executive_gate >= 0.5,
            "components": {
                "ambiguity": ambiguous,
                "open_dependencies": open_dependencies,
                "working_memory_load": memory_load,
                "candidate_conflict": conflict,
                "time_pressure": time_pressure,
                "task_control": task_control,
            },
            "status": "conditionally_recruited" if executive_gate >= 0.5 else "low_additional_control_demand",
        }
        return mental, executive

    def _extract_outputs(self, outputs: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Extract separate sensory, lexical, syntactic, proposition, and memory candidates."""
        return {
            "perceptual": _stage_candidates(
                outputs, ("sensory", "perceptual", "point1", "form_hypotheses"),
                ("candidates", "form_hypotheses", "alternatives", "items"), "perceptual"
            ),
            "lexical": _stage_candidates(
                outputs, ("lexical", "lexical_candidates", "point2"),
                ("lexical_candidates", "candidates", "alternatives", "items"), "lexical"
            ),
            "syntactic": _stage_candidates(
                outputs, ("syntactic", "syntax", "parse_hypotheses", "point3"),
                ("parse_hypotheses", "syntactic_hypotheses", "candidates", "alternatives", "structures"), "syntactic"
            ),
            "propositions": _stage_candidates(
                outputs, ("propositions", "literal_propositions", "semantic_composition", "point4"),
                ("propositions", "literal_propositions", "events", "candidates", "alternatives"), "proposition"
            ),
            "discourse": _stage_candidates(
                outputs, ("discourse", "contextual", "point5", "integrated"),
                ("enriched_interpretation", "situation_model", "discourse_model", "candidates"), "discourse"
            ),
            "working_memory": _stage_candidates(
                outputs, ("working_memory", "working_memory_state", "point6"),
                ("active_items", "propositions", "buffers", "items"), "working_memory"
            ),
            "memory": _stage_candidates(
                outputs, ("memory", "retrieved_memories", "point7"),
                ("retrieved_memories", "episodes", "semantic_records", "candidates"), "memory"
            ),
        }

    def _retrieve_knowledge(
        self,
        request: Point9Request,
        proposition_candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Rank caller-supplied conceptual knowledge by cue relevance, source, and confidence."""
        proposition_text = [item.get("content") for item in proposition_candidates]
        context = request.context
        ranked = []
        for index, raw in enumerate(request.knowledge_candidates):
            content = raw.get("content", raw.get("knowledge", raw.get("fact", raw)))
            proposition_fit = max((_similarity(content, item) for item in proposition_text), default=0.0)
            context_fit = _similarity(content, context)
            caller_relevance = _bounded(raw.get("relevance", raw.get("activation", 0.5)))
            confidence = _bounded(raw.get("confidence", 0.5))
            score = 0.40 * proposition_fit + 0.25 * context_fit + 0.20 * caller_relevance + 0.15 * confidence
            ranked.append({
                "candidate_id": str(raw.get("candidate_id", raw.get("id", f"knowledge-{index}"))),
                "content": deepcopy(content),
                "source": str(raw.get("source", raw.get("provenance", "caller_supplied_conceptual_knowledge"))),
                "confidence": confidence,
                "relevance_score": score,
                "status": str(raw.get("status", "candidate")),
                "evidence": {
                    "proposition_fit": proposition_fit,
                    "context_fit": context_fit,
                    "caller_relevance": caller_relevance,
                },
            })
        ranked.sort(key=lambda item: item["relevance_score"], reverse=True)
        return ranked[: self.knowledge_capacity]

    def _mentalizing_candidates(
        self,
        request: Point9Request,
        demand: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Score explicitly supplied agent belief, goal, perspective, and intention candidates."""
        if not demand.get("engaged"):
            return []
        ranked = []
        for index, raw in enumerate(request.mental_state_candidates):
            belief_fit = _bounded(raw.get("belief_fit", raw.get("confidence", 0.5)))
            context_fit = _bounded(raw.get("context_fit", _similarity(raw.get("content", raw), request.context)))
            utterance_fit = _bounded(raw.get("utterance_fit", _similarity(raw.get("content", raw), request.stage_outputs)))
            contradiction = _bounded(raw.get("contradiction", 0.0))
            score = 0.35 * belief_fit + 0.30 * context_fit + 0.25 * utterance_fit - 0.30 * contradiction
            ranked.append({
                "candidate_id": str(raw.get("candidate_id", raw.get("id", f"mental-state-{index}"))),
                "agent": raw.get("agent"),
                "content": deepcopy(raw.get("content", raw.get("belief", raw))),
                "source": str(raw.get("source", "agent_belief_or_intention_inference")),
                "confidence": _bounded(score),
                "status": "candidate_supported" if score >= 0.55 else "unresolved",
                "evidence": {
                    "belief_fit": belief_fit,
                    "context_fit": context_fit,
                    "utterance_fit": utterance_fit,
                    "contradiction": contradiction,
                },
                "literal_content_remains_separate": True,
            })
        ranked.sort(key=lambda item: item["confidence"], reverse=True)
        return ranked[: self.candidate_capacity]

    def _update_system_states(
        self,
        drives: Mapping[str, float],
        representations: Mapping[str, Any],
        sources: Mapping[str, str],
        confidence: Mapping[str, float],
        mentalizing_gate: float,
        executive_gate: float,
        communication_graph: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Apply one recurrent graph update using prior-time activity and gated sparse messages."""
        weights = {
            (str(edge["source"]), str(edge["target"])): float(edge["weight"])
            for edge in communication_graph.get("edges", [])
        }
        previous = {
            name: _bounded(record.get("activation", 0.0))
            for name, record in self.state.systems.items()
        }
        updated: dict[str, float] = {}
        messages: dict[str, list[dict[str, Any]]] = {name: [] for name in POINT9_SYSTEM_NAMES}
        gate_by_system = {
            "mentalizing": mentalizing_gate,
            "multiple_demand": executive_gate,
            "medial_temporal": _bounded(drives.get("medial_temporal", 0.0)),
        }
        for (source, target), weight in weights.items():
            gate = gate_by_system.get(source, 1.0)
            message = weight * gate * previous[source]
            messages[target].append({"source": source, "weighted_activation": message, "weight": weight, "gate": gate})
        for name in POINT9_SYSTEM_NAMES:
            incoming = sum(item["weighted_activation"] for item in messages[name])
            self_recurrence = 0.12 * previous[name]
            drive = _bounded(drives.get(name, 0.0))
            raw = drive + self_recurrence + self.recurrent_gain * incoming
            activation = math.tanh(raw)
            updated[name] = _bounded(activation)
        for name in POINT9_SYSTEM_NAMES:
            self._update_system(
                name,
                updated[name],
                representations.get(name, {}),
                sources.get(name, "network_state_update"),
                confidence.get(name, updated[name]),
            )
        self.state.activity_history.append(dict(updated))
        return {
            "prior_activity": previous,
            "new_activity": updated,
            "incoming_messages": messages,
            "update_rule": "tanh(drive + 0.12*prior + recurrent_gain*sum(gated_weighted_messages))",
        }

    def _prediction_errors(self, predictions: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, float]:
        """Compare matched predicted and observed channels without inventing missing observations."""
        errors: dict[str, float] = {}
        for channel, expected in predictions.items():
            if channel not in observed:
                continue
            actual = observed[channel]
            if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
                errors[channel] = min(1.0, abs(float(actual) - float(expected)))
            else:
                errors[channel] = 0.0 if expected == actual else 1.0
        return errors

    def _route_errors(self, errors: Mapping[str, float]) -> list[dict[str, Any]]:
        """Map each observed mismatch type to its primary representation and dependent systems."""
        targets = {
            "sensory": ("perceptual", ("language",)),
            "perceptual": ("perceptual", ("language",)),
            "lexical": ("language", ("working_memory",)),
            "syntactic": ("language", ("working_memory",)),
            "semantic": ("conceptual_knowledge", ("language", "working_memory")),
            "referential": ("conceptual_knowledge", ("mentalizing", "language")),
            "pragmatic": ("mentalizing", ("language", "conceptual_knowledge")),
            "social": ("mentalizing", ("conceptual_knowledge", "language")),
            "memory": ("medial_temporal", ("conceptual_knowledge",)),
        }
        notices = []
        for channel, error in errors.items():
            primary, dependent = targets.get(channel, ("language", ("working_memory",)))
            notice = {
                "channel": channel,
                "mismatch": error,
                "primary_system": primary,
                "dependent_systems_to_notify": list(dependent),
                "status": "revision_notice" if error > 0.0 else "matched_prediction",
            }
            notices.append(notice)
        self.state.revision_notices.extend(deepcopy(notices))
        return notices

    def _integrate_interpretation(
        self,
        request: Point9Request,
        propositions: Sequence[Mapping[str, Any]],
        knowledge: Sequence[Mapping[str, Any]],
        mental_states: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Rank supplied interpretation candidates and preserve separate evidence provenance."""
        candidates: list[dict[str, Any]] = []
        supplied = list(request.candidate_interpretations)
        if not supplied:
            supplied = [
                {
                    "candidate_id": item["candidate_id"],
                    "content": item["content"],
                    "source": item["source"],
                    "confidence": item["confidence"],
                    "evidence_channels": ["language"],
                } for item in propositions
            ]
        for index, raw in enumerate(supplied):
            channels = set(raw.get("evidence_channels", ("language",)))
            language_evidence = _bounded(raw.get("language_evidence", raw.get("confidence", 0.5)))
            conceptual_evidence = _bounded(raw.get("conceptual_evidence", max(
                (item["relevance_score"] for item in knowledge), default=0.0
            ))) if ("conceptual" in channels or knowledge) else 0.0
            mentalizing_evidence = _bounded(raw.get("mentalizing_evidence", max(
                (item["confidence"] for item in mental_states), default=0.0
            ))) if ("mentalizing" in channels or mental_states) else 0.0
            task = request.task
            weights = {
                "language": _bounded(task.get("language_evidence_weight", 1.0), 1.0),
                "conceptual": _bounded(task.get("conceptual_evidence_weight", 0.35)),
                "mentalizing": _bounded(task.get("mentalizing_evidence_weight", 0.25)),
            }
            total = (
                weights["language"] * language_evidence
                + weights["conceptual"] * conceptual_evidence
                + weights["mentalizing"] * mentalizing_evidence
            )
            denominator = sum(weights.values()) or 1.0
            candidates.append({
                "candidate_id": str(raw.get("candidate_id", raw.get("id", f"interpretation-{index}"))),
                "content": deepcopy(raw.get("content", raw.get("interpretation", raw))),
                "source": str(raw.get("source", raw.get("provenance", "caller_supplied_interpretation"))),
                "status": str(raw.get("status", "candidate")),
                "model_score": total / denominator,
                "evidence_channels": {
                    "language": language_evidence,
                    "conceptual_knowledge": conceptual_evidence,
                    "mentalizing": mentalizing_evidence,
                },
                "evidence_weights": weights,
                "literal_and_inferred_content_remain_distinguishable": True,
            })
        candidates.sort(key=lambda item: item["model_score"], reverse=True)
        selected = candidates[0] if candidates else None
        return {
            "literal_propositions": _plain(list(propositions)),
            "conceptual_knowledge": _plain(list(knowledge)),
            "mental_state_hypotheses": _plain(list(mental_states)),
            "candidate_interpretations": candidates,
            "selected_candidate": deepcopy(selected),
            "status": "candidate_ranked" if selected else "no_supported_interpretation_candidate",
            "provenance_policy": "literal, conceptual, and inferred mental-state evidence remain separately identified",
        }

    def _sufficiency(
        self,
        interpretation: Mapping[str, Any],
        executive: Mapping[str, Any],
        request: Point9Request,
    ) -> dict[str, Any]:
        """Decide whether the best supplied interpretation is adequate for the caller's task."""
        selected = interpretation.get("selected_candidate")
        confidence = _bounded(selected.get("model_score")) if isinstance(selected, Mapping) else 0.0
        unresolved_critical = [
            item for item in request.unresolved
            if bool(item.get("critical", False)) and _bounded(item.get("uncertainty", 0.5), 0.5) >= 0.5
        ]
        conflict = max(
            [_bounded(item.get("conflict", 0.0)) for item in request.unresolved] + [0.0]
        )
        threshold = _bounded(
            request.task.get(
                "interpretation_threshold",
                request.person_profile.get("interpretation_threshold", self.interpretation_threshold),
            ),
            self.interpretation_threshold,
        )
        adequate = bool(selected) and confidence >= threshold and conflict < 0.75 and not unresolved_critical
        return {
            "adequate": adequate,
            "confidence": confidence,
            "threshold": threshold,
            "remaining_conflict": conflict,
            "critical_unresolved": [item.get("issue_id", item.get("content")) for item in unresolved_critical],
            "status": "sufficient_for_current_task" if adequate else "continue_or_preserve_uncertainty",
        }

    def _routing(
        self,
        request: Point9Request,
        interpretation: Mapping[str, Any],
        sufficiency: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Prepare optional downstream routes without executing memory writes or external actions."""
        task = request.task
        routes = ["working_memory"] if interpretation.get("selected_candidate") else []
        if task.get("encode_episode", False):
            routes.append("episodic_memory_handoff")
        if task.get("generalizable_knowledge", False):
            routes.append("semantic_memory_candidate")
        if task.get("requires_answer", False):
            routes.append("question_answering")
        if task.get("requires_action_plan", False):
            routes.append("decision_support")
        selected = interpretation.get("selected_candidate")
        return {
            "routes": routes,
            "working_memory_payload": deepcopy(selected) if "working_memory" in routes else None,
            "memory_handoff": {
                "target": "Point7MemoryDecisionProcessor",
                "proposition": deepcopy(selected.get("content")) if isinstance(selected, Mapping) else {},
                "components": deepcopy(interpretation.get("literal_propositions", [])),
                "context": deepcopy(request.context),
                "task": {
                    "encode_episode": "episodic_memory_handoff" in routes,
                    "generalizable_knowledge": "semantic_memory_candidate" in routes,
                    "semantic_verified": False,
                },
                "durable_storage_performed": False,
            } if any(route in routes for route in ("episodic_memory_handoff", "semantic_memory_candidate")) else None,
            "ready_for_downstream_use": bool(sufficiency.get("adequate")),
            "action_execution_performed": False,
            "note": "Routes prepare downstream input; no durable write, answer, or physical action is executed here.",
        }

    def process(self, request: Point9Request) -> Point9Result:
        """Execute all 20 network-coordination operations and update persistent system state."""
        if not isinstance(request, Point9Request):
            raise TypeError("request must be a Point9Request")
        self.state.time_index += 1
        state = self.state
        outputs = request.stage_outputs
        trace: list[dict[str, Any]] = []
        candidates = self._extract_outputs(outputs)

        # Step 1: keep seven system classes and their records distinguishable.
        self._record(trace, 1, {
            "systems": list(POINT9_SYSTEM_NAMES),
            "state_fields": ["representation", "activation", "source", "confidence", "last_updated", "status"],
            "time_index": state.time_index,
        })

        # Step 2: prefer supplied participant-level localizer evidence; do not fabricate it.
        localization = self._localization(request.person_profile)
        self._record(trace, 2, localization)

        # Step 3: pass sensory form evidence without converting it to meaning.
        perceptual = candidates["perceptual"]
        if request.increment is not None and not perceptual:
            perceptual = [_candidate(request.increment, "perceptual", 0)]
        self._update_system(
            "perceptual",
            self._system_drive("perceptual", perceptual),
            perceptual,
            "sensory_or_point1_evidence" if perceptual else "no_sensory_candidate_supplied",
            self._system_drive("perceptual", perceptual),
        )
        self._record(trace, 3, {
            "candidate_count": len(perceptual),
            "sensory_source_preserved": True,
            "meaning_claimed_from_sensory_only": False,
        })

        # Step 4: use ORE's existing IFG/MTG/ATL recurrent language-network component.
        lexical = candidates["lexical"]
        syntactic = candidates["syntactic"]
        lexical_candidate = str(lexical[0]["content"]) if lexical else None
        semantic_features = _numeric_features(_stage_value(outputs, ("semantic_features", "semantic", "semantics")))
        context_features = _numeric_features(request.context)
        core_network = self.language_network.process(
            lexical_candidate=lexical_candidate,
            semantic_features=semantic_features,
            context_features=context_features,
        )
        language_confidence = max(
            self._system_drive("language", lexical),
            self._system_drive("language", syntactic),
        )
        language_representation = {
            "lexical_candidates": lexical,
            "syntactic_hypotheses": syntactic,
            "regional_activity_summary": core_network["regional_activity_summary"],
            "region_order": core_network["region_order"],
            "language_selectivity_not_inferred_from_single_task": True,
        }
        self._update_system(
            "language", language_confidence, language_representation,
            "Point2_Point3_and_existing_ore_brain_networks_language_network",
            language_confidence,
        )
        self._record(trace, 4, {
            "lexical_candidate_count": len(lexical),
            "syntactic_hypothesis_count": len(syntactic),
            "reused_language_regions": list(core_network["region_order"]),
            "regional_activity_summary": core_network["regional_activity_summary"],
        })

        # Step 5: preserve the literal proposition and its open roles.
        propositions = candidates["propositions"]
        if not propositions:
            integrated = _stage_value(outputs, ("integrated", "point8"))
            if isinstance(integrated, Mapping):
                final = integrated.get("finalized_interpretation")
                if isinstance(final, Mapping):
                    propositions = _stage_candidates(
                        {"p": final}, ("p",),
                        ("literal_propositions", "selected_literal_proposition"), "proposition"
                    )
        state.literal_propositions = deepcopy(propositions)
        self._record(trace, 5, {
            "literal_proposition_count": len(propositions),
            "open_slots_preserved": sum(
                bool(item.get("metadata", {}).get("open_slots")
                     or item.get("metadata", {}).get("unresolved", False))
                for item in propositions
            ),
            "speaker_intention_inference_performed": False,
        })

        # Step 6: retrieve only caller-supplied knowledge candidates.
        knowledge = self._retrieve_knowledge(request, propositions)
        state.knowledge_candidates = deepcopy(knowledge)
        knowledge_activation = max(
            (item["relevance_score"] for item in knowledge), default=0.0
        )
        self._update_system(
            "conceptual_knowledge", knowledge_activation, knowledge,
            "caller_supplied_conceptual_knowledge",
            knowledge_activation,
        )
        self._record(trace, 6, {
            "retrieved_candidate_count": len(knowledge),
            "top_relevance": knowledge[0]["relevance_score"] if knowledge else 0.0,
            "unprovided_world_knowledge_invented": False,
        })

        # Step 7: activate mentalizing only when task or clear cues support it.
        mental_demand, executive_demand = self._demand_estimates(
            request, syntactic, request.predictions
        )
        mental_states = self._mentalizing_candidates(request, mental_demand)
        state.mental_state_candidates = deepcopy(mental_states)
        mental_activation = (
            max((item["confidence"] for item in mental_states), default=0.0)
            * _bounded(mental_demand["gate"])
        )
        self._update_system(
            "mentalizing", mental_activation, mental_states,
            "conditional_mPFC_TPJ_functional_reference",
            mental_activation,
        )
        state.mentalizing_demand = deepcopy(mental_demand)
        self._record(trace, 7, {
            **mental_demand,
            "candidate_count": len(mental_states),
            "beliefs_goals_and_intentions_remain_hypotheses": True,
        })

        # Step 8: estimate frontoparietal control requirements independently.
        state.executive_demand = deepcopy(executive_demand)
        self._record(trace, 8, executive_demand)

        # Step 9: construct a sparse functional communication graph.
        profile_weights = request.person_profile.get("communication_weights", {})
        combined_weights = dict(profile_weights) if isinstance(profile_weights, Mapping) else {}
        combined_weights.update(request.communication_weights)
        graph = self._communication_graph(combined_weights)
        state.communication_graph = deepcopy(graph)
        self._record(trace, 9, graph)

        # Step 10: update all system activities together from prior-time state.
        memory_task = any(request.task.get(key, False) for key in ("encode_episode", "generalizable_knowledge"))
        wm_candidates = candidates["working_memory"] + [
            _candidate(item, "unresolved", index) for index, item in enumerate(request.unresolved)
        ]
        drives = {
            "perceptual": self._system_drive("perceptual", perceptual),
            "language": language_confidence,
            "conceptual_knowledge": knowledge_activation,
            "mentalizing": mental_activation,
            "multiple_demand": _bounded(executive_demand["gate"]),
            "working_memory": max(self._system_drive("working_memory", wm_candidates),
                                  _bounded(request.task.get("working_memory_load", 0.0))),
            "medial_temporal": 0.8 if memory_task else self._system_drive("medial_temporal", candidates["memory"]),
        }
        representations = {
            "perceptual": perceptual,
            "language": language_representation,
            "conceptual_knowledge": knowledge,
            "mentalizing": mental_states,
            "multiple_demand": executive_demand,
            "working_memory": wm_candidates,
            "medial_temporal": candidates["memory"],
        }
        sources = {name: "step_10_recurrent_system_coordination" for name in POINT9_SYSTEM_NAMES}
        confidence = {name: drives[name] for name in POINT9_SYSTEM_NAMES}
        recurrent = self._update_system_states(
            drives, representations, sources, confidence,
            mental_demand["gate"], executive_demand["gate"], graph,
        )
        self._record(trace, 10, recurrent)

        # Step 11: report language and mentalizing state as different representational channels.
        self._record(trace, 11, {
            "language_state": deepcopy(state.systems["language"]),
            "mentalizing_state": deepcopy(state.systems["mentalizing"]),
            "representations_merged": False,
            "shared_final_use_is_not_shared_computation": True,
        })

        # Step 12: apply the multiple-demand gate without making it the language processor.
        md_active = bool(executive_demand["engaged"])
        self._update_system(
            "multiple_demand",
            drives["multiple_demand"],
            {**executive_demand, "control_function": "attention_selection_conflict_and_task_goal_support"},
            "conditional_multiple_demand_support",
            drives["multiple_demand"],
        )
        self._record(trace, 12, {
            "recruited": md_active,
            "gate": executive_demand["gate"],
            "core_grammar_assigned_to_MD": False,
            "executive_contribution": "conditional_control_not_primary_grammar_storage",
        })

        # Step 13: estimate connectivity only from caller-provided recorded time series.
        connectivity = _system_series_connectivity(request.system_time_series)
        self._record(trace, 13, connectivity)

        # Step 14: route typed prediction errors to the representation that was challenged.
        errors = self._prediction_errors(request.predictions, request.observed_evidence)
        state.prediction_errors = errors
        notices = self._route_errors(errors)
        self._record(trace, 14, {
            "prediction_errors": errors,
            "revision_notices": notices,
            "all_networks_activated_for_every_error": False,
        })

        # Step 15: combine supplied evidence channels without losing source identity.
        interpretation = self._integrate_interpretation(
            request, propositions, knowledge, mental_states
        )
        self._record(trace, 15, interpretation)

        # Step 16: decide whether the current result is adequate for this task.
        sufficiency = self._sufficiency(interpretation, executive_demand, request)
        self._record(trace, 16, sufficiency)

        # Step 17: prepare the appropriate destination(s) without side effects.
        routing = self._routing(request, interpretation, sufficiency)
        self._update_system(
            "medial_temporal",
            drives["medial_temporal"],
            routing.get("memory_handoff") or candidates["memory"],
            "memory_routing_preparation_not_durable_storage",
            drives["medial_temporal"],
        )
        self._record(trace, 17, routing)

        # Step 18: recompute next-step demand using this increment's unresolved conflicts and context changes.
        next_mental, next_executive = self._demand_estimates(
            request,
            syntactic,
            {**request.predictions, **{f"{k}_error": v for k, v in errors.items()}},
        )
        state.demand_history.append({
            "time_index": state.time_index,
            "mentalizing": next_mental,
            "executive": next_executive,
        })
        self._record(trace, 18, {
            "next_mentalizing_gate": next_mental["gate"],
            "next_executive_gate": next_executive["gate"],
            "demand_recalculated": True,
        })

        # Step 19: retain individual profile and localizer evidence without demographic inference.
        self._record(trace, 19, {
            "individual_profile_fields": sorted(state.individual_profile.keys()),
            "personalized_thresholds": {
                "mentalizing": mental_demand.get("threshold_used"),
                "executive": executive_demand.get("threshold_used"),
                "interpretation": sufficiency.get("threshold"),
            },
            "subject_specific_communication_overrides": sorted(
                (request.person_profile.get("communication_weights", {}) or {}).keys()
            ) if isinstance(request.person_profile.get("communication_weights", {}), Mapping) else [],
            "localization_status": localization["status"],
            "lateralization_claim": state.individual_profile.get("lateralization", "not_supplied"),
            "demographic_anatomy_inference_performed": False,
        })

        # Step 20: produce a provenance-aware audit of system roles and interpretation.
        contributions = {
            name: {
                "activation": state.systems[name]["activation"],
                "source": state.systems[name]["source"],
                "status": state.systems[name]["status"],
                "role": {
                    "perceptual": "acoustic_or_orthographic_form_evidence",
                    "language": "lexical_access_and_linguistic_combination",
                    "conceptual_knowledge": "world_and_event_knowledge_constraints",
                    "mentalizing": "conditional_agent_belief_goal_intention_hypotheses",
                    "multiple_demand": "conditional_attention_conflict_and_task_control",
                    "working_memory": "temporary_relevant_content_and_dependencies",
                    "medial_temporal": "optional_event_and_relational_memory_handoff",
                }[name],
            } for name in POINT9_SYSTEM_NAMES
        }
        final_interpretation = {
            **interpretation,
            "sufficiency": sufficiency,
            "system_contributions": contributions,
            "literal_propositions": deepcopy(propositions),
            "conceptual_knowledge_sources": [item["source"] for item in knowledge],
            "mental_state_sources": [item["source"] for item in mental_states],
            "uncertainty": {
                "unresolved_items": deepcopy(list(request.unresolved)),
                "candidate_scores_are_calibrated_probabilities": False,
            },
            "localization_status": localization["status"],
        }
        self._record(trace, 20, final_interpretation)

        biological = self.lif.run(drives, len(POINT9_STEP_SPECS)) if self.lif else {}
        biological_trace = biological.get("trace", []) if biological else []
        diagnostics = {
            "valid": len(trace) == len(POINT9_STEP_SPECS)
                and all(item["operation_id"] == POINT9_STEP_SPECS[index]["operation_id"]
                        for index, item in enumerate(trace)),
            "step_count": len(trace),
            "operation_count": len(POINT9_STEP_SPECS),
            "time_index": state.time_index,
            "system_names": list(POINT9_SYSTEM_NAMES),
            "active_systems": [
                name for name, data in state.systems.items()
                if data["activation"] >= 0.25
            ],
            "mentalizing_conditionally_recruited": bool(mental_demand["engaged"]),
            "multiple_demand_conditionally_recruited": md_active,
            "participant_localizer_supplied": localization["status"] == "participant_specific_estimate",
            "functional_connectivity_uses_caller_measurements": connectivity["status"] == "estimated_from_caller_supplied_time_series",
            "durable_memory_write_performed": False,
            "biological_reference_is_measured_human_data": False,
            "warnings": [
                "Network activation, localization, and coupling are model estimates unless participant measurements were explicitly supplied.",
                "Correlation among supplied system time series is statistical dependence, not causal or anatomical connectivity.",
                "The existing IFG/MTG/ATL language-network processor is reused as one component; other system representations remain separate.",
                "The optional LIF reference is a simplified population simulation, not a fitted human brain or evidence of biological validation.",
            ],
        }
        return Point9Result(
            sequence_id=request.sequence_id,
            time_index=state.time_index,
            state=state.to_dict(),
            functional_trace=trace,
            localization=deepcopy(localization),
            communication_graph=deepcopy(graph),
            functional_connectivity=connectivity,
            interpretation=final_interpretation,
            routing=routing,
            diagnostics=diagnostics,
            biological_summary=biological,
            biological_trace=biological_trace,
        )


# Alias for consistent composition with Points 1–8.
Point9Processor = Point9NeuralArchitectureCoordinator
