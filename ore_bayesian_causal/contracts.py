"""Validated data contracts for Point 11 Bayesian, probabilistic and causal reasoning.

Every record keeps probability, source reliability, measurement uncertainty, logical
truth and causal confidence as separate dimensions. A posterior close to one is not a
proven fact, a semantic relation is not a causal edge, and a reliable source is not an
accurate measurement.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

# Epistemic classes used across evidence, hypotheses and causal edges.
EVIDENCE_CLASSIFICATIONS = (
    "observation", "intervention", "prior_belief", "derived_claim", "source_assertion",
    "unknown",
)
CAUSAL_CONFIDENCE_CLASSES = (
    "interventional", "observational_associational", "theoretical", "postulated",
)
HYPOTHESIS_STATUSES = ("candidate", "evaluated", "published", "rejected", "inconclusive")
QUERY_KINDS = ("observational", "intervention", "counterfactual")


def _text(name: str, value: str) -> str:
    """Return non-empty normalized text or raise a contract error."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _unit(name: str, value: float) -> float:
    """Validate a finite scalar in the closed unit interval."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return number


def _distribution(name: str, values: Mapping[str, Any], states: Sequence[str],
                  tolerance: float = 1e-9) -> dict[str, float]:
    """Validate a categorical distribution over the exact states of one variable."""
    if set(values) != set(states):
        raise ValueError(f"{name} must define exactly the states {list(states)}")
    result: dict[str, float] = {}
    total = 0.0
    for state in states:
        number = float(values[state])
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{name}[{state}] must be finite and in [0, 1]")
        result[state] = number
        total += number
    if abs(total - 1.0) > tolerance:
        raise ValueError(f"{name} must sum to 1 (got {total})")
    return result


@dataclass(frozen=True, slots=True)
class VariableSpec:
    """Define one reasoning variable with an explicit finite categorical domain."""

    name: str
    states: tuple[str, ...]
    description: str = ""
    usg_record_ids: tuple[str, ...] = ()
    usg_unresolved_alternatives: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Reject ambiguous or malformed variable semantics."""
        object.__setattr__(self, "name", _text("variable name", self.name))
        if not self.states or len(set(self.states)) != len(self.states):
            raise ValueError(f"variable {self.name} needs unique non-empty states")
        object.__setattr__(self, "states", tuple(_text("state", s) for s in self.states))
        object.__setattr__(self, "usg_record_ids", tuple(self.usg_record_ids))
        object.__setattr__(self, "usg_unresolved_alternatives", tuple(self.usg_unresolved_alternatives))
        if len(set(self.usg_record_ids)) != len(self.usg_record_ids):
            raise ValueError(f"variable {self.name} has duplicate USG record ids")


@dataclass(frozen=True, slots=True)
class SourceProfile:
    """Record source identity and reliability independently of measurement quality."""

    source_id: str
    reliability: float = 0.5
    provenance: str = "unspecified"
    depends_on: tuple[str, ...] = ()
    dependence_strength: float = 0.0

    def __post_init__(self) -> None:
        """Validate source identity, reliability and declared dependence."""
        object.__setattr__(self, "source_id", _text("source_id", self.source_id))
        object.__setattr__(self, "reliability", _unit("source reliability", self.reliability))
        object.__setattr__(self, "provenance", _text("source provenance", self.provenance))
        object.__setattr__(self, "depends_on", tuple(self.depends_on))
        object.__setattr__(self, "dependence_strength",
                           _unit("dependence_strength", self.dependence_strength))
        if self.source_id in self.depends_on:
            raise ValueError("a source cannot depend on itself")


@dataclass(frozen=True, slots=True)
class EvidenceRecord:
    """Represent one normalized observation with provenance and epistemic classification.

    ``hard_value`` binds one state of the variable (certain observation); ``likelihood``
    instead maps states to non-negative evidence weights (soft/likelihood evidence).
    ``measurement_uncertainty`` describes measurement quality; ``source.reliability``
    describes the reporting source; the two are never merged.
    """

    evidence_id: str
    variable: str
    classification: str = "observation"
    hard_value: str | None = None
    likelihood: Mapping[str, float] | None = None
    timestamp: str = ""
    unit: str | None = None
    missing_value_code: str | None = None
    measurement_uncertainty: float | None = None
    measurement_interval: tuple[float, float] | None = None
    source: SourceProfile | None = None
    context: Mapping[str, Any] = field(default_factory=dict)
    lineage: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate classification, evidence binding and separated uncertainty dimensions."""
        object.__setattr__(self, "evidence_id", _text("evidence_id", self.evidence_id))
        object.__setattr__(self, "variable", _text("variable", self.variable))
        if self.classification not in EVIDENCE_CLASSIFICATIONS:
            raise ValueError(f"unsupported evidence classification: {self.classification}")
        if self.hard_value is not None:
            object.__setattr__(self, "hard_value", _text("hard_value", self.hard_value))
            if self.likelihood is not None:
                raise ValueError("evidence cannot be both hard and soft")
        if self.likelihood is not None:
            weights = {str(k): float(v) for k, v in self.likelihood.items()}
            if not weights:
                raise ValueError("soft evidence needs at least one state weight")
            for state, weight in weights.items():
                if not math.isfinite(weight) or weight < 0.0:
                    raise ValueError(f"likelihood[{state}] must be finite and non-negative")
            object.__setattr__(self, "likelihood", weights)
        if self.hard_value is None and self.likelihood is None and self.missing_value_code is None:
            raise ValueError("evidence must be hard, soft, or explicitly missing")
        if self.measurement_uncertainty is not None:
            object.__setattr__(self, "measurement_uncertainty",
                               _unit("measurement_uncertainty", self.measurement_uncertainty))
        if self.measurement_interval is not None:
            low, high = self.measurement_interval
            if not (math.isfinite(low) and math.isfinite(high)) or not low <= high:
                raise ValueError("measurement_interval must be an ordered finite pair")
        object.__setattr__(self, "context", dict(self.context))
        object.__setattr__(self, "lineage", tuple(self.lineage))
        if self.evidence_id in self.lineage:
            raise ValueError("evidence cannot reference itself in its lineage")


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    """Represent one candidate explanation with an explicit, documented prior."""

    hypothesis_id: str
    statement: str
    target_variable: str
    target_state: str
    prior: float
    prior_source: str = "unspecified"
    scope: str = "request"
    assumptions: tuple[str, ...] = ()
    usg_concept_ids: tuple[str, ...] = ()
    status: str = "candidate"

    def __post_init__(self) -> None:
        """Validate hypothesis identity, prior and status labels."""
        for name in ("hypothesis_id", "statement", "target_variable", "target_state"):
            object.__setattr__(self, name, _text(name, getattr(self, name)))
        object.__setattr__(self, "prior", _unit("hypothesis prior", self.prior))
        object.__setattr__(self, "prior_source", _text("prior_source", self.prior_source))
        object.__setattr__(self, "scope", _text("scope", self.scope))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "usg_concept_ids", tuple(self.usg_concept_ids))
        if self.status not in HYPOTHESIS_STATUSES:
            raise ValueError(f"unsupported hypothesis status: {self.status}")


@dataclass(frozen=True, slots=True)
class CPTSpec:
    """Specify one conditional probability table with its parent set and prior source."""

    variable: str
    parents: tuple[str, ...]
    table: Mapping[str, Mapping[str, float]]
    prior_source: str = "unspecified"

    def __post_init__(self) -> None:
        """Validate row structure; semantic and numeric validation run in the engine."""
        object.__setattr__(self, "variable", _text("cpt variable", self.variable))
        object.__setattr__(self, "parents", tuple(self.parents))
        object.__setattr__(self, "prior_source", _text("cpt prior_source", self.prior_source))
        if self.variable in self.parents:
            raise ValueError("a variable cannot be its own parent")
        if len(set(self.parents)) != len(self.parents):
            raise ValueError("parent sets must not contain duplicates")
        normalized: dict[str, dict[str, float]] = {}
        for key, row in dict(self.table).items():
            if not isinstance(key, str):
                raise ValueError("CPT rows must be keyed by parent-state tuples")
            # Root variables use the empty string as the single parent-state tuple key.
            if not key.strip() and self.parents:
                raise ValueError("CPT rows must be keyed by parent-state tuples")
            if not isinstance(row, Mapping) or not row:
                raise ValueError("CPT rows must be non-empty mappings")
            normalized[key] = {str(k): float(v) for k, v in row.items()}
        if not self.parents and len(normalized) != 1:
            raise ValueError("root CPTs must contain exactly one prior row")


@dataclass(frozen=True, slots=True)
class BayesianNetworkSpec:
    """Represent a discrete Bayesian network DAG with parameter provenance."""

    network_id: str
    variables: tuple[VariableSpec, ...]
    cpts: tuple[CPTSpec, ...]
    assumptions: tuple[str, ...] = ()
    dynamic_unrolled: bool = False
    version: int = 1
    provenance: str = "user_supplied_model"

    def __post_init__(self) -> None:
        """Validate variable identity; structural validation runs in the engine."""
        object.__setattr__(self, "network_id", _text("network_id", self.network_id))
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "cpts", tuple(self.cpts))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        object.__setattr__(self, "provenance", _text("network provenance", self.provenance))
        if self.version < 1:
            raise ValueError("network version must be at least 1")
        names = [variable.name for variable in self.variables]
        if len(set(names)) != len(names):
            raise ValueError("network variable names must be unique")
        if any(cpt.variable not in names for cpt in self.cpts):
            raise ValueError("every CPT variable must exist in the network")
        if len({cpt.variable for cpt in self.cpts}) != len(self.cpts):
            raise ValueError("each variable may declare at most one CPT")


@dataclass(frozen=True, slots=True)
class EvidenceBinding:
    """Bind evidence to a network node as hard state or soft likelihood evidence."""

    variable: str
    evidence_id: str
    hard_value: str | None = None
    likelihood: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        """Validate that exactly one binding style is used."""
        object.__setattr__(self, "variable", _text("binding variable", self.variable))
        object.__setattr__(self, "evidence_id", _text("evidence_id", self.evidence_id))
        if (self.hard_value is None) == (self.likelihood is None):
            raise ValueError("binding must be either hard or soft")
        if self.hard_value is not None:
            object.__setattr__(self, "hard_value", _text("hard_value", self.hard_value))
        else:
            weights = {str(k): float(v) for k, v in dict(self.likelihood).items()}
            if any(not math.isfinite(v) or v < 0.0 for v in weights.values()):
                raise ValueError("binding likelihood weights must be finite and non-negative")
            object.__setattr__(self, "likelihood", weights)


@dataclass(frozen=True, slots=True)
class HMMModelSpec:
    """Define hidden states, transitions, emissions and initial distribution for filtering."""

    model_id: str
    hidden_states: tuple[str, ...]
    observation_states: tuple[str, ...]
    initial_distribution: Mapping[str, float]
    transition_matrix: Mapping[str, Mapping[str, float]]
    emission_matrix: Mapping[str, Mapping[str, float]]
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate shape only; stochastic validation runs in the engine."""
        object.__setattr__(self, "model_id", _text("hmm model_id", self.model_id))
        object.__setattr__(self, "hidden_states", tuple(self.hidden_states))
        object.__setattr__(self, "observation_states", tuple(self.observation_states))
        object.__setattr__(self, "initial_distribution", dict(self.initial_distribution))
        object.__setattr__(self, "transition_matrix",
                           {str(a): dict(row) for a, row in dict(self.transition_matrix).items()})
        object.__setattr__(self, "emission_matrix",
                           {str(a): dict(row) for a, row in dict(self.emission_matrix).items()})
        object.__setattr__(self, "assumptions", tuple(self.assumptions))


@dataclass(frozen=True, slots=True)
class CausalEdge:
    """Represent one causal edge kept separate from semantic association edges."""

    source: str
    target: str
    confidence_class: str = "postulated"
    evidence_basis: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate endpoints and the causal confidence class."""
        object.__setattr__(self, "source", _text("edge source", self.source))
        object.__setattr__(self, "target", _text("edge target", self.target))
        if self.confidence_class not in CAUSAL_CONFIDENCE_CLASSES:
            raise ValueError(f"unsupported causal confidence class: {self.confidence_class}")
        object.__setattr__(self, "evidence_basis", tuple(self.evidence_basis))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))


@dataclass(frozen=True, slots=True)
class CausalDAGSpec:
    """Represent a causal DAG with annotated edges and explicit model assumptions."""

    model_id: str
    variables: tuple[str, ...]
    edges: tuple[CausalEdge, ...]
    assumptions: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        """Validate variable identity and edge endpoints."""
        object.__setattr__(self, "model_id", _text("causal model_id", self.model_id))
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        if self.version < 1:
            raise ValueError("causal model version must be at least 1")
        if len(set(self.variables)) != len(self.variables):
            raise ValueError("causal model variable names must be unique")
        if any(edge.source not in self.variables or edge.target not in self.variables
               for edge in self.edges):
            raise ValueError("every causal edge endpoint must exist in the model")


@dataclass(frozen=True, slots=True)
class StructuralEquation:
    """Define one structural equation X_i = f_i(PA_i, U_i) as a discrete conditional."""

    variable: str
    parents: tuple[str, ...]
    mechanism: Mapping[str, Mapping[str, float]]
    exogenous: str | None = None
    function_description: str = "discrete_conditional"

    def __post_init__(self) -> None:
        """Validate the mechanism table shape."""
        object.__setattr__(self, "variable", _text("equation variable", self.variable))
        object.__setattr__(self, "parents", tuple(self.parents))
        object.__setattr__(self, "mechanism",
                           {str(k): dict(row) for k, row in dict(self.mechanism).items()})
        object.__setattr__(self, "function_description",
                           _text("function_description", self.function_description))
        if self.variable in self.parents:
            raise ValueError("a structural equation cannot list its own variable as a parent")


@dataclass(frozen=True, slots=True)
class SCMSpec:
    """Represent a structural causal model with exogenous distributions and assumptions."""

    model_id: str
    variables: tuple[str, ...]
    exogenous: tuple[str, ...]
    equations: tuple[StructuralEquation, ...]
    exogenous_distributions: Mapping[str, Mapping[str, float]]
    assumptions: tuple[str, ...] = ()
    version: int = 1

    def __post_init__(self) -> None:
        """Validate SCM shape; functional validation runs in the engine."""
        object.__setattr__(self, "model_id", _text("scm model_id", self.model_id))
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "exogenous", tuple(self.exogenous))
        object.__setattr__(self, "equations", tuple(self.equations))
        object.__setattr__(self, "exogenous_distributions",
                           {str(k): dict(v) for k, v in dict(self.exogenous_distributions).items()})
        object.__setattr__(self, "assumptions", tuple(self.assumptions))
        if self.version < 1:
            raise ValueError("SCM version must be at least 1")
        if set(self.exogenous) & set(self.variables):
            raise ValueError("exogenous names must be disjoint from endogenous variables")
        if len({equation.variable for equation in self.equations}) != len(self.equations):
            raise ValueError("each endogenous variable needs at most one structural equation")
        if set(self.exogenous_distributions) < set(self.exogenous):
            raise ValueError("every exogenous variable needs a distribution")


@dataclass(frozen=True, slots=True)
class QuerySpec:
    """Represent observational, intervention or counterfactual queries explicitly."""

    query_id: str
    kind: str
    target: str
    conditions: Mapping[str, str] = field(default_factory=dict)
    interventions: Mapping[str, str] = field(default_factory=dict)
    evidence: Mapping[str, str] = field(default_factory=dict)
    decision_threshold: float | None = None

    def __post_init__(self) -> None:
        """Validate the query kind and required fields per kind."""
        object.__setattr__(self, "query_id", _text("query_id", self.query_id))
        if self.kind not in QUERY_KINDS:
            raise ValueError(f"unsupported query kind: {self.kind}")
        object.__setattr__(self, "target", _text("query target", self.target))
        object.__setattr__(self, "conditions", dict(self.conditions))
        object.__setattr__(self, "interventions", dict(self.interventions))
        object.__setattr__(self, "evidence", dict(self.evidence))
        if self.kind == "intervention" and not self.interventions:
            raise ValueError("intervention queries must declare do(...) values")
        if self.kind == "counterfactual" and not (self.evidence and self.interventions):
            raise ValueError("counterfactual queries need factual evidence and interventions")
        if self.decision_threshold is not None:
            object.__setattr__(self, "decision_threshold",
                               _unit("decision_threshold", self.decision_threshold))


@dataclass(frozen=True, slots=True)
class BayesianCausalConfig:
    """Configure thresholds, budgets, inference limits and reproducibility settings."""

    decision_threshold: float = 0.75
    sensitivity_perturbation: float = 0.2
    reversal_flag_margin: float = 0.02
    conflict_threshold: float = 0.4
    dependence_threshold: float = 0.5
    variable_elimination_node_limit: int = 12
    enumeration_state_limit: int = 20000
    resource_budget: int = 4000
    random_seed: int = 11
    prior_predictive_samples: int = 200
    posterior_predictive_samples: int = 200
    calibration_bins: int = 5
    require_calibration_status: bool = False
    min_source_reliability: float = 0.05
    missing_evidence_is_not_negative: bool = True
    independent_evidence_required: bool = True

    def __post_init__(self) -> None:
        """Reject invalid thresholds, budgets and sampling settings."""
        for name in ("decision_threshold", "sensitivity_perturbation", "reversal_flag_margin",
                     "conflict_threshold", "dependence_threshold", "min_source_reliability"):
            _unit(name, getattr(self, name))
        if not 2 <= self.variable_elimination_node_limit <= 512:
            raise ValueError("variable_elimination_node_limit must be between 2 and 512")
        if not 64 <= self.enumeration_state_limit <= 10_000_000:
            raise ValueError("enumeration_state_limit must be between 64 and 10^7")
        if not 100 <= self.resource_budget <= 1_000_000:
            raise ValueError("resource_budget must be between 100 and 10^6")
        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative")
        if not 16 <= self.prior_predictive_samples <= 100_000:
            raise ValueError("prior_predictive_samples must be between 16 and 10^5")
        if not 16 <= self.posterior_predictive_samples <= 100_000:
            raise ValueError("posterior_predictive_samples must be between 16 and 10^5")
        if not 2 <= self.calibration_bins <= 20:
            raise ValueError("calibration_bins must be between 2 and 20")


@dataclass
class BayesianCausalState:
    """Maintain the hypothesis registry, models, evidence and governed learning history."""

    runner_id: str
    hypotheses: dict[str, HypothesisRecord] = field(default_factory=dict)
    networks: dict[str, BayesianNetworkSpec] = field(default_factory=dict)
    causal_models: dict[str, CausalDAGSpec] = field(default_factory=dict)
    scms: dict[str, SCMSpec] = field(default_factory=dict)
    evidence: dict[str, EvidenceRecord] = field(default_factory=dict)
    sources: dict[str, SourceProfile] = field(default_factory=dict)
    runs: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    learning_updates: list[dict[str, Any]] = field(default_factory=list)
    run_count: int = 0

    def __post_init__(self) -> None:
        """Validate runner identity."""
        self.runner_id = _text("runner_id", self.runner_id)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly copy of the complete reasoning state."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BayesianCausalRequest:
    """Specify one inference run: question, models, evidence, hypotheses and queries."""

    question: str
    time_horizon: str = "single_inference"
    scope: str = "request"
    decision_threshold: float | None = None
    resource_budget: int | None = None
    variables: tuple[VariableSpec, ...] = ()
    network: BayesianNetworkSpec | None = None
    hypotheses: tuple[HypothesisRecord, ...] = ()
    evidence: tuple[EvidenceRecord, ...] = ()
    bindings: tuple[EvidenceBinding, ...] = ()
    queries: tuple[QuerySpec, ...] = ()
    hmm: HMMModelSpec | None = None
    hmm_observations: tuple[str | None, ...] = ()
    hmm_goal: str = "filter"
    causal_dag: CausalDAGSpec | None = None
    scm: SCMSpec | None = None
    observation_candidates: tuple[str, ...] = ()
    acquisition_costs: Mapping[str, float] = field(default_factory=dict)
    usg_resource: Any = None
    symbolic_constraints: tuple[Mapping[str, Any], ...] = ()
    verifier_hook: Any = None
    cancel_check: Any = None
    calibration_labels: tuple[tuple[float, int], ...] = ()

    def __post_init__(self) -> None:
        """Normalize the request and validate cross-references."""
        object.__setattr__(self, "question", _text("question", self.question))
        object.__setattr__(self, "time_horizon", _text("time_horizon", self.time_horizon))
        object.__setattr__(self, "scope", _text("scope", self.scope))
        if self.decision_threshold is not None:
            object.__setattr__(self, "decision_threshold",
                               _unit("request decision_threshold", self.decision_threshold))
        if self.resource_budget is not None and self.resource_budget < 1:
            raise ValueError("request resource_budget must be positive")
        object.__setattr__(self, "variables", tuple(self.variables))
        object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        object.__setattr__(self, "evidence", tuple(self.evidence))
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "queries", tuple(self.queries))
        object.__setattr__(self, "hmm_observations", tuple(self.hmm_observations))
        object.__setattr__(self, "observation_candidates", tuple(self.observation_candidates))
        object.__setattr__(self, "acquisition_costs",
                           {str(k): float(v) for k, v in dict(self.acquisition_costs).items()})
        object.__setattr__(self, "symbolic_constraints", tuple(self.symbolic_constraints))
        object.__setattr__(self, "calibration_labels",
                           tuple((float(p), int(y)) for p, y in self.calibration_labels))
        if len({record.evidence_id for record in self.evidence}) != len(self.evidence):
            raise ValueError("evidence IDs must be unique")
        if len({record.hypothesis_id for record in self.hypotheses}) != len(self.hypotheses):
            raise ValueError("hypothesis IDs must be unique")
        if self.hmm_goal not in {"filter", "smooth", "decode", "monitor"}:
            raise ValueError("hmm_goal must be one of filter, smooth, decode, monitor")


@dataclass(frozen=True, slots=True)
class BayesianCausalResult:
    """Capture the complete 40-step trace, posteriors, causal findings and publication state."""

    runner_id: str
    operation_trace: tuple[Mapping[str, Any], ...]
    intake_report: Mapping[str, Any]
    hypothesis_evaluations: tuple[Mapping[str, Any], ...]
    posterior_distributions: Mapping[str, Any]
    diagnostics: Mapping[str, Any]
    sequential_results: Mapping[str, Any]
    causal_findings: tuple[Mapping[str, Any], ...]
    counterfactual_results: tuple[Mapping[str, Any], ...]
    conflict_report: Mapping[str, Any]
    publication_report: Mapping[str, Any]
    confidence_and_uncertainty: Mapping[str, Any]
    biological: Mapping[str, Any]
    final_state: Mapping[str, Any]
    output: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Require the exact ordered 40-step Point 11 registry in every result."""
        expected = [f"P11-{index:02d}" for index in range(1, 41)]
        actual = [str(item.get("operation_id", "")) for item in self.operation_trace]
        if actual != expected:
            raise ValueError("Point 11 results must contain exactly 40 ordered operations")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete result as ordinary nested Python mappings."""
        return asdict(self)
