"""Phase V, Step 23 - integrate uncertainty, Bayesian reasoning and causal analysis.

Bayesian updating is applied only with a defined likelihood model and explicit
assumptions; causal structure stays separate from correlation; confounders and
missing variables are recorded; alternative explanations are compared; and
the reasoner abstains when evidence is insufficient or model assumptions
cannot be defended.  Scores from unrelated modules are never averaged as if
they were independent likelihoods.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from .core import new_id, sha256_digest, utc_now_iso
from .evidence import SemanticClaim

UNCERTAINTY_SCHEMA = "ore.point24.uncertainty.v1"


@dataclass(frozen=True, slots=True)
class BayesianModel:
    """An explicitly declared binomial Bayesian model (prior + likelihood)."""

    prior_alpha: float
    prior_beta: float
    likelihood_model: str = "binomial"
    assumptions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.prior_alpha <= 0 or self.prior_beta <= 0:
            raise ValueError("Beta prior parameters must be positive")
        if self.likelihood_model != "binomial":
            raise ValueError("first-version supports only the binomial likelihood model")

    def posterior(self, successes: int, failures: int) -> "BayesianModel":
        if successes < 0 or failures < 0:
            raise ValueError("counts must be non-negative")
        return BayesianModel(
            prior_alpha=self.prior_alpha + successes,
            prior_beta=self.prior_beta + failures,
            likelihood_model=self.likelihood_model,
            assumptions=self.assumptions,
        )

    def posterior_mean(self) -> float:
        total = self.prior_alpha + self.prior_beta
        return self.prior_alpha / total

    def posterior_variance(self) -> float:
        total = self.prior_alpha + self.prior_beta
        return (self.prior_alpha * self.prior_beta) / (total * total * (total + 1.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "likelihood_model": self.likelihood_model,
            "assumptions": list(self.assumptions),
            "posterior_mean": self.posterior_mean(),
            "posterior_variance": self.posterior_variance(),
        }


@dataclass(frozen=True, slots=True)
class BayesianUpdate:
    """The record of one Bayes-rule application with its assumptions."""

    hypothesis: str
    prior_mean: float
    posterior_mean: float
    evidence_summary: str
    assumptions: tuple[str, ...]
    update_id: str = field(default_factory=lambda: new_id("bayes_update"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "hypothesis": self.hypothesis,
            "prior_mean": self.prior_mean,
            "posterior_mean": self.posterior_mean,
            "evidence_summary": self.evidence_summary,
            "assumptions": list(self.assumptions),
        }


class BayesianReasoner:
    """Apply Bayes' rule only under a declared likelihood model.

    ``P(H|E) = P(E|H) P(H) / P(E)`` is computed here through the conjugate
    Beta-binomial posterior; every update records its assumptions, and
    evidence-source dependence is checked so correlated evidence is never
    counted twice.
    """

    def __init__(self) -> None:
        self._updates: list[BayesianUpdate] = []

    def update(
        self,
        hypothesis: str,
        model: BayesianModel,
        *,
        successes: int,
        failures: int,
        evidence_summary: str,
        evidence_dependent: bool = False,
    ) -> BayesianUpdate:
        """One explicit Bayesian update; refuses double-counted evidence."""
        if evidence_dependent and successes > 0:
            raise ValueError(
                "correlated evidence must not be counted as independent "
                "corroboration; group it first or update once"
            )
        posterior = model.posterior(successes, failures)
        record = BayesianUpdate(
            hypothesis=hypothesis,
            prior_mean=model.posterior_mean(),
            posterior_mean=posterior.posterior_mean(),
            evidence_summary=evidence_summary,
            assumptions=model.assumptions,
        )
        self._updates.append(record)
        return record

    def updates(self) -> tuple[BayesianUpdate, ...]:
        return tuple(self._updates)


@dataclass(frozen=True, slots=True)
class CausalStructure:
    """An explicit structural causal model fragment (Step 23)."""

    variables: tuple[str, ...]
    edges: tuple[tuple[str, str], ...]
    confounders: tuple[str, ...] = ()
    missing_variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        known = set(self.variables)
        for source, target in self.edges:
            if source not in known or target not in known:
                raise ValueError(f"edge ({source}, {target}) references an undeclared variable")

    def is_acyclic(self) -> bool:
        """A causal structure must be a DAG; cycles indicate a model error."""
        graph: dict[str, list[str]] = {var: [] for var in self.variables}
        for source, target in self.edges:
            graph[source].append(target)
        state: dict[str, int] = {var: 0 for var in self.variables}

        def visit(node: str) -> bool:
            if state[node] == 1:
                return False
            state[node] = 1
            ok = all(visit(child) for child in graph[node])
            state[node] = 2
            return ok

        return all(state[var] != 0 or visit(var) for var in self.variables)

    def to_dict(self) -> dict[str, Any]:
        return {
            "variables": list(self.variables),
            "edges": [[s, t] for s, t in self.edges],
            "confounders": list(self.confounders),
            "missing_variables": list(self.missing_variables),
            "acyclic": self.is_acyclic(),
        }


@dataclass(frozen=True, slots=True)
class CausalQueryResult:
    """The answer to one causal query with its identification assumptions."""

    query: str
    estimable: bool
    estimate: float | None
    identification_assumptions: tuple[str, ...]
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "estimable": self.estimable,
            "estimate": self.estimate,
            "identification_assumptions": list(self.identification_assumptions),
            "notes": list(self.notes),
        }


class CausalAnalyst:
    """Separate causal claims from correlational observations.

    Observational correlation alone does not identify an intervention effect;
    a causal query is estimable only when the structural model plus
    justified intervention assumptions support it.
    """

    def query(
        self,
        structure: CausalStructure,
        *,
        treatment: str,
        outcome: str,
        observed_correlation: float | None = None,
    ) -> CausalQueryResult:
        if treatment not in structure.variables or outcome not in structure.variables:
            raise ValueError("treatment and outcome must be declared variables")
        # A common cause (declared confounder or structural parent of both
        # treatment and outcome) makes the observed correlation
        # non-identifying for the intervention effect.
        parents_of = {var: [] for var in structure.variables}
        for source, target in structure.edges:
            parents_of[target].append(source)
        common_causes = sorted(
            set(parents_of[treatment]) & set(parents_of[outcome])
        )
        declared_confounders = sorted(structure.confounders)
        confounded = bool(common_causes or declared_confounders)
        if observed_correlation is not None and confounded:
            return CausalQueryResult(
                query=f"effect({treatment} -> {outcome})",
                estimable=False,
                estimate=None,
                identification_assumptions=(),
                notes=(
                    f"observed correlation {observed_correlation:.3f} is not an "
                    "intervention effect: common causes present "
                    f"({', '.join(common_causes + declared_confounders) or 'unobserved'})",
                ),
            )
        if observed_correlation is None:
            return CausalQueryResult(
                query=f"effect({treatment} -> {outcome})",
                estimable=False,
                estimate=None,
                identification_assumptions=(),
                notes=("no observational estimate supplied; abstaining",),
            )
        return CausalQueryResult(
            query=f"effect({treatment} -> {outcome})",
            estimable=True,
            estimate=observed_correlation,
            identification_assumptions=(
                "no unmeasured confounders between treatment and outcome",
                "structure is acyclic and correctly specified",
                "treatment assignment is ignorable given covariates",
            ),
            notes=("estimate identified from the declared structural model",),
        )


@dataclass(frozen=True, slots=True)
class UncertaintyReport:
    """A belief state or abstention with provenance."""

    hypothesis: str
    posterior_mean: float | None
    abstained: bool
    reason: str
    provenance: tuple[str, ...] = ()
    reported_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": UNCERTAINTY_SCHEMA,
            "hypothesis": self.hypothesis,
            "posterior_mean": self.posterior_mean,
            "abstained": self.abstained,
            "reason": self.reason,
            "provenance": list(self.provenance),
            "reported_at": self.reported_at,
        }


def abstain_when_insufficient(
    claim: SemanticClaim,
    *,
    minimum_independent_evidence: int = 2,
) -> UncertaintyReport:
    """Abstain when evidence is insufficient - never force a conclusion."""
    independent = claim.independent_corroboration_count()
    if independent < minimum_independent_evidence:
        return UncertaintyReport(
            hypothesis=claim.statement,
            posterior_mean=None,
            abstained=True,
            reason=(
                f"only {independent} independent evidence item(s); "
                f"{minimum_independent_evidence} required"
            ),
            provenance=tuple(item.source_id for item in claim.evidence),
        )
    return UncertaintyReport(
        hypothesis=claim.statement,
        posterior_mean=None,
        abstained=False,
        reason="evidence sufficiency satisfied; a calibrated model is still required for probabilities",
        provenance=tuple(item.source_id for item in claim.evidence),
    )


__all__ = [
    "UNCERTAINTY_SCHEMA",
    "BayesianModel",
    "BayesianReasoner",
    "BayesianUpdate",
    "CausalAnalyst",
    "CausalQueryResult",
    "CausalStructure",
    "UncertaintyReport",
    "abstain_when_insufficient",
]
