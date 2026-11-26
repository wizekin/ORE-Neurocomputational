"""Probabilistic evidence adapter (Phase E, Steps E1-E5; math ops 6.4-6.6).

Design rules enforced here:

- model predictions are wrapped as ``EvidenceItem`` records carrying model
  identity, version, input hash, score semantics and calibration metadata;
  raw softmax is *not* automatically a probability (Step E1);
- probabilistic observations are stored with ``EpistemicStatus.PROBABILISTIC``
  and never as ordinary true facts (Step E2);
- evidence is combined under an explicitly declared conditional-independence
  assumption: ``P(H|e_1:n) ∝ P(H) * prod_i P(e_i|H)`` (6.4). Dependent evidence
  uses a declared dependency group with a joint likelihood, or falls back to
  qualitative support — never default multiplication (Step E3);
- unsupported probabilistic semantics report qualitative support instead of
  fabricated precision (Step E4);
- the confidence decomposition ``q = (q_source, q_observation, q_rule,
  q_coverage, q_conflict)`` is kept as separate dimensions (6.5, Step E5).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .contracts import (
    ConfidenceVector,
    EpistemicStatus,
    EvidenceItem,
    make_id,
)


class ProbabilisticSemanticsError(ValueError):
    """Raised when evidence combination is requested without declared semantics."""


@dataclass(frozen=True, slots=True)
class IndependenceDeclaration:
    """One declared (conditional-)independence group over evidence IDs."""

    group_id: str
    member_ids: tuple[str, ...]
    assumption: str  # "conditional_independence" | "dependent_joint"
    joint_likelihood: float | None = None

    def __post_init__(self) -> None:
        """Validate the assumption label and joint likelihood range."""
        if not isinstance(self.group_id, str) or not self.group_id:
            raise ValueError("IndependenceDeclaration.group_id must be a non-empty string")
        if not self.member_ids or not isinstance(self.member_ids, tuple):
            raise ValueError("IndependenceDeclaration.member_ids must be a non-empty tuple")
        allowed = {"conditional_independence", "dependent_joint", "unmodelled"}
        if self.assumption not in allowed:
            raise ValueError(f"assumption must be one of {sorted(allowed)}")
        if self.assumption == "dependent_joint":
            if self.joint_likelihood is None or not 0.0 <= self.joint_likelihood <= 1.0:
                raise ValueError(
                    "dependent_joint groups must declare a joint likelihood in [0, 1]",
                )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "group_id": self.group_id,
            "member_ids": list(self.member_ids),
            "assumption": self.assumption,
            "joint_likelihood": self.joint_likelihood,
        }


@dataclass(frozen=True, slots=True)
class EvidenceSupport:
    """The combined support for one hypothesis, kept qualitative when needed."""

    hypothesis: str
    likelihood_ratio: float | None
    posterior: float | None
    qualitative: str | None
    combined_ids: tuple[str, ...]
    semantics: str

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "hypothesis": self.hypothesis,
            "likelihood_ratio": self.likelihood_ratio,
            "posterior": self.posterior,
            "qualitative": self.qualitative,
            "combined_ids": list(self.combined_ids),
            "semantics": self.semantics,
        }


@dataclass(slots=True)
class ProbabilisticReport:
    """Outcome of one probabilistic evidence pass (Steps E1-E5)."""

    wrapped_evidence: tuple[EvidenceItem, ...] = ()
    separated_probabilistic: tuple[str, ...] = ()
    supports: tuple[EvidenceSupport, ...] = ()
    qualitative_fallbacks: tuple[str, ...] = ()
    confidence_vectors: dict[str, ConfidenceVector] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible record."""
        return {
            "wrapped_count": len(self.wrapped_evidence),
            "separated_probabilistic": list(self.separated_probabilistic),
            "supports": [item.to_dict() for item in self.supports],
            "qualitative_fallbacks": list(self.qualitative_fallbacks),
            "confidence_vectors": {
                key: value.to_dict() for key, value in sorted(self.confidence_vectors.items())
            },
        }


def _logit(posterior_odds: float) -> float:
    """Return the natural-log odds used by the log-odds update (6.4)."""
    if posterior_odds <= 0:
        return float("-inf")
    return math.log(posterior_odds)


class ProbabilisticEvidenceAdapter:
    """Wrap, separate, combine and propagate uncertain evidence (Phase E)."""

    def __init__(self, *, min_reliability: float = 0.05) -> None:
        """Bind the minimum reliability an evidence item must carry."""
        if not 0.0 <= min_reliability <= 1.0:
            raise ValueError("min_reliability must be in [0, 1]")
        self.min_reliability = min_reliability

    # ---------------------------------------------------------------- Step E1

    def wrap_model_evidence(
        self,
        *,
        source: str,
        observation: Mapping[str, Any],
        reliability: float,
        model_id: str,
        model_version: str,
        input_hash: str,
        score_semantics: str,
        calibration: Mapping[str, Any] | None = None,
        lineage: tuple[str, ...] = (),
    ) -> EvidenceItem:
        """Wrap one model prediction as explicit evidence (Step E1).

        ``score_semantics`` must state what the numbers mean (e.g.
        ``softmax_logit``, ``calibrated_probability``); when semantics is
        ``softmax_logit`` the observation is explicitly flagged
        ``is_calibrated_probability: False`` because raw softmax is not
        automatically a probability.
        """
        if not score_semantics:
            raise ProbabilisticSemanticsError(
                "model evidence requires an explicit score_semantics declaration",
            )
        observation_record = {
            **dict(observation),
            "model_id": model_id,
            "model_version": model_version,
            "input_hash": input_hash,
            "score_semantics": score_semantics,
            "is_calibrated_probability": score_semantics == "calibrated_probability",
            "calibration": dict(calibration or {}),
        }
        return EvidenceItem(
            evidence_id=make_id("evidence"),
            source=source,
            observation=observation_record,
            reliability=reliability,
            uncertainty_model={"declared": score_semantics},
            lineage=lineage,
        )

    # ---------------------------------------------------------------- Step E2

    def separate_uncertain_from_crisp(
        self,
        evidence: tuple[EvidenceItem, ...],
    ) -> tuple[tuple[EvidenceItem, ...], tuple[EvidenceItem, ...]]:
        """Split evidence into probabilistic vs crisp records (Step E2).

        Probabilistic observations (any uncertainty model declared, or a
        ``probability`` field in the observation) must not be stored as
        ordinary true facts; callers receive the two sets explicitly.
        """
        crisp: list[EvidenceItem] = []
        probabilistic: list[EvidenceItem] = []
        for item in evidence:
            is_probabilistic = bool(
                item.uncertainty_model
                or item.observation.get("probability") is not None
                or item.observation.get("score_semantics"),
            )
            (probabilistic if is_probabilistic else crisp).append(item)
        return tuple(crisp), tuple(probabilistic)

    # ---------------------------------------------------------------- Step E3

    def combine_evidence(
        self,
        hypothesis: str,
        evidence: tuple[EvidenceItem, ...],
        priors: Mapping[str, float],
        declarations: tuple[IndependenceDeclaration, ...] = (),
        *,
        likelihood_fn: Any = None,
    ) -> EvidenceSupport:
        """Combine evidence under explicit assumptions (Steps E3-E4, math op 6.4).

        Only members of ``conditional_independence`` groups are multiplied:
        ``P(H|e_1:n) ∝ P(H) * prod_i P(e_i|H)``. Members of ``dependent_joint``
        groups use the declared joint likelihood once per group. Undeclared
        dependencies (``unmodelled`` or evidence outside any group) never
        multiply — they produce qualitative support instead of fabricated
        precision.
        """
        prior = priors.get(hypothesis)
        if prior is None or not 0.0 < prior <= 1.0:
            raise ProbabilisticSemanticsError(
                f"hypothesis {hypothesis!r} requires a declared prior in (0, 1]",
            )
        by_id = {item.evidence_id: item for item in evidence}
        assigned: set[str] = set()
        log_odds = _logit(prior / (1.0 - prior))
        combined: list[str] = []
        qualitative: list[str] = []
        semantics_parts: list[str] = []

        for declaration in declarations:
            members = [item for item in declaration.member_ids if item in by_id]
            if not members:
                continue
            if declaration.assumption == "conditional_independence":
                for member in members:
                    likelihood = self._likelihood(member, by_id, hypothesis, likelihood_fn)
                    log_odds += math.log(
                        max(likelihood, 1e-12) / max(1.0 - likelihood, 1e-12),
                    )
                    assigned.add(member)
                    combined.append(member)
                semantics_parts.append("conditional_independence_product")
            elif declaration.assumption == "dependent_joint":
                joint = float(declaration.joint_likelihood or 0.0)
                log_odds += math.log(max(joint, 1e-12) / max(1.0 - joint, 1e-12))
                assigned.update(members)
                combined.extend(members)
                semantics_parts.append("declared_joint_likelihood")
            else:
                for member in members:
                    assigned.add(member)
                    qualitative.append(member)
                semantics_parts.append("unmodelled_dependency_qualitative")

        for item in evidence:
            if item.evidence_id not in assigned:
                qualitative.append(item.evidence_id)

        posterior = None
        likelihood_ratio = None
        if combined:
            posterior = 1.0 / (1.0 + math.exp(-log_odds)) if log_odds < 700 else 1.0
            likelihood_ratio = math.exp(log_odds) / max(
                math.exp(math.log(prior / (1.0 - prior))), 1e-12,
            )
        if not combined and qualitative:
            return EvidenceSupport(
                hypothesis=hypothesis,
                likelihood_ratio=None,
                posterior=None,
                qualitative="qualitative_support_only",
                combined_ids=tuple(dict.fromkeys(qualitative)),
                semantics="no_declared_independent_evidence",
            )
        return EvidenceSupport(
            hypothesis=hypothesis,
            likelihood_ratio=likelihood_ratio,
            posterior=posterior,
            qualitative=("with_qualitative_additions" if qualitative else None),
            combined_ids=tuple(dict.fromkeys(combined)),
            semantics="+".join(semantics_parts) or "prior_only",
        )

    def _likelihood(
        self,
        evidence_id: str,
        by_id: Mapping[str, EvidenceItem],
        hypothesis: str,
        likelihood_fn: Any,
    ) -> float:
        """Resolve P(e|H) for one evidence item through the declared likelihood fn."""
        item = by_id.get(evidence_id)
        if item is None:
            raise ProbabilisticSemanticsError(f"unknown evidence id {evidence_id!r}")
        if likelihood_fn is not None:
            value = float(likelihood_fn(item, hypothesis))
        else:
            value = float(item.observation.get("probability", item.reliability))
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ProbabilisticSemanticsError(
                f"evidence {evidence_id!r} produced an out-of-range likelihood {value!r}",
            )
        return min(1.0 - 1e-12, max(1e-12, value))

    # ---------------------------------------------------------------- Step E5

    def build_confidence_vector(
        self,
        *,
        source_reliability: float,
        observation_uncertainty: float,
        rule_strength: float,
        evidence_coverage: float,
        conflict_level: float,
    ) -> ConfidenceVector:
        """Assemble the five-dimension confidence vector (6.5, Step E5).

        The dimensions stay separate; nothing here collapses them into one
        opaque score.
        """
        for name, value in (
            ("source_reliability", source_reliability),
            ("observation_uncertainty", observation_uncertainty),
            ("rule_strength", rule_strength),
            ("evidence_coverage", evidence_coverage),
            ("conflict_level", conflict_level),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        return ConfidenceVector(
            q_source=source_reliability,
            q_observation=observation_uncertainty,
            q_rule=rule_strength,
            q_coverage=evidence_coverage,
            q_conflict=conflict_level,
        )

    # ------------------------------------------------------- math op 6.6 (optional)

    @staticmethod
    def differentiable_consistency(
        truth_values: Mapping[str, float],
        weights: Mapping[str, float],
        *,
        data_loss: float = 0.0,
        lam: float = 1.0,
    ) -> float:
        """Compute the optional LTN-like consistency signal (6.6).

        ``L_logic = sum_j omega_j * l(Truth(phi_j), 1) + lambda * L_data`` using
        squared-hinge satisfaction loss ``l = (1 - Truth)^2``. This is a soft
        optimization signal for optional learned components; it is explicitly
        *not* a formal proof and never gates publication.
        """
        total = 0.0
        for formula_id, truth in truth_values.items():
            if not math.isfinite(truth) or not 0.0 <= truth <= 1.0:
                raise ValueError(f"truth value for {formula_id!r} must be in [0, 1]")
            weight = float(weights.get(formula_id, 1.0))
            total += weight * (1.0 - truth) ** 2
        return total + lam * float(data_loss)


differentiable_consistency = ProbabilisticEvidenceAdapter.differentiable_consistency


__all__ = [
    "EvidenceSupport",
    "IndependenceDeclaration",
    "ProbabilisticEvidenceAdapter",
    "ProbabilisticReport",
    "ProbabilisticSemanticsError",
    "differentiable_consistency",
]
