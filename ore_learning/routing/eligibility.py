"""Capability resolution and eligibility filtering (Point 14, Steps 6-7).

A candidate is eligible only if schema compatibility (S), capability fit (C),
provenance/policy validity (P), resource feasibility (R) and license/use
compatibility (L) all hold: E(m,t) = S(m,t) and C(m,t) and P(m,t) and R(m,t) and
L(m,t).  Hard eligibility is computed before and independently of any utility
score; a higher utility can never repair a failed factor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import LearningConfig
from ..contracts import (
    BenchmarkResult,
    Capability,
    CapabilityRequest,
    LifecycleState,
    Modality,
    ModelManifest,
    RiskClass,
    SchemaDescriptor,
)
from ..registry import ModelRegistry

__all__ = ("CapabilityResolver", "EligibilityFactor", "ModelCandidate",
           "RoutingPolicy", "default_routing_policy")


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """Hard policy constraints that override utility (acceptance criterion 7).

    Licenses forbidden for the risk class, security verdicts, and provenance
    completeness are policy facts: a candidate violating them is excluded no matter
    how high its measured utility is.
    """

    licenses_forbidden_high_risk: tuple[str, ...] = ("research-only", "proprietary")
    licenses_forbidden_always: tuple[str, ...] = ()
    security_verdicts_accepted: tuple[str, ...] = ("approved", "not_required")
    security_verdicts_accepted_low_risk: tuple[str, ...] = ("approved", "not_required",
                                                            "pending")
    require_benchmark_evidence: bool = True
    allowed_execution_permissions: tuple[str, ...] = ("inference:local",)

    def licenses_forbidden(self, risk_class: RiskClass) -> tuple[str, ...]:
        """Licenses forbidden for the given risk class."""
        if risk_class in (RiskClass.HIGH, RiskClass.CRITICAL):
            return tuple(self.licenses_forbidden_high_risk) + tuple(
                self.licenses_forbidden_always)
        return tuple(self.licenses_forbidden_always)

    def security_verdicts_allowed(self, risk_class: RiskClass) -> tuple[str, ...]:
        """Accepted security verdicts for the given risk class."""
        if risk_class in (RiskClass.HIGH, RiskClass.CRITICAL):
            return tuple(self.security_verdicts_accepted)
        return tuple(self.security_verdicts_accepted_low_risk)


def default_routing_policy() -> RoutingPolicy:
    """The canonical policy."""
    return RoutingPolicy()


@dataclass(frozen=True, slots=True)
class EligibilityFactor:
    """One factor of the eligibility conjunction with its evidence."""

    factor: str  # S | C | P | R | L
    passed: bool
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"factor": self.factor, "passed": self.passed, "evidence": self.evidence}


@dataclass(slots=True)
class ModelCandidate:
    """Section 5.1 -- a registry reference plus eligibility evidence."""

    manifest: ModelManifest
    eligible: bool
    factors: tuple[EligibilityFactor, ...]
    utility: float | None = None
    utility_features: dict[str, float] = field(default_factory=dict)
    rank: int = 0
    exclusion_reason: str = ""

    @property
    def identity(self) -> str:
        """Candidate identity string."""
        return self.manifest.identity

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"model_id": self.manifest.model_id,
                "model_version": self.manifest.version,
                "identity": self.identity, "eligible": self.eligible,
                "factors": [factor.to_dict() for factor in self.factors],
                "utility": self.utility,
                "utility_features": dict(self.utility_features),
                "rank": self.rank, "exclusion_reason": self.exclusion_reason}


class CapabilityResolver:
    """Step 6-7 -- map a task contract to required capabilities and filter candidates."""

    def __init__(self, policy: RoutingPolicy | None = None,
                 config: LearningConfig | None = None) -> None:
        """Configure the resolver with a routing policy and governance config."""
        self.policy = policy or default_routing_policy()
        self.config = config or LearningConfig()

    def resolve(self, request: CapabilityRequest, registry: ModelRegistry) -> list[ModelCandidate]:
        """Return every registry match with a full eligibility verdict.

        The registry first applies its selectable-state and coarse capability filter
        (Step 6); this method then evaluates the full conjunction E = S and C and P
        and R and L per candidate (Step 7).
        """
        query = registry.query(request)
        candidates = []
        for manifest in query.matched:
            factors = (
                self._schema_factor(manifest, request),
                self._capability_factor(manifest, request),
                self._provenance_policy_factor(manifest, request),
                self._resource_factor(manifest, request),
                self._license_factor(manifest, request),
            )
            eligible = all(factor.passed for factor in factors)
            failed = [factor for factor in factors if not factor.passed]
            reason = "; ".join(f"{factor.factor}: {factor.evidence}" for factor in failed)
            candidates.append(ModelCandidate(
                manifest=manifest, eligible=eligible, factors=factors,
                exclusion_reason=reason if not eligible else ""))
        return candidates

    # ------------------------------------------------------------- factors

    def _schema_factor(self, manifest: ModelManifest,
                       request: CapabilityRequest) -> EligibilityFactor:
        """S(m,t): output schema satisfies the task's output contract."""
        contract = request.output_contract
        if manifest.output_schema.schema_type == contract.schema_type and \
                manifest.output_schema.schema_version == contract.schema_version:
            return EligibilityFactor("S", True, "output schema satisfies the task contract")
        return EligibilityFactor(
            "S", False,
            f"output schema {manifest.output_schema.schema_type}@"
            f"{manifest.output_schema.schema_version} does not satisfy contract "
            f"{contract.schema_type}@{contract.schema_version}")

    def _capability_factor(self, manifest: ModelManifest,
                           request: CapabilityRequest) -> EligibilityFactor:
        """C(m,t): declared capabilities, modality, and measured quality fit the task."""
        required = {item.value for item in request.capabilities}
        declared = {item.value if hasattr(item, "value") else str(item)
                    for item in manifest.supported_tasks}
        if not required.issubset(declared):
            missing = sorted(required - declared)
            return EligibilityFactor("C", False, f"missing capabilities: {missing}")
        if request.modality not in manifest.modalities:
            return EligibilityFactor("C", False,
                                     f"modality {request.modality.value} unsupported")
        if self.policy.require_benchmark_evidence:
            quality = best_task_quality(manifest, request)
            if quality is None:
                return EligibilityFactor(
                    "C", False,
                    "no held-out benchmark evidence for the requested capability")
            if quality + 1e-9 < request.accuracy_target:
                return EligibilityFactor(
                    "C", False,
                    f"measured quality {quality:.3f} below the accuracy target "
                    f"{request.accuracy_target:.3f}")
        return EligibilityFactor("C", True, "capabilities, modality and quality fit")

    def _provenance_policy_factor(self, manifest: ModelManifest,
                                  request: CapabilityRequest) -> EligibilityFactor:
        """P(m,t): provenance completeness, security review, execution permissions."""
        provenance = manifest.training_data_provenance
        if provenance is None or not provenance.dataset_id or not provenance.license:
            return EligibilityFactor("P", False, "provenance incomplete")
        verdicts = self.policy.security_verdicts_allowed(request.risk_class)
        if manifest.security_review.verdict not in verdicts:
            return EligibilityFactor(
                "P", False,
                f"security verdict {manifest.security_review.verdict} not accepted for "
                f"{request.risk_class.value}-risk tasks")
        if not set(manifest.execution_permissions).issubset(
                set(self.policy.allowed_execution_permissions)):
            return EligibilityFactor("P", False, "execution permissions exceed policy")
        return EligibilityFactor("P", True, "provenance complete and policy accepted")

    def _resource_factor(self, manifest: ModelManifest,
                         request: CapabilityRequest) -> EligibilityFactor:
        """R(m,t): declared resources fit the declared budget envelope.

        The full CROS reservation happens in Step 11 (routing.resources); this factor
        checks the manifest against the budget envelope declared with the request
        (``latency_target_ms`` and a conservative per-task resource ceiling derived
        from the config safety margin).
        """
        profile = manifest.resource_profile
        if profile.estimated_latency_ms > request.latency_target_ms * (
                1.0 + self.config.resource_safety_margin):
            return EligibilityFactor(
                "R", False,
                f"declared latency {profile.estimated_latency_ms:.0f} ms exceeds the "
                f"target {request.latency_target_ms:.0f} ms")
        return EligibilityFactor("R", True, "declared resources fit the budget envelope")

    def _license_factor(self, manifest: ModelManifest,
                        request: CapabilityRequest) -> EligibilityFactor:
        """L(m,t): license/use compatibility for the task risk class."""
        forbidden = self.policy.licenses_forbidden(request.risk_class)
        if manifest.license in forbidden:
            return EligibilityFactor(
                "L", False,
                f"license {manifest.license!r} forbidden for {request.risk_class.value}-"
                "risk tasks")
        return EligibilityFactor("L", True, f"license {manifest.license!r} compatible")


def best_task_quality(manifest: ModelManifest,
                      request: CapabilityRequest) -> float | None:
    """Best held-out quality measurement relevant to the requested capabilities."""
    relevant = {item.value for item in request.capabilities}
    values: list[float] = []
    for benchmark in manifest.benchmark_results:
        if not benchmark.held_out:
            continue
        if benchmark.suite in relevant or benchmark.suite == "generic":
            if benchmark.lower_is_better:
                continue
            values.append(float(benchmark.value))
    return max(values) if values else None
