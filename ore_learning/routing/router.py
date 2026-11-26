"""The Model Router facade (Point 14, Steps 6-11).

resolve capabilities -> enforce eligibility -> rank by utility -> select one model
or a bounded ensemble -> reserve resources through CROS -> or defer / fall back.
The router returns a :class:`ModelSelection` with the full ranking rationale and
fallback path; it is deterministic for identical state and inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Sequence

from ..config import LearningConfig
from ..contracts import CapabilityRequest, LifecycleState, ModelManifest
from ..registry import ModelRegistry
from .eligibility import CapabilityResolver, ModelCandidate, RoutingPolicy
from .ensemble import BoundedEnsemble, ensemble_justified
from .ranking import RankingRecord, rank_candidates, ranking_table, weights_for_risk
from .resources import CROSPort, LocalResourceCoordinator, ReservationLease

__all__ = ("ModelRouter", "ModelSelection", "SelectionDecision")


class SelectionDecision(str, Enum):
    """The terminal decision of one routing run."""

    SELECTED = "selected"
    ENSEMBLE = "ensemble"
    FALLBACK = "fallback"
    DEFERRED = "deferred"


@dataclass(slots=True)
class ModelSelection:
    """Section 5.1 -- selected candidate(s), ranking rationale, and fallback."""

    task_id: str
    decision: SelectionDecision
    selected: tuple[ModelManifest, ...]
    ranking: list[RankingRecord] = field(default_factory=list)
    candidates: list[ModelCandidate] = field(default_factory=list)
    fallback_reason: str = ""
    reservation_id: str | None = None
    resource_units: float = 0.0
    utility_weights: dict[str, float] = field(default_factory=dict)
    selection_rationale: str = ""
    selected_at: float = field(default_factory=time.time)

    @property
    def primary(self) -> ModelManifest | None:
        """The primary selected manifest, if any."""
        return self.selected[0] if self.selected else None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"task_id": self.task_id, "decision": self.decision.value,
                "selected": [manifest.identity for manifest in self.selected],
                "ranking": [record.to_dict() for record in self.ranking],
                "candidates": [candidate.to_dict() for candidate in self.candidates],
                "fallback_reason": self.fallback_reason,
                "reservation_id": self.reservation_id,
                "resource_units": round(self.resource_units, 4),
                "utility_weights": dict(self.utility_weights),
                "selection_rationale": self.selection_rationale,
                "selected_at": self.selected_at}


class ModelRouter:
    """Steps 6-11 -- deterministic capability-based routing with hard policy gates."""

    def __init__(self, registry: ModelRegistry, *,
                 resolver: CapabilityResolver | None = None,
                 coordinator: CROSPort | None = None,
                 config: LearningConfig | None = None,
                 policy: RoutingPolicy | None = None,
                 fallback_model_id: str | None = None,
                 fallback_model_version: str = "1.0.0",
                 allow_ensemble: bool = False) -> None:
        """Compose the router from its collaborators."""
        self.registry = registry
        self.config = config or LearningConfig()
        self.resolver = resolver or CapabilityResolver(policy=policy, config=self.config)
        self.coordinator: CROSPort = coordinator or LocalResourceCoordinator()
        self.fallback_model_id = fallback_model_id
        self.fallback_model_version = fallback_model_version
        self.allow_ensemble = bool(allow_ensemble)
        self.ensemble = BoundedEnsemble(self.config)
        self.selections: list[ModelSelection] = []
        self._leases: dict[int, ReservationLease] = {}

    def select(self, request: CapabilityRequest) -> ModelSelection:
        """Run Steps 6-11 for one capability request."""
        candidates = self.resolver.resolve(request, self.registry)
        candidates = rank_candidates(candidates, request, self.config)
        ranking = ranking_table(candidates)
        weights = weights_for_risk(self.config, request.risk_class)
        selection = ModelSelection(
            task_id=request.task_id, decision=SelectionDecision.DEFERRED,
            selected=(), ranking=ranking, candidates=list(candidates),
            utility_weights=weights.to_dict())

        eligible = sorted((candidate for candidate in candidates if candidate.eligible),
                          key=lambda candidate: candidate.rank)
        if not eligible:
            return self._try_fallback(selection, request,
                                      "no candidate passed the eligibility conjunction")

        # Step 10 -- one specialist preferred; a bounded ensemble runs only when the
        # caller supplied ensemble evidence (allow_ensemble) AND the justification
        # check passes: independent candidates within the utility margin.
        top = eligible[0]
        members: list[ModelCandidate] = [top]
        if self.allow_ensemble and len(eligible) >= 2:
            justified, why = ensemble_justified(
                eligible[:self.config.max_ensemble_size], self.config,
                top.utility or 0.0, eligible[1].utility or 0.0)
            if justified:
                members = eligible[:self.config.max_ensemble_size]
                selection.selection_rationale = (
                    f"bounded ensemble of {len(members)} independent candidates: {why}")
        if not selection.selection_rationale:
            selection.selection_rationale = (
                "single specialist selected: "
                + ("top utility within the ensemble margin but ensemble not justified"
                   if len(eligible) >= 2 else "single eligible candidate")
                + f"; utility={top.utility:.4f}")

        # Step 11 -- reserve resources through CROS for each member, best-first.
        for candidate in members:
            lease = self._reserve(selection, request, candidate.manifest)
            if lease is not None:
                selection.decision = (SelectionDecision.ENSEMBLE
                                      if len(members) > 1 else SelectionDecision.SELECTED)
                selection.selected = tuple(member.manifest for member in members)
                self.selections.append(selection)
                self._leases[id(selection)] = lease
                return selection
        return self._try_fallback(
            selection, request, "resource reservation refused for every candidate")

    def release(self, selection: ModelSelection, reason: str = "completed") -> bool:
        """Release the reservation held by a selection (timeouts call this)."""
        lease = self._leases.pop(id(selection), None)
        if lease is None:
            return False
        return lease.release(reason)

    # ------------------------------------------------------------- internals

    def _reserve(self, selection: ModelSelection, request: CapabilityRequest,
                 manifest: ModelManifest) -> ReservationLease | None:
        """Step 11 -- attempt one reservation, best candidate first."""
        reservation = self.coordinator.request(
            manifest.model_id, manifest.resource_profile,
            margin=self.config.resource_safety_margin)
        if reservation.granted:
            selection.reservation_id = reservation.reservation_id
            selection.resource_units = reservation.units
            return ReservationLease(
                reservation, release_hook=lambda reservation_id, reason:
                self.coordinator.release(reservation_id, reason))
        selection.fallback_reason = reservation.reason
        return None

    def _try_fallback(self, selection: ModelSelection, request: CapabilityRequest,
                      reason: str) -> ModelSelection:
        """Safe fallback or deferral when selection or reservation fails."""
        selection.fallback_reason = reason
        if self.fallback_model_id is not None:
            manifest = self.registry.get(self.fallback_model_id,
                                         self.fallback_model_version)
            if manifest is not None and manifest.lifecycle_state in (
                    LifecycleState.VALIDATED, LifecycleState.CANARY,
                    LifecycleState.ACTIVE):
                lease = self._reserve(selection, request, manifest)
                if lease is not None:
                    selection.decision = SelectionDecision.FALLBACK
                    selection.selected = (manifest,)
                    selection.selection_rationale = (
                        f"deterministic fallback engaged: {reason}")
                    self.selections.append(selection)
                    self._leases[id(selection)] = lease
                    return selection
        selection.decision = SelectionDecision.DEFERRED
        selection.selection_rationale = f"deferred: {reason}"
        self.selections.append(selection)
        return selection

    def describe(self) -> dict[str, Any]:
        """Summarize the router configuration."""
        return {"registry_models": self.registry.describe()["models"],
                "coordinator": type(self.coordinator).__name__,
                "fallback_model": (f"{self.fallback_model_id}@"
                                   f"{self.fallback_model_version}"
                                   if self.fallback_model_id else None),
                "selections": len(self.selections)}
