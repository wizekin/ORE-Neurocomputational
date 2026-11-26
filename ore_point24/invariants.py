"""Phase I, Step 5 - system-wide invariants.

Mandatory invariants are modelled as predicates ``I_k(s)`` over an inspectable
system-state mapping.  A system state ``s`` is admissible only when
``AND_k I_k(s) = true``; all later phases must preserve these invariants.  The
checker never consults documentation - each predicate receives independently
evidenced facts from the registry, ledger and execution records.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .core import canonical_json, new_id, sha256_digest, utc_now_iso

INVARIANT_SCHEMA = "ore.point24.invariants.v1"

INVARIANT_IDS: tuple[str, ...] = (
    "typed_versioned_contracts",
    "producer_identity_and_lineage",
    "controlled_publication",
    "no_implicit_knowledge_promotion",
    "availability_reported_separately",
    "bounded_execution",
    "independent_revalidation",
    "optional_module_isolation",
)


@dataclass(frozen=True, slots=True)
class InvariantResult:
    """The evaluation of one invariant predicate over a system state."""

    invariant_id: str
    description: str
    satisfied: bool
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariant_id": self.invariant_id,
            "description": self.description,
            "satisfied": self.satisfied,
            "evidence": list(self.evidence),
        }


INVARIANT_DESCRIPTIONS: Mapping[str, str] = {
    "typed_versioned_contracts":
        "Every inter-module exchange uses a typed, versioned contract.",
    "producer_identity_and_lineage":
        "Every accepted artifact has producer identity and lineage.",
    "controlled_publication":
        "Shared-state mutations pass through controlled publication.",
    "no_implicit_knowledge_promotion":
        "Candidate knowledge is not promoted to authoritative knowledge by default.",
    "availability_reported_separately":
        "Module availability and runtime executability are separately reported.",
    "bounded_execution":
        "Execution has resource and iteration limits.",
    "independent_revalidation":
        "Critical results can be independently revalidated.",
    "optional_module_isolation":
        "A failure in one optional module cannot silently corrupt unrelated state.",
}


class InvariantSet:
    """Evaluate the eight mandatory invariants over evidenced system state.

    The system state mapping is supplied by callers (registry snapshot, ledger,
    execution records); predicates are pure functions of that mapping so the
    check is reproducible and independently auditable.
    """

    def __init__(self) -> None:
        self._checks: dict[str, Callable[[Mapping[str, Any]], InvariantResult]] = {
            "typed_versioned_contracts": self._check_typed_contracts,
            "producer_identity_and_lineage": self._check_lineage,
            "controlled_publication": self._check_controlled_publication,
            "no_implicit_knowledge_promotion": self._check_knowledge_promotion,
            "availability_reported_separately": self._check_availability,
            "bounded_execution": self._check_bounded_execution,
            "independent_revalidation": self._check_revalidation,
            "optional_module_isolation": self._check_isolation,
        }

    # -- individual predicates -------------------------------------------------

    @staticmethod
    def _evidence(state: Mapping[str, Any], key: str) -> Any:
        return state.get(key)

    def _check_typed_contracts(self, state: Mapping[str, Any]) -> InvariantResult:
        exchanges = state.get("module_exchanges", [])
        untyped = [
            item for item in exchanges
            if not (item.get("contract_id") and item.get("schema_version"))
        ]
        return InvariantResult(
            "typed_versioned_contracts", INVARIANT_DESCRIPTIONS["typed_versioned_contracts"],
            not untyped,
            tuple(f"exchange without typed contract: {item.get('exchange_id')}" for item in untyped[:5]),
        )

    def _check_lineage(self, state: Mapping[str, Any]) -> InvariantResult:
        artifacts = state.get("artifacts", [])
        missing = [
            item for item in artifacts
            if not (item.get("producer_id") and item.get("run_id"))
        ]
        return InvariantResult(
            "producer_identity_and_lineage", INVARIANT_DESCRIPTIONS["producer_identity_and_lineage"],
            not missing,
            tuple(f"artifact without producer/run identity: {item.get('artifact_id')}" for item in missing[:5]),
        )

    def _check_controlled_publication(self, state: Mapping[str, Any]) -> InvariantResult:
        mutations = state.get("shared_state_mutations", [])
        uncontrolled = [item for item in mutations if not item.get("through_controlled_publication")]
        return InvariantResult(
            "controlled_publication", INVARIANT_DESCRIPTIONS["controlled_publication"],
            not uncontrolled,
            tuple(f"uncontrolled mutation: {item.get('mutation_id')}" for item in uncontrolled[:5]),
        )

    def _check_knowledge_promotion(self, state: Mapping[str, Any]) -> InvariantResult:
        promotions = state.get("knowledge_promotions", [])
        implicit = [item for item in promotions if not item.get("authorized_transaction")]
        return InvariantResult(
            "no_implicit_knowledge_promotion",
            INVARIANT_DESCRIPTIONS["no_implicit_knowledge_promotion"],
            not implicit,
            tuple(f"implicit promotion: {item.get('promotion_id')}" for item in implicit[:5]),
        )

    def _check_availability(self, state: Mapping[str, Any]) -> InvariantResult:
        modules = state.get("modules", [])
        conflated = [
            item for item in modules
            if item.get("runnable") and not (
                item.get("manifest_registered") and item.get("backend_bound")
                and item.get("validated")
            )
        ]
        return InvariantResult(
            "availability_reported_separately",
            INVARIANT_DESCRIPTIONS["availability_reported_separately"],
            not conflated,
            tuple(f"runnable reported without bound+validated backend: {item.get('module_id')}"
                  for item in conflated[:5]),
        )

    def _check_bounded_execution(self, state: Mapping[str, Any]) -> InvariantResult:
        budget = state.get("execution_budget")
        satisfied = (
            isinstance(budget, Mapping)
            and budget.get("max_workers", 0) >= 1
            and budget.get("timeout_seconds") is not None
            and budget.get("timeout_seconds", 0) > 0
            and budget.get("max_iterations", 0) >= 0
        )
        return InvariantResult(
            "bounded_execution", INVARIANT_DESCRIPTIONS["bounded_execution"], satisfied,
            () if satisfied else ("execution budget missing or unbounded",),
        )

    def _check_revalidation(self, state: Mapping[str, Any]) -> InvariantResult:
        critical = state.get("critical_results", [])
        unverifiable = [
            item for item in critical
            if not item.get("independent_verification_available", True)
            and item.get("accepted", False)
        ]
        return InvariantResult(
            "independent_revalidation", INVARIANT_DESCRIPTIONS["independent_revalidation"],
            not unverifiable,
            tuple(f"accepted without revalidation path: {item.get('result_id')}"
                  for item in unverifiable[:5]),
        )

    def _check_isolation(self, state: Mapping[str, Any]) -> InvariantResult:
        failures = state.get("module_failures", [])
        corrupted = [
            item for item in failures
            if item.get("optional", True) and item.get("corrupted_unrelated_state", False)
        ]
        return InvariantResult(
            "optional_module_isolation", INVARIANT_DESCRIPTIONS["optional_module_isolation"],
            not corrupted,
            tuple(f"optional failure corrupted unrelated state: {item.get('module_id')}"
                  for item in corrupted[:5]),
        )

    # -- public API ------------------------------------------------------------

    def evaluate(self, system_state: Mapping[str, Any]) -> tuple[InvariantResult, ...]:
        """Evaluate every mandatory invariant; results keep their evidence."""
        return tuple(self._checks[invariant_id](system_state) for invariant_id in INVARIANT_IDS)

    def admissible(self, system_state: Mapping[str, Any]) -> bool:
        """A system state is admissible only when every invariant holds."""
        return all(result.satisfied for result in self.evaluate(system_state))

    def report(self, system_state: Mapping[str, Any]) -> dict[str, Any]:
        """Return a JSON-compatible invariant report with a stable digest."""
        results = self.evaluate(system_state)
        payload = {
            "schema": INVARIANT_SCHEMA,
            "report_id": new_id("point24_invariants"),
            "evaluated_at": utc_now_iso(),
            "admissible": all(result.satisfied for result in results),
            "results": [result.to_dict() for result in results],
        }
        payload["digest"] = sha256_digest(payload)
        return payload


__all__ = [
    "INVARIANT_DESCRIPTIONS",
    "INVARIANT_IDS",
    "INVARIANT_SCHEMA",
    "InvariantResult",
    "InvariantSet",
]
