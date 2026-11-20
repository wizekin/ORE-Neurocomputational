"""Expected-information-gain scheduling and duplicate-action protection."""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping

from .contracts import ClarificationOption, CorrectionRequest
from .functional import expected_information_gain
from .recurrent_contracts import FeedbackAction, RecurrentFeedbackConfig


_ACTION_MODULES = {
    "infer": "cognitive_heuristics",
    "clarify": "feedback_seeking",
    "check": "memory_retrieval",
    "test": "semantic_reprocessing",
}


def _fingerprint(value: Any) -> str:
    """Hash JSON-like action context deterministically for repeat-attempt protection."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str, ensure_ascii=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


class FeedbackActionScheduler:
    """Rank action options by expected information gain minus cost under one episode budget."""

    def __init__(self, config: RecurrentFeedbackConfig | None = None) -> None:
        """Initialize per-episode action-attempt counts and shared resource accounting."""
        self.config = config or RecurrentFeedbackConfig()
        self._attempts: dict[str, int] = {}
        self._history: list[dict[str, Any]] = []
        self._spent_cost = 0.0
        self._total_actions = 0

    @property
    def spent_cost(self) -> float:
        """Return the cost charged to the shared feedback episode."""
        return self._spent_cost

    @property
    def total_actions(self) -> int:
        """Return the number of actions dispatched across all participating modules."""
        return self._total_actions

    def action_history(self) -> tuple[dict[str, Any], ...]:
        """Return detached attempt records for the feedback episode."""
        return tuple(dict(item) for item in self._history)

    def _action_signature(
        self,
        option: ClarificationOption,
        *,
        target: str,
        context: Mapping[str, Any],
        evidence_ids: tuple[str, ...],
    ) -> tuple[str, str]:
        """Fingerprint module, action, target, context, and evidence-set identity."""
        module = _ACTION_MODULES[option.kind]
        context_hash = _fingerprint(context)
        evidence_hash = _fingerprint(sorted(evidence_ids))
        signature = _fingerprint({
            "module_id": module,
            "action_id": option.action_id,
            "action_type": option.kind,
            "action_description": option.description,
            "target_id": target,
            "context_hash": context_hash,
            "evidence_set_hash": evidence_hash,
        })
        input_hash = _fingerprint({
            "action_id": option.action_id,
            "target": target,
            "context_hash": context_hash,
            "evidence_set_hash": evidence_hash,
        })
        return signature, input_hash

    def rank(
        self,
        request: CorrectionRequest,
        probabilities: Mapping[str, float],
        *,
        target: str,
        context: Mapping[str, Any],
        evidence_ids: tuple[str, ...],
        remaining_cost: float | None = None,
    ) -> tuple[FeedbackAction, ...]:
        """Build eligible actions using EIG(a)=H(H)-E_r[H(H|r,a)] and cost-weighted utility."""
        if self._total_actions >= self.config.max_total_actions:
            return ()
        budget = self.config.max_total_action_cost - self._spent_cost
        if remaining_cost is not None:
            budget = min(budget, max(0.0, float(remaining_cost)))
        scored: list[tuple[float, str, FeedbackAction]] = []
        for option in request.clarification_options:
            module = _ACTION_MODULES[option.kind]
            signature, input_hash = self._action_signature(
                option, target=target, context=context, evidence_ids=evidence_ids,
            )
            attempts = self._attempts.get(signature, 0)
            if attempts >= self.config.max_attempts_per_action_signature:
                continue
            if option.cost > budget:
                continue
            gain = expected_information_gain(probabilities, option)
            utility = gain - self.config.action_cost_weight * option.cost
            if utility <= 0.0:
                continue
            action = FeedbackAction(
                action_id=option.action_id,
                responsible_module=module,
                action_type=option.kind,
                target=target,
                purpose=option.description,
                expected_information_gain=gain,
                estimated_cost=option.cost,
                input_fingerprint=input_hash,
                attempt_number=attempts + 1,
                action_fingerprint=signature,
            )
            scored.append((utility, option.action_id, action))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return tuple(item[2] for item in scored)

    def select(
        self,
        request: CorrectionRequest,
        probabilities: Mapping[str, float],
        *,
        target: str,
        context: Mapping[str, Any],
        evidence_ids: tuple[str, ...],
    ) -> FeedbackAction | None:
        """Select the highest-utility action still within the shared action and cost budgets."""
        ranked = self.rank(
            request, probabilities, target=target, context=context, evidence_ids=evidence_ids,
        )
        return ranked[0] if ranked else None

    def record_attempt(self, action: FeedbackAction, result_status: str) -> FeedbackAction:
        """Charge one shared action/cost unit and remember its outcome under its exact fingerprint."""
        status = str(result_status).strip().casefold()
        if not status:
            raise ValueError("result_status must be non-empty")
        self._attempts[action.action_fingerprint] = self._attempts.get(action.action_fingerprint, 0) + 1
        self._total_actions += 1
        self._spent_cost += action.estimated_cost
        updated = FeedbackAction(
            action_id=action.action_id,
            responsible_module=action.responsible_module,
            action_type=action.action_type,
            target=action.target,
            purpose=action.purpose,
            expected_information_gain=action.expected_information_gain,
            estimated_cost=action.estimated_cost,
            input_fingerprint=action.input_fingerprint,
            attempt_number=action.attempt_number,
            result_status=status,
            action_fingerprint=action.action_fingerprint,
        )
        self._history.append(updated.to_dict())
        return updated

    def diagnostics(self) -> dict[str, Any]:
        """Report bounded resource usage and why identical unsuccessful attempts cannot recur."""
        return {
            "total_actions": self._total_actions,
            "max_total_actions": self.config.max_total_actions,
            "spent_cost": self._spent_cost,
            "remaining_cost": max(0.0, self.config.max_total_action_cost - self._spent_cost),
            "distinct_action_signatures": len(self._attempts),
            "attempts_per_signature": dict(self._attempts),
            "failed_attempts_are_remembered": True,
            "new_context_or_evidence_changes_action_fingerprint": True,
        }
