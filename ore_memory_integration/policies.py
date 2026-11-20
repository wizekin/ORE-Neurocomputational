"""Evidence-quality, source-independence, encoding, and retention policies."""
from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

from ore_workspace.working_memory import WorkingMemoryPolicy

from .contracts import MemoryEvidence, MemoryIntegrationConfig


def _clip(value: float) -> float:
    """Clamp one finite model score to the unit interval."""
    if not math.isfinite(float(value)):
        raise ValueError("score must be finite")
    return max(0.0, min(1.0, float(value)))


def _text(value: Any) -> str:
    """Flatten a supplied value into deterministic text for simple overlap features."""
    if isinstance(value, Mapping):
        return " ".join(f"{key} {_text(item)}" for key, item in value.items())
    if isinstance(value, (tuple, list, set)):
        return " ".join(_text(item) for item in value)
    return "" if value is None else str(value)


def _tokens(value: Any) -> set[str]:
    """Return normalized tokens for transparent, non-embedding feature comparisons."""
    return {
        item.casefold().strip(".,!?;:()[]{}\"'")
        for item in _text(value).split()
        if item.strip(".,!?;:()[]{}\"'")
    }


def _jaccard(left: Any, right: Any) -> float:
    """Compute J(A,B)=|A∩B|/|A∪B|, returning zero when either cue set is empty."""
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a and b else 0.0


class SourceIndependencePolicy:
    """Prevent derived results with a shared origin from being counted as independent support."""

    def group(self, evidence: Iterable[MemoryEvidence]) -> dict[str, list[MemoryEvidence]]:
        """Group memory records by their declared origin or independence group."""
        result: dict[str, list[MemoryEvidence]] = defaultdict(list)
        for item in evidence:
            result[item.independence_group].append(item)
        return dict(result)

    def independent_representatives(
        self, evidence: Iterable[MemoryEvidence],
    ) -> tuple[MemoryEvidence, ...]:
        """Choose one strongest representative per source group instead of summing duplicates."""
        representatives: dict[str, MemoryEvidence] = {}
        for item in evidence:
            current = representatives.get(item.independence_group)
            candidate_strength = item.relevance_score * item.confidence
            current_strength = current.relevance_score * current.confidence if current else -1.0
            if current is None or candidate_strength > current_strength:
                representatives[item.independence_group] = item
        return tuple(sorted(representatives.values(), key=lambda item: item.memory_id))


class EvidenceValidationPolicy:
    """Score an interpretation from independent support, conflict, and unresolved evidence."""

    def __init__(self, config: MemoryIntegrationConfig | None = None) -> None:
        """Retain explicit penalties and source grouping settings for repeatable evaluation."""
        self.config = config or MemoryIntegrationConfig()
        self.independence = SourceIndependencePolicy()

    def score_hypothesis(
        self,
        contributions: Sequence[Mapping[str, Any]],
        *,
        unresolved_count: int = 0,
        conflict_count: int = 0,
    ) -> dict[str, Any]:
        """Compute S(h)=sum_g w_g E_g(h)-lambda*C(h)-mu*U(h) over unique evidence groups."""
        if unresolved_count < 0 or conflict_count < 0:
            raise ValueError("conflict and unresolved counts cannot be negative")
        grouped: dict[str, dict[str, Any]] = {}
        for index, contribution in enumerate(contributions):
            group = str(contribution.get("independence_group") or contribution.get("source_id") or f"source-{index}")
            support = float(contribution.get("support", 0.0))
            reliability = _clip(float(contribution.get("reliability", 0.5)))
            if not math.isfinite(support) or not -1.0 <= support <= 1.0:
                raise ValueError("support must be finite and in [-1, 1]")
            weighted = support * reliability
            previous = grouped.get(group)
            if previous is None or abs(weighted) > abs(previous["weighted_support"]):
                grouped[group] = {
                    "independence_group": group,
                    "support": support,
                    "reliability": reliability,
                    "weighted_support": weighted,
                }
        support_total = sum(item["weighted_support"] for item in grouped.values())
        score = (
            support_total
            - self.config.evidence_conflict_penalty * conflict_count
            - self.config.unresolved_penalty * unresolved_count
        )
        return {
            "score": score,
            "independent_group_count": len(grouped),
            "support_total": support_total,
            "conflict_penalty": self.config.evidence_conflict_penalty * conflict_count,
            "unresolved_penalty": self.config.unresolved_penalty * unresolved_count,
            "groups": [grouped[key] for key in sorted(grouped)],
            "selected": False,
            "selection_note": "A score alone does not select a hypothesis; compare alternatives and task-specific thresholds.",
        }


class MemoryRelevancePolicy:
    """Prioritize references for working memory using the existing workspace retention equation."""

    def __init__(
        self,
        config: MemoryIntegrationConfig | None = None,
        policy: WorkingMemoryPolicy | None = None,
    ) -> None:
        """Compose existing goal/recency/context/unresolved weights with an explicit capacity."""
        self.config = config or MemoryIntegrationConfig()
        self.policy = policy or WorkingMemoryPolicy(capacity=self.config.max_active_memories)

    def select(
        self,
        evidence: Sequence[MemoryEvidence],
        *,
        goal: str,
        query_text: str,
    ) -> tuple[tuple[str, float], ...]:
        """Rank evidence with P_i=(.35G_i+.25R_i+.25C_i+.15U_i)/sum(weights)."""
        items: dict[str, dict[str, Any]] = {}
        for item in evidence:
            content = item.content
            goal_score = _jaccard(goal, content)
            context_score = max(_jaccard(query_text, content), item.relevance_score)
            recency = _clip(float(item.context.get("recency_score", 0.5)))
            unresolved = _clip(float(item.context.get("unresolved_importance", 0.0)))
            items[item.memory_id] = {
                "is_active": True,
                "retention_factors": {
                    "goal": goal_score,
                    "recency": recency,
                    "context": context_score,
                    "unresolved": unresolved,
                },
            }
        return self.policy.prioritize(items)[: self.config.max_active_memories]


class MemoryEncodingPolicy:
    """Require an explicit post-recall learning decision before writing a new episodic trace."""

    _SUPPORTED_SOURCES = {"explicit", "entailed", "reported", "witnessed", "observed"}

    def evaluate(self, *, requested: bool, payload: Mapping[str, Any] | None) -> dict[str, Any]:
        """Allow encoding only for an explicit, provenance-bearing interpretation payload."""
        if not requested:
            return {"allowed": False, "reason": "encoding_not_requested", "write_performed": False}
        if not isinstance(payload, Mapping) or not payload:
            return {"allowed": False, "reason": "missing_episode_payload", "write_performed": False}
        provenance = payload.get("source_provenance", ())
        if isinstance(provenance, str):
            provenance = (provenance,)
        sources = {str(item).casefold() for item in provenance}
        confidence = _clip(float(payload.get("confidence", 0.0)))
        if not sources or not sources.issubset(self._SUPPORTED_SOURCES):
            return {"allowed": False, "reason": "source_is_not_direct_or_supported", "write_performed": False}
        if confidence <= 0.0:
            return {"allowed": False, "reason": "missing_positive_confidence", "write_performed": False}
        if payload.get("unresolved", False) or payload.get("hypothetical", False):
            return {"allowed": False, "reason": "unresolved_or_hypothetical_content", "write_performed": False}
        return {"allowed": True, "reason": "explicit_supported_learning_request", "write_performed": False}
