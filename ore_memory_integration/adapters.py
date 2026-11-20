"""Module-specific handoff builders for ORE's existing cognitive components."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ore_workspace.working_memory import WorkingMemoryPolicy

from .contracts import MemoryEvidence, MemoryIntegrationConfig


_VERIFIED = {"accepted", "verified"}


def _content(item: MemoryEvidence) -> dict[str, Any]:
    """Expose content as a mapping while preserving scalar claims under a value key."""
    if isinstance(item.content, Mapping):
        return dict(item.content)
    return {"value": item.content}


class WorkingMemoryAdapter:
    """Translate evidence into bounded working-memory references and accepted semantic context."""

    def __init__(self, config: MemoryIntegrationConfig | None = None) -> None:
        """Configure retention capacity while reusing the shared workspace's policy."""
        self.config = config or MemoryIntegrationConfig()
        self.policy = WorkingMemoryPolicy(capacity=self.config.max_active_memories)

    def select(self, evidence: Sequence[MemoryEvidence], goal: str, query_text: str) -> tuple[MemoryEvidence, ...]:
        """Keep the highest weighted goal/recency/context/unresolved references without copying the store."""
        rows: dict[str, dict[str, Any]] = {}
        for item in evidence:
            context = item.context
            rows[item.memory_id] = {
                "is_active": True,
                "retention_factors": {
                    "goal": item.relevance_score if goal else 0.0,
                    "recency": float(context.get("recency_score", 0.5)),
                    "context": max(item.relevance_score, float(context.get("context_score", 0.0))),
                    "unresolved": float(context.get("unresolved_importance", 0.0)),
                },
            }
        selected = set(self.policy.select(rows))
        return tuple(item for item in evidence if item.memory_id in selected)

    def to_context(self, evidence: Sequence[MemoryEvidence]) -> dict[str, Any]:
        """Build WM request context with references and verified-only semantic knowledge."""
        accepted: dict[str, Any] = {}
        pending: list[dict[str, Any]] = []
        episodic: list[dict[str, Any]] = []
        references: list[dict[str, Any]] = []
        for item in evidence:
            references.append({
                "memory_id": item.memory_id,
                "memory_type": item.memory_type,
                "relevance_score": item.relevance_score,
                "confidence": item.confidence,
                "verification_status": item.verification_status,
                "source_reference": item.source_reference,
                "independence_group": item.independence_group,
            })
            if item.memory_type == "semantic":
                if item.verification_status.casefold() in _VERIFIED:
                    accepted[item.memory_id] = {
                        **_content(item),
                        "memory_id": item.memory_id,
                        "verification_status": item.verification_status,
                        "confidence": item.confidence,
                        "provenance": dict(item.provenance),
                    }
                else:
                    pending.append(item.to_dict())
            elif item.memory_type == "episodic":
                episodic.append(item.to_dict())
        return {
            "semantic_knowledge": accepted,
            "pending_semantic_candidates": pending,
            "episodic_memory_evidence": episodic,
            "retrieved_memory_references": references,
            "memory_evidence_count": len(evidence),
        }


class SemanticMemoryAdapter:
    """Separate accepted semantic facts from claims that still require verification."""

    def to_payload(self, evidence: Sequence[MemoryEvidence]) -> dict[str, Any]:
        """Return verified claims and pending candidates in separate non-interchangeable fields."""
        return {
            "accepted_knowledge": [
                item.to_dict() for item in evidence
                if item.memory_type == "semantic" and item.verification_status.casefold() in _VERIFIED
            ],
            "pending_verification": [
                item.to_dict() for item in evidence
                if item.memory_type == "semantic" and item.verification_status.casefold() not in _VERIFIED
            ],
        }


class SchemaMemoryAdapter:
    """Convert episode records into source-labelled examples for schema formation."""

    def to_payload(self, evidence: Sequence[MemoryEvidence]) -> dict[str, Any]:
        """Expose recurring event structures as learning examples, never as universal rules."""
        return {
            "episode_examples": [
                {
                    "episode_id": item.memory_id,
                    "event": _content(item),
                    "entities": dict(item.entities),
                    "relations": list(item.relations),
                    "context": dict(item.context),
                    "source_reference": item.source_reference,
                    "independence_group": item.independence_group,
                    "confidence": item.confidence,
                    "evidence_status": "retrieved_episode",
                }
                for item in evidence if item.memory_type == "episodic"
            ],
            "learning_authorization_required": True,
        }


class SchemaGuidedInferenceAdapter:
    """Supply existing schemas and episodic analogues as distinct inference inputs."""

    def to_payload(self, evidence: Sequence[MemoryEvidence]) -> dict[str, Any]:
        """Separate semantic priors from retrieved event examples and preserve their origins."""
        return {
            "semantic_priors": [
                item.to_dict() for item in evidence
                if item.memory_type == "semantic" and item.verification_status.casefold() in _VERIFIED
            ],
            "episodic_examples": [
                item.to_dict() for item in evidence if item.memory_type == "episodic"
            ],
            "pending_claims_excluded_from_priors": [
                item.memory_id for item in evidence
                if item.memory_type == "semantic" and item.verification_status.casefold() not in _VERIFIED
            ],
        }


class AnalogicalMemoryAdapter:
    """Prepare relational situation evidence for the existing analogical-transfer processor."""

    def to_payload(self, evidence: Sequence[MemoryEvidence]) -> dict[str, Any]:
        """Preserve stored relation direction and source IDs without inventing mappings or causality."""
        return {
            "source_situations": [
                {
                    "case_id": item.memory_id,
                    "entities": dict(item.entities),
                    "relations": list(item.relations),
                    "domain": item.context.get("domain", "unspecified"),
                    "source_reference": item.source_reference,
                    "independence_group": item.independence_group,
                }
                for item in evidence
                if item.memory_type == "episodic" and item.relations
            ],
            "transfer_is_hypothetical_until_target_validation": True,
        }


class ReconstructiveMemoryAdapter:
    """Supply episode fragments and source metadata to the Point 20 reconstruction boundary."""

    def to_payload(self, evidence: Sequence[MemoryEvidence]) -> dict[str, Any]:
        """Mark every retrieved detail as episodic evidence and leave missing fields unknown."""
        fragments = []
        for item in evidence:
            if item.memory_type != "episodic":
                continue
            content = _content(item)
            for index, (feature, value) in enumerate(content.items()):
                fragments.append({
                    "detail_id": f"{item.memory_id}:{feature}:{index}",
                    "feature_type": str(feature),
                    "content": value,
                    "status": "episodically_retrieved",
                    "source_type": "episodic_retrieval",
                    "source_id": item.memory_id,
                    "episode_id": item.memory_id,
                    "retrieval_strength": item.relevance_score,
                    "recollection": item.confidence,
                    "provenance_ids": [item.memory_id, *item.parent_memory_ids],
                    "independence_group": item.independence_group,
                })
        return {
            "episodic_fragments": fragments,
            "source_history": [
                {
                    "source_id": item.memory_id,
                    "source_type": "episodic_retrieval" if item.memory_type == "episodic" else "reported",
                    "detail_id": item.memory_id,
                    "content": item.content,
                    "reliability": item.confidence,
                    "independence_group": item.independence_group,
                    "parent_source_ids": list(item.parent_memory_ids),
                }
                for item in evidence
            ],
            "unprovided_details_remain_unknown": True,
        }
