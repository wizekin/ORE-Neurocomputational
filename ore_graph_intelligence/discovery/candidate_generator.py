"""Typed candidate generation over a USG graph projection (ORE Point 8 steps 25-26, 29).

Query anchors resolve to stable node ids only through USG identity (label/alias sets);
ambiguous anchors abstain rather than choosing by nearest embedding. Candidate pairs are
typed *before* neural scoring: subject/object kind, relation domain/range, direction, and
cardinality are enforced from the schema, and type-invalid candidates are hard-rejected
pre-ranking with an explicit reason. Duplicates, inverse duplicates, contradictions, and
previously rejected candidates are filtered with recorded reasons.
"""
from __future__ import annotations

from typing import Iterable, Mapping

from ..projection.typed_graph import TypedGraphSnapshot
from ..schemas.candidate_schema import (
    CandidateRejectionReason,
    CandidateStatus,
    GraphCandidate,
    GraphQuery,
    GraphQueryAnchor,
    QueryAnchorResolution,
)

RejectionLedger = Mapping[tuple[str, str, str], str]


class CandidateGenerator:
    """Generate schema-constrained relation hypotheses for one query (steps 25-26)."""

    def __init__(
        self,
        snapshot: TypedGraphSnapshot,
        *,
        previously_rejected: RejectionLedger | None = None,
    ) -> None:
        self.snapshot = snapshot
        self.schema = snapshot.schema
        self.node_by_id = {node.node_id: node for node in snapshot.nodes}
        self.existing = snapshot.existing_triples()
        self.previously_rejected = dict(previously_rejected or {})

    def resolve_anchor(self, anchor: GraphQueryAnchor) -> QueryAnchorResolution:
        """Resolve one anchor to stable node ids; ambiguity abstains (step 25)."""
        if anchor.node_id is not None:
            if anchor.node_id in self.node_by_id:
                return QueryAnchorResolution((anchor.node_id,))
            return QueryAnchorResolution((), ambiguous_labels=())
        label = (anchor.label or "").strip()
        matches = self.snapshot.node_id_by_alias(label)
        if not matches:
            return QueryAnchorResolution((), ambiguous_labels=())
        if len(matches) > 1:
            return QueryAnchorResolution(matches, ambiguous_labels=(label,))
        return QueryAnchorResolution(matches)

    def generate(
        self,
        query: GraphQuery,
        *,
        candidate_objects: Iterable[str] | None = None,
    ) -> tuple[list[GraphCandidate], list[GraphCandidate]]:
        """Return (kept, hard-rejected) candidates for one query under its budget.

        Every kept candidate is schema-typed with provenance fields populated; rejected
        candidates carry ``rejected_pre_rank`` status and a rejection reason. Generation
        order is deterministic (sorted node ids) so results are reproducible.
        """
        resolution = self.resolve_anchor(query.anchor)
        if resolution.abstained or not resolution.node_ids:
            return [], []
        kept: list[GraphCandidate] = []
        rejected: list[GraphCandidate] = []
        relations = (
            tuple(query.relation_types) if query.relation_types
            else self.snapshot.relation_types()
        )
        objects = candidate_objects
        if objects is None:
            objects = [node.node_id for node in sorted(
                self.snapshot.nodes, key=lambda item: item.node_id,
            )]
        budget = query.budget_candidates
        for subject_id in sorted(resolution.node_ids):
            for relation_type in sorted(relations):
                signature = self.schema.signature(relation_type)
                subject_kind = self.node_by_id[subject_id].node_kind
                if signature is not None and not signature.domain_allows(subject_kind):
                    continue
                for object_id in sorted(objects):
                    object_kind = self.node_by_id[object_id].node_kind
                    candidate = GraphCandidate(
                        subject_id=subject_id,
                        relation_type=relation_type,
                        object_id=object_id,
                        status=CandidateStatus.PROPOSED,
                        model_version="candidate_generator/1.0.0",
                        snapshot_hash=self.snapshot.graph_hash,
                    )
                    reason = self._rejection_reason(candidate, signature, subject_kind, object_kind)
                    if reason is not None:
                        candidate.status = CandidateStatus.REJECTED_PRE_RANK
                        candidate.rejection_reasons = (reason,)
                        rejected.append(candidate)
                        continue
                    kept.append(candidate)
                    if len(kept) >= budget:
                        return kept, rejected
        return kept, rejected

    def _rejection_reason(
        self,
        candidate: GraphCandidate,
        signature,
        subject_kind: str,
        object_kind: str,
    ) -> str | None:
        """Hard semantic violations reject before ranking; they are not score penalties."""
        if signature is not None:
            if not signature.range_allows(object_kind):
                return f"{CandidateRejectionReason.TYPE_VIOLATION}:object:{object_kind}"
            if signature.direction == "DIRECTED" and candidate.subject_id == candidate.object_id \
                    and signature.cardinality == "one_to_many":
                return CandidateRejectionReason.CARDINALITY_VIOLATION
            required_roles = signature.required_roles()
            if required_roles:
                return f"{CandidateRejectionReason.TYPE_VIOLATION}:missing_roles:{','.join(required_roles)}"
        triple = candidate.triple()
        if triple in self.existing:
            return CandidateRejectionReason.DUPLICATE
        if signature is not None and signature.inverse_of is not None:
            inverted = (candidate.object_id, signature.inverse_of, candidate.subject_id)
            if inverted in self.existing and not signature.symmetric:
                return CandidateRejectionReason.INVERSE_DUPLICATE
        if triple in self.previously_rejected:
            return CandidateRejectionReason.PREVIOUSLY_REJECTED
        return None


class GraphCandidateService:
    """Bound candidate proposal service exposing the Point 8 contract (step 32 helper).

    ``propose(query, budget)`` returns hypotheses only: no authoritative USG writes are
    possible from this service by construction.
    """

    def __init__(
        self,
        snapshot: TypedGraphSnapshot,
        *,
        previously_rejected: RejectionLedger | None = None,
    ) -> None:
        self.generator = CandidateGenerator(snapshot, previously_rejected=previously_rejected)
        self.snapshot = snapshot

    def propose(
        self,
        query: GraphQuery,
        *,
        budget: int | None = None,
    ) -> tuple[list[GraphCandidate], list[GraphCandidate]]:
        """Propose typed candidates for one bounded query; hypotheses only."""
        if budget is not None:
            if budget < 1:
                raise ValueError("budget must be positive")
            query = GraphQuery(
                anchor=query.anchor,
                relation_types=query.relation_types,
                target_node_id=query.target_node_id,
                task_hint=query.task_hint,
                budget_candidates=min(query.budget_candidates, budget),
                neighborhood_depth=query.neighborhood_depth,
            )
        return self.generator.generate(query)

    def anchor_resolution(self, anchor: GraphQueryAnchor) -> QueryAnchorResolution:
        """Expose anchor resolution for abstention reporting."""
        return self.generator.resolve_anchor(anchor)


__all__ = ["CandidateGenerator", "GraphCandidateService", "RejectionLedger"]
