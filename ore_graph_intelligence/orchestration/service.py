"""GraphIntelligenceCoordinator: the Point 8 end-to-end service contract (steps 1-56).

``execute(task) -> GraphIntelligenceResult`` runs the integrated 56-step algorithm:

Phase A (1-12)   bounded task, eligibility, read-only typed projection with provenance
                 and integrity, partitioning/leakage-safe split bookkeeping.
Phase B (13-24)  deterministic baselines plus optional trained embeddings and R-GCN
                 scorers selected under resource admission.
Phase C (25-32)  anchor resolution, typed candidate generation, complementary scoring,
                 explanation neighborhoods, utility ranking, abstention, hypothesis
                 publication into a dedicated workspace namespace.
Phase D (33-40)  symbolic type/rule/evidence/counter-evidence verification, promotion
                 gate, disposition publication and full lifecycle event logging.
Phase E (41-48)  isolated NAR advisory proposals with deterministic checks.
Phase F (49-56)  path selection, CROS admission, bounded caching, monitoring metrics,
                 shadow/registry integration and cycle closure.

Core invariant (enforced structurally): learned scores guide search; they never
establish truth; the prediction service cannot write authoritative USG state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from ..discovery.abstention import AbstentionDecision, AbstentionPolicy
from ..discovery.candidate_generator import GraphCandidateService
from ..discovery.explainer import NeighborhoodExplainer
from ..discovery.link_predictor import ComplementaryScorer, LinkPredictor
from ..discovery.ranker import UtilityRanker
from ..embeddings.trainer import EmbeddingScorer, TrainedEmbedding
from ..gnn.inference import (
    DeterministicGraphBaselines,
    RGCNLinkPredictor,
    RelationGraph,
    build_snapshot_features,
)
from ..nar.algorithmic_gnn import AlgorithmicProposal, NeuralAlgorithmicService
from ..nar.task_generators import NarrowTask
from ..observability.events import GraphEventRecorder
from ..projection.integrity import IntegrityReport, IntegrityValidator
from ..projection.typed_graph import TypedGraphSnapshot
from ..projection.usg_snapshot import USGGraphProjector
from ..schemas.candidate_schema import (
    GraphCandidate,
    GraphQuery,
    new_hypothesis_namespace,
)
from ..verification.promotion_gate import (
    AuthorizedUpdateProposal,
    CandidateVerifier,
    PromotionGate,
    VerificationResult,
)
from .cache import CacheKey, GraphResultCache
from .resource_estimator import (
    AdmissionDecision,
    AdaptiveResourceCoordinator,
    GraphResourceEstimator,
    ResourceBudget,
    ResourceEstimate,
)

#: Paths the coordinator can select between (step 49).
COORDINATOR_PATHS = (
    "exact_lookup",
    "learned_graph_intelligence",
    "nar_advisory_only",
    "rejected_by_resources",
)

#: Workspace finding type for hypotheses; distinct from verified facts (step 32, test 14).
HYPOTHESIS_FINDING_TYPE = "graph_hypothesis"

#: Workspace finding type for verification outcomes (step 39).
VERIFICATION_FINDING_TYPE = "graph_verification"


def _correlation_id() -> str:
    return f"graph-cycle:{uuid4().hex[:20]}"


class CancelToken:
    """Cooperative cancellation flag the coordinator checks between phases."""

    def __init__(self) -> None:
        self._cancelled = False

    def cancel(self) -> None:
        """Request cancellation; the running cycle stops at the next phase boundary."""
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        """Whether cancellation was requested."""
        return self._cancelled


@dataclass
class GraphIntelligenceTask:
    """Bounded task submitted by the Main Brain Coordinator (step 1)."""

    task_type: str  # exact_lookup | link_discovery
    query: GraphQuery
    bundle: Mapping[str, Any]
    deadline_ms: float = 500.0
    budget: ResourceBudget = field(default_factory=ResourceBudget)
    learned_inference_permitted: bool = True
    nar_task: NarrowTask | None = None
    cancel_token: CancelToken | None = None
    correlation_id: str = field(default_factory=_correlation_id)

    def __post_init__(self) -> None:
        if self.task_type not in {"exact_lookup", "link_discovery"}:
            raise ValueError("task_type must be exact_lookup or link_discovery")
        if self.deadline_ms <= 0:
            raise ValueError("deadline_ms must be positive")


@dataclass
class GraphIntelligenceResult:
    """Full outcome of one coordinated graph intelligence cycle (steps 39, 56)."""

    correlation_id: str
    task_type: str
    selected_path: str
    admission: AdmissionDecision | None = None
    snapshot: TypedGraphSnapshot | None = None
    integrity: IntegrityReport | None = None
    resource_estimate: ResourceEstimate | None = None
    exact_lookup: tuple[str, ...] = ()
    candidates: list[GraphCandidate] = field(default_factory=list)
    rejected: list[GraphCandidate] = field(default_factory=list)
    abstention: AbstentionDecision | None = None
    verification: list[VerificationResult] = field(default_factory=list)
    update_proposals: list[AuthorizedUpdateProposal] = field(default_factory=list)
    nar_proposal: AlgorithmicProposal | None = None
    workspace_publications: list[dict[str, object]] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)
    cache_hits: int = 0
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "correlation_id": self.correlation_id, "task_type": self.task_type,
            "selected_path": self.selected_path,
            "admission": self.admission.as_dict() if self.admission else None,
            "snapshot_hash": self.snapshot.graph_hash if self.snapshot else None,
            "integrity_passed": self.integrity.passed if self.integrity else None,
            "quarantine": [
                {"record_id": q.record_id, "reason": q.reason}
                for q in (self.snapshot.quarantine if self.snapshot else ())
            ],
            "exact_lookup": list(self.exact_lookup),
            "candidates": [c.as_payload() for c in self.candidates],
            "rejected": [c.as_payload() for c in self.rejected],
            "abstention": self.abstention.as_dict() if self.abstention else None,
            "verification": [v.as_dict() for v in self.verification],
            "update_proposals": [p.as_dict() for p in self.update_proposals],
            "nar_proposal": self.nar_proposal.as_dict() if self.nar_proposal else None,
            "workspace_publications": list(self.workspace_publications),
            "metrics": dict(self.metrics), "cache_hits": self.cache_hits,
            "errors": list(self.errors), "cancelled": self.cancelled,
        }


class GraphIntelligenceCoordinator:
    """Select services under budget and run the integrated Point 8 pipeline."""

    def __init__(
        self,
        *,
        projector: USGGraphProjector | None = None,
        estimator: GraphResourceEstimator | None = None,
        resource_coordinator: AdaptiveResourceCoordinator | None = None,
        cache: GraphResultCache | None = None,
        event_recorder: GraphEventRecorder | None = None,
        embedding: TrainedEmbedding | None = None,
        embedding_scorer: EmbeddingScorer | None = None,
        gnn_predictor: RGCNLinkPredictor | None = None,
        nar_service: NeuralAlgorithmicService | None = None,
        workspace_publisher: Any | None = None,
        abstention_policy: AbstentionPolicy | None = None,
        model_version: str = "coordinator/1.0.0",
    ) -> None:
        self.projector = projector or USGGraphProjector()
        self.estimator = estimator or GraphResourceEstimator()
        self.resource_coordinator = resource_coordinator or AdaptiveResourceCoordinator()
        self.cache = cache or GraphResultCache()
        self.event_recorder = event_recorder or GraphEventRecorder()
        self.embedding = embedding
        self.embedding_scorer = embedding_scorer or (
            embedding.scorer() if embedding is not None else None
        )
        self.gnn_predictor = gnn_predictor
        self.nar_service = nar_service
        self.workspace_publisher = workspace_publisher
        self.abstention_policy = abstention_policy or AbstentionPolicy()
        self.model_version = model_version
        self._gnn_context: tuple[Any, Any, Mapping[str, int], Mapping[str, int]] | None = None

    # ------------------------------------------------------------------ contract

    def execute(self, task: GraphIntelligenceTask) -> GraphIntelligenceResult:
        """Run the integrated 56-step cycle for one bounded task."""
        correlation = task.correlation_id
        recorder = self.event_recorder
        recorder.emit("task_received", correlation, {
            "task_type": task.task_type,
            "query": {
                "anchor": task.query.anchor.label or task.query.anchor.node_id,
                "relation_types": list(task.query.relation_types),
                "budget_candidates": task.query.budget_candidates,
            },
            "deadline_ms": task.deadline_ms,
            "learned_inference_permitted": task.learned_inference_permitted,
        })
        result = GraphIntelligenceResult(
            correlation_id=correlation, task_type=task.task_type,
            selected_path="exact_lookup",
        )
        try:
            self._run_cycle(task, result)
            recorder.emit("cycle_closed", correlation, {
                "selected_path": result.selected_path,
                "candidates": len(result.candidates),
                "verified": len(result.verification),
                "update_proposals": len(result.update_proposals),
                "errors": list(result.errors),
            })
        except Exception as error:  # noqa: BLE001 - never block reporting
            message = f"{type(error).__name__}: {error}"
            result.errors.append(message)
            recorder.emit("error", correlation, {"message": message})
        if recorder.run_id:
            recorder.finish("completed" if not result.errors else "completed_with_errors")
        return result

    # ------------------------------------------------------------------ cycle

    def _cancelled(self, task: GraphIntelligenceTask) -> bool:
        """Cooperative cancellation check (contract requirement)."""
        return task.cancel_token is not None and task.cancel_token.cancelled

    def _run_cycle(
        self, task: GraphIntelligenceTask, result: GraphIntelligenceResult,
    ) -> None:
        correlation = task.correlation_id
        recorder = self.event_recorder

        # Steps 1-2: eligibility - exact rules serve simple queries (step 2).
        eligible_for_learning = (
            task.task_type == "link_discovery" and task.learned_inference_permitted
        )
        selected_path = (
            "learned_graph_intelligence" if eligible_for_learning else "exact_lookup"
        )
        recorder.emit("eligibility_decided", correlation, {
            "eligible_for_learning": eligible_for_learning,
        })

        # Steps 3-9: read-only typed projection with provenance and integrity.
        snapshot = self.projector.project(task.bundle)
        result.snapshot = snapshot
        recorder.emit("snapshot_projected", correlation, {
            "snapshot_hash": snapshot.graph_hash,
            "schema_version": snapshot.schema.schema_version,
            "nodes": snapshot.node_count, "edges": snapshot.edge_count,
            "relation_types": list(snapshot.relation_types()),
        })
        integrity = IntegrityValidator().validate(snapshot)
        result.integrity = integrity
        recorder.emit("integrity_validated", correlation, {
            "passed": integrity.passed,
            "quarantine": [
                {"record_id": q.record_id, "reason": q.reason}
                for q in snapshot.quarantine
            ],
        })
        if self._check_cancel(task, result):
            return

        # Steps 49-50: path selection and CROS admission.
        estimate = self.estimator.estimate(snapshot, task.query)
        result.resource_estimate = estimate
        admission = self.resource_coordinator.admit(estimate)
        result.admission = admission
        recorder.emit("admission_decided", correlation, admission.as_dict())
        if admission.decision == "reject":
            result.selected_path = "rejected_by_resources"
            recorder.emit("path_selected", correlation, {
                "selected_path": result.selected_path,
                "reason": admission.reason,
            })
            return
        recorder.emit("path_selected", correlation, {"selected_path": selected_path})
        result.selected_path = selected_path
        if self._check_cancel(task, result):
            return

        # Step 51: bounded cache keyed by snapshot hash and model version.
        cache_key = CacheKey(
            snapshot_hash=snapshot.graph_hash,
            model_version=self.model_version,
            namespace=f"candidates:{task.task_type}",
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            result.cache_hits += 1
            recorder.emit("cache_lookup", correlation, {
                "hit": True, "key": cache_key.namespace,
                "snapshot_hash": cache_key.snapshot_hash,
            })
            self._reuse_cached(cached, result)
            self._maybe_nar(task, result)
            return
        recorder.emit("cache_lookup", correlation, {
            "hit": False, "key": cache_key.namespace,
            "snapshot_hash": cache_key.snapshot_hash,
        })

        if selected_path == "exact_lookup":
            self._run_exact(task, result)
        else:
            self._run_learned(task, result, admission)
        if result.cancelled:
            return
        self._maybe_nar(task, result)
        self.cache.put(cache_key, self._cacheable_state(result))

    def _check_cancel(
        self, task: GraphIntelligenceTask, result: GraphIntelligenceResult,
    ) -> bool:
        """Stop the cycle when cancellation was requested; keep partial results."""
        if self._cancelled(task):
            result.cancelled = True
            self.event_recorder.emit("error", task.correlation_id, {
                "message": "cycle cancelled at a phase boundary",
            })
            return True
        return False

    # ------------------------------------------------------------------ exact path

    def _run_exact(
        self, task: GraphIntelligenceTask, result: GraphIntelligenceResult,
    ) -> None:
        """Exact typed neighborhood lookup via deterministic rules (steps 2, 13, 49)."""
        snapshot = result.snapshot
        assert snapshot is not None
        service = GraphCandidateService(snapshot)
        resolution = service.anchor_resolution(task.query.anchor)
        anchor_id = (
            task.query.anchor.node_id
            if task.query.anchor.node_id is not None and not resolution.abstained
            else (resolution.node_ids[0] if resolution.node_ids else None)
        )
        found: tuple[str, ...] = ()
        if anchor_id is not None:
            relations = task.query.relation_types or snapshot.relation_types()
            baselines = DeterministicGraphBaselines(
                [(e.source_id, e.relation_type, e.target_id) for e in snapshot.edges],
            )
            for relation in sorted(relations):
                lookup = baselines.typed_neighborhood_lookup(anchor_id, relation)
                if lookup:
                    found = lookup
                    break
        result.exact_lookup = found
        self.event_recorder.emit("candidates_generated", result.correlation_id, {
            "path": "exact_lookup", "kept": [], "exact_lookup_results": list(found),
            "anchor": anchor_id,
        })

    # ------------------------------------------------------------------ learned path

    def _run_learned(
        self,
        task: GraphIntelligenceTask,
        result: GraphIntelligenceResult,
        admission: AdmissionDecision,
    ) -> None:
        """Phases B-D: scoring, discovery, verification and promotion gating."""
        correlation = task.correlation_id
        snapshot = result.snapshot
        assert snapshot is not None

        budget = admission.sampled_candidate_budget or task.query.budget_candidates
        gnn_scorer = self._build_gnn_scorer(snapshot)

        # Steps 25-26: anchor resolution and typed candidate generation.
        service = GraphCandidateService(snapshot)
        kept, rejected = service.propose(task.query, budget=budget)
        result.candidates, result.rejected = kept, rejected
        self.event_recorder.emit("candidates_generated", correlation, {
            "kept": [
                {"candidate_id": c.candidate_id, "triple": list(c.triple())} for c in kept
            ],
            "rejected": [
                {"candidate_id": c.candidate_id, "reasons": list(c.rejection_reasons)}
                for c in rejected
            ],
            "budget": budget,
        })

        # Steps 27: complementary scoring; learned failure degrades to baselines (test 28).
        scorer = ComplementaryScorer(
            snapshot,
            embedding_scorer=self.embedding_scorer,
            gnn_scorer=gnn_scorer,
        )
        predictor = LinkPredictor(scorer)
        try:
            kept = predictor.score_all(kept)
            result.candidates = kept
        except Exception as error:  # noqa: BLE001 - learned components never block
            message = f"learned scoring failed; baseline-only fallback: {error}"
            result.errors.append(message)
            self.event_recorder.emit("error", correlation, {"message": message})
            baseline_only = ComplementaryScorer(snapshot)
            kept = LinkPredictor(baseline_only).score_all(kept)
            result.candidates = kept
        self.event_recorder.emit("candidates_scored", correlation, {
            "scored": [
                {"candidate_id": c.candidate_id,
                 "component_scores": c.component_scores.as_dict()}
                for c in kept
            ],
        })

        # Step 28: bounded explanation neighborhoods.
        explainer = NeighborhoodExplainer(snapshot)
        neighborhoods = explainer.attach(kept)
        for candidate, neighborhood in zip(kept, neighborhoods, strict=True):
            candidate.explanation_paths = neighborhood.paths
            if neighborhood.evidence_ids and not candidate.evidence_ids:
                candidate.evidence_ids = neighborhood.evidence_ids
            if neighborhood.source_ids and not candidate.source_ids:
                candidate.source_ids = neighborhood.source_ids

        # Steps 29-30: duplicates/novelty and utility ranking.
        ranker = UtilityRanker(snapshot)
        ranked = ranker.rank(kept)
        result.candidates = ranked
        self.event_recorder.emit("candidates_ranked", correlation, {
            "ranked": [
                {"candidate_id": c.candidate_id, "utility": c.utility}
                for c in ranked
            ],
        })

        # Step 31: abstention for low confidence / ties / insufficient evidence.
        abstained, decision = self.abstention_policy.apply(ranked)
        result.abstention = decision
        self.event_recorder.emit("abstention_evaluated", correlation, decision.as_dict())
        if decision.abstained:
            result.candidates = abstained
            return

        # Step 32: publish hypotheses to the dedicated workspace namespace.
        result.candidates = self._publish_hypotheses(ranked, result)

        # Steps 33-39: symbolic verification of every published candidate.
        verifier = CandidateVerifier(snapshot)
        results = verifier.verify_all(ranked)
        result.verification = results
        self.event_recorder.emit("verification_completed", correlation, {
            "verified": [
                {
                    "candidate_id": v.candidate_id,
                    "disposition": v.disposition,
                    "evidence_classification": v.evidence_classification,
                    "evidence_ids": list(v.evidence_ids),
                    "rules_applied": list(v.rules_applied),
                    "notes": list(v.notes),
                }
                for v in results
            ],
        })

        # Steps 38-39: promotion gate; hypotheses vs verified facts stay distinct.
        proposals = PromotionGate().gate(results)
        result.update_proposals = proposals
        self.event_recorder.emit("promotion_gated", correlation, {
            "update_proposals": [p.as_dict() for p in proposals],
            "authorized_workflow": "authorized_usg_update_workflow",
        })
        self._publish_verifications(results, result)

    # ------------------------------------------------------------------ NAR path

    def _maybe_nar(
        self, task: GraphIntelligenceTask, result: GraphIntelligenceResult,
    ) -> None:
        """Phase E: advisory NAR proposal with a deterministic check (steps 41-48)."""
        if task.nar_task is None or self.nar_service is None:
            return
        proposal = self.nar_service.predict(task.nar_task)
        result.nar_proposal = proposal
        self.event_recorder.emit("nar_proposed", result.correlation_id, proposal.as_dict())

    # ------------------------------------------------------------------ workspace

    def _publish_hypotheses(
        self,
        ranked: Sequence[GraphCandidate],
        result: GraphIntelligenceResult,
    ) -> list[GraphCandidate]:
        """Publish hypotheses into the dedicated namespace (step 32)."""
        if self.workspace_publisher is None or not ranked:
            return list(ranked)
        namespace = new_hypothesis_namespace(self.model_version, result.snapshot.graph_hash)
        published: list[dict[str, object]] = []
        for candidate in ranked:
            receipt = self.workspace_publisher.publish_hypothesis(
                candidate, namespace,
            )
            if receipt is not None:
                published.append(receipt)
        result.workspace_publications.extend(published)
        self.event_recorder.emit("hypotheses_published", result.correlation_id, {
            "namespace": namespace,
            "published": [
                {"candidate_id": c.candidate_id, "triple": list(c.triple())} for c in ranked
            ],
            "finding_type": HYPOTHESIS_FINDING_TYPE,
        })
        return list(ranked)

    def _publish_verifications(
        self, results: Sequence[VerificationResult], result: GraphIntelligenceResult,
    ) -> None:
        """Publish dispositions distinctly from hypotheses (step 39, test 14)."""
        if self.workspace_publisher is None or not results:
            return
        receipts = self.workspace_publisher.publish_verifications(results)
        result.workspace_publications.extend(receipts)

    # ------------------------------------------------------------------ helpers

    def _build_gnn_scorer(
        self, snapshot: TypedGraphSnapshot,
    ) -> Callable[[str, str, str], float] | None:
        """Bind the trained R-GCN to string node/relation IDs (step 17, step 27)."""
        if self.gnn_predictor is None:
            return None
        if self._gnn_context is None or self._gnn_context[0] is not snapshot:
            node_ids = sorted(node.node_id for node in snapshot.nodes)
            relations = sorted(snapshot.relation_types())
            node_index = {node_id: index for index, node_id in enumerate(node_ids)}
            relation_index = {relation: index for index, relation in enumerate(relations)}
            features = build_snapshot_features(snapshot)
            triples = [
                (node_index[edge.source_id], relation_index[edge.relation_type],
                 node_index[edge.target_id])
                for edge in snapshot.edges
            ]
            graph = RelationGraph.from_triples(
                triples, relations, n_nodes=snapshot.node_count,
            )
            self._gnn_context = (snapshot, (features, graph), node_index, relation_index)
        features, graph = self._gnn_context[1]
        node_index, relation_index = self._gnn_context[2], self._gnn_context[3]
        predictor = self.gnn_predictor

        def scorer(subject: str, relation: str, obj: str) -> float:
            if subject not in node_index or obj not in node_index \
                    or relation not in relation_index:
                raise KeyError((subject, relation, obj))
            return predictor.score(
                features, graph, node_index[subject],
                relation_index[relation], node_index[obj],
            )

        return scorer

    @staticmethod
    def _cacheable_state(result: GraphIntelligenceResult) -> dict[str, object]:
        """Snapshot of a cycle reusable for cache hits (never USG state)."""
        return {
            "exact_lookup": result.exact_lookup,
            "candidate_ids": [c.candidate_id for c in result.candidates],
            "verification_ids": [v.candidate_id for v in result.verification],
            "metrics": dict(result.metrics),
        }

    @staticmethod
    def _reuse_cached(state: Mapping[str, object], result: GraphIntelligenceResult) -> None:
        cached_lookup = state.get("exact_lookup", ())
        result.exact_lookup = tuple(cached_lookup)  # type: ignore[arg-type]
        result.metrics.setdefault("cache_reused", True)


__all__ = [
    "COORDINATOR_PATHS", "GraphIntelligenceCoordinator", "GraphIntelligenceResult",
    "GraphIntelligenceTask", "HYPOTHESIS_FINDING_TYPE", "VERIFICATION_FINDING_TYPE",
]
