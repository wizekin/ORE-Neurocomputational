"""Bounded coordinator and event router joining memory with ORE's cognitive workspace."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
import uuid
from typing import Any

from ore_language_comprehension.point07 import Point7MemoryStore
from ore_workspace import FindingPublisher, WorkspaceManager
from ore_working_memory import WorkingMemoryConfig, WorkingMemoryProcessor, WorkingMemoryRequest

from .adapters import (
    AnalogicalMemoryAdapter, ReconstructiveMemoryAdapter, SchemaGuidedInferenceAdapter,
    SchemaMemoryAdapter, SemanticMemoryAdapter, WorkingMemoryAdapter,
)
from .annotations import register_annotations
from .contracts import MemoryBatch, MemoryEvidence, MemoryHandoff, MemoryIntegrationConfig, MemoryIntegrationResult, MemoryQuery
from .policies import MemoryEncodingPolicy
from .retrieval import MemoryRetrievalCoordinator


class MemoryEventHandler:
    """Convert committed workspace events into stable module-specific memory handoffs."""

    def __init__(self, router: "CognitiveMemoryRouter") -> None:
        """Bind the handler to one router and avoid publishing during subscriber callbacks."""
        self.router = router

    def handle(self, module_id: str, event: Any) -> MemoryHandoff | None:
        """Route the latest snapshot only when the workspace event includes relevant memory state."""
        return self.router.route(module_id, event)


class CognitiveMemoryRouter:
    """Subscribe cognitive structures and route normalized evidence through typed adapters."""

    MODULES = (
        "working_memory", "semantic_processor", "schema_formation",
        "schema_guided_inference", "analogical_transfer", "reconstructive_memory",
        "cognitive_control", "discourse_processor", "working_memory_overload",
    )

    def __init__(
        self, workspace: Any, *,
        module_handlers: Mapping[str, Callable[[MemoryHandoff], Any]] | None = None,
    ) -> None:
        """Prepare adapter instances and optional callbacks for existing ORE cognitive modules."""
        self.workspace = workspace
        self.module_handlers = dict(module_handlers or {})
        self.working_memory_adapter = WorkingMemoryAdapter()
        self.semantic_adapter = SemanticMemoryAdapter()
        self.schema_adapter = SchemaMemoryAdapter()
        self.schema_guided_adapter = SchemaGuidedInferenceAdapter()
        self.analogical_adapter = AnalogicalMemoryAdapter()
        self.reconstructive_adapter = ReconstructiveMemoryAdapter()
        self.handler = MemoryEventHandler(self)
        self._handoffs: dict[str, MemoryHandoff] = {}
        self._errors: list[dict[str, str]] = []
        self._attached: list[str] = []

    def attach(self) -> tuple[str, ...]:
        """Register one event callback for every module with a declared workspace subscription."""
        for module_id in self.MODULES:
            if self.workspace.subscriptions.get(module_id) is None:
                continue
            self.workspace.register_consumer(module_id, self._make_callback(module_id))
            self._attached.append(module_id)
        return tuple(self._attached)

    def _make_callback(self, module_id: str) -> Callable[[Any], None]:
        """Create a module-scoped subscriber that delegates without mutating workspace state."""
        def callback(event: Any) -> None:
            """Handle one committed event and preserve callback failures as router diagnostics."""
            self.handler.handle(module_id, event)
        return callback

    def _module_payload(self, module_id: str, evidence: Sequence[MemoryEvidence]) -> dict[str, Any]:
        """Build a target-specific data shape without changing original memory provenance."""
        if module_id == "working_memory":
            return self.working_memory_adapter.to_context(evidence)
        if module_id == "semantic_processor":
            return {
                **self.semantic_adapter.to_payload(evidence),
                "episodic_context": [item.to_dict() for item in evidence if item.memory_type == "episodic"],
            }
        if module_id == "schema_formation":
            return self.schema_adapter.to_payload(evidence)
        if module_id == "schema_guided_inference":
            return self.schema_guided_adapter.to_payload(evidence)
        if module_id == "analogical_transfer":
            return self.analogical_adapter.to_payload(evidence)
        if module_id == "reconstructive_memory":
            return self.reconstructive_adapter.to_payload(evidence)
        if module_id == "cognitive_control":
            groups: dict[str, list[str]] = {}
            for item in evidence:
                groups.setdefault(item.independence_group, []).append(item.memory_id)
            return {
                "evidence": [item.to_dict() for item in evidence],
                "independence_groups": groups,
                "unique_source_group_count": len(groups),
                "verification_conflicts": [
                    item.memory_id for item in evidence
                    if item.verification_status.casefold() not in {"accepted", "verified", "stored_episode"}
                ],
            }
        if module_id == "working_memory_overload":
            return {
                "memory_reference_count": len(evidence),
                "memory_reference_ids": [item.memory_id for item in evidence],
                "reference_types": [item.memory_type for item in evidence],
                "long_term_payloads_copied": False,
            }
        return {
            "evidence": [item.to_dict() for item in evidence],
            "source_count": len({item.independence_group for item in evidence}),
        }

    def route(self, module_id: str, event: Any) -> MemoryHandoff | None:
        """Build and optionally deliver one handoff from the current immutable workspace snapshot."""
        snapshot = self.workspace.snapshot()
        state = snapshot.state
        records = state.get("retrieved_memories", ())
        evidence = []
        for raw in records:
            normalized = {
                **dict(raw),
                "provenance": raw.get("memory_provenance", raw.get("provenance", {})),
            }
            try:
                evidence.append(MemoryEvidence.from_mapping(normalized))
            except (KeyError, TypeError, ValueError):
                continue
        if not evidence:
            return None
        if module_id in {"schema_formation", "analogical_transfer", "reconstructive_memory"}:
            selected = [item for item in evidence if item.memory_type == "episodic"]
        elif module_id == "schema_guided_inference":
            selected = [
                item for item in evidence
                if item.memory_type == "episodic" or item.verification_status.casefold() in {"accepted", "verified"}
            ]
        else:
            selected = evidence
        if not selected:
            return None
        sequence_id = next(
            (str(item.get("sequence_id")) for item in state.get("working_memory", {}).values()
             if item.get("sequence_id")),
            next((item.sequence_id for item in selected if item.sequence_id), "unspecified"),
        )
        handoff = MemoryHandoff(
            module_id=module_id,
            workspace_id=snapshot.workspace_id,
            workspace_revision=snapshot.revision,
            sequence_id=sequence_id,
            memory_ids=tuple(item.memory_id for item in selected),
            payload=self._module_payload(module_id, selected),
            event_id=str(getattr(event, "event_id", "")),
        )
        self._handoffs[module_id] = handoff
        callback = self.module_handlers.get(module_id)
        if callback is not None:
            try:
                callback(handoff)
            except Exception as exc:
                self._errors.append({"module_id": module_id, "error": f"{type(exc).__name__}: {exc}"})
        return handoff

    def handoffs(self) -> dict[str, dict[str, Any]]:
        """Return the latest successful handoff per module as detached dictionaries."""
        return {key: value.to_dict() for key, value in sorted(self._handoffs.items())}

    def diagnostics(self) -> dict[str, Any]:
        """Expose attached subscribers, routed modules, and isolated callback failures."""
        return {
            "attached_modules": list(self._attached),
            "routed_modules": sorted(self._handoffs),
            "callback_errors": list(self._errors),
        }


class MemoryIntegrationCoordinator:
    """Run initial working memory, recall, workspace publication, routing, and one feedback pass.

    Existing Point 7 SQLite tables remain the long-term store. The workspace is the shared
    communication layer. The logger database is only the symbol-annotation/execution catalogue.
    """

    def __init__(
        self,
        memory_store: Any | None = None,
        *,
        working_memory_processor: WorkingMemoryProcessor | None = None,
        workspace_manager: WorkspaceManager | None = None,
        config: MemoryIntegrationConfig | None = None,
        module_handlers: Mapping[str, Callable[[MemoryHandoff], Any]] | None = None,
        annotation_database_path: str | Path | None = None,
        register_logger_annotations: bool = True,
    ) -> None:
        """Bind existing stores/processors or create one persistent Point 7 store and local workspace manager."""
        self.config = config or MemoryIntegrationConfig()
        if memory_store is not None and hasattr(memory_store, "store"):
            memory_store = memory_store.store
        self._owns_memory_store = memory_store is None
        self.memory_store = memory_store or Point7MemoryStore(self.config.memory_database_path)
        if not callable(getattr(self.memory_store, "add_episode", None)):
            raise TypeError("memory_store must expose the Point7MemoryStore interface")
        self.working_memory_processor = working_memory_processor or WorkingMemoryProcessor(
            WorkingMemoryConfig(simulate_biology=False)
        )
        self._owns_workspace_manager = workspace_manager is None
        self.workspace_manager = workspace_manager or WorkspaceManager(enable_logging=False)
        self.module_handlers = dict(module_handlers or {})
        self.encoding_policy = MemoryEncodingPolicy()
        self.annotation_report = (
            register_annotations(database_path=annotation_database_path)
            if register_logger_annotations else {}
        )
        self._closed = False

    def _make_working_memory_request(
        self, sequence_id: str, text: str, goal: str, context: Mapping[str, Any],
    ) -> WorkingMemoryRequest:
        """Create a working-memory request while keeping upstream syntax/evidence caller-supplied."""
        merged_context = {**dict(context), "current_goal": goal}
        return WorkingMemoryRequest(
            sequence_id=sequence_id, text=text,
            goals={"current_goal": goal, "goal": goal},
            context=merged_context,
        )

    def _publish_working_memory(
        self, workspace: Any, sequence_id: str, result: Any, *,
        stage: str, memory_ids: Sequence[str] = (),
    ) -> Any:
        """Publish one working-memory snapshot through its existing field permission."""
        return FindingPublisher(workspace, "working_memory").publish(
            "working_memory",
            {
                "item_id": sequence_id, "sequence_id": sequence_id,
                "stage": stage, "result": result.to_dict(),
                "active_memory_references": list(memory_ids), "is_active": True,
            },
            key=sequence_id,
            finding_type="working_memory_snapshot",
            confidence=1.0,
        )

    def _publish_memories(self, workspace: Any, batch: MemoryBatch) -> tuple[Any, ...]:
        """Publish ranked records while retaining original source provenance beside the workspace envelope."""
        publisher = FindingPublisher(workspace, "memory_retrieval")
        outcomes = []
        for evidence in batch.items:
            payload = evidence.to_dict()
            payload["memory_provenance"] = dict(evidence.provenance)
            outcomes.append(publisher.publish(
                "retrieved_memories", payload,
                key=evidence.memory_id,
                finding_type="retrieved_memory_evidence",
                confidence=evidence.confidence,
            ))
        return tuple(outcomes)

    def process(
        self,
        text: str,
        goal: str = "understand_sentence",
        *,
        sequence_id: str | None = None,
        context: Mapping[str, Any] | None = None,
        event_cues: Mapping[str, Any] | None = None,
        context_cues: Mapping[str, Any] | None = None,
        temporal_cues: Mapping[str, Any] | None = None,
        current_timestamp: float | None = None,
        encode_current: bool = False,
        episode_payload: Mapping[str, Any] | None = None,
    ) -> MemoryIntegrationResult:
        """Process current input, recall only older records, route evidence, then optionally encode."""
        if self._closed:
            raise RuntimeError("MemoryIntegrationCoordinator is closed")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("text must be non-empty")
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be non-empty")
        sequence = sequence_id or f"memory_sequence_{uuid.uuid4().hex}"
        caller_context = dict(context or {})
        workspace = self.workspace_manager.create_workspace(text, goal)
        router = CognitiveMemoryRouter(workspace, module_handlers=self.module_handlers)
        router.attach()

        # First pass: establish the current state before querying long-term stores.
        initial = self.working_memory_processor.process(
            self._make_working_memory_request(sequence, text, goal, caller_context)
        )
        initial_publication = self._publish_working_memory(
            workspace, sequence, initial, stage="initial",
        )

        # Point 7's normal processor performs encoding before later retrieval operations.
        # Recall therefore reads its existing tables directly and explicitly excludes this sequence.
        query = MemoryQuery(
            sequence_id=sequence, text=text, goal=goal,
            lexical_cues=tuple(sorted(_query_tokens(text))),
            event_cues=dict(event_cues or {}),
            context_cues=dict(context_cues or caller_context),
            temporal_cues=dict(temporal_cues or {}),
            current_timestamp=current_timestamp,
            retrieval_limit=self.config.max_retrieval_items,
        )
        batch = MemoryRetrievalCoordinator(self.memory_store, self.config).retrieve(query)
        selected = WorkingMemoryAdapter(self.config).select(batch.items, goal, text)
        memory_publications = self._publish_memories(workspace, batch)

        # Exactly one optional feedback pass: pending semantic claims remain separate candidates.
        integrated_result = initial
        integrated_publication = None
        rounds = 1
        if selected and self.config.max_integration_rounds > 1:
            wm_context = {
                **caller_context,
                **WorkingMemoryAdapter(self.config).to_context(selected),
                "current_goal": goal,
            }
            integrated_result = self.working_memory_processor.process(
                self._make_working_memory_request(sequence, text, goal, wm_context)
            )
            integrated_publication = self._publish_working_memory(
                workspace, sequence, integrated_result,
                stage="memory_integrated",
                memory_ids=tuple(item.memory_id for item in selected),
            )
            rounds = 2

        # Encoding is strictly after recall and interpretation, and only for explicitly supplied
        # observed/entailed/reported content. Reconstructed hypotheses are not raw observations.
        encoding = self.encoding_policy.evaluate(
            requested=encode_current,
            payload=episode_payload,
        )
        if encoding["allowed"]:
            trace_id = self.memory_store.add_episode(sequence, dict(episode_payload or {}))
            encoding.update({"write_performed": True, "memory_id": trace_id, "stage": "after_recall"})

        snapshot = workspace.snapshot()
        diagnostics = {
            "working_memory_passes": rounds,
            "feedback_rounds": max(0, rounds - 1),
            "feedback_round_limit": self.config.max_integration_rounds - 1,
            "initial_publication": initial_publication.decision.value,
            "retrieval_publications": [item.decision.value for item in memory_publications],
            "integrated_publication": (
                integrated_publication.decision.value if integrated_publication is not None else "not_run"
            ),
            "selected_memory_references": [item.memory_id for item in selected],
            "memory_database_is_separate_from_logger": True,
            "same_sequence_retrieval_excluded": True,
            "recall_precedes_encoding": True,
            "biological_simulation_enabled": False,
            "annotation_registration": self.annotation_report,
            "router": router.diagnostics(),
        }
        return MemoryIntegrationResult(
            workspace_id=snapshot.workspace_id,
            sequence_id=sequence,
            initial_working_memory=initial.to_dict(),
            integrated_working_memory=integrated_result.to_dict(),
            memory_batch=batch,
            workspace_snapshot=snapshot.to_dict(),
            handoffs=router.handoffs(),
            encoding=encoding,
            diagnostics=diagnostics,
        )

    def close(self) -> None:
        """Release resources created by this coordinator without closing caller-owned stores."""
        if self._closed:
            return
        if self._owns_workspace_manager:
            self.workspace_manager.close()
        if self._owns_memory_store:
            self.memory_store.close()
        self._closed = True

    def __enter__(self) -> "MemoryIntegrationCoordinator":
        """Return this coordinator for context-managed operation."""
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, traceback: object) -> bool:
        """Close owned resources while allowing caller exceptions to propagate."""
        self.close()
        return False


def _query_tokens(text: str) -> set[str]:
    """Extract non-trivial lexical retrieval cues from the current input."""
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "of", "to", "in", "on", "at", "for", "by", "with", "from", "as",
        "and", "or", "but", "if", "then", "that", "this", "these", "those",
        "it", "he", "she", "they", "we", "you", "i", "me", "my", "your",
    }
    return {
        item.casefold().strip(".,!?;:()[]{}\"'")
        for item in text.split()
        if item.strip(".,!?;:()[]{}\"'") and item.casefold().strip(".,!?;:()[]{}\"'") not in stopwords
    }
