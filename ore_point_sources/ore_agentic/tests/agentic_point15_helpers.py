"""Shared builders for the Point 15 agentic acceptance tests.

Mirrors the repository's ``learning_point14_helpers.py`` convention: plain
builder functions imported directly by the test modules.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import time

from ore_agentic import (
    AgenticConfig,
    AgenticEngine,
    GoalRequest,
    LocalWorker,
    ResourceVector,
    SideEffectClass,
    SubgoalSpec,
    WorkerManifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

CAP_A = "test.capability_a"
CAP_B = "test.capability_b"


def make_config(**overrides: Any) -> AgenticConfig:
    """Build a fast test configuration and apply overrides."""
    values: dict[str, Any] = dict(
        retry_initial_delay_seconds=0.005,
        retry_max_delay_seconds=0.02,
        retry_jitter_seconds=0.0,
        retry_total_time_budget_seconds=2.0,
        worker_pool_size=2,
        per_workflow_concurrency=2,
        per_worker_concurrency=2,
        queue_capacity=8,
        checkpoint_every_n_tasks=1,
    )
    values.update(overrides)
    return AgenticConfig(**values)


def make_subgoal(name: str, capability: str = CAP_A, **overrides: Any) -> SubgoalSpec:
    """Build one subgoal and apply overrides."""
    values: dict[str, Any] = dict(
        name=name,
        capability=capability,
        description=f"subgoal {name}",
        inputs={"value": f"{name}-result"},
        depends_on=(),
        side_effect_class=SideEffectClass.READ_ONLY,
        output_schema={},
    )
    values.update(overrides)
    return SubgoalSpec(**values)


def make_request(**overrides: Any) -> GoalRequest:
    """Build a fully populated goal request and apply overrides."""
    values: dict[str, Any] = dict(
        objective="Test objective: analyze the provided evidence.",
        requester="test-requester",
        subgoals=(make_subgoal("alpha"), make_subgoal("beta", capability=CAP_B)),
        constraints={"intent": "test"},
        priority=0.5,
        idempotency_key=None,
    )
    values.update(overrides)
    return GoalRequest(**values)


def echo_handler(value: str | None = None,
                 delay_seconds: float = 0.0,
                 evidence: Sequence[str] = ("test-evidence-1",)
                 ) -> Callable[[Any], Mapping[str, Any]]:
    """Build a deterministic worker handler returning a fixed payload."""
    def _handler(envelope: Any) -> Mapping[str, Any]:
        if delay_seconds:
            time.sleep(delay_seconds)
        return {"value": value if value is not None else
                envelope.inputs["payload"].get("value", "result"),
                "evidence": list(evidence)}
    return _handler


def make_manifest(worker_id: str, capabilities: Iterable[str] = (CAP_A,),
                  **overrides: Any) -> WorkerManifest:
    """Build a worker manifest and apply overrides."""
    values: dict[str, Any] = dict(
        worker_id=worker_id,
        version="1.0.0",
        capabilities=tuple(capabilities),
        accepted_schemas=("object",),
        permission_scopes=(),
        resource_profile=ResourceVector(cpu_units=4.0, memory_mb=512.0,
                                        io_units=4.0, network_units=4.0,
                                        seconds=10.0),
        reliability=0.9,
        latency_seconds=0.5,
        cost_per_task=0.5,
        load=0.0,
    )
    values.update(overrides)
    return WorkerManifest(**values)


def register_default_worker(engine: AgenticEngine,
                            capabilities: Iterable[str] = (CAP_A, CAP_B),
                            worker_id: str = "test-worker-1",
                            handler: Callable[[Any], Mapping[str, Any]]
                            | None = None) -> None:
    """Register one echo worker covering the standard test capabilities."""
    engine.register_worker(
        make_manifest(worker_id, capabilities),
        LocalWorker(worker_id, "1.0.0", tuple(capabilities),
                    handler or echo_handler()))


def make_engine(config: AgenticConfig | None = None, *,
                database_path: str | Path | None = None,
                workers: Sequence[tuple[WorkerManifest, LocalWorker]] = (),
                **engine_kwargs: Any) -> AgenticEngine:
    """Build an engine with the standard test worker registered by default."""
    engine = AgenticEngine(config or make_config(), database_path=database_path,
                           **engine_kwargs)
    if not workers:
        register_default_worker(engine)
    for manifest, worker in workers:
        engine.register_worker(manifest, worker)
    return engine


class FakeUSGIndex:
    """A duck-typed USG knowledge index for retrieval tests.

    Records every call so tests can prove the semantic memory never writes.
    """

    def __init__(self, labels: Mapping[str, str] | None = None,
                 relations: Sequence[tuple[str, str, str]] = ()) -> None:
        self.records = {
            key: _FakeItem(record_id=key, label=label, kind="SEMANTIC_CONCEPT")
            for key, label in (labels or {"attention": "attention (cognitive process)"}).items()}
        self.relations = tuple(
            _FakeRelation(f"rel-{index}", subject, predicate, obj)
            for index, (subject, predicate, obj) in enumerate(relations))
        self.write_calls: list[str] = []

    def lookup(self, query: str) -> Any:
        """Return exact-match records (strip/casefold semantics)."""
        normalized = query.strip().casefold()
        items = [item for key, item in self.records.items()
                 if key.strip().casefold() == normalized]
        return _FakeResult(tuple(items), exact_match=bool(items))

    def relations_for(self, record_id: str) -> tuple[Any, ...]:
        """Return relations touching a record."""
        return tuple(relation for relation in self.relations
                     if relation.subject_id == record_id
                     or relation.object_id == record_id)


class _FakeItem:
    def __init__(self, record_id: str, label: str, kind: str) -> None:
        self.record_id = record_id
        self.label = label
        self.kind = kind
        self.source_reference = "usg:test-bundle"


class _FakeResult:
    def __init__(self, items: tuple[_FakeItem, ...], exact_match: bool) -> None:
        self.items = items
        self.exact_match = exact_match


class _FakeRelation:
    def __init__(self, relation_id: str, subject_id: str, predicate: str,
                 object_id: str) -> None:
        self.relation_id = relation_id
        self.subject_id = subject_id
        self.predicate = predicate
        self.object_id = object_id
        self.source_reference = "usg:test-bundle"
