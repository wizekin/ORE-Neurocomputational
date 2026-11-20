"""Adapters that safely connect recurrent feedback to existing cognitive processors."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from .contracts import FeedbackEvidence
from .recurrent_contracts import _plain


@dataclass(frozen=True, slots=True)
class ModuleAdapterResult:
    """Describe one module invocation and only the evidence it explicitly supplied."""

    module_id: str
    status: str
    output: Mapping[str, Any]
    feedback_evidence: tuple[FeedbackEvidence, ...] = ()
    clarification_options: tuple[Any, ...] = ()
    diagnostics: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the adapter outcome without serializing live processor instances."""
        return {
            "module_id": self.module_id,
            "status": self.status,
            "output": _plain(self.output),
            "feedback_evidence": [asdict(item) for item in self.feedback_evidence],
            "clarification_options": [_plain(item) for item in self.clarification_options],
            "diagnostics": _plain(self.diagnostics or {}),
        }


class RecurrentModuleAdapters:
    """Invoke existing ORE structures only through supplied typed requests or callable adapters."""

    def __init__(
        self,
        handlers: Mapping[str, Callable[[Mapping[str, Any]], Any]] | None = None,
        processors: Mapping[str, Any] | None = None,
    ) -> None:
        """Copy module bindings without taking ownership of the supplied processor objects."""
        self.handlers = dict(handlers or {})
        self.processors = dict(processors or {})

    @staticmethod
    def _source_type(value: str) -> str:
        """Map compatible Point 07 source labels to Point 24 evidence categories."""
        candidate = str(value).casefold()
        if candidate in {"measurement", "expert", "primary_document", "reported", "memory", "model", "other"}:
            return candidate
        return "reported"

    @staticmethod
    def _p07_evidence(request: Any) -> tuple[FeedbackEvidence, ...]:
        """Convert Point 07 evidence only when it already carries explicit hypothesis likelihoods."""
        sources = {item.source_id: item for item in getattr(request, "feedback_sources", ())}
        converted = []
        for item in getattr(request, "feedback_evidence", ()):
            likelihoods = dict(getattr(item, "hypothesis_likelihoods", {}) or {})
            if not likelihoods:
                continue
            source = sources.get(item.source_id)
            if source is None:
                continue
            converted.append(FeedbackEvidence(
                evidence_id=item.evidence_id,
                description=item.correction or item.statement,
                likelihoods=likelihoods,
                source_name=source.name,
                source_type=RecurrentModuleAdapters._source_type(source.domain),
                reliability=float(source.reliability),
                relevance=float(item.relevance),
                diagnosticity=float(source.specificity),
                directness="indirect",
                independence_key=str(item.independence_key or source.source_id),
                context=str(getattr(request, "domain", "")),
            ))
        return tuple(converted)

    @staticmethod
    def _p21_evidence(request: Any) -> tuple[FeedbackEvidence, ...]:
        """Convert Point 21 observations to Point 24 evidence without inventing their likelihoods."""
        sources = {item.source_id: item for item in getattr(request, "sources", ())}
        converted = []
        for item in getattr(request, "evidence", ()):
            source = sources.get(item.source_id)
            if source is None or not item.likelihoods:
                continue
            maximum = max(float(value) for value in item.likelihoods.values())
            if maximum <= 0.0:
                continue
            # Point 21 accepts positive relative likelihoods; scale all by their common maximum.
            likelihoods = {key: float(value) / maximum for key, value in item.likelihoods.items()}
            source_kind = "measurement" if item.direct_observation else "reported"
            if str(source.source_type).casefold() in {"expert", "primary_document"}:
                source_kind = str(source.source_type).casefold()
            converted.append(FeedbackEvidence(
                evidence_id=item.evidence_id,
                description=item.content,
                likelihoods=likelihoods,
                source_name=item.source_id,
                source_type=source_kind,
                reliability=float(source.reliability),
                relevance=float(item.relevance),
                diagnosticity=float(item.measurement_quality),
                directness="direct" if item.direct_observation else "indirect",
                independence_key=str(source.independence_group or source.source_id),
                context=str(item.feature),
            ))
        return tuple(converted)

    @staticmethod
    def _evidence_from_handler(output: Mapping[str, Any]) -> tuple[FeedbackEvidence, ...]:
        """Accept only existing Point 24 evidence objects returned by an adapter."""
        values = output.get("feedback_evidence", output.get("evidence", ()))
        if values is None:
            return ()
        if isinstance(values, FeedbackEvidence):
            values = (values,)
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise TypeError("adapter evidence must be a sequence of FeedbackEvidence objects")
        invalid = [type(item).__name__ for item in values if not isinstance(item, FeedbackEvidence)]
        if invalid:
            raise TypeError(f"adapter returned non-FeedbackEvidence entries: {invalid}")
        return tuple(values)

    def _invoke_existing_processor(
        self,
        module_id: str,
        payload: Mapping[str, Any],
        module_inputs: Mapping[str, Any],
    ) -> ModuleAdapterResult:
        """Invoke a processor only when its own validated request object is supplied."""
        if module_id == "feedback_seeking":
            request = module_inputs.get("feedback_request")
            if request is None:
                return ModuleAdapterResult(module_id, "unavailable", {}, diagnostics={"reason": "typed_feedback_request_not_supplied"})
            processor = self.processors.get(module_id)
            if processor is None:
                from ore_feedback_seeking import Point07FeedbackProcessor
                processor = Point07FeedbackProcessor(simulate_biology=False)
            from ore_feedback_seeking import FeedbackRequest
            if not isinstance(request, FeedbackRequest):
                raise TypeError("module_inputs.feedback_request must be a FeedbackRequest")
            result = processor.process(request)
            return ModuleAdapterResult(
                module_id, "completed", {"result": result.to_dict()},
                self._p07_evidence(request),
                tuple(module_inputs.get("clarification_options", ())),
                {"source_result_diagnostics": dict(result.diagnostics)},
            )

        if module_id == "confirmation_bias":
            request = module_inputs.get("point21_request")
            if request is None:
                return ModuleAdapterResult(module_id, "unavailable", {}, diagnostics={"reason": "typed_point21_request_not_supplied"})
            processor = self.processors.get(module_id)
            if processor is None:
                from ore_confirmation_bias import Point21ConfirmationBiasProcessor, Point21Config
                processor = Point21ConfirmationBiasProcessor(config=Point21Config(simulate_biology=False))
            from ore_confirmation_bias import Point21Request
            if not isinstance(request, Point21Request):
                raise TypeError("module_inputs.point21_request must be a Point21Request")
            result = processor.process(request)
            return ModuleAdapterResult(
                module_id, "completed", {"result": result.to_dict()},
                self._p21_evidence(request),
                tuple(module_inputs.get("clarification_options", ())),
                {"validation": _plain(result.validation), "beliefs": dict(result.beliefs)},
            )

        if module_id == "cognitive_heuristics":
            request = module_inputs.get("judgment_problem")
            if request is None:
                return ModuleAdapterResult(module_id, "unavailable", {}, diagnostics={"reason": "typed_judgment_problem_not_supplied"})
            processor = self.processors.get(module_id)
            if processor is None:
                from ore_cognitive_heuristics import Point22CognitiveHeuristicsProcessor, Point22Config
                processor = Point22CognitiveHeuristicsProcessor(config=Point22Config(simulate_biology=False))
            from ore_cognitive_heuristics import JudgmentProblem
            if not isinstance(request, JudgmentProblem):
                raise TypeError("module_inputs.judgment_problem must be a JudgmentProblem")
            result = processor.process(request)
            return ModuleAdapterResult(module_id, "completed", {"result": _plain(result)}, diagnostics={"warnings": list(result.warnings), "diagnostics": _plain(result.diagnostics)})

        if module_id == "curiosity_motivation":
            request = module_inputs.get("learning_request")
            if request is None:
                return ModuleAdapterResult(module_id, "unavailable", {}, diagnostics={"reason": "typed_learning_request_not_supplied"})
            processor = self.processors.get(module_id)
            if processor is None:
                from ore_curiosity_motivation import Point08MotivatedLearningProcessor
                processor = Point08MotivatedLearningProcessor(simulate_biology=False)
            from ore_curiosity_motivation import LearningRequest
            if not isinstance(request, LearningRequest):
                raise TypeError("module_inputs.learning_request must be a LearningRequest")
            result = processor.process(request)
            return ModuleAdapterResult(module_id, "completed", {"result": result.to_dict()}, clarification_options=tuple(module_inputs.get("clarification_options", ())), diagnostics=_plain(result.diagnostics))

        if module_id == "semantic_reprocessing":
            request = module_inputs.get("n9_request")
            processor = self.processors.get(module_id, self.processors.get("n9"))
            if request is None or processor is None:
                return ModuleAdapterResult(module_id, "unavailable", {}, diagnostics={"reason": "n9_processor_or_typed_request_not_supplied"})
            from ore_n9.feedback_adapter import N9FeedbackAdapter
            adapter = N9FeedbackAdapter(processor)
            result = adapter.reprocess(
                request,
                payload.get("revised_interpretation", {}),
                payload.get("affected_components", ()),
            )
            return ModuleAdapterResult(module_id, str(result.get("status", "unavailable")), result, diagnostics={"learning_authorized": False})

        if module_id == "memory_retrieval":
            query = module_inputs.get("memory_query")
            store = module_inputs.get("memory_store")
            coordinator = module_inputs.get("memory_coordinator")
            if store is None and coordinator is not None:
                store = getattr(coordinator, "memory_store", None)
            if query is None or store is None:
                return ModuleAdapterResult(module_id, "unavailable", {}, diagnostics={"reason": "memory_query_or_store_not_supplied"})
            from ore_memory_integration import MemoryEvidence, MemoryQuery, MemoryRetrievalCoordinator
            if not isinstance(query, MemoryQuery):
                raise TypeError("module_inputs.memory_query must be a MemoryQuery")
            batch = MemoryRetrievalCoordinator(store).retrieve(query)
            evidence = []
            for item in batch.items:
                likelihoods = item.provenance.get("hypothesis_likelihoods")
                if not isinstance(likelihoods, Mapping) or not likelihoods:
                    continue
                from .contracts import FeedbackEvidence as P24Evidence
                evidence.append(P24Evidence(
                    evidence_id=item.memory_id,
                    description=str(item.content),
                    likelihoods={str(k): float(v) for k, v in likelihoods.items()},
                    source_name=item.source_reference,
                    source_type="memory",
                    reliability=item.confidence,
                    relevance=item.relevance_score,
                    diagnosticity=float(item.provenance.get("diagnosticity", 0.5)),
                    directness="indirect",
                    independence_key=item.independence_group,
                    context=item.verification_status,
                ))
            return ModuleAdapterResult(
                module_id, "completed", {"memory_batch": batch.to_dict()},
                tuple(evidence),
                tuple(module_inputs.get("clarification_options", ())),
                {"candidate_count": len(batch.items), "converted_likelihood_records": len(evidence), "recall_is_read_only": True},
            )

        return ModuleAdapterResult(module_id, "unavailable", {}, diagnostics={"reason": "no_typed_adapter_registered"})

    def invoke(
        self,
        module_id: str,
        payload: Mapping[str, Any],
        module_inputs: Mapping[str, Any] | None = None,
    ) -> ModuleAdapterResult:
        """Invoke a caller handler or typed processor and isolate missing/failed module outcomes."""
        inputs = dict(module_inputs or {})
        handler = self.handlers.get(module_id)
        if handler is not None:
            try:
                raw = handler(dict(payload))
                if raw is None:
                    return ModuleAdapterResult(module_id, "completed_no_output", {})
                if not isinstance(raw, Mapping):
                    raise TypeError("module handler must return a mapping")
                evidence = self._evidence_from_handler(raw)
                return ModuleAdapterResult(
                    module_id,
                    str(raw.get("status", "completed")),
                    dict(raw),
                    evidence,
                    tuple(raw.get("clarification_options", ())),
                    dict(raw.get("diagnostics", {})) if isinstance(raw.get("diagnostics", {}), Mapping) else {},
                )
            except Exception as exc:
                return ModuleAdapterResult(module_id, "failed", {}, diagnostics={"error": f"{type(exc).__name__}: {exc}"})
        try:
            return self._invoke_existing_processor(module_id, payload, inputs)
        except Exception as exc:
            return ModuleAdapterResult(module_id, "failed", {}, diagnostics={"error": f"{type(exc).__name__}: {exc}"})

    def reprocess(
        self,
        module_ids: Sequence[str],
        payload: Mapping[str, Any],
        module_inputs: Mapping[str, Any] | None = None,
    ) -> tuple[ModuleAdapterResult, ...]:
        """Dispatch only the supplied affected module identifiers and preserve availability status."""
        results = []
        for module_id in dict.fromkeys(str(item) for item in module_ids):
            results.append(self.invoke(module_id, payload, module_inputs))
        return tuple(results)
