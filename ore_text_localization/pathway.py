"""Cooperate across visual localization, N9 language processing and working memory."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .annotations import register_visual_language_annotations
from .config import TextLocalizationConfig
from .contracts import TextImageRequest, TextLocalizationResult, TrainingGlyph
from .localization import (
    TextLocalizer,
    _make_ink_mask,
    _resize_to_canvas,
    normalize_grayscale_image,
)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """Read an existing result mapping without mutating its owner."""
    if isinstance(value, Mapping):
        return value
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, Mapping):
            return converted
    raise TypeError("result must be a mapping or expose to_dict()")


def _unit_score(value: Any) -> float | None:
    """Return a finite normalized score, or None when evidence is absent or malformed."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and 0.0 <= number <= 1.0 else None


@dataclass(frozen=True, slots=True)
class VisualLanguagePathwayResult:
    """Expose each pathway result separately so downstream modules can inspect provenance."""

    sequence_id: str
    status: str
    localization: Mapping[str, Any]
    n9: Mapping[str, Any] | None
    working_memory: Mapping[str, Any] | None
    integration: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the multi-stage result without copying the original page pixel arrays."""
        return {
            "schema": "ore.visual_language_pathway.v1",
            "sequence_id": self.sequence_id,
            "status": self.status,
            "localization": dict(self.localization),
            "n9": dict(self.n9) if self.n9 is not None else None,
            "working_memory": dict(self.working_memory) if self.working_memory is not None else None,
            "integration": dict(self.integration),
        }


class N9WorkingMemoryAdapter:
    """Convert N9 language outputs into the established working-memory request contract."""

    def __init__(self, processor: Any | None = None) -> None:
        """Use a caller-configured processor or construct the repository working-memory engine."""
        if processor is None:
            from ore_working_memory import WorkingMemoryProcessor
            processor = WorkingMemoryProcessor()
        if not callable(getattr(processor, "process", None)):
            raise TypeError("working-memory processor must expose process(WorkingMemoryRequest)")
        self.processor = processor

    def process(
        self,
        n9_result: Any,
        *,
        language: str = "en",
        task_sequence: Sequence[str] = ("visual_language_comprehension",),
    ) -> dict[str, Any]:
        """Preserve lexical order, syntax, propositions and unresolved dependencies in working memory."""
        from ore_working_memory import WorkingMemoryRequest

        payload = _as_mapping(n9_result)
        syntax_value = payload.get("syntax", {})
        syntax = _as_mapping(syntax_value) if syntax_value else {}
        lexical_rows = [
            dict(item) for item in (payload.get("lexical_hypotheses", ()) or ())
            if isinstance(item, Mapping)
        ]

        def lexical_position(item: Mapping[str, Any]) -> tuple[int, int]:
            """Return a sortable token-span key even when an adapter omits optional offsets."""
            try:
                start = int(item.get("start", 0))
            except (TypeError, ValueError):
                start = 0
            try:
                end = int(item.get("end", 0))
            except (TypeError, ValueError):
                end = 0
            return start, end

        lexical_rows.sort(key=lexical_position)
        tokens = [
            str(item.get("form", item.get("token", ""))).strip()
            for item in lexical_rows
            if str(item.get("form", item.get("token", ""))).strip()
        ]
        if not tokens:
            syntax_tokens = syntax.get("tokens", ())
            if isinstance(syntax_tokens, Sequence) and not isinstance(syntax_tokens, (str, bytes)):
                tokens = [str(item) for item in syntax_tokens if str(item).strip()]
        if not tokens:
            return {
                "status": "unavailable_no_lexical_tokens",
                "working_memory": None,
                "handoff": "not_performed",
                "reason": "N9 did not emit an ordered lexical token sequence",
            }

        decision = payload.get("decision", {})
        decision = decision if isinstance(decision, Mapping) else {}
        global_confidence = _unit_score(decision.get("confidence"))
        propositions = [
            dict(item) for item in (payload.get("propositions", ()) or ())
            if isinstance(item, Mapping)
        ]
        interpretations: list[dict[str, Any]] = []
        for index, proposition in enumerate(propositions):
            candidate: dict[str, Any] = {
                "hypothesis_id": str(proposition.get("proposition_id", f"n9-proposition-{index + 1}")),
                "text": str(proposition.get("text", proposition.get("predicate", ""))),
                "evidence_source": "ore_n9.semantic_composition",
            }
            explicit_confidence = _unit_score(
                proposition.get("confidence", proposition.get("score"))
            )
            if explicit_confidence is not None:
                candidate["prior"] = explicit_confidence
            elif global_confidence is not None:
                candidate["prior"] = global_confidence
            if "well_formed" in proposition:
                candidate["semantic_score"] = 1.0 if proposition["well_formed"] else 0.0
            feature_map = proposition.get("semantic_features", {})
            if isinstance(feature_map, Mapping):
                bounded = {
                    str(key): score for key, raw in feature_map.items()
                    if (score := _unit_score(raw)) is not None
                }
                if bounded:
                    candidate["features"] = bounded
            interpretations.append(candidate)

        unresolved_dependencies = [
            dict(item) for item in (syntax.get("long_distance_dependencies", ()) or ())
            if isinstance(item, Mapping) and (
                item.get("unresolved") is True
                or item.get("resolved") is False
                or item.get("status") == "open"
            )
        ]
        partial_structures = [
            dict(item) for item in (syntax.get("incomplete_structures", ()) or ())
            if isinstance(item, Mapping)
        ]
        graph_value = payload.get("semantic_graph", {})
        graph = dict(graph_value) if isinstance(graph_value, Mapping) else {}
        context_value = payload.get("context_model", {})
        context_model = dict(context_value) if isinstance(context_value, Mapping) else {}
        context = {
            "current_goal": "understand_visual_language_input",
            "end_of_input": True,
            "n9_decision": dict(decision),
            "semantic_graph": graph,
            "propositions": propositions,
            "discourse_context": context_model,
            "unresolved_dependencies": unresolved_dependencies,
            "ambiguity_sources": (
                [{"kind": "n9_abstention_or_ambiguity", "status": decision.get("status")}]
                if decision.get("status") in {"ABSTAIN", "AMBIGUOUS"} else []
            ),
            "source_reference": "ore_n9.VisualLanguagePathway",
        }
        request = WorkingMemoryRequest(
            sequence_id=str(payload.get("sequence_id", "visual-language")),
            tokens=tuple(tokens),
            syntax_result=dict(syntax) if syntax else None,
            partial_structures=tuple(partial_structures),
            interpretations=tuple(interpretations),
            task_sequence=tuple(task_sequence),
            goals={"name": "understand_visual_language_input", "priority": 0.7},
            context=context,
            language=language,
            source_reference="ore_text_localization.N9WorkingMemoryAdapter",
        )
        result = self.processor.process(request)
        result_payload = dict(_as_mapping(result))
        return {
            "status": "completed",
            "handoff": "working_memory_request_processed",
            "tokens": tokens,
            "interpretation_count": len(interpretations),
            "unresolved_dependency_count": len(unresolved_dependencies),
            "working_memory": result_payload,
        }


class VisualLanguagePathway:
    """Coordinate visual localization, N9 language modules and the working-memory handoff.

    The class is a pathway-level orchestrator, not a replacement for the main brain coordinator.
    A caller may supply that coordinator through a callback and/or publish a validated finding
    through the shared ORE workspace. The N3 recognizer and lexical/semantic inventories remain
    caller-provided resources; the pathway never fabricates training labels or concept records.
    """

    def __init__(
        self,
        *,
        localizer: TextLocalizer | None = None,
        n9_processor: Any | None = None,
        working_memory_processor: Any | None = None,
        localization_config: TextLocalizationConfig | None = None,
        coordinator_handoff: Callable[[Mapping[str, Any]], Any] | None = None,
        workspace: Any | None = None,
        finding_publisher: Any | None = None,
        register_annotations: bool = True,
        repository_root: str | None = None,
        database_path: str | None = None,
    ) -> None:
        """Bind stage processors and optionally index pathway/dependency symbols in ORE SQLite."""
        from ore_n9 import N9Processor

        self.localizer = localizer or TextLocalizer(localization_config)
        self.n9_processor = n9_processor or N9Processor()
        if not callable(getattr(self.n9_processor, "process", None)):
            raise TypeError("n9_processor must expose process(N9Request)")
        self.working_memory = N9WorkingMemoryAdapter(working_memory_processor)
        self.coordinator_handoff = coordinator_handoff
        self.workspace = workspace
        self.finding_publisher = finding_publisher
        if workspace is not None and finding_publisher is not None:
            raise ValueError("provide workspace or finding_publisher, not both")
        self.annotation_report = (
            register_visual_language_annotations(repository_root, database_path)
            if register_annotations else {"status": "annotation_registration_disabled"}
        )

    @classmethod
    def from_processors(
        cls,
        *,
        lexical_access: Any,
        syntax: Any,
        semantic_composition: Any,
        discourse: Any | None = None,
        n1: Any | None = None,
        n2: Any | None = None,
        n3: Any | None = None,
        n4: Any | None = None,
        language: str = "en",
        **options: Any,
    ) -> "VisualLanguagePathway":
        """Build existing N9 N6/N7/N8/syntax adapters around caller-owned language inventories."""
        from ore_n9 import N9Adapters, N9Processor, build_n9_language_adapters

        base = N9Adapters(n1=n1, n2=n2, n3=n3, n4=n4)
        adapters = build_n9_language_adapters(
            lexical_access=lexical_access,
            syntax=syntax,
            semantic_composition=semantic_composition,
            discourse=discourse,
            base_adapters=base,
            language=language,
        )
        return cls(n9_processor=N9Processor(adapters=adapters), **options)

    def fit_grapheme_model(
        self,
        training_samples: Sequence[TrainingGlyph],
        *,
        grapheme_sequences: Sequence[Sequence[str]] = (),
        n3_config: Any | None = None,
    ) -> dict[str, Any]:
        """Fit N3 from caller-labelled glyph images using the same N1/N2 descriptor path as inference."""
        from ore_n1 import N1Processor
        from ore_n2 import N2Processor
        from ore_n3 import LabeledN2Patch, N3Processor

        samples = tuple(training_samples)
        if not samples or any(not isinstance(item, TrainingGlyph) for item in samples):
            raise ValueError("training_samples must contain TrainingGlyph records")
        adapters = self.n9_processor.adapters
        if adapters.n3 is None:
            adapters.n3 = N3Processor(n3_config)
        n1_processor = adapters.n1 or N1Processor()
        n2_processor = adapters.n2 or N2Processor()
        adapters.n1 = n1_processor
        adapters.n2 = n2_processor
        labelled: list[LabeledN2Patch] = []
        for sample in samples:
            gray = normalize_grayscale_image(sample.image)
            _, _, polarity = _make_ink_mask(gray, self.localizer.config.epsilon)
            if polarity == "light_on_dark":
                gray = 1.0 - gray
            crop = _resize_to_canvas(gray, self.localizer.config)
            n1_result = n1_processor.process(crop)
            n2_result = n2_processor.process(n1_result)
            labelled.append(LabeledN2Patch(grapheme=sample.grapheme, n2_result=n2_result))
        adapters.n3.fit(
            labelled,
            grapheme_sequences=tuple(tuple(sequence) for sequence in grapheme_sequences),
        )
        return {
            "status": "fitted",
            "sample_count": len(labelled),
            "class_names": list(adapters.n3.class_names),
            "fingerprint": getattr(adapters.n3, "fingerprint", None),
            "training_is_caller_supplied": True,
            "parent_inventory_mutated": False,
        }

    def process(
        self,
        request: TextImageRequest,
        *,
        task: str = "read",
        context: Mapping[str, Any] | None = None,
        task_sequence: Sequence[str] = ("visual_language_comprehension",),
    ) -> VisualLanguagePathwayResult:
        """Run localization, N9 recognition/composition, working memory and optional coordinator handoff."""
        if not isinstance(request, TextImageRequest):
            raise TypeError("request must be a TextImageRequest")
        localization_result: TextLocalizationResult = self.localizer.localize(request)
        localization_payload = localization_result.to_dict()
        if not localization_result.localized_glyphs:
            integration = {
                "status": "no_visual_text_candidates",
                "modules": {"localization": "completed", "n9": "not_run", "working_memory": "not_run"},
                "coordinator_handoff": "not_configured",
                "workspace_publication": "not_attempted",
                "annotation_catalog": dict(self.annotation_report),
            }
            result = VisualLanguagePathwayResult(
                sequence_id=request.sequence_id,
                status="UNAVAILABLE",
                localization=localization_payload,
                n9=None,
                working_memory=None,
                integration=integration,
            )
            self._send_downstream(result)
            return result

        n9_request = localization_result.to_n9_request(
            task=task,
            context={
                **dict(context or {}),
                "localization": localization_payload,
                "script_profile": dict(request.script_profile),
            },
            source_id=request.source_id,
        )
        n9_result = self.n9_processor.process(n9_request)
        n9_payload = dict(_as_mapping(n9_result))
        language = str(request.script_profile.get("language", "en"))
        memory_payload = self.working_memory.process(
            n9_payload, language=language, task_sequence=task_sequence,
        )
        memory_result = memory_payload.get("working_memory")
        class_names = [str(item) for item in n9_payload.get("class_names", ())]
        decision = n9_payload.get("decision", {})
        decision = decision if isinstance(decision, Mapping) else {}
        lexical_rows = n9_payload.get("lexical_hypotheses", ()) or ()
        semantic_graph = n9_payload.get("semantic_graph", {})
        semantic_graph = semantic_graph if isinstance(semantic_graph, Mapping) else {}
        semantic_evidence = bool(n9_payload.get("propositions")) or bool(semantic_graph.get("nodes"))
        adapters = self.n9_processor.adapters
        n3_fitted = (
            adapters.n3 is not None and bool(getattr(adapters.n3, "_fitted", False))
        )
        recognition_available = len(class_names) > 1 and n3_fitted
        language_adapters_available = all((
            adapters.n6 is not None,
            adapters.syntax is not None,
            adapters.n8 is not None,
            adapters.n7 is not None,
        ))
        complete = (
            decision.get("status") == "CANDIDATE"
            and bool(lexical_rows)
            and semantic_evidence
            and memory_result is not None
            and recognition_available
            and language_adapters_available
        )
        status = "COMPLETED" if complete else "PARTIAL"
        integration = {
            "status": "all_required_evidence_present" if complete else "partial_or_abstaining",
            "modules": {
                "text_localization": "completed",
                "n1": "configured" if adapters.n1 is not None else "N9_default_processor",
                "n2": "configured" if adapters.n2 is not None else "N9_default_processor",
                "n3": "fitted" if n3_fitted else "not_fitted_or_not_configured",
                "n4": "configured" if adapters.n4 is not None else (
                    "N9_default_when_N3_available" if recognition_available else "not_run_without_N3"
                ),
                "n6_lexical_access": "configured" if adapters.n6 is not None else "not_configured",
                "syntax": "configured" if adapters.syntax is not None else "not_configured",
                "n8_semantic_composition": "configured" if adapters.n8 is not None else "not_configured",
                "n7_discourse": "configured" if adapters.n7 is not None else "not_configured",
                "working_memory": memory_payload.get("status", "not_run"),
            },
            "n9_decision_status": decision.get("status"),
            "recognition_available": recognition_available,
            "lexical_hypothesis_count": len(lexical_rows),
            "semantic_evidence_available": semantic_evidence,
            "working_memory_available": memory_result is not None,
            "visual_evidence_preserved_separately": True,
            "main_coordinator_api_assumed": False,
            "coordinator_handoff": "not_configured",
            "workspace_publication": "not_attempted",
            "annotation_catalog": dict(self.annotation_report),
            "limitations": [
                "COMPLETED means the configured software handoffs returned evidence; it is not a claim of human-level accuracy",
                "N3 must be fitted with caller-labelled examples and lexical/semantic inventories must be supplied",
                "image localization is a constrained projection proposal, not a general OCR system",
            ],
        }
        result = VisualLanguagePathwayResult(
            sequence_id=request.sequence_id,
            status=status,
            localization=localization_payload,
            n9=n9_payload,
            working_memory=memory_payload,
            integration=integration,
        )
        self._send_downstream(result)
        return result

    def _send_downstream(self, result: VisualLanguagePathwayResult) -> None:
        """Offer a detached result to the main coordinator callback or shared workspace publisher."""
        integration = result.integration
        payload = result.to_dict()
        if self.coordinator_handoff is not None:
            try:
                self.coordinator_handoff({
                    "producer_module": "ore_text_localization",
                    "sequence_id": result.sequence_id,
                    "finding": payload,
                })
                if isinstance(integration, dict):
                    integration["coordinator_handoff"] = "sent"
            except Exception as exc:
                if isinstance(integration, dict):
                    integration["coordinator_handoff"] = f"failed:{type(exc).__name__}"
        if self.finding_publisher is not None or self.workspace is not None:
            try:
                publisher = self.finding_publisher
                if publisher is None:
                    from ore_workspace import FindingPublisher
                    publisher = FindingPublisher(self.workspace, "ore_text_localization")
                decision = (result.n9 or {}).get("decision", {})
                confidence = _unit_score(
                    decision.get("confidence") if isinstance(decision, Mapping) else None
                )
                from ore_workspace import Provenance
                published = publisher.publish(
                    "visual_language_pathway",
                    payload,
                    finding_type="visual_language_interpretation",
                    provenance=Provenance(
                        source="ore_text_localization.VisualLanguagePathway",
                        rules_applied=(
                            "projection_based_text_localization",
                            "N9_evidence_preserving_language_integration",
                            "explicit_working_memory_handoff",
                        ),
                        notes="Computational pathway output; not measured neural activity.",
                    ),
                    confidence=confidence,
                )
                decision_value = getattr(published, "decision", None)
                if isinstance(integration, dict):
                    integration["workspace_publication"] = (
                        str(getattr(decision_value, "value", decision_value))
                        if decision_value is not None else "submitted"
                    )
            except Exception as exc:
                if isinstance(integration, dict):
                    integration["workspace_publication"] = f"failed:{type(exc).__name__}"


__all__ = [
    "N9WorkingMemoryAdapter",
    "VisualLanguagePathway",
    "VisualLanguagePathwayResult",
]
