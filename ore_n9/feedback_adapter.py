"""Read-only adapter for targeted semantic reprocessing through the N9 coordinator."""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping, Sequence
from typing import Any

from .contracts import N9Request


class N9FeedbackAdapter:
    """Re-enter N9 with a revised context while preserving the original request and model state."""

    def __init__(self, processor: Any) -> None:
        """Bind an injected N9 processor without taking ownership of its model resources."""
        if not callable(getattr(processor, "process", None)):
            raise TypeError("processor must expose process(N9Request)")
        self.processor = processor

    def reprocess(
        self,
        request: N9Request,
        revised_interpretation: Mapping[str, Any],
        affected_components: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Run one N9 pass on a detached request and explicitly disable inference-time learning."""
        if not isinstance(request, N9Request):
            raise TypeError("request must be an N9Request")
        if not isinstance(revised_interpretation, Mapping):
            raise TypeError("revised_interpretation must be a mapping")
        detached = deepcopy(request)
        context = deepcopy(dict(detached.context))
        context["recurrent_feedback"] = {
            "revised_interpretation": deepcopy(dict(revised_interpretation)),
            "affected_components": list(dict.fromkeys(str(item) for item in affected_components)),
            "source_sequence_id": request.sequence_id,
            "operation": "targeted_semantic_reprocessing",
        }
        detached.context = context
        detached.task = "recurrent_feedback_semantic_revision"
        detached.learning_authorized = False
        result = self.processor.process(detached)
        output = result.to_dict() if callable(getattr(result, "to_dict", None)) else result
        if not isinstance(output, Mapping):
            raise TypeError("N9 processor must return a mapping-compatible result")
        learning = dict(output.get("learning", {})) if isinstance(output.get("learning", {}), Mapping) else {}
        diagnostics = dict(output.get("diagnostics", {})) if isinstance(output.get("diagnostics", {}), Mapping) else {}
        return {
            "status": "completed",
            "source_sequence_id": request.sequence_id,
            "reprocessed_sequence_id": detached.sequence_id,
            "affected_components": list(dict.fromkeys(str(item) for item in affected_components)),
            "revised_n9_result": dict(output),
            "model_learning_authorized": False,
            "parent_learning_status": learning.get("status", "not_requested"),
            "parent_model_mutated": learning.get("parent_model_mutated", False),
            "diagnostics": diagnostics,
        }


__all__ = ["N9FeedbackAdapter"]
