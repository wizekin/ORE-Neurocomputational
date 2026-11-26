"""Run a small self-contained Cognitive World Model demonstration from the CLI."""
from __future__ import annotations

import json

from ore_workspace import WorkspaceManager

from . import (
    AdvisoryMode,
    CognitiveOperation,
    CognitiveWorldModel,
    CWMBudget,
    CWMConfig,
    OperationType,
)


def main() -> None:
    """Execute and print one read-only CWM prediction and advisory planning round."""
    config = CWMConfig(min_support=1, max_candidates=4)
    model = CognitiveWorldModel(config=config)
    with WorkspaceManager(enable_logging=False) as manager:
        workspace = manager.create_workspace(
            current_input="The bilingual student translated the ambiguous sentence.",
            current_goal="understand_sentence",
        )
        candidates = (
            CognitiveOperation(
                operation_type=OperationType.MODULE_EXECUTION,
                capability_id="language.lexical_processing",
                module_id="lexical_processor",
                output_schema=("ORE.LEXICON@1.0.0",),
                produced_fields=("lexical_hypotheses",),
                timeout_ms=400.0,
            ),
            CognitiveOperation(
                operation_type=OperationType.MODULE_EXECUTION,
                capability_id="language.semantic_composition",
                module_id="semantic_processor",
                output_schema=("ORE.KNOWLEDGE@1.0.0",),
                produced_fields=("semantic_hypotheses",),
                timeout_ms=600.0,
            ),
        )
        snapshot = workspace.snapshot()
        records = model.predict_outcomes(snapshot, candidates, advisory_mode="shadow")
        plan = model.plan(
            snapshot, candidates,
            budget=CWMBudget(horizon=2, max_candidates=4),
            advisory_mode=AdvisoryMode.SHADOW.value,
        )
        report = {
            "workspace_revision": snapshot.revision,
            "predictions": [
                {
                    "operation_id": item.operation_id,
                    "predicted_success": round(item.predicted.success_probability, 3),
                    "sample_count": item.predicted.uncertainty.sample_count,
                    "record_kind": item.record_kind,
                }
                for item in records
            ],
            "plan": {
                "fallback": (
                    None if plan.fallback is None else plan.fallback.trigger.value),
                "first_action": (
                    None if plan.first_action is None else plan.first_action.module_id),
                "feasible_sequences": len(plan.ranked_sequences),
                "explanation": plan.explanation,
            },
            "workspace_revision_after_planning": workspace.revision,
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
