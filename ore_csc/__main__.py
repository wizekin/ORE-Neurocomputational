"""Run the CSC ambiguous-word demonstration and populate the shared SQLite logger."""
from __future__ import annotations

from pathlib import Path
import json

from ore_semantic.contracts import SemanticConcept

from . import CSCConfig, CSCProcessor, CSCRequest, index_csc_annotations, record_csc_result


def _demo_concepts() -> list[SemanticConcept]:
    """Build two provenance-tagged meanings of the ambiguous word bank for the demo."""
    return [
        SemanticConcept(
            concept_id="bank-financial",
            label="financial bank",
            aliases=("bank", "financial institution"),
            semantic_features={
                "FINANCIAL_INSTITUTION": 1.0, "MONEY": 0.95,
                "TRANSACTIONS": 0.9, "CUSTOMERS": 0.7, "BUILDING": 0.6,
            },
            frequency=80.0,
            verified_by="CSC demonstration fixture",
            source_reference="demo://bank/financial",
        ),
        SemanticConcept(
            concept_id="bank-river",
            label="river bank",
            aliases=("bank", "riverbank", "shore"),
            semantic_features={
                "RIVER": 1.0, "WATER": 0.95, "SHORE": 0.9,
                "LANDSCAPE": 0.8, "OUTDOORS": 0.7,
            },
            frequency=12.0,
            verified_by="CSC demonstration fixture",
            source_reference="demo://bank/river",
        ),
    ]


def main() -> None:
    """Run one controlled semantic trial, index symbol annotations, and save its event trace."""
    root = Path(__file__).resolve().parents[1]
    concepts = _demo_concepts()
    processor = CSCProcessor(concepts, CSCConfig(simulate_biology=True))
    request = CSCRequest(
        sequence_id="csc-bank-demo",
        text="The fisherman sat beside the bank.",
        task="interpret the geographical meaning",
        context_features={"RIVER": 0.95, "WATER": 0.9, "FISHERMAN": 0.8},
        task_features={"RIVER": 0.9, "WATER": 0.9, "SHORE": 0.85},
        source_reference="Controlled Semantic Cognition (CSC).txt",
    )
    result = processor.process(request)
    annotation_report = index_csc_annotations(root)
    logger_report = record_csc_result(result, root)
    print(json.dumps({
        "selected_concept": result.selected_concept,
        "response": result.response,
        "stages": len(result.stage_trace),
        "operations": len(result.operation_trace),
        "diagnostics": result.diagnostics,
        "annotation_catalog": annotation_report,
        "logger": logger_report,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
