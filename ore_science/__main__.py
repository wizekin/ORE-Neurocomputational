"""Run the scientific reasoning demo from the command line."""
from __future__ import annotations

import argparse
import json
from typing import Any

from .contracts import ScienceRequest
from .pipeline import ScientificReasoner

_DEMO_QUESTIONS: tuple[str, ...] = (
    "Predict the temperature field of a 1 m rod after 60 s with thermal diffusivity "
    "alpha = 1e-4 m2/s, initial temperature 100 degC, ends held at 0 degC",
    "Forecast the ensemble weather distribution of wind speed 48 h ahead",
    "Use SLAM with odometry and landmarks to localize a robot pose",
    "Discover stable crystal materials with a perovskite oxide lattice for battery electrodes",
    "Predict the structure of a protein backbone and its interaction with a ligand",
    "Check the rotation equivariance of the geometric transform for a point cloud",
)


def main(argv: list[str] | None = None) -> int:
    """Execute the demonstration questions and print a compact report for each."""
    parser = argparse.ArgumentParser(description="ORE scientific reasoning layer demo")
    parser.add_argument("--output", type=str, default=None, help="write results JSON to this path")
    parser.add_argument("--question", type=str, default=None, help="run a single custom question")
    args = parser.parse_args(argv)

    reasoner = ScientificReasoner()
    questions = (args.question,) if args.question else _DEMO_QUESTIONS
    all_results: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        result = reasoner.process(ScienceRequest(question=question, request_id=f"demo_{index:02d}"))
        summary = {
            "request_id": result.request_id,
            "question": question,
            "outcome": result.outcome.value,
            "result_status": result.result_status,
            "valid": result.diagnostics.get("valid"),
            "domains": (result.domain_decision or {}).get("domains"),
            "predictions": [item["model_id"] for item in result.predictions],
            "operation_count": result.diagnostics.get("operation_count"),
            "executed_count": result.diagnostics.get("executed_count"),
            "validation_passed": [report.get("passed") for report in result.validation_reports],
            "provenance": (result.provenance_ref or {}).get("provenance_id"),
        }
        all_results.append(summary)
        print(f"[{index}/{len(questions)}] {question[:70]}...")
        print(
            f"    outcome={summary['outcome']} status={summary['result_status']} "
            f"valid={summary['valid']} models={summary['predictions']}"
        )
    if args.output:
        payload = {"schema": "ore.science.demo.v1", "results": all_results}
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"Wrote {len(all_results)} result summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
