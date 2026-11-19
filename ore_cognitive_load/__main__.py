"""Run a small Point 06 example from the command line."""
from __future__ import annotations

import json

from . import Point06CognitiveLoadProcessor, Point06Request


def main() -> None:
    """Execute and print a self-contained cognitive-load demonstration."""
    request = Point06Request(
        sequence_id="point06-demo", goal="explain cellular respiration stages",
        units=["Glycolysis breaks down glucose.", "Electron carriers transport high-energy electrons.",
               "The electron transport chain builds a proton gradient.",
               "ATP synthase uses the gradient to produce ATP."],
        dependencies=[{"source": "u0001", "target": "u0002"},
                      {"source": "u0002", "target": "u0003"},
                      {"source": "u0003", "target": "u0004"}],
        metrics={"local_accuracy": 0.8, "relation_accuracy": 0.7,
                 "transfer_accuracy": 0.6, "explanation_accuracy": 0.8},
    )
    result = Point06CognitiveLoadProcessor().process(request)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()