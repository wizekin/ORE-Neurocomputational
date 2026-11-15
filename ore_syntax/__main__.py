"""Run a compact syntax-parser demonstration."""
from __future__ import annotations

import argparse
import json

from . import SyntaxConfig, SyntaxProcessor, SyntaxRequest


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ORE nine-stage syntax parser.")
    parser.add_argument("text", nargs="*", default=["The cat eats fish."])
    parser.add_argument("--no-biology", action="store_true", help="disable LIF reference dynamics")
    args = parser.parse_args()
    text = " ".join(args.text)
    result = SyntaxProcessor(config=SyntaxConfig(simulate_biology=not args.no_biology)).process(
        SyntaxRequest(sequence_id="syntax-demo", text=text, language="en")
    )
    payload = result.to_dict()
    print(json.dumps({
        "tokens": payload["tokens"],
        "pos": payload["selected_pos"],
        "phrases": payload["phrases"],
        "dependencies": payload["dependency_arcs"],
        "constituency": payload["constituency_tree"],
        "arguments": payload["predicate_arguments"],
        "thematic_roles": payload["thematic_roles"],
        "long_distance": payload["long_distance_dependencies"],
        "diagnostics": payload["diagnostics"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
