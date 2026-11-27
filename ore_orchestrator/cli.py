"""Command-line entry point for the unified ORE orchestrator."""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from .catalog import OREPackageCatalog
from .orchestrator import OREOrchestrator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ore",
        description="Discover and run the ORE Neurocomputational repository.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory", help="list discovered Python packages")
    inventory.add_argument("--category", help="optional package category filter")

    text = subparsers.add_parser("text", help="run the integrated text-comprehension workflow")
    text.add_argument("text", help="text to process")
    text.add_argument("--language", default="en")
    text.add_argument("--sequence-id")
    text.add_argument("--no-working-memory", action="store_true")
    text.add_argument("--no-discourse", action="store_true")
    text.add_argument("--simulate-biology", action="store_true")
    text.add_argument("--no-point-integration", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inventory":
        catalog = OREPackageCatalog()
        packages = catalog.list_packages(args.category)
        payload = {
            "repository_root": str(catalog.repository_root),
            "package_count": len(packages),
            "packages": [item.to_dict() for item in packages],
        }
    else:
        orchestrator = OREOrchestrator(
            simulate_biology=args.simulate_biology,
            integrate_point_modules=not args.no_point_integration,
        )
        report = orchestrator.run_text(
            args.text,
            language=args.language,
            sequence_id=args.sequence_id,
            include_working_memory=not args.no_working_memory,
            include_discourse=not args.no_discourse,
        )
        payload = report.to_dict()

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0 if args.command == "inventory" or payload.get("status") in {"completed", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
