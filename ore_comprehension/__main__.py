"""Command-line demonstration for the 38-operation comprehension pipeline."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ore_logger.catalog import sync_repository
from ore_logger.database import LoggerDatabase

from . import ComprehensionConfig, ComprehensionProcessor, ComprehensionRequest


def main() -> None:
    """Run sentence comprehension, synchronize all source annotations, and print a JSON report.

    Each run is indexed by the repository's standard logger. The default annotation and event
    database is ore_logs/ore_logger.sqlite3; set ORE_LOGGER_DB or --db-path to override it.
    """
    parser = argparse.ArgumentParser(
        description="Run all 38 TDPIA operations for each token in a sentence."
    )
    parser.add_argument(
        "text",
        nargs="*",
        default=["The chef prepared the meal with a knife."],
        help="Sentence text to comprehend.",
    )
    parser.add_argument("--sequence-id", default="tdpia-cli-demo")
    parser.add_argument("--language", default="en")
    parser.add_argument("--no-biology", action="store_true", help="Disable the LIF reference simulation.")
    parser.add_argument("--db-path", default=os.environ.get("ORE_LOGGER_DB"))
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    database_path = Path(args.db_path).expanduser() if args.db_path else (
        repository_root / "ore_logs" / "ore_logger.sqlite3"
    )
    database = LoggerDatabase(database_path)
    annotation_report = sync_repository(database, repository_root)

    processor = ComprehensionProcessor(
        ComprehensionConfig(simulate_biology=not args.no_biology)
    )
    request = ComprehensionRequest(
        sequence_id=args.sequence_id,
        text=" ".join(args.text),
        language=args.language,
    )
    result = processor.process(request)
    payload = result.to_dict()
    payload["logger"] = {
        "database_path": str(database.path),
        "annotation_sync": annotation_report,
        "annotation_counts": database.summary(),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
