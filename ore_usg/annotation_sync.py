"""Synchronize AST annotations for USG/ORE integration code into ORE's SQLite DB."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ore_logger.catalog import sync_repository
from ore_logger.database import LoggerDatabase

DEFAULT_SCOPES = {
    "ore_usg", "ore_blackboard", "ore_contracts", "ore_logger",
}


def synchronize_annotations(
    repository_root: str | Path,
    *,
    database_path: str | Path | None = None,
    include_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Index module, class, function and method annotations for integration packages."""
    root = Path(repository_root).resolve()
    database = LoggerDatabase(database_path or root / "ore_logs" / "ore_logger.sqlite3")
    report = sync_repository(database, root, include_paths or DEFAULT_SCOPES)
    with database.connection() as connection:
        rows = connection.execute(
            """SELECT source_path, symbol_kind, COUNT(*) AS count
               FROM annotations
               WHERE source_path LIKE 'ore_usg/%'
                  OR source_path LIKE 'ore_blackboard/%'
                  OR source_path LIKE 'ore_contracts/%'
                  OR source_path LIKE 'ore_logger/%'
               GROUP BY source_path, symbol_kind
               ORDER BY source_path, symbol_kind"""
        ).fetchall()
    coverage = [
        {"source_path": row["source_path"], "symbol_kind": row["symbol_kind"], "count": row["count"]}
        for row in rows
    ]
    return {
        **report,
        "database_path": str(database.path),
        "annotated_source_groups": coverage,
        "annotated_symbols": sum(item["count"] for item in coverage),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--database", default=None)
    args = parser.parse_args()
    report = synchronize_annotations(args.repository_root, database_path=args.database)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report["parse_errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_SCOPES", "synchronize_annotations", "main"]
