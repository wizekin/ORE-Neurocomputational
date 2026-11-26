"""Synchronize AST annotations for USG/ORE integration code into ORE's SQLite DB."""
from __future__ import annotations

import argparse
import ast
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
    indexed_counts = {}
    for item in coverage:
        indexed_counts[item["source_path"]] = indexed_counts.get(item["source_path"], 0) + item["count"]

    expected_by_file: dict[str, int] = {}
    parse_failures: list[str] = []
    scopes = include_paths or DEFAULT_SCOPES
    for scope in scopes:
        scope_path = root / scope
        candidates = [scope_path] if scope_path.is_file() else (
            list(scope_path.rglob("*.py")) if scope_path.is_dir() else []
        )
        for candidate in candidates:
            if any(part.startswith(".") or part in {"__pycache__", ".venv", "venv"} for part in candidate.parts):
                continue
            relative = candidate.relative_to(root).as_posix()
            try:
                tree = ast.parse(candidate.read_text(encoding="utf-8"), filename=relative)
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                parse_failures.append(f"{relative}: {type(exc).__name__}: {exc}")
                continue
            definitions = sum(
                isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                for node in ast.walk(tree)
            )
            expected_by_file[relative] = definitions + 1  # the module-level annotation
    coverage_gaps = [
        {
            "source_path": source_path,
            "expected_symbols": expected,
            "annotated_symbols": indexed_counts.get(source_path, 0),
            "missing_symbols": expected - indexed_counts.get(source_path, 0),
        }
        for source_path, expected in sorted(expected_by_file.items())
        if indexed_counts.get(source_path, 0) != expected
    ]
    return {
        **report,
        "database_path": str(database.path),
        "annotated_source_groups": coverage,
        "annotated_symbols": sum(item["count"] for item in coverage),
        "expected_symbols": sum(expected_by_file.values()),
        "coverage_gaps": coverage_gaps,
        "coverage_parse_errors": parse_failures,
        "coverage_complete": not coverage_gaps and not parse_failures and report["parse_errors"] == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--database", default=None)
    args = parser.parse_args()
    report = synchronize_annotations(args.repository_root, database_path=args.database)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if not report["coverage_complete"] else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_SCOPES", "synchronize_annotations", "main"]
