"""SQLite annotation and event-linkage tests for the universal module registry."""
from pathlib import Path

from ore_contracts.module_logging import (
    SQLiteModuleEventLogger, register_module_annotations,
)
from ore_logger.database import LoggerDatabase


def test_every_registry_definition_is_annotated_and_events_link_back_to_symbols(tmp_path):
    """Persist module_registry classes, functions, and methods in the existing logger database."""
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "ore_logger.sqlite3"
    catalogue = register_module_annotations(database_path, root)
    assert catalogue["parse_errors"] == 0
    assert catalogue["symbols_written"] > 0

    database = LoggerDatabase(database_path)
    class_row = database.get_annotation(
        "ore_contracts.module_registry", "ModuleRegistry", "class",
        "ore_contracts/module_registry.py",
    )
    method_row = database.get_annotation(
        "ore_contracts.module_registry", "ModuleRegistry.register", "function",
        "ore_contracts/module_registry.py",
    )
    assert class_row is not None
    assert method_row is not None
    run_id = None
    with SQLiteModuleEventLogger(database_path, root) as logger:
        run_id = logger.run_id
        logger.record("register", {"module_id": "ore.test"})
    with database.connection() as connection:
        event = connection.execute(
            """SELECT e.symbol_id, e.class_symbol_id, f.qualified_name AS function_name,
                      c.qualified_name AS class_name
               FROM events e
               LEFT JOIN annotations f ON f.symbol_id=e.symbol_id
               LEFT JOIN annotations c ON c.symbol_id=e.class_symbol_id
               WHERE e.run_id=?""",
            (run_id,),
        ).fetchone()
    assert event is not None
    assert event["function_name"] == "ModuleRegistry.register"
    assert event["class_name"] == "ModuleRegistry"
