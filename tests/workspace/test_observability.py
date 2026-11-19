"""Tests that workspace functions and classes are annotated in the existing ORE SQLite logger."""
import ast
from pathlib import Path

from ore_logger.database import LoggerDatabase
from ore_workspace import WorkspaceManager, register_workspace_annotations


def test_ast_annotation_catalogue_contains_workspace_classes_and_methods(tmp_path):
    """The existing logger catalogue indexes workspace module, class and callable definitions."""
    root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "ore_logger.sqlite3"
    report = register_workspace_annotations(database_path, root)
    assert report["parse_errors"] == 0
    assert report["files_seen"] >= 10
    database = LoggerDatabase(database_path)
    class_row = database.get_annotation(
        "ore_workspace.workspace", "WorkspaceManager", "class",
        "ore_workspace/workspace.py",
    )
    method_row = database.get_annotation(
        "ore_workspace.workspace", "WorkspaceManager.create_workspace", "function",
        "ore_workspace/workspace.py",
    )
    state_manager_row = database.get_annotation(
        "ore_workspace.state_manager", "WorkspaceStateManager.submit", "function",
        "ore_workspace/state_manager.py",
    )
    assert class_row and class_row["annotation"]
    assert method_row and method_row["annotation"]
    assert state_manager_row and state_manager_row["annotation"]

    source = root / "ore_workspace" / "scheduler.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definitions = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    with database.connection() as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM annotations WHERE source_path=? AND symbol_kind IN ('class','function','async_function')",
            ("ore_workspace/scheduler.py",),
        ).fetchone()[0]
    assert count >= len(definitions)


def test_workspace_manager_records_method_linked_runtime_events(tmp_path):
    """Creating and publishing through the manager creates event rows linked to annotations."""
    root = Path(__file__).resolve().parents[2]
    database_path = tmp_path / "ore_logger.sqlite3"
    with WorkspaceManager(repository_root=root, logger_database_path=database_path) as manager:
        workspace = manager.create_workspace("boy opened door")
        from ore_workspace import FindingPublisher
        finding = FindingPublisher(workspace, "lexical_processor").create_finding(
            "lexical_hypotheses", {"candidate": "PERSON"}, key="boy",
        )
        workspace.publish(finding)
        run_id = manager.observer.run_id
    database = LoggerDatabase(database_path)
    run = database.get_run(run_id)
    events = database.list_events(run_id)
    assert run["status"] == "completed"
    assert any(
        item.get("qualified_name") == "WorkspaceManager.create_workspace"
        and item.get("annotation") for item in events
    )
    assert any(
        item.get("qualified_name") == "WorkspaceStateManager.submit"
        and item.get("annotation") for item in events
    )