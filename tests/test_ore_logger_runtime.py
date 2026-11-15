from pathlib import Path

from ore_logger.database import LoggerDatabase
from ore_logger.runtime import RuntimeSession, _candidate_script


def _runtime_probe(value):
    return value + 10


def test_direct_script_and_module_entry_point_detection():
    root = Path(__file__).resolve().parents[1]
    script = root / "examples" / "basic_pipeline.py"
    detected, kind = _candidate_script([str(script)], root)
    assert detected == script.resolve()
    assert kind == "script"

    module_script, module_kind = _candidate_script(["python", "-m", "ore_n1"], root)
    assert module_script is not None
    assert module_script.name == "__main__.py"
    assert module_kind == "module"


def test_runtime_session_records_calls_and_returns_for_script_symbols(tmp_path):
    root = Path(__file__).resolve().parents[1]
    database_path = tmp_path / "runtime.sqlite3"
    session = RuntimeSession(
        repository_root=root,
        script_path=Path(__file__),
        database_path=database_path,
    )
    assert _runtime_probe(5) == 15
    session.close("completed")

    database = LoggerDatabase(database_path)
    run = database.get_run(session.run_id)
    events = database.list_events(session.run_id)
    probe_events = [
        event for event in events
        if event.get("qualified_name") == "_runtime_probe"
    ]

    assert run["status"] == "completed"
    assert run["started_at_utc"]
    assert run["started_at_local"]
    assert any(event["event_kind"] == "call" for event in probe_events)
    assert any(event["event_kind"] == "return" for event in probe_events)
    assert all(event.get("annotation") for event in probe_events)



def test_repository_catalog_indexes_the_full_python_source_tree(tmp_path):
    from ore_logger.catalog import sync_repository

    root = Path(__file__).resolve().parents[1]
    database = LoggerDatabase(tmp_path / "full-catalog.sqlite3")
    report = sync_repository(database, root)

    assert report["files_seen"] >= 100
    assert report["files_indexed"] >= 100
    assert report["parse_errors"] == 0
    assert database.get_annotation("ore_n9.pipeline", "N9Processor", "class")
    assert database.get_annotation("ore_n9.pipeline", "N9Processor.process", "function")
    assert database.get_annotation("ore_n1.pipeline", "N1Processor.process", "function")
    assert database.get_annotation("ore_syntax.pipeline", "SyntaxProcessor.process", "function") or database.search_annotations("SyntaxProcessor", limit=20)


def test_streamlit_viewer_has_valid_python_syntax():
    import ast

    root = Path(__file__).resolve().parents[1]
    viewer = root / "ore_logger" / "viewer.py"
    ast.parse(viewer.read_text(encoding="utf-8"), filename=str(viewer))
