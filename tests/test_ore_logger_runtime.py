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
