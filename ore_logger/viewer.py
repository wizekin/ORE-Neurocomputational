"""Streamlit viewer for ORE script runs and the SQLite annotation catalogue."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ore_logger.database import LoggerDatabase

DB_PATH = Path(os.getenv("ORE_LOGGER_DB") or ROOT / "ore_logs" / "ore_logger.sqlite3")
db = LoggerDatabase(DB_PATH)

st.set_page_config(page_title="ORE Execution Observatory", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {background:#0b1020;}
    [data-testid="stSidebar"] {background:#111a2e;border-right:1px solid #26344e;}
    .block-container {max-width:1700px;padding-top:1.3rem;}
    [data-testid="stMetric"] {background:#111a2e;border:1px solid #26344e;padding:10px;border-radius:12px;}
    </style>
    """, unsafe_allow_html=True,
)

st.title("🧠 ORE Execution Observatory")
st.caption("A searchable history of Python executions, with each event linked to its reusable class or function explanation.")

summary = db.summary()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Recorded runs", summary["runs"])
m2.metric("Events", f'{summary["events"]:,}')
m3.metric("Indexed symbols", f'{summary["symbols"]:,}')
m4.metric("Classes", summary["classes"])
m5.metric("Database size", f'{summary["database_bytes"] / 1024:.1f} KB')
st.caption(f"SQLite database: {DB_PATH}")

runs = db.list_runs(500)
run_tab, annotation_tab, about_tab = st.tabs(["Execution history", "Annotation catalogue", "How the logger works"])

with run_tab:
    if not runs:
        st.info("No Python script runs have been recorded yet. Install the repository in the environment and execute a Python file from the repository.")
    else:
        labels = {
            f'{run["display_label"]} · {run["script_path"]} · {run["status"]}': run["run_id"]
            for run in runs
        }
        selected_label = st.selectbox(
            "Choose a script execution", list(labels),
            help="Every launch receives its own timestamp and run identifier.",
        )
        run_id = labels[selected_label]
        selected_run = db.get_run(run_id)
        events = db.list_events(run_id)
        metric_cols = st.columns(4)
        metric_cols[0].metric("Status", selected_run["status"].upper())
        metric_cols[1].metric("Duration", f'{selected_run["duration_ms"] or 0:.2f} ms')
        metric_cols[2].metric("Recorded events", selected_run["event_count"])
        metric_cols[3].metric("Process ID", selected_run["process_id"] or "—")
        st.write(f'**Script:** {selected_run["script_path"]}')
        st.write(f'**Started (local):** {selected_run["started_at_local"]}')
        st.write(f'**Started (UTC):** {selected_run["started_at_utc"]}')
        if selected_run.get("module_name"):
            st.caption(f'Module: {selected_run["module_name"]} · Entry type: {selected_run["entry_kind"]}')
        if selected_run.get("error_summary"):
            st.error(selected_run["error_summary"])

        kind_options = sorted({event["event_kind"] for event in events})
        selected_kinds = st.multiselect(
            "Filter event types", kind_options, default=kind_options,
            key=f"event_kinds_{run_id}",
        )
        visible = [event for event in events if event["event_kind"] in selected_kinds]
        st.markdown("### Execution timeline")
        timeline_rows = []
        for event in visible:
            annotation = event.get("annotation") or ""
            if event.get("class_annotation"):
                annotation = event["class_annotation"] + " " + annotation
            timeline_rows.append({
                "#": event["sequence_no"],
                "elapsed (ms)": event["elapsed_ms"],
                "event": event["event_kind"],
                "symbol": event.get("qualified_name") or event.get("source_path") or "",
                "source": f'{event.get("source_path") or ""}:{event.get("line_number") or ""}',
                "duration (ms)": event.get("duration_ms") or "",
                "explanation": annotation or event.get("exception_message") or "",
            })
        if timeline_rows:
            st.dataframe(timeline_rows, use_container_width=True, hide_index=True)
            event_labels = [
                f'#{event["sequence_no"]:05d} · {event["event_kind"]} · '
                f'{event.get("qualified_name") or event.get("source_path") or "run event"}'
                for event in visible
            ]
            chosen_event_label = st.selectbox(
                "Inspect an event", event_labels, key=f"event_inspect_{run_id}",
            )
            event = visible[event_labels.index(chosen_event_label)]
            st.markdown("### Explanation")
            if event.get("qualified_name"):
                st.subheader(event["qualified_name"])
                st.caption(
                    f'{event.get("module_name", "")} · {event.get("symbol_kind", "")} · '
                    f'{event.get("source_path", "")}:{event.get("line_number") or ""}'
                )
                if event.get("annotation_source"):
                    st.caption(f'Annotation source: {event["annotation_source"]}')
                st.write(event.get("annotation") or "No annotation is currently stored for this symbol.")
            else:
                st.write(event.get("event_kind", "Event").replace("_", " ").title())
            if event.get("class_annotation"):
                st.markdown("**Containing class**")
                st.write(event["class_annotation"])
            if event.get("exception_message"):
                st.error(f'{event.get("exception_type") or "Exception"}: {event["exception_message"]}')
            if event.get("metadata"):
                with st.expander("Technical event details"):
                    st.json(event["metadata"])
        else:
            st.info("No events match the selected filters.")

        st.download_button(
            "Export this run with linked annotations",
            data=json.dumps(db.export_run(run_id), indent=2, ensure_ascii=False, default=str),
            file_name=f"ore-run-{run_id[:12]}.json",
            mime="application/json",
            key=f"export_{run_id}",
        )

with annotation_tab:
    st.markdown("### All indexed Python definitions")
    query = st.text_input(
        "Search modules, classes, functions, files or explanations",
        placeholder="For example: N1Processor, normalize, ore_n9.pipeline",
    )
    symbols = db.search_annotations(query, limit=5000)
    if not symbols:
        st.info("No annotations match that search.")
    else:
        symbol_labels = [
            f'{item["symbol_kind"]} · {item["module_name"]}.{item["qualified_name"]} '
            f'· {item["source_path"]}:{item["line_start"] or ""}'
            for item in symbols
        ]
        chosen_symbol = st.selectbox(
            f"Select a definition ({len(symbols)} result(s))",
            symbol_labels, key="annotation_symbol_selector",
        )
        symbol = symbols[symbol_labels.index(chosen_symbol)]
        st.subheader(symbol["qualified_name"])
        st.caption(
            f'{symbol["symbol_kind"]} · {symbol["module_name"]} · '
            f'{symbol["source_path"]}:{symbol["line_start"] or ""}–{symbol["line_end"] or ""}'
        )
        if symbol.get("signature"):
            st.code(symbol["qualified_name"] + symbol["signature"], language="python")
        if symbol.get("stale"):
            st.warning("The source code changed after this annotation was manually reviewed. Please review the explanation.")
        if symbol.get("source_docstring"):
            with st.expander("Original Python docstring"):
                st.write(symbol["source_docstring"])
        with st.form(f'edit_annotation_{symbol["symbol_id"]}'):
            edited = st.text_area(
                "Plain-English annotation", value=symbol["annotation"], height=220,
                help="This text is stored once in SQLite. Run events link to this record instead of copying the explanation into every log event.",
            )
            submitted = st.form_submit_button("Save annotation")
        if submitted:
            try:
                db.save_manual_annotation(symbol["symbol_id"], edited)
                st.success("Annotation saved in the SQLite catalogue.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save annotation: {type(exc).__name__}: {exc}")
        st.caption(f'Current source: {symbol["annotation_source"]} · Stale: {bool(symbol["stale"])}')

with about_tab:
    st.markdown(
        """
        ### How annotations are created

        The catalogue scans Python source using the Python AST parser. It indexes module definitions, classes, methods, nested functions, ordinary functions and asynchronous functions across the repository.

        Existing docstrings are used as the first explanation source. Where no docstring exists, the scanner builds a transparent source-based explanation using the symbol name, parameters, declared return type, helper calls and control-flow structures. Those generated descriptions are marked as source analysis; they are not presented as manually verified documentation.

        ### How run history is stored

        Each process launch or Workbench component execution receives a unique run ID and local/UTC timestamps. Function-call, return and exception events store a foreign-key reference to the annotation. The annotation itself is stored once in the annotations table; individual events do not duplicate its text.

        ### Automatic coverage

        The repository installs a Python startup hook when installed with pip. It starts tracing when the launched script is inside this repository. Set ORE_AUTO_LOG=0 to disable the hook for a process. Workbench executions are recorded as individual runs rather than being mixed into the long-lived Streamlit server process.

        ### Privacy and storage

        The logger records function identity, event type, source location, timestamps, duration, exception summaries and bounded technical metadata. It does not log Python argument values, raw prompts, images or full NumPy arrays. The database uses SQLite with WAL mode, so annotation text is stored once and events refer to it by ID.
        """
    )
