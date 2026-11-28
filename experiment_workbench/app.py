from __future__ import annotations

import json
import queue
import sys
import threading
import time
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from adapter import accepts_parameter, discover_callables, execute, json_safe
from auto_test import discover_ore_packages, run_auto_test_suite, save_manual_run_report

COMPONENTS = discover_ore_packages(ROOT)
st.set_page_config(page_title="ORE Experiment Workbench", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")


st.markdown("""
<style>
/* Main application */
[data-testid="stAppViewContainer"] {
    background: #FFFFFF !important;
    color: #172033 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #F1F5F9 !important;
    border-right: 1px solid #D8E0EA;
}

/* Main content */
.block-container {
    max-width: 1600px;
    padding-top: 1.35rem;
}

/* Subtitle */
.ore-subtitle {
    color: #475569 !important;
    margin-top: -.7rem;
}

/* Component labels */
.ore-pill {
    display: inline-block;
    padding: 4px 10px;
    border: 1px solid #CBD5E1;
    border-radius: 20px;
    color: #334155 !important;
    font-size: 12px;
    margin-right: 6px;
    background: #F8FAFC;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #F1F5F9 !important;
    color: #172033 !important;
    border: 1px solid #D8E0EA;
    padding: 10px;
    border-radius: 12px;
}

/* General text */
[data-testid="stMarkdownContainer"],
[data-testid="stText"],
[data-testid="stCaptionContainer"] {
    color: #172033;
}

/* Inputs */
[data-baseweb="input"],
[data-baseweb="textarea"],
[data-baseweb="select"] {
    background: #FFFFFF !important;
    color: #172033 !important;
}
</style>
""", unsafe_allow_html=True)


#st.markdown(

#<style>
#[data-testid="stAppViewContainer"] {background:#0b1020;}
#[data-testid="stSidebar"] {background:#111a2e;border-right:1px solid #26344e;}
#.block-container {max-width:1600px;padding-top:1.35rem;}
#.ore-subtitle {color:#9aabc6;margin-top:-.7rem;}
#[data-testid="stChatMessage"] {border:1px solid #25334d;border-radius:14px;}
#.ore-pill {display:inline-block;padding:4px 10px;border:1px solid #334663;
#border-radius:20px;color:#b7c9e5;font-size:12px;margin-right:6px;}
#[data-testid="stMetric"] {background:#111a2e;border:1px solid #26344e;padding:10px;border-radius:12px;}
#</style>
#, unsafe_allow_html=True)
st.title("🧠 ORE Experiment Workbench")
st.markdown('<p class="ore-subtitle">A conversational lab for executing ORE components and observing what the Python implementation actually computes.</p>', unsafe_allow_html=True)

DEFAULTS = {"messages": [], "last_result": None, "last_output_object": None,
            "discovery_cache": {}, "artifacts": [], "run_count": 0,
            "automatic_test_result": None}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

def _image_to_array(upload):
    if upload is None:
        return None
    try:
        from io import BytesIO
        from PIL import Image
        import numpy as np
        return np.asarray(Image.open(BytesIO(upload.getvalue())).convert("L"), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"Could not decode uploaded image: {type(exc).__name__}: {exc}") from exc

def _compact(value, max_items=40, depth=0):
    if depth > 7:
        return "…"
    if isinstance(value, dict):
        return {str(k): _compact(v, max_items, depth + 1)
                for k, v in list(value.items())[:max_items]}
    if isinstance(value, list):
        items = [_compact(v, max_items, depth + 1) for v in value[:max_items]]
        if len(value) > max_items:
            items.append(f"… {len(value) - max_items} more item(s)")
        return items
    return value

with st.sidebar:
    st.header("Experiment setup")
    component = st.selectbox("ORE component", COMPONENTS)
    st.caption("One selected callable per run. Pipeline chaining is explicit through saved output artifacts.")
    st.divider()
    st.subheader("Input attachments")
    image_upload = st.file_uploader(
        "Image input (for image-based stages such as N1)",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="Passed as a NumPy grayscale array to callables that accept an image argument.",
    )
    if image_upload is not None:
        st.caption(f"Attached: {image_upload.name}")
    st.divider()
    show_trace = st.toggle("Show detailed Python call trace", value=True)
    live_debug = st.toggle("Show live execution debugger", value=True)
    trace_lines = st.toggle("Trace individual source lines (slower)", value=False,
        help="Line-by-line tracing inside the selected ORE package. More detailed, but can noticeably slow execution.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()
    if st.button("Clear saved artifacts", use_container_width=True):
        st.session_state.artifacts = []
        st.session_state.last_output_object = None
        st.rerun()

with st.expander("Randomized theory-and-algorithm testing", expanded=False):
    st.markdown(
        f"Randomly exercise **all eligible callables** found in the {len(COMPONENTS)} packages. "
        "Inputs and pseudo-text are newly generated for every trial. No fixed prompt bank, target answer, "
        "or expected output is used."
    )
    st.caption(
        "The runner checks execution, numerical invariants, declared operation registries, and observed traces. "
        "It does not substitute for empirical neuroscience evidence or domain-specific theory review."
    )
    trial_count = st.number_input(
        "Random trials per callable", min_value=1, max_value=500, value=25, step=5,
        help="Every discovered eligible callable is tested this many times with fresh random inputs."
    )
    callable_cap = st.number_input(
        "Maximum callables per package (0 = all)", min_value=0, max_value=1000,
        value=0, step=5,
        help="Zero includes every discovered safe algorithm entry point, including supported learning methods."
    )
    retain_generated_artifacts = st.toggle(
        "Keep generated SQLite and temporary files",
        value=False,
        help=(
            "Off keeps the randomized report small by deleting generated database and temporary files "
            "after each package. If a file is still open, cleanup is retried after the batch. Turn it on "
            "only when you need to inspect those artifacts."
        ),
    )
    seed_text = st.text_input(
        "Random seed (optional)", value="",
        help="Leave blank for a new unpredictable seed. Save the seed from the report to reproduce a run."
    )
    st.caption(
        f"Configured run: up to {int(trial_count)} random trials for each selected callable. "
        "The actual total is calculated after package discovery."
    )
    if st.button("Start randomized testing of all packages", type="primary",
                  use_container_width=True, key="run_all_ore_randomized_tests"):
        progress_bar = st.progress(0.0, text="Discovering package algorithms...")
        progress_detail = st.empty()
        status_panel = st.status("Randomized evaluation running", expanded=True)
        random_seed = None
        try:
            random_seed = int(seed_text.strip()) if seed_text.strip() else None
        except ValueError:
            st.error("The random seed must be an integer or left empty.")
            st.stop()

        def _update_random_progress(payload):
            fraction = float(payload.get("fraction", 0.0))
            progress_bar.progress(min(1.0, max(0.0, fraction)))
            package_name = payload.get("package") or "All packages"
            progress_detail.write(
                f"Package {payload.get('package_index', 0)} of {payload.get('package_count', len(COMPONENTS))}: "
                f"**{package_name}** · Callable {payload.get('callable_index', 0)} of "
                f"{payload.get('callable_count', 0)} · Trial {payload.get('trial_index', 0)} of "
                f"{payload.get('trial_count', 0)} · {payload.get('phase', '')} · "
                f"{payload.get('completed', 0)} / {payload.get('total', 0)} trials"
            )

        try:
            auto_result = run_auto_test_suite(
                ROOT,
                progress_callback=_update_random_progress,
                trials_per_callable=int(trial_count),
                max_callables_per_package=int(callable_cap),
                random_seed=random_seed,
                retain_generated_artifacts=retain_generated_artifacts,
            )
            st.session_state.automatic_test_result = auto_result
            status_panel.update(label="Randomized evaluation completed", state="complete", expanded=False)
            st.rerun()
        except Exception as exc:
            status_panel.update(label="Randomized evaluation stopped with an error", state="error", expanded=True)
            st.error(f"{type(exc).__name__}: {exc}")

    auto_result = st.session_state.get("automatic_test_result")
    if auto_result:
        summary = auto_result["summary"]
        st.divider()
        st.subheader("Latest randomized evaluation")
        metric_cols = st.columns(6)
        metric_cols[0].metric("Packages", summary["package_count"])
        metric_cols[1].metric("Callables", summary["callables_exercised"])
        metric_cols[2].metric("Random trials", summary["completed_trials"])
        metric_cols[3].metric("Runtime errors", summary["execution_errors"])
        metric_cols[4].metric("Input / gate rejections", summary["input_rejections"])
        metric_cols[5].metric("Invariant flags", summary["candidate_invariant_violations"])
        st.caption(
            f"Run {auto_result['run_id']} · Random seed {auto_result['random_seed']} · "
            f"{summary['completed_trials']} / {summary['planned_trials']} trials · "
            f"Reports saved to {auto_result['report_directory']}"
        )
        cleanup_warnings = summary.get("artifact_cleanup_warnings", [])
        if cleanup_warnings:
            st.warning(
                "The batch completed, but generated files for these packages could not be deleted "
                "(for example, because a file remained open): " + ", ".join(cleanup_warnings)
            )
        zip_path = Path(auto_result["archive_path"])
        if zip_path.is_file():
            st.download_button(
                "Download all package reports (ZIP)",
                data=zip_path.read_bytes(),
                file_name=zip_path.name,
                mime="application/zip",
                use_container_width=True,
                key="download_all_randomized_package_reports",
            )
        summary_path = Path(auto_result["summary_markdown"])
        if summary_path.is_file():
            st.download_button(
                "Download overall summary (Markdown)",
                data=summary_path.read_bytes(),
                file_name=summary_path.name,
                mime="text/markdown",
                key="download_randomized_package_summary",
            )
        package_reports = auto_result.get("package_reports", [])
        if package_reports:
            st.markdown("**Individual package reports**")
            package_names = [item["package"] for item in package_reports]
            chosen_report_name = st.selectbox(
                "Select a package report", package_names, key="randomized_report_package"
            )
            chosen_report = next(item for item in package_reports
                                 if item["package"] == chosen_report_name)
            part = chosen_report["summary"]
            st.caption(
                f"Import: {chosen_report['import_status']} · "
                f"Callables: {part.get('callables_exercised', 0)} · "
                f"Trials: {part.get('trials_executed', 0)} · "
                f"Errors: {part.get('execution_errors', 0)} · "
                f"Candidate invariant flags: {part.get('candidate_invariant_violations', 0)}"
            )
            report_json_path = Path(chosen_report["json_path"]) if chosen_report.get("json_path") else None
            report_md_path = Path(chosen_report["markdown_path"]) if chosen_report.get("markdown_path") else None
            report_cols = st.columns(2)
            if report_json_path and report_json_path.is_file():
                report_cols[0].download_button(
                    "Download package JSON",
                    data=report_json_path.read_bytes(),
                    file_name=report_json_path.name,
                    mime="application/json",
                    key=f"download_random_json_{chosen_report_name}",
                    use_container_width=True,
                )
            if report_md_path and report_md_path.is_file():
                report_cols[1].download_button(
                    "Download package Markdown",
                    data=report_md_path.read_bytes(),
                    file_name=report_md_path.name,
                    mime="text/markdown",
                    key=f"download_random_md_{chosen_report_name}",
                    use_container_width=True,
                )


try:
    if component not in st.session_state.discovery_cache:
        st.session_state.discovery_cache[component] = discover_callables(component)
    callables, discovery_errors = st.session_state.discovery_cache[component]
except Exception as exc:
    callables, discovery_errors = [], [f"{type(exc).__name__}: {exc}"]

artifact_options = ["None"] + [
    f'{a["component"]} · {a["callable_name"]} · {a["output_type"]} · {a["run_id"][:8]}'
    for a in reversed(st.session_state.artifacts)
]
artifact_lookup = {
    f'{a["component"]} · {a["callable_name"]} · {a["output_type"]} · {a["run_id"][:8]}': a
    for a in reversed(st.session_state.artifacts)
}

left, right = st.columns([1.55, 1.15], gap="large")
with left:
    st.subheader("Conversation")
    st.markdown(f'<span class="ore-pill">{component}</span><span class="ore-pill">{len(callables)} callable(s) discovered</span>', unsafe_allow_html=True)
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            payload = item.get("result")
            if payload:
                observation = payload.get("observation", {})
                if payload["status"] == "success":
                    st.success(f'Executed in {payload["elapsed_ms"]} ms · {observation.get("trace_count", 0)} trace records · {observation.get("array_count", 0)} numerical arrays observed')
                    for warning in observation.get("warnings", []):
                        st.warning(warning)
                else:
                    st.error(payload.get("error") or "Execution failed")
                    if payload.get("traceback"):
                        with st.expander("Traceback"):
                            st.code(payload["traceback"])

    if not callables:
        st.warning("No compatible public callables were discovered. This package may expose contracts/constants only, or require a package-specific adapter.")
        if discovery_errors:
            with st.expander("Import diagnostics"):
                st.code("\n".join(discovery_errors))
    else:
        with st.expander("Select the operation", expanded=True):
            labels = [item.label for item in callables]
            chosen_label = st.selectbox("Callable / processor method", labels, key=f"callable_{component}")
            selected = next(item for item in callables if item.label == chosen_label)
            st.caption(selected.doc)
            st.code(selected.signature, language="python")

        upstream_key = f"upstream_{component}"
        if st.session_state.get(upstream_key, "None") not in artifact_options:
            st.session_state[upstream_key] = "None"
        st.selectbox(
            "Upstream artifact (optional)", artifact_options,
            key=upstream_key,
            help="Choose a real result from an earlier run. For example, pass an N1Result to N2 or an N3Result to N4.",
        )
        selected_artifact_label = st.session_state.get(f"upstream_{component}", "None")
        upstream_artifact = artifact_lookup.get(selected_artifact_label)
        if upstream_artifact:
            st.caption(f'Will supply {upstream_artifact["output_type"]} from {upstream_artifact["component"]}.')

        with st.expander("Advanced arguments (JSON)", expanded=False):
            raw_args = st.text_area(
                "Keyword arguments", value="{}",
                help='Use {"request_data": {...}} for dataclass requests and {"__init__": {...}} for processor constructor options. Iterable dataclass inputs accept JSON lists.',
                key=f"args_{component}", height=150,
            )
        if image_upload is not None:
            st.image(image_upload.getvalue(), caption="Uploaded image input", width=280)

        prompt = st.chat_input(f"Send an experimental prompt to {component}…")
        if prompt:
            try:
                supplied = json.loads(raw_args or "{}")
                if not isinstance(supplied, dict):
                    raise ValueError("Keyword arguments must be a JSON object.")
                runtime_kwargs = {}
                image_is_supported = accepts_parameter(selected, "image")
                if image_upload is not None and image_is_supported:
                    runtime_kwargs["image"] = _image_to_array(image_upload)
                st.session_state.messages.append({"role": "user", "content": prompt})
                event_queue = queue.Queue()
                execution_holder = {}

                def _execute_in_worker():
                    execution_holder["result"] = execute(
                        component, selected, prompt, supplied,
                        previous_output=upstream_artifact["output"] if upstream_artifact else None,
                        runtime_kwargs=runtime_kwargs,
                        on_event=event_queue.put,
                        trace_lines=trace_lines,
                        log_directory=ROOT / "ore_logs",
                    )

                worker = threading.Thread(target=_execute_in_worker, daemon=True)
                worker.start()
                live_events = []
                last_render = 0.0
                with st.status(f"Running {component}...", expanded=live_debug) as live_status:
                    metrics_slot = st.empty()
                    active_slot = st.empty()
                    timeline_slot = st.empty()
                    while worker.is_alive() or not event_queue.empty():
                        received = False
                        while True:
                            try:
                                live_events.append(event_queue.get_nowait())
                                received = True
                            except queue.Empty:
                                break
                        now = time.monotonic()
                        if live_debug and (received or now - last_render >= 0.35):
                            last_render = now
                            latest = live_events[-1] if live_events else None
                            with metrics_slot.container():
                                m1, m2, m3, m4 = st.columns(4)
                                m1.metric("Observed events", len(live_events))
                                m2.metric("ORE steps", sum(1 for e in live_events if e.get("kind") == "operation"))
                                m3.metric("Python calls", sum(1 for e in live_events if e.get("kind") == "call"))
                                m4.metric("Errors raised", sum(1 for e in live_events if e.get("kind") == "exception"))
                            if latest:
                                stack = latest.get("active_stack", [])
                                current = stack[-1]["function"] if stack else "Between component calls or finishing."
                                latest_explanation = latest.get("explanation", "")
                                active_slot.markdown(
                                    ("**Latest ORE explanation**\n\n" + latest_explanation + "\n\n" if latest_explanation else "")
                                    + "**What Python is doing now**\n\n" + current
                                    + "\n\n**Active call stack**\n"
                                    + ("\n".join(f'{i+1}. {item["function"]}' for i, item in enumerate(stack[-8:])) if stack else "No ORE function is currently active.")
                                )
                                rows = [{
                                    "time (ms)": e.get("elapsed_ms"),
                                    "event": e.get("kind"),
                                    "step": e.get("step", ""),
                                    "operation / function": e.get("operation") or e.get("function", ""),
                                    "what happened": e.get("explanation"),
                                    "location": f'{e.get("file", "")}:{e.get("line", "")}',
                                    "duration (ms)": e.get("duration_ms", ""),
                                } for e in live_events[-10:]]
                                timeline_slot.dataframe(rows, use_container_width=True, hide_index=True)
                            else:
                                active_slot.info("Waiting for the selected component to enter its first Python function...")
                        time.sleep(0.06)
                    worker.join()
                    while True:
                        try:
                            live_events.append(event_queue.get_nowait())
                        except queue.Empty:
                            break
                    result = execution_holder.get("result")
                    if result is None:
                        raise RuntimeError("The execution worker ended without returning a result.")
                    result["live_events"] = live_events
                    live_status.update(
                        label=("Execution completed" if result["status"] == "success" else "Execution failed"),
                        state=("complete" if result["status"] == "success" else "error"),
                        expanded=live_debug,
                    )
                try:
                    result["automatic_report_path"] = save_manual_run_report(ROOT, result)
                except Exception as report_exc:
                    result["automatic_report_error"] = f"{type(report_exc).__name__}: {report_exc}"
                st.session_state.last_result = result
                if result["status"] == "success":
                    raw_output = result["output"]
                    st.session_state.last_output_object = raw_output
                    st.session_state.run_count += 1
                    safe_output = json_safe(raw_output)
                    artifact = {
                        "run_id": result["run_id"], "component": component,
                        "callable_name": selected.label, "output_type": type(raw_output).__name__,
                        "created_at": result["started_at"], "output": raw_output,
                        "output_preview": _compact(safe_output),
                    }
                    st.session_state.artifacts.append(artifact)
                    st.session_state.artifacts = st.session_state.artifacts[-30:]
                    result["output_preview"] = artifact["output_preview"]
                    result["upstream_run_id"] = upstream_artifact["run_id"] if upstream_artifact else None
                summary = (f'Execution completed: {result["observation"].get("output_type", "output")}.'
                           if result["status"] == "success" else f'Execution failed: {result["error"]}')
                st.session_state.messages.append({"role": "assistant", "content": summary, "result": result})
            except Exception as exc:
                st.session_state.messages.append({"role": "assistant", "content": f"Workbench input error: {type(exc).__name__}: {exc}"})
            st.rerun()

with right:
    st.subheader("Execution & observation")
    result = st.session_state.last_result
    if result is None:
        st.info("Run a component to populate execution telemetry, stage traces, numerical summaries, and validation checks.")
        st.metric("Saved upstream artifacts", len(st.session_state.artifacts))
    else:
        obs = result.get("observation", {})
        profile = result.get("profile", {})
        status = result.get("status", "unknown")
        (st.success if status == "success" else st.error)(status.upper())
        metrics = st.columns(3)
        metrics[0].metric("Elapsed time", f'{result.get("elapsed_ms", 0)} ms')
        metrics[1].metric("Python calls", profile.get("observed_call_count", 0))
        metrics[2].metric("Numeric arrays", obs.get("array_count", 0))
        tabs = st.tabs(["Overview", "Live debugger", "Narration log", "Stage trace", "Python calls", "Numerics", "Raw result"])
        with tabs[0]:
            st.write("**Run identity**")
            st.code(result.get("run_id", ""), language="text")
            st.write("**Executed callable**")
            st.code(result.get("callable", ""), language="text")
            st.write("**Output type**")
            st.write(obs.get("output_type"))
            if result.get("automatic_report_path"):
                st.caption("Automatically saved report: " + result["automatic_report_path"])
            if result.get("automatic_report_error"):
                st.warning("Automatic report could not be saved: " + result["automatic_report_error"])
            if result.get("upstream_run_id"):
                st.caption(f'Upstream run: {result["upstream_run_id"]}')
            st.write("**Validation checks**")
            if obs.get("checks"):
                st.dataframe(obs["checks"], use_container_width=True, hide_index=True)
            else:
                st.caption("The result did not expose machine-readable validation flags.")
            if obs.get("warnings"):
                st.write("**Warnings emitted by the component**")
                for warning in obs["warnings"]:
                    st.warning(warning)
            if result.get("error"):
                st.error(result["error"])
                if result.get("traceback"):
                    with st.expander("Full traceback", expanded=True):
                        st.code(result["traceback"])
            st.write("**Input observations**")
            input_obs = result.get("input_observation", {})
            if input_obs.get("numerical_summaries"):
                st.dataframe(input_obs["numerical_summaries"], use_container_width=True, hide_index=True)
            else:
                st.caption("No NumPy arrays were found in the arguments for this run.")
            st.write("**Constructor arguments**")
            st.json(_compact(json_safe(result.get("constructor_arguments", {}))))
            if result.get("reused_instance"):
                st.info("Reused the selected processor instance from the chosen upstream artifact.")
            st.write("**Arguments actually passed**")
            st.json(_compact(json_safe(result.get("arguments", {}))))

        with tabs[1]:
            st.markdown("**Readable execution timeline**")
            st.caption("Events stream from the selected package while it runs. Calls mean a function started; returns mean it finished. Source-line events appear only when line tracing is enabled.")
            timeline = result.get("live_events") or profile.get("timeline", [])
            if timeline:
                event_rows = [{
                    "time (ms)": e.get("elapsed_ms"),
                    "activity": e.get("kind", "event"),
                    "plain-language explanation": e.get("explanation", ""),
                    "function": e.get("function", ""),
                    "source": f'{e.get("file", "")}:{e.get("line", "")}',
                    "duration (ms)": e.get("duration_ms", ""),
                } for e in timeline]
                st.dataframe(event_rows, use_container_width=True, hide_index=True)
                if profile.get("dropped_event_count"):
                    st.warning(f'{profile["dropped_event_count"]} events were omitted at the capture limit.')
                peak_stack = max((e.get("active_stack", []) for e in timeline), key=len, default=[])
                st.write("**Deepest observed call stack**")
                if peak_stack:
                    for level, item in enumerate(peak_stack):
                        st.write(f'{level + 1}. {"  " * min(level, 8)}{item["function"]}')
                else:
                    st.caption("No active stack was captured.")
                exceptions = [e for e in timeline if e.get("kind") == "exception"]
                if exceptions:
                    st.write("**Exceptions observed during execution**")
                    for e in exceptions:
                        st.error(f'{e.get("exception_type")}: {e.get("exception_message", "")} · {e.get("file")}:{e.get("line")}')
                line_events = [e for e in timeline if e.get("kind") == "line"]
                if line_events:
                    with st.expander("Individual source lines", expanded=False):
                        st.code("\n".join(f'{e.get("file")}:{e.get("line")}  {e.get("source", "")}' for e in line_events), language="python")
            else:
                st.info("No live Python events were captured for this run. Check that the callable executes code inside the selected package.")
            st.markdown("**How to read this panel**")
            st.markdown("- Call: a function starts. Its position in the stack shows which function called it.")
            st.markdown("- Return: a function finishes. Duration includes nested work.")
            st.markdown("- Exception: Python raised an error at that point; the traceback gives full context.")
            st.markdown("- Line: the specific source line Python is executing. Enable line tracing for this slower view.")

        with tabs[2]:
            narrative_events = [
                e for e in result.get("live_events", [])
                if e.get("kind") in {"operation", "component_trace"}
            ]
            st.markdown("**Plain-language account of ORE's processing steps**")
            st.caption("N9 operation events are written as their mark() calls occur. Trace rows from other packages are narrated when the component returns them.")
            if narrative_events:
                narrative_rows = [{
                    "elapsed (ms)": e.get("elapsed_ms"),
                    "step": e.get("step", ""),
                    "stage": e.get("phase", ""),
                    "operation": e.get("operation", ""),
                    "status": e.get("status", ""),
                    "explanation": e.get("explanation", ""),
                } for e in narrative_events]
                st.dataframe(narrative_rows, use_container_width=True, hide_index=True)
                labels = [
                    f'{e.get("sequence", i + 1):04d} · {e.get("step", "trace")} · {e.get("title", e.get("operation", "component step"))}'
                    for i, e in enumerate(narrative_events)
                ]
                selected_label = st.selectbox("Inspect a narrated step", labels, key=f'narration_{result.get("run_id")}')
                selected_event = narrative_events[labels.index(selected_label)]
                st.subheader(selected_event.get("title", selected_event.get("operation", "Step details")))
                st.write(selected_event.get("explanation", "No plain-language explanation was supplied."))
                if selected_event.get("phase_explanation"):
                    st.caption(selected_event["phase_explanation"])
                if selected_event.get("status_explanation"):
                    st.info(selected_event["status_explanation"])
                if selected_event.get("details"):
                    st.write("**Recorded technical details**")
                    st.json(_compact(selected_event["details"]))
            else:
                st.info("No declared operation trace was captured for this run. The Python call timeline remains available in the Live debugger tab.")
            log_meta = result.get("logs", {})
            if log_meta:
                st.divider()
                st.markdown("**Saved in the shared SQLite execution database**")
                st.caption(
                    f'Events stored: {log_meta.get("event_count", 0)} · '
                    f'Run ID: {log_meta.get("run_id", result.get("run_id"))}'
                )
                st.code(log_meta.get("database_path", ""), language="text")
                st.caption(
                    "Annotation text is stored once in SQLite. Timeline events reference the matching class/function annotation by ID."
                )
                st.markdown("Open the searchable run history and annotation editor with:")
                st.code("streamlit run ore_logger/viewer.py", language="powershell")
                if log_meta.get("logger_errors"):
                    st.warning("The execution completed, but the logger reported: " + "; ".join(log_meta["logger_errors"]))
            elif result.get("logging_error"):
                st.warning("SQLite logging could not be started: " + result["logging_error"])

        with tabs[3]:
            rows = obs.get("trace_rows", [])
            st.caption(f'{obs.get("trace_count", 0)} records extracted from component traces and diagnostics.')
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No explicit stage/operation trace was exposed by this result object.")

        with tabs[4]:
            events = profile.get("events", [])
            if profile.get("dropped_event_count"):
                st.warning(f'{profile["dropped_event_count"]} Python call events were omitted at the capture limit.')
            if events:
                if show_trace:
                    st.dataframe(events, use_container_width=True, hide_index=True)
                st.write("**Functions ranked by total observed time**")
                st.dataframe(profile.get("function_summary", [])[:40], use_container_width=True, hide_index=True)
            else:
                st.info("No Python calls inside the selected package were captured.")

        with tabs[5]:
            arrays = obs.get("numerical_summaries", [])
            if arrays:
                st.caption("Summaries are calculated from returned NumPy arrays; raw arrays remain unchanged.")
                st.dataframe(arrays, use_container_width=True, hide_index=True)
            else:
                st.info("The output did not contain inspectable NumPy arrays.")

        with tabs[6]:
            st.caption("Preview is truncated for readability. Download the full JSON report.")
            st.json(result.get("output_preview", _compact(json_safe(result.get("output")))))
            report = {
                "run": {k: v for k, v in result.items() if k not in {"output", "arguments"}},
                "arguments": json_safe(result.get("arguments", {})),
                "output": json_safe(result.get("output")),
                "input_observation": result.get("input_observation", {}),
                "observation": obs, "profile": profile,
                "live_events": result.get("live_events", []),
            }
            report_text = json.dumps(report, indent=2, ensure_ascii=False, default=str)
            st.download_button(
                "Download full experiment report", data=report_text,
                file_name=f'ore-experiment-{result.get("run_id", "run")[:12]}.json',
                mime="application/json", use_container_width=True,
            )

st.divider()
with st.expander("Saved upstream artifacts"):
    if not st.session_state.artifacts:
        st.caption("Successful run outputs become reusable artifacts here.")
    else:
        for artifact in reversed(st.session_state.artifacts[-12:]):
            st.markdown(f'**{artifact["component"]}** · {artifact["output_type"]}  \n<small>{artifact["callable_name"]}</small> · <small>{artifact["run_id"]}</small>', unsafe_allow_html=True)

with st.expander("Input and execution notes"):
    st.markdown("""
- Prompt: mapped to a compatible text-like parameter or to the text field in a dataclass request.
- Structured request: enter fields under request_data; missing sequence_id values are generated for convenience.
- Image: upload an image for callables that explicitly accept an image parameter, such as N1Processor.process(image=...). The uploader is not injected into unrelated functions.
- Upstream artifact: explicitly hand a prior raw Python result to a compatible downstream input, such as an N1Result to N2Processor.process.
- Constructor options: enter configuration under __init__. Inventory JSON records are converted to known ORE dataclasses where supported.
- Live diagnostic: Python function-call activity is streamed while a run is executing. The component’s own named operation/stage records appear after its result is returned. Optional source-line tracing is slower.
- Observation: stage records come from component traces/diagnostics. Python call timings are captured by a temporary profiler; NumPy summaries are calculated after execution.
- Scientific interpretation: timing and numerical statistics are software observations. They do not establish biological validity.
- Security: imports execute Python code. Keep the Workbench local or protect and isolate any hosted deployment.
""")
with st.expander("Use with another Python implementation"):
    st.markdown("""
Add an importable package name to COMPONENTS in app.py and install it in the same Python environment.
The discovery and observation layers are package-agnostic. For custom request lifecycles, stateful APIs,
or nonstandard output contracts, implement a package adapter in adapter.py and preserve the observation schema.
""")
