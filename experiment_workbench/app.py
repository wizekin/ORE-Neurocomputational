from __future__ import annotations

import json
import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from adapter import accepts_parameter, discover_callables, execute, json_safe

COMPONENTS = [
    "ore_contracts", "ore_lexical_access", "ore_lexical_context",
    "ore_n1", "ore_n2", "ore_n3", "ore_n4", "ore_n5", "ore_n6", "ore_n9",
    "ore_predictive_orthography", "ore_reading", "ore_semantic",
]
st.set_page_config(page_title="ORE Experiment Workbench", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:#0b1020;}
[data-testid="stSidebar"] {background:#111a2e;border-right:1px solid #26344e;}
.block-container {max-width:1600px;padding-top:1.35rem;}
.ore-subtitle {color:#9aabc6;margin-top:-.7rem;}
[data-testid="stChatMessage"] {border:1px solid #25334d;border-radius:14px;}
.ore-pill {display:inline-block;padding:4px 10px;border:1px solid #334663;
border-radius:20px;color:#b7c9e5;font-size:12px;margin-right:6px;}
[data-testid="stMetric"] {background:#111a2e;border:1px solid #26344e;padding:10px;border-radius:12px;}
</style>
""", unsafe_allow_html=True)
st.title("🧠 ORE Experiment Workbench")
st.markdown('<p class="ore-subtitle">A conversational lab for executing ORE components and observing what the Python implementation actually computes.</p>', unsafe_allow_html=True)

DEFAULTS = {"messages": [], "last_result": None, "last_output_object": None,
            "discovery_cache": {}, "artifacts": [], "run_count": 0}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

def _image_to_array(upload):
    if upload is None:
        return None
    try:
        from PIL import Image
        import numpy as np
        return np.asarray(Image.open(upload).convert("L"), dtype=np.uint8)
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
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()
    if st.button("Clear saved artifacts", use_container_width=True):
        st.session_state.artifacts = []
        st.session_state.last_output_object = None
        st.rerun()

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
            st.image(image_upload, caption="Uploaded image input", width=280)

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
                result = execute(
                    component, selected, prompt, supplied,
                    previous_output=upstream_artifact["output"] if upstream_artifact else None,
                    runtime_kwargs=runtime_kwargs,
                )
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
        tabs = st.tabs(["Overview", "Stage trace", "Python calls", "Numerics", "Raw result"])
        with tabs[0]:
            st.write("**Run identity**")
            st.code(result.get("run_id", ""), language="text")
            st.write("**Executed callable**")
            st.code(result.get("callable", ""), language="text")
            st.write("**Output type**")
            st.write(obs.get("output_type"))
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
            st.write("**Constructor arguments**")
            st.json(_compact(json_safe(result.get("constructor_arguments", {}))))
            if result.get("reused_instance"):
                st.info("Reused the selected processor instance from the chosen upstream artifact.")
            st.write("**Arguments actually passed**")
            st.json(_compact(json_safe(result.get("arguments", {}))))

        with tabs[1]:
            rows = obs.get("trace_rows", [])
            st.caption(f'{obs.get("trace_count", 0)} records extracted from component traces and diagnostics.')
            if rows:
                st.dataframe(rows, use_container_width=True, hide_index=True)
            else:
                st.info("No explicit stage/operation trace was exposed by this result object.")

        with tabs[2]:
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

        with tabs[3]:
            arrays = obs.get("numerical_summaries", [])
            if arrays:
                st.caption("Summaries are calculated from returned NumPy arrays; raw arrays remain unchanged.")
                st.dataframe(arrays, use_container_width=True, hide_index=True)
            else:
                st.info("The output did not contain inspectable NumPy arrays.")

        with tabs[4]:
            st.caption("Preview is truncated for readability. Download the full JSON report.")
            st.json(result.get("output_preview", _compact(json_safe(result.get("output")))))
            report = {
                "run": {k: v for k, v in result.items() if k not in {"output", "arguments"}},
                "arguments": json_safe(result.get("arguments", {})),
                "output": json_safe(result.get("output")),
                "observation": obs, "profile": profile,
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
