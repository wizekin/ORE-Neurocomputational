from __future__ import annotations
import json, sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from adapter import discover_callables, execute, json_safe

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
.block-container {max-width:1500px;padding-top:1.5rem;}
.ore-subtitle {color:#9aabc6;margin-top:-.7rem;}
[data-testid="stChatMessage"] {border:1px solid #25334d;border-radius:14px;}
.ore-pill {display:inline-block;padding:4px 10px;border:1px solid #334663;
border-radius:20px;color:#b7c9e5;font-size:12px;margin-right:6px;}
</style>
""", unsafe_allow_html=True)
st.title("🧠 ORE Experiment Workbench")
st.markdown('<p class="ore-subtitle">A conversational laboratory for inspecting, testing, and comparing neurocomputational Python components.</p>', unsafe_allow_html=True)

for key, default in [("messages", []), ("last_result", None), ("discovery_cache", {})]:
    if key not in st.session_state: st.session_state[key] = default

with st.sidebar:
    st.header("Experiment setup")
    component = st.selectbox("Component", COMPONENTS)
    st.caption("Each experiment invokes one selected component. No full ORE pipeline is run implicitly.")
    st.divider()
    show_trace = st.toggle("Show detailed execution trace", value=True)
    st.caption("Python executes on the machine hosting this Streamlit app.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages, st.session_state.last_result = [], None
        st.rerun()

try:
    if component not in st.session_state.discovery_cache:
        st.session_state.discovery_cache[component] = discover_callables(component)
    callables, discovery_errors = st.session_state.discovery_cache[component]
except Exception as exc:
    callables, discovery_errors = [], [f"{type(exc).__name__}: {exc}"]

left, right = st.columns([1.65, 1], gap="large")
with left:
    st.subheader("Conversation")
    st.markdown(f'<span class="ore-pill">{component}</span><span class="ore-pill">{len(callables)} callable(s) discovered</span>', unsafe_allow_html=True)
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            payload = item.get("result")
            if payload:
                if payload["status"] == "success":
                    st.success(f'Executed in {payload["elapsed_ms"]} ms')
                    st.code(json.dumps(json_safe(payload["output"]), indent=2, ensure_ascii=False), language="json")
                else:
                    st.error(payload.get("error") or "Execution failed")
                    if payload.get("traceback"):
                        with st.expander("Traceback"): st.code(payload["traceback"])
    if not callables:
        st.warning("No public Python functions were discovered. Inspect this component's contracts/classes or add a deliberate adapter.")
        if discovery_errors:
            with st.expander("Import diagnostics"): st.code("\n".join(discovery_errors))
    else:
        with st.expander("Choose the function to call", expanded=True):
            labels = [x.label for x in callables]
            chosen = st.selectbox("Callable", labels, key=f"callable_{component}")
            selected = next(x for x in callables if x.label == chosen)
            st.caption(selected.doc)
            st.code(selected.signature, language="python")
        with st.expander("Advanced arguments", expanded=False):
            raw_args = st.text_area("Arguments (JSON object)", value="{}",
                help="Prompt maps to text-like parameters. For dataclass request objects use {\"request_data\":{...}}; optional constructor arguments can be supplied under __init__.",
                key=f"args_{component}", height=110)
        prompt = st.chat_input(f"Send a test prompt to {component}…")
        if prompt:
            try:
                supplied = json.loads(raw_args or "{}")
                if not isinstance(supplied, dict): raise ValueError("Arguments must be a JSON object.")
                st.session_state.messages.append({"role":"user", "content":prompt})
                result = execute(component, selected, prompt, supplied)
                st.session_state.last_result = result
                summary = "Component execution completed." if result["status"] == "success" else f'Component execution failed: {result["error"]}'
                st.session_state.messages.append({"role":"assistant","content":summary,"result":result})
            except Exception as exc:
                st.session_state.messages.append({"role":"assistant","content":f"Workbench input error: {type(exc).__name__}: {exc}"})
            st.rerun()

with right:
    st.subheader("Component inspector")
    st.metric("Discovered functions", len(callables))
    st.metric("Import warnings", len(discovery_errors))
    if discovery_errors:
        with st.expander("Import warnings"): st.code("\n".join(discovery_errors))
    st.divider()
    st.subheader("Latest execution")
    result = st.session_state.last_result
    if result:
        (st.success if result["status"] == "success" else st.error)(result["status"].upper())
        st.metric("Execution time", f'{result["elapsed_ms"]} ms')
        st.write("**Callable**")
        st.code(result["callable"], language="text")
        if show_trace:
            st.write("**Arguments**")
            st.json(json_safe(result["arguments"]))
            st.write("**Output**")
            st.json(json_safe(result["output"]))
            if result.get("traceback"):
                with st.expander("Full traceback", expanded=True): st.code(result["traceback"])
    else:
        st.info("Submit a prompt to inspect the selected component's output and execution details.")

st.divider()
with st.expander("Using another Python implementation"):
    st.markdown("""
The Workbench is not hard-coded to ORE. Add another importable package name to COMPONENTS in app.py.
Install its source in the same Python environment as Streamlit. The adapter discovers public functions
and signatures. For stateful classes or unusual entry points, add a deliberate adapter in adapter.py.

The prompt is passed to a compatible text-like parameter. Supply other required arguments as JSON.
The Workbench displays the actual Python return value; it does not use an LLM to invent or rewrite outputs.
""")
