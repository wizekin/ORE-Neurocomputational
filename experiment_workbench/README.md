# ORE Experiment Workbench

A reusable Streamlit chat-style interface for invoking and inspecting individual Python components in ORE-Neurocomputational.

## Features
- Independently select and test each ORE package.
- Chat-style prompts and per-session conversation history.
- Discover public functions in a package and its importable submodules.
- Map prompts to common text parameter names.
- Supply additional required keyword arguments as JSON.
- Inspect actual return values, execution time, import warnings, and tracebacks.
- Reuse the discovery/dispatch adapter with other installed Python packages.
- No LLM or API key required: outputs come from the selected Python implementation.

## Included components
ore_contracts, ore_lexical_access, ore_lexical_context, ore_n1, ore_n2, ore_n3, ore_n4, ore_n5, ore_n6, ore_n9, ore_predictive_orthography, ore_reading, ore_semantic.

## Run locally
From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r experiment_workbench/requirements.txt
streamlit run experiment_workbench/app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Invocation model
The adapter imports the selected package, walks importable submodules, and discovers public functions defined there. Select a function, submit a prompt, and the adapter maps the prompt to a recognized text-like parameter. Supply other required keyword arguments in the JSON panel.

This is a transparent test harness, not an LLM. It does not silently substitute mock answers when a component cannot be imported or executed. Errors and tracebacks are shown for debugging.

Processor, simulator, and model classes are also inspected for common methods such as process, run, infer, predict, train, simulate, and validate. The Workbench instantiates these classes with their default constructor. If a constructor requires configuration, provide it in the JSON panel under __init__. If a method expects a dataclass request object, provide its fields under request_data; the prompt is automatically assigned to a recognized text-like field when one exists. Classes with unusual lifecycle or nested input contracts may still need an explicit adapter.

## Reuse with another implementation
Install the other Python package in the same environment, add its import name to COMPONENTS in app.py, and use the discovery/dispatch behavior. For a different invocation contract, extend adapter.py.

## Security
Only run this workbench with trusted Python packages. Importing a package executes its module initialization code. Do not expose this development workbench publicly without authentication and isolation.
