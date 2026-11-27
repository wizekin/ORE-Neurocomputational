# ORE Experiment Workbench

A reusable Streamlit conversational laboratory for executing individual Python components and inspecting their actual behavior.

## What the execution and observation engine does

- Gives each run a unique run ID and UTC start timestamp.
- Executes the selected Python callable and captures its real return object.
- Profiles Python function calls made inside the selected package, recording call order, nesting depth, source file/line, return type, and elapsed time.
- Extracts component-authored operation, stage, iteration, and training traces where present.
- Extracts diagnostic stage records, including dataclass-backed records.
- Summarizes NumPy arrays found in both inputs and outputs: shape, dtype, size, finite-value status, minimum, maximum, mean, standard deviation, L2 norm, and nonzero count.
- Surfaces warnings and validation flags exposed by the component.
- Retains successful raw Python output objects as selectable in-session upstream artifacts.
- Exports a JSON experiment report containing run metadata, arguments, output, traces, numerical summaries, and Python call profile.

The observer does not modify arrays or component outputs. It does not infer unreported stages, and it does not claim that software timing or numerical values establish biological validity. Function timings are inclusive; nested calls overlap and should not be summed as if they were independent wall-clock intervals.

## Components

ore_contracts, ore_lexical_access, ore_lexical_context, ore_n1, ore_n2, ore_n3, ore_n4, ore_n5, ore_n6, ore_n9, ore_predictive_orthography, ore_reading, ore_semantic.

## Install and run

From the repository root:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r experiment_workbench/requirements.txt
streamlit run experiment_workbench/app.py
```

Open the local URL printed by Streamlit (normally http://localhost:8501).

## Typical experiments

### Run an image through N1 and N2

1. Select ore_n1 and N1Processor.process.
2. Upload a grayscale or color image; it is supplied as a grayscale NumPy array.
3. Send any short prompt (the image itself is the computational input).
4. Select ore_n2 and N2Processor.process.
5. Choose the N1Result from the Upstream artifact selector and run. The original Python result object, not a JSON reconstruction, is passed to N2.

### Train and use N3

1. Select ore_n3 and N3Processor.fit.
2. Supply JSON keyword arguments containing reference_patches and optionally grapheme_sequences. Each reference patch may contain a grapheme and a numerical features array.
3. Run fit. N3 returns its fitted processor instance.
4. Select N3Processor.recognize and choose the fitted N3Processor artifact as the upstream artifact. Supply request_data with a sequence_id and patches; each patch should contain a patch_id and features or an N2 result.

### Use text-oriented systems

For ore_reading and ore_semantic, choose the processor's process method and provide text in the prompt. The request object's sequence_id is generated automatically. For lexical access, the prompt populates the request's word field; supply a lexicon through the processor's __init__ JSON arguments.

### Supply structured request data

Enter a JSON object under request_data. For example:

```json
{
  "request_data": {
    "language": "en",
    "context": {"science": 0.8}
  }
}
```

The prompt is mapped to a recognized text field, and sequence_id is generated when the request contract has that required field. Explicit request_data values override automatic prompt values.

### Continue a stateful experiment

Choose a successful upstream artifact before executing the next operation. When the artifact is an instance of the selected processor class (for example, an N3Processor returned by fit), the existing instance is reused, preserving its fitted state. Otherwise, the artifact can be injected into compatible upstream-result parameters or request fields (for example, n1_result or n3_result).

N5 model snapshots can be restored from their exported ore.n5.plasticity_model.v1 JSON representation. N5 learning still enforces its own independent-verification and user-authorization gates.

## Reuse with other Python implementations

Add an importable package name to COMPONENTS in app.py and install it in the same Python environment. The generic discovery, execution profiling, output inspection, and report export remain reusable. For unusual constructors, stateful lifecycles, or custom input/output schemas, extend adapter.py while preserving the observation schema.

## Important limits

- A raw text prompt is not a substitute for an image, patch sequence, lexicon, verified training sample, or upstream result required by a component.
- A component may expose contracts and registries but no executable callable; the Workbench reports that rather than inventing a fake output.
- Python profiling captures calls in the selected package on the current execution thread. It does not capture NumPy/SciPy internal C-level operations or work dispatched to other threads/processes.
- Some classes may need custom lifecycle setup or domain-specific JSON inputs. Use the displayed signature, docstring, and traceback to configure them.
- This interface is a test harness, not an LLM. It returns actual Python output and does not generate natural-language answers in place of component results.

## Randomized evaluation and failure interpretation

The randomized theory runner generates synthetic inputs from each callable's declared Python types and ORE data contracts. It does not use fixed prompts or expected answers.

The shared adapter:
- converts nested dataclasses, enums, literals, typed mappings/sequences, TypedDict, NamedTuple, paths, dates and UUID values;
- validates callable and constructor arguments before invocation;
- preserves explicit values rather than allowing runtime inputs to overwrite them;
- accepts a request_data helper only when exactly one dataclass request parameter can receive it;
- restores versioned N5 model snapshots through their package-owned schema loader.

Randomized file outputs are redirected to the current report's side_effects/<package>/ directory. Read-only repository-root parameters continue to refer to the project itself. Input/output observation summarizes numerical arrays in bounded-memory chunks, including complex arrays.

### How to read randomized failures

The report keeps three categories separate:

- **Input/authorization rejections:** the generated values reached an ORE contract or an explicit permission gate and were rejected. These are not counted as successful executions.
- **Workbench errors:** the harness could not generate or convert a valid value, bind a call, or dispatch the requested input. These are defects in the test harness or unsupported contract shapes.
- **ORE runtime errors:** the component itself raised an unexpected exception after dispatch. These are candidates for package-level implementation defects.

This classification uses the execution phase and traceback as well as the exception type. A TypeError raised inside an ORE function is not automatically treated as a harmless input rejection.

### Validation limits

Type annotations and dataclass contracts do not describe every semantic dependency between fields. The generator has explicit fixtures for known cross-object ORE contracts and validates other dataclasses incrementally, but domain-specific constraints not represented in Python types or constructors can still require a package-specific generator. Synthetic randomized execution is software evidence; it does not prove cognitive, biological, or scientific validity.

For local verification, run the focused Workbench regression tests before the full repository suite:

```powershell
python -m pytest -q tests/test_workbench_randomized_contracts.py tests/test_workbench_adapter.py tests/test_workbench_narration.py tests/test_workbench_observer.py tests/test_workbench_randomized_output_paths.py
python -m pytest -q
```

## Security

Only run trusted Python packages. Importing a package executes its module initialization code. Keep the Workbench local unless an exposed deployment has authentication, isolation, resource limits, and appropriate controls for uploaded data.


## Live Diagnostic (debugger-style view)

The chat run now executes in a worker thread while Streamlit polls a thread-safe event queue. The Live debugger view can show recent activity before the selected component finishes:

- A **Call** event means a Python function inside the selected ORE package has started.
- A **Return** event means that function finished; its duration includes any nested calls.
- An **Exception** event marks an exception raised inside an ORE function. The run result also preserves the final traceback.
- The **Active call stack** shows the chain of ORE functions currently executing.
- Each event includes a plain-language explanation, module, file, and source line.
- The optional **Trace individual source lines** setting adds line-by-line events and source text. It is slower and should be enabled only when investigating a specific issue.
- After execution, the debugger retains the full event timeline and deepest observed call stack. The separate Stage trace view displays operation/stage records emitted by the component itself.


## Plain-language background execution logger

Each Workbench component execution is stored in the shared SQLite database at `ore_logs/ore_logger.sqlite3` by default. The database contains three linked kinds of information:

- **Annotations:** one reusable plain-English explanation per module, class, method, function or registered ORE operation.
- **Runs:** a unique run ID, local and UTC start time, end time, script/component identity, duration and status.
- **Events:** calls, returns, exceptions, operation milestones and their source locations. Each event stores an annotation foreign key, not another copy of the annotation text.

Raw prompts, image pixels, full input arrays and Python argument values are not written to the event database. Technical metadata is bounded before storage.

### What gets narrated

- **Repository-wide Python symbols:** an AST catalogue indexes module definitions, classes, methods, nested functions, synchronous functions and asynchronous functions across the Python source files.
- **Existing explanations:** existing docstrings are used as the starting annotation. Where a docstring does not exist, a source-based explanation is generated using the definition name, arguments, return type, helper calls and control-flow structures. The viewer identifies these as source analysis, not reviewed documentation.
- **N9:** its 120 registered operations are emitted through the optional `on_operation` callback as each operation is recorded. Each operation is linked to its reusable description and grouped into the ten N9 phases.
- **Other ORE components:** operation, stage and iteration records exposed by a component result are narrated after that component returns. Their Python function calls are observed while it runs.
- **Python execution:** call, return, exception and optional line events retain function identity, source location, call depth and duration.

The logger distinguishes completed work from `not_applicable`, `not_configured`, fallback and abstention statuses. A missing N6/N7/N8 adapter is reported as missing, not described as though that stage had executed.

### View and edit the catalogue

Launch the independent Streamlit observatory:

```powershell
streamlit run ore_logger/viewer.py
```

The viewer provides timestamped run history, event filtering, function/class explanations, source locations, technical details, run export, annotation search and a manual annotation editor. Manual explanations are retained in SQLite; when the source changes, the viewer marks the explanation for review.

### Direct Python use

The generic adapter accepts a log directory; it creates or reuses the SQLite database inside that directory:

```python
from experiment_workbench.adapter import execute

result = execute(
    package_name,
    callable_info,
    prompt="",
    supplied={"request_data": request_payload},
    log_directory="ore_logs",
)
print(result.get("logs", {}).get("database_path"))
```

The N9 processor can also stream operation records to an application callback without using Streamlit:

```python
events = []
result = N9Processor().process(request, on_operation=events.append)
```

N9 operation events are logged as they are recorded. For packages that only expose traces in their final result, those trace rows cannot truthfully be displayed as live operation events; they are added after return.

### What the debugger can and cannot see

The live timeline is based on Python's function profiler (and optional Python line tracing) for the selected package. It can show Python function boundaries and source lines, but it cannot show individual C-level NumPy/SciPy instructions, GPU kernels, or work dispatched to other threads/processes. Those operations are instead inspected through input/output array summaries and component-provided diagnostics after they return.

A component may perform substantial work inside one Python call. In that case, the active function remains visible until the call returns; this is expected and is not evidence that the debugger has frozen. Stage-specific progress can only be shown live when the component emits progress events itself. The Workbench does not fabricate intermediate stage updates.

Line tracing adds overhead. Use call tracing as the default; enable line tracing for a focused debugging run.


## Automatic logging for other Python scripts

Installing the repository with `python -m pip install -e .` installs the small Python startup hook. When a Python script or module entry point inside this repository starts, it creates a timestamped run and observes repository-defined Python calls, returns and uncaught exceptions. Workbench runs are logged individually by the adapter rather than as part of the long-lived Streamlit server process.

The database defaults to `ore_logs/ore_logger.sqlite3`. Set `ORE_LOGGER_DB` to override its location, `ORE_LOG_MAX_EVENTS` to change the per-process cap (default 200,000), or `ORE_AUTO_LOG=0` to disable automatic tracing for a process. The startup hook is diagnostic only and fails open: a logger failure must not prevent the user's script from running.

Python profiling does not expose native NumPy/SciPy instructions, GPU kernels or work performed in existing external processes. It records Python-level operations and the component-authored diagnostics available from ORE.



## Randomized theory-and-algorithm evaluation

The Streamlit Workbench discovers first-party `ore_*` packages and their public module-level functions and class methods dynamically. It exercises both high-level processors and callable algorithm/stage helpers, excluding known file, network, persistence, deletion, and lifecycle operations. Its **Start randomized testing of all packages** action runs random trials against each discovered callable, one callable and one package at a time.

### No fixed prompts or expected answers

The randomized runner deliberately does not use a pre-authored prompt bank, canned example inputs, fixed target labels, or expected output answers. It generates a fresh random seed, random pseudo-text, random numerical arrays, random structured values, and typed request records using the inspected Python signatures and dataclass contracts. Each trial receives new input and a new processor instance. The seed is included in the report so the random input sequence can be repeated; run IDs and timestamps will naturally differ.

The default is 25 random trials per discovered callable. You can increase this to 500, and set the callable cap to zero to include all discovered eligible entry points. The runner includes selected in-memory learning/update methods when they are discoverable, but intentionally excludes file-writing, persistence, network, deletion, reset, and shutdown methods. Input/guard rejections (for example, a deliberately random input being rejected by a contract or an authorization gate) are reported separately from unexpected runtime exceptions. Any learning/update trial uses a fresh in-memory model and randomly generated synthetic records; randomized authorization flags only exercise the permission gate and do not represent genuine human verification or consent. No persistent model write is performed.

Random strings and arbitrary structured data are useful for discovering crashes, bad assumptions, unstable numerical operations, type-contract gaps, and unexpected edge behaviour. They are not intended to simulate meaningful human language or to provide a human-readable expected answer.

### How it evaluates algorithms without an answer key

The runner gathers three kinds of implementation evidence:

1. **Randomized execution:** it records each callable, random input, actual arguments, returned type/value summary, errors, elapsed time, Python call profile, and component-provided traces.
2. **Oracle-free invariant checks:** it checks numerical finiteness and reports candidate violations of common probability/confidence bounds or probability normalization. Name-based checks are explicitly marked as candidates because the variable's exact theoretical meaning must be reviewed before calling them defects.
3. **Theory and algorithm coverage:** it discovers package operation/stage registries, captures declared mathematical forms and biological references where present, compares registered operation identifiers with identifiers observed in execution traces, inventories public source functions/classes, and attaches matching architecture documents plus the source-to-code crosswalk.

The automatic runner cannot prove a theory from random input alone. For example, random numbers cannot establish that a semantic representation has the correct human meaning, or that a neural approximation matches measured brain responses. Those claims require domain-specific controlled experiments, reference measurements, and human theory review. This randomized evaluation is designed to uncover weaknesses and direct that next stage, not to overstate what has been demonstrated.

### Reports

Each run creates a timestamped directory under `ore_reports/randomized_theory_tests/` with:

- An overall `summary.json` and `summary.md`.
- One separate JSON and Markdown report for every package.
- Callable-by-callable results, randomized trials, input summaries, output summaries, runtime errors, invariant checks, operation-registry coverage, and matched theory-document excerpts.
- A ZIP archive containing the overall summary and all package reports.

Ordinary manual Workbench runs continue to create automatic JSON and Markdown records under `ore_reports/manual_runs/`. Generated reports are local artifacts and do not need to be committed to Git.

### Run locally

From the repository root:

    python -m pip install -e .
    python -m pip install -r experiment_workbench/requirements.txt
    streamlit run experiment_workbench/app.py

The randomized runner is separate from pytest. Clicking its Streamlit button does not start pytest, GitHub Actions, or a pull request.
