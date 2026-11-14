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

### What the debugger can and cannot see

The live timeline is based on Python's function profiler (and optional Python line tracing) for the selected package. It can show Python function boundaries and source lines, but it cannot show individual C-level NumPy/SciPy instructions, GPU kernels, or work dispatched to other threads/processes. Those operations are instead inspected through input/output array summaries and component-provided diagnostics after they return.

A component may perform substantial work inside one Python call. In that case, the active function remains visible until the call returns; this is expected and is not evidence that the debugger has frozen. Stage-specific progress can only be shown live when the component emits progress events itself. The Workbench does not fabricate intermediate stage updates.

Line tracing adds overhead. Use call tracing as the default; enable line tracing for a focused debugging run.
