# ORE Experiment Workbench Guide

**Package release:** 0.8.0  
**Scope:** ORE stages A–J plus the experiment-inspection and controlled-comparison layer.

The Experiment Workbench is a research interface over persisted ORE artifacts. It does not replace, retrain, or alter the stage implementations. It reads the artifacts written by the stages, creates reproducible experiment records, and lets you inspect how evidence changes across the pipeline.

## 1. Launch

Install the core and optional UI dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip install -r requirements-ui.txt
python ore_main.py serve
```

Or launch Streamlit directly:

```powershell
python -m streamlit run streamlit_app.py
```

The browser UI has two main areas:

- **Chat:** sends commands and attachments to the selected Python backend.
- **Experiments:** inspects saved ORE stage artifacts, compares outputs, runs controlled recognition experiments, and exports experiment evidence.

Set `ORE_WORKSPACE` to choose where run artifacts and experiment records are stored. The default workspace is a sibling directory next to the package.

## 2. Stage inspection

Open **Experiments → Stage inspection**, choose a stage A–J, and enter either the stage's output directory or a parent run directory containing its artifacts.

The inspector searches for the known artifact filenames for that stage. It reports the actual source files it found, metrics derived from those files, structured tables, and warnings for absent or unreadable evidence.

| Stage | Main inspection content |
|---|---|
| A | Run configuration, script profile, trace events and run identity |
| B | Accepted/rejected samples, prepared patches, split counts and target records |
| C | Character posterior candidates, unknown mass, training history and split evaluation |
| D | Logical positions, ordinal ranks, pairwise geometry and order evaluation |
| E | Transition statistics, chunk candidates, promotion and description-length records |
| F | Frozen model metadata, calibration, evaluation, ablations and memory audit |
| G | Ranked candidate sequences, abstention, per-patch visual/fused candidates and seven evidence channels |
| H | Canonical representation, alternatives, uncertainty and provenance |
| I | Incremental-memory changes, verified example counts and learning boundaries |
| J | Software acceptance checks, scientific-evidence state and release evidence |

The interface displays probability-like quantities only when an artifact actually contains them. It does not derive calibrated sequence probabilities from Stage G beam scores.

## 3. Run-trace analysis

Open **Experiments → Run trace** and provide either a run directory containing `trace.jsonl` or the trace file itself.

The trace reader displays:

- total event count and elapsed timestamp span when available;
- event counts grouped by stage and event type;
- the parsed event stream, including sequence, timestamp, step, run ID and payload;
- malformed-line and truncation warnings.

The reader has a maximum event limit to prevent an unbounded trace from exhausting memory. The limit is visible as a warning when reached.

## 4. Compare outputs

Open **Experiments → Compare outputs**. Supported comparisons are:

### Stage H bundle versus Stage H bundle

Both bundles are loaded through Stage H's integrity-checking loader. The report aligns candidate strings and compares rank, beam mass, log score, top-candidate margin and entropy fields when present. It records whether the Stage F parent and input fingerprint match.

Beam masses are uncalibrated relative scores over the returned bounded candidate set. They are not probabilities that the recognized text is correct.

### Stage I overlay versus Stage I overlay

Both overlay file hashes are verified. Comparisons are rejected when the overlays belong to different frozen Stage F parents. The report identifies added, removed and changed chunk records and summarizes update indices and transition-key counts.

### Stage C versus Stage G character evidence

The comparison joins candidate evidence using the stable `(patch_id, grapheme)` key. It displays Stage C visual posterior, Stage G visual candidate value, Stage G fused candidate value, and the descriptive fused-minus-visual delta. It also reports unmatched patch IDs. A delta is meaningful only when the same patch and grapheme occur in both evidence sets; it is not proof that one channel alone caused a change.

### General stage-inspection comparison

Two stage artifact directories can be compared by metric label. Arithmetic deltas are emitted only for numeric values with matching units. Similar labels across different stages do not imply that the metrics have the same scientific meaning.

## 5. Controlled Stage G parameter experiments

Open **Experiments → Controlled experiments**.

1. Upload a Stage G recognition request JSON containing caller-supplied pre-segmented patch matrices and bounding boxes.
2. Point to a frozen, integrity-checked Stage F model directory.
3. Choose a script profile and deterministic seed.
4. Configure A and B. The visible controls include beam width, top K, minimum sequence margin and maximum mean unknown mass.
5. Run the comparison.

Both executions use the same input request, frozen Stage F parent, script profile and seed. Only the supplied Stage G configuration differs. Each run and the Stage H bundle comparison are saved separately.

The underlying service accepts any field declared by `StageGConfig`, while the visible UI presents the commonly varied controls. Invalid configuration names or values are rejected by the Stage G configuration contract.

## 6. Same-input before/after Stage I comparison

The same controlled-experiment panel can compare recognition with no overlay (or a baseline overlay) against an updated Stage I overlay.

Both executions keep the patch request, Stage F model, script profile, seed and Stage G settings fixed. Only the Stage I overlay changes. The comparison verifies the Stage H bundle integrity and reports candidate/rank changes and uncertainty fields.

Stage I overlays must be generated from explicitly verified grapheme labels. An unverified Stage G prediction is not accepted as training truth. The experiment runner does not modify the frozen Stage F checkpoint or the original Stage E memory snapshot.

## 7. Experiment records and export

Each inspection or comparison produces an experiment directory under:

```text
<ORE_WORKSPACE>/experiments/<experiment-id>/
```

Each record includes:

- `experiment.json` — structured record with inputs, fingerprints, outputs, metrics, tables, warnings and notes;
- `report.md` — human-readable experiment report;
- `metrics.csv` — tabular metrics suitable for spreadsheet analysis.

The **Experiment history** tab lists saved records and supports downloading JSON, Markdown and CSV. It can also create a ZIP export of one complete experiment record. ZIP integrity is checked before the download is offered.

Controlled Stage G runs create their stage output directories under the experiment workspace's `controlled_runs` folder. The experiment record preserves their directory paths and fingerprints so that the artifacts can be inspected later.

## 8. Python API

```python
from ore_experiments import ExperimentWorkbench

workbench = ExperimentWorkbench(
    package_root=".",
    workspace="./ore_experiment_workspace",
)

inspection, record_directory = workbench.inspect_stage(
    "C", "./path/to/stage_c_artifacts",
    name="Visual posterior inspection",
)

trace, trace_record = workbench.inspect_trace("./path/to/ore_run")

comparison, comparison_record = workbench.compare_c_to_g_evidence(
    "./path/to/stage_c_artifacts",
    "./path/to/stage_g_artifacts",
)
```

The public API is exported from `ore_experiments`. The service class also exposes Stage H, Stage I, general inspection, controlled-recognition, and before/after Stage I comparisons, plus record listing and ZIP export.

## 9. Interpretation and scope limits

- **No automatic page segmentation:** Stage G consumes pre-segmented patches. Uploading a PNG/JPG does not make the workbench detect text regions or split characters automatically.
- **No semantic language-model correction:** this tool inspects the orthographic evidence produced by ORE. A free-form natural-language prompt does not turn it into a general-purpose LLM.
- **No invented measurements:** missing metrics remain missing. Unmatched patch IDs remain unmatched. Invalid or missing artifact sets produce warnings/errors rather than fabricated values.
- **No calibrated sequence confidence claim:** Stage G beam masses and Stage H candidate masses are not calibrated correctness probabilities.
- **No automatic ground-truth creation:** Stage I requires explicit verified labels. Predictions are not labels.
- **No silent model mutation:** controlled recognition is read-only with respect to frozen upstream model releases. Stage I stores an isolated incremental overlay.
- **No scientific acceptance by dashboard:** software tests and visualization do not establish recognition accuracy. Independent, rights-cleared real-script/real-font evaluation and an external review process remain necessary.

## 10. Reusing the chat front end

The Chat tab remains backend-neutral through `ore_workbench`'s backend contract. An alternate Python implementation can be selected using `ORE_BACKEND=module.path:factory` as described in `WORKBENCH_GUIDE.md`.

The **Experiments** tab is specifically an ORE A–J artifact inspector. A different backend can use the same Chat front end; to use the ORE Experiments tab it must produce compatible ORE artifacts or be adapted to the `ore_experiments` service APIs.


## Stage A: create a run before inspecting it

Stage A artifacts are generated by `initialize_ore`; they are not shipped as pre-existing files in the source-code package. To create them from the Streamlit interface:

1. Open **Experiments → Stage inspection**.
2. Select **A**.
3. Choose `latin_basic` or `arabic_basic`, set a seed, and choose an output directory.
4. Click **Initialize Stage A run**.
5. After the page refreshes, the new run directory is filled into the artifact path. Click **Inspect artifacts**.

Alternatively, initialize a run from Python:

```python
from ore_workbench.orchestrator import OREOrchestrator

orchestrator = OREOrchestrator("/path/to/ORE/package")
context = orchestrator.initialize_stage_a(
    output_directory="./ore_runs",
    script_profile="latin_basic",
    seed=1337,
)
print(context.run_directory)
```

Pass the printed run directory to the Stage A inspector. Pointing Stage A inspection at the package source directory alone is expected to produce an actionable message because source code is not a runtime run artifact. Nested Stage A traces are read from the directory containing the discovered `trace.jsonl` file.

