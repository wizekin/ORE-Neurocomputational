# ORE A–J Main Orchestrator and Reusable Chat Workbench

**Workbench release:** 0.1.0  
**ORE package:** 0.7.0 (the workbench adds an execution layer; it does not change the underlying recognition algorithms)

## 1. What this adds

- `ore_main.py`: command-line entry point for status, isolated stage tests, full tests, Streamlit launch, and Stage J acceptance.
- `ore_workbench/orchestrator.py`: programmatic coordinator for the real Stage A–J functions.
- `streamlit_app.py`: chat-style Streamlit interface.
- `ore_workbench/backend.py`: backend-neutral plug-in loader.
- `ore_workbench/contracts.py`: small data contracts shared by the front end and any Python implementation.
- `ore_workbench/ore_backend.py`: the ORE-specific chat adapter.

The Streamlit interface does not import or call ORE algorithms directly. It only calls `handle_prompt(prompt, attachments, context)` on the active backend. That is the reusable boundary for other Python projects.

## 2. Install and launch

From this directory, use a virtual environment when possible:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt -r requirements-ui.txt
python ore_main.py status
python ore_main.py serve
```

Open the local Streamlit URL printed by Streamlit (normally `http://127.0.0.1:8501`).

You can also launch the UI directly:

```bash
streamlit run streamlit_app.py
```

## 3. Chat commands

| Prompt | Operation |
|---|---|
| `help` | Show supported commands and input requirements |
| `status` | Inspect all ten stage modules, tests, version and acceptance status |
| `test stage A` … `test stage J` | Run only that stage's dedicated unittest file in a child process |
| `test all` | Run the complete regression suite |
| `run acceptance` | Run Stage J software acceptance and release checks |
| `train attached dataset` | Train A–F from an attached dataset-manifest JSON |
| `recognize attached patches` | Run G recognition then H serialization from patch JSON |
| `learn verified feedback` | Run I from explicitly verified labels linked to a Stage H bundle |

Every stage-specific test command uses a separate invocation of Python's unittest discovery with a single test filename, for example:

```bash
python -m unittest discover -s tests -p test_ore_stage_g.py -v
```

This isolates the *test suite* for a stage. Some tests necessarily load earlier-stage artifacts or shared contracts, because the implementation stages depend on them. The orchestrator does not pretend a dependent stage can run without its declared prerequisites.

## 4. Recognition request format

Stage G currently accepts pre-segmented patch arrays, not a full page image. It does not yet locate text lines or segment an uploaded page automatically. A recognition request is a JSON object such as:

```json
{
  "script_profile": "latin_basic",
  "stage_f_directory": "./path/to/frozen_stage_f_release",
  "hypotheses": [
    {
      "hypothesis_id": "segmentation-1",
      "prior_probability": 1.0,
      "patches": [
        {
          "patch_id": "patch-1",
          "pixels": [[0.0, 0.1, 0.2], [0.0, 0.8, 1.0], [0.1, 0.2, 0.0]],
          "bbox": [10, 10, 13, 13],
          "segmentation_confidence": 0.95
        }
      ]
    }
  ]
}
```

Each patch contains a rectangular grayscale matrix with values in `[0, 1]` and a source-image bounding box `[left, top, right, bottom]`. You can also provide a top-level `patches` array for a single segmentation hypothesis. The script profile selected in the sidebar must match the Stage F model. A Stage F directory is required; the workbench will not invent or train a model during recognition.

The default interface does not convert a raw PNG/JPG page into patches. Doing that correctly requires a separately validated text detector and segmentation component, which should eventually be integrated as an explicit upstream stage rather than hidden inside this UI.

## 5. Dataset manifest for A–F training

The `train attached dataset` command expects a JSON object with a non-empty `samples` array. A sample contains a local image path and optional pre-segmented patch matrices. Example:

```json
{
  "dataset_id": "my_dataset",
  "dataset_version": "1.0",
  "rights_license": "CC-BY-4.0",
  "source": "dataset source description",
  "samples": [
    {
      "image_path": "images/sample_001.png",
      "label": "cat",
      "source_record_id": "record-001",
      "patches": [
        {
          "patch_id": "p1",
          "pixels": [[0.0, 0.2], [0.8, 1.0]],
          "bbox": [0, 0, 2, 2],
          "label_grapheme_index": 0,
          "alignment_confidence": 1.0
        }
      ]
    }
  ]
}
```

The example is a schema illustration only; a meaningful train/validation/test partition needs enough rights-cleared, appropriately diverse samples. The caller must ensure `image_path` resolves in the workbench inbox or from the dataset manifest's directory. For large datasets, using a manifest that references files on disk is preferable to uploading every file through Streamlit.

## 6. Stage I verified-feedback request

Stage I can be reached from the chat interface with `learn verified feedback` and a JSON attachment. The Stage H bundle directory and frozen Stage F directory can be entered in the sidebar or included in the JSON. Every example must provide pre-segmented graphemes, a split (`train`, `validation`, or `test`), a permitted verification status, and source/reference identifiers. See [`examples/workbench/stage_i_feedback.example.json`](examples/workbench/stage_i_feedback.example.json).

Permitted verification statuses are `HUMAN_CONFIRMED`, `INDEPENDENTLY_VERIFIED`, and `RIGHTS_CLEARED_CORPUS`. These are auditable declarations made by the caller; the software cannot independently establish whether a label is correct. Stage H linkage is verified and recorded. Only training examples update counts; validation examples gate chunk promotion; test examples are reporting-only. A prior Stage I overlay can optionally be supplied in the sidebar.

## 7. Use the orchestrator from another Python script

```python
from ore_workbench.orchestrator import OREOrchestrator

ore = OREOrchestrator(".")

# Test one stage independently.
report = ore.run_stage_tests("G")
print(report.passed, report.tests_run)

# Test the complete package.
full = ore.run_all_tests()
print(full["passed"], full["tests_run"])
```

The orchestrator also exposes `initialize_stage_a`, `prepare_stage_b`, `train_stage_c`, `train_stage_d`, `train_stage_e`, `train_stage_f`, `train_through_f`, `recognize_stage_g`, `serialize_stage_h`, `learn_stage_i`, `learn_from_feedback_json`, and `accept_stage_j`.

## 8. Plug in another Python implementation

Create an importable module that returns an object with two methods:

```python
from ore_workbench.contracts import BackendReply

class MyBackend:
    def describe(self):
        return {"name": "My Python Engine", "capabilities": ["summarize", "analyze"]}

    def handle_prompt(self, prompt, attachments, context):
        # attachments: Sequence[Attachment] with name, content_type and data bytes
        # context: ChatContext with workspace, settings and session_id
        answer = my_engine.process(prompt)
        return BackendReply(text=str(answer), status="ok")

def create_backend(package_root=None):
    return MyBackend()
```

Run the same Streamlit front end with:

```bash
# PowerShell
$env:ORE_BACKEND = "my_project.backend:create_backend"
python ore_main.py serve

# macOS/Linux
# ORE_BACKEND=my_project.backend:create_backend python ore_main.py serve
```

Alternatively, change the backend factory in the Streamlit sidebar and reload it. The backend module must be importable in the Python environment running Streamlit. The UI accepts `BackendReply`, a string, or a compatible dictionary response.

## 9. Safety and scientific boundaries

- Uploaded `.py`, executable, pickle, and archive files are not accepted by the Streamlit upload widget, and the backend does not execute attachment contents.
- Stage I still requires explicitly verified labels. An ORE hypothesis is not ground truth.
- Stage J software acceptance is distinct from scientific acceptance.
- Synthetic test fixtures show code paths and contracts work; they are not a benchmark of reading accuracy.
- Streamlit is an optional interface dependency listed in `requirements-ui.txt`. Install it only when you want the browser UI; the core orchestrator and CLI can be used without launching the UI.
