# ORE Main Orchestrator and Reusable Streamlit Workbench — Implementation Report

**Package release:** 0.7.0  
**Scope:** Main orchestration, chat backend contract, Streamlit front end, stage-isolated test execution  
**Status:** Implemented; core regression tests passed. Streamlit browser execution could not be launched in this build environment because the `streamlit` package is not installed and package-network access was unavailable.

## Deliverables

| File | Purpose |
|---|---|
| `ore_main.py` | Main CLI entry point: status, isolated tests, full tests, server launch and Stage J acceptance |
| `ore_workbench/orchestrator.py` | Programmatic A–J stage wrappers and ordered A–F training / G→H recognition workflows |
| `ore_workbench/contracts.py` | UI-neutral attachment, request context, backend response and stage test-report types |
| `ore_workbench/backend.py` | Dynamically loads any `module:factory` conforming to the backend contract |
| `ore_workbench/ore_backend.py` | ORE prompt router for help, status, per-stage tests, all tests, training, recognition, verified Stage I feedback and Stage J acceptance |
| `streamlit_app.py` | Chat-style front end with session history, file attachments, backend selector, profile/checkpoint settings and output download buttons |
| `WORKBENCH_GUIDE.md` | Installation, commands, patch JSON, dataset manifest, Stage I feedback and plugin contract |
| `requirements-ui.txt` | Optional Streamlit UI dependency, separate from core algorithm dependencies |
| `tests/test_ore_workbench.py` | Adapter, routing, parser, test isolation and manifest-input tests |

## Orchestration coverage

- A: initialize a deterministic ORE context and script profile.
- B: prepare a rights-/provenance-recorded data manifest and patch set.
- C–F: execute the ordered learning pipeline using the actual stage implementations.
- G: run visual recognition from caller-supplied `VisualPatchInput` / segmentation hypotheses and a verified frozen Stage F release.
- H: serialize the Stage G result into the orthographic representation, ranked hypotheses, uncertainty and provenance bundle.
- I: build Stage H-linked examples from separately verified labels and create/extend the isolated incremental-memory overlay.
- J: execute package acceptance and optional release archive generation.

The orchestrator also runs the dedicated unittest file for exactly one selected stage. The full suite remains available as a separate operation. A dependent stage's test suite may internally use upstream fixtures or contracts; the orchestrator does not bypass declared dependencies or claim that a trained stage can execute without required artifacts.

## Chat/backend separation

The Streamlit UI depends only on `handle_prompt(prompt, attachments, context)` and `describe()`. The backend contract is deliberately independent of ORE dataclasses and stage modules. Any other Python implementation can be selected using `ORE_BACKEND=some.module:create_backend` or the backend field in the sidebar.

## Supported ORE chat operations

- `help`, `status`
- `test stage A` through `test stage J`
- `test all`
- `train attached dataset` with a Stage B-compatible dataset manifest
- `recognize attached patches` with Stage G patch JSON; the selected Stage F release is required
- `learn verified feedback` with a Stage H bundle and independently verified grapheme labels
- `run acceptance` for Stage J

## Input boundary

ORE is an orthographic visual recognition and learning pipeline, not a conversational language model. The prompt field selects operations and supplies instructions; it does not transform arbitrary prose into visual data. Stage G requires pre-segmented grayscale patch matrices and bounding boxes. A raw page image alone is not recognized because page-region detection and automatic segmentation are outside the current A–J implementation. The UI explains this rather than inventing an OCR result.

## Test evidence

- Ten isolated stage suites passed individually: A 12, B 10, C 14, D 20, E 18, F 12, G 10, H 14, I 9, J 11 — **130/130** stage tests.
- Workbench contract/routing tests: **15/15**.
- Combined A–J plus workbench regression suite: **145/145**.
- Python compilation checks passed for the new CLI, Streamlit source, backend package and tests.

The Streamlit package could not be installed in this environment due unavailable package-network access. The UI source compiled successfully, but an actual browser session still needs to be launched in an environment with `requirements-ui.txt` installed.

## Scientific and product limitations

Software tests demonstrate operation of the current Python contracts and orchestration. They do not validate real-world reading accuracy. The current recognizer is patch-input based; a future full-page workflow requires separately tested detection and segmentation components. Stage I only consumes caller-declared verified labels and does not automatically promote model predictions to ground truth. Stage J continues to keep scientific acceptance pending external rights-cleared corpus evaluation.
