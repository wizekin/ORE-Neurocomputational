# ORE Experiment Workbench — Implementation Report

**Release:** 0.8.0  
**Implementation date:** 2026-11-13  
**Scope:** Research and controlled-experiment layer over ORE A–J.

## Delivered capabilities

1. **Stage-aware artifact inspection (A–J).** The inspector reads known artifacts emitted by the individual stages and exposes metrics, tables, warnings, and source paths. Specialized views expose Stage C character posteriors, Stage D logical positions, and Stage G/H candidate evidence.
2. **Trace inspection.** The trace reader summarizes `trace.jsonl` by stage and event type, counts events, calculates a timestamp span when available, and reports malformed records or truncation.
3. **Integrity-aware comparison.** Stage H bundle comparison uses the Stage H loader; Stage I overlay comparison verifies the overlay SHA-256 and requires a shared frozen Stage F parent. C→G evidence comparison aligns rows by patch ID and grapheme. General stage comparisons only produce arithmetic deltas for matching units.
4. **Controlled recognition.** The experiment service executes the same patch request against one frozen Stage F model and seed with caller-selected Stage G configuration. It stores both run fingerprints and output locations.
5. **Before/after incremental learning.** The service runs identical recognition inputs without/with an explicitly supplied Stage I overlay and compares the resulting Stage H bundles.
6. **Experiment persistence/export.** Each experiment stores JSON, Markdown and CSV outputs; complete experiment directories can be exported as CRC-verified ZIPs.
7. **Streamlit research panel.** The existing chat UI now includes Stage inspection, Run trace, Compare outputs, Controlled experiments and Experiment history panels.

## Architecture and separation

The Experiment Workbench is a separate Python package, `ore_experiments`, layered over the existing A–J artifacts. It does not replace the recognition pipeline or retrain the frozen model as a side effect of inspection. The Streamlit panel is a presentation layer over `ExperimentWorkbench` service methods.

## Data integrity and scientific interpretation

- Stage H inputs are verified through Stage H's integrity loader before bundle comparison.
- Stage I overlay bytes are checked against the declared manifest hash, and comparisons reject overlays attached to different Stage F parents.
- Per-patch evidence is aligned only by stable patch ID and grapheme; unmatched rows are retained and counted.
- Candidate beam masses are labelled uncalibrated; no recognition correctness probabilities are invented.
- Missing artifacts and measurements are surfaced as warnings or errors.
- Experiment records fingerprint source inputs and preserve configuration, output locations, metrics and warnings.

## Test coverage

The dedicated `tests/test_ore_experiments.py` suite covers Stage C/D/E/G inspection, Stage H comparison/integrity, Stage I overlay checks, trace parsing, durable export, service persistence, configuration validation and path fingerprints. Existing stage and chat-workbench tests remain in the combined suite.

## Known limits

- Stage G still requires pre-segmented visual patches and does not detect text regions or segment a full page image.
- The Experiment Workbench is an ORE artifact research UI, not a generic visualizer for arbitrary Python output schemas. Other projects can reuse the Chat tab through the backend contract; their artifacts need an adapter to appear in ORE-specific experiment inspectors.
- The full Streamlit browser session could not be launched in the implementation environment because Streamlit was unavailable and package-network access was blocked. The application source compiled and its service/inspection APIs were exercised through automated tests. Install `requirements-ui.txt` to launch it locally.
- Software acceptance does not demonstrate OCR accuracy, real-world generalization, biological fidelity or scientific validity. Independent rights-cleared benchmark evaluation is still outstanding.
