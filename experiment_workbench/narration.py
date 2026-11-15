"""Plain-language narration and durable logging for ORE execution."""
from __future__ import annotations

import json
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from ore_n9.pipeline import N9_ALGORITHM_STEPS
except Exception:  # Keep the logger usable for isolated component experiments.
    N9_ALGORITHM_STEPS = ()

N9_PHASES = (
    (1, 12, "A — Input, provenance and state", "ORE checks incoming evidence, identifies its format, preserves source identity, and prepares processing state and module connections."),
    (13, 24, "B — Visual evidence and organization", "ORE extracts or reuses visual signals, organizes local features and contours, and packages available visual evidence. Character segmentation is supplied by the caller; this controller does not invent character crops."),
    (25, 36, "C — Grapheme identity and sequence", "ORE forms grapheme alternatives, records positions, compares features, uses transition memory when available, and builds candidate sequences and word spans."),
    (37, 48, "D — Attention and precision", "ORE estimates uncertainty and prediction error, allocates attention across positions, adjusts sensory and prior precision, and prepares these signals for later processing."),
    (49, 60, "E — Sequence, lexical and phonological access", "ORE combines forward and backward sequence expectations with visual evidence, then requests lexical, morphological and pronunciation information where an adapter is available."),
    (61, 72, "F — Syntax and semantic composition", "ORE looks for grammatical structure and composes supported roles, phrases, propositions and typed semantic relations. Missing adapters remain explicitly unavailable."),
    (73, 84, "G — Discourse and situation context", "ORE gathers supplied discourse, semantic-memory and episodic context, then requests intent, coherence, temporal/causal links and uncertainty from the context adapter."),
    (85, 96, "H — Bounded feedback and conflict resolution", "ORE sends higher-level predictions back toward lexical and grapheme alternatives, measures disagreement, performs bounded belief revision and preserves separate evidence channels."),
    (97, 108, "I — Integration, decision and output", "ORE combines available representations, evaluates consistency and confidence, chooses a candidate or abstains, constructs the result and performs integrity checks."),
    (109, 120, "J — Learning governance and scientific acceptance", "ORE freezes the inference result before considering learning. Any model update requires explicit authorization and must be proposed on a detached child model; validation requirements are recorded separately."),
)

_N9_STEP_DESCRIPTIONS = (
"Checks that every patch has one permitted evidence source, a usable identifier and valid confidence metadata. Invalid input is rejected rather than silently reinterpreted.",
"Identifies whether evidence arrives as an image, N2 representation, feature vector or grapheme probabilities, and reads the declared script and reading direction.",
"Builds a stable digest from the sequence, source and patch evidence so later results can be linked to this observation without copying image pixels into the log.",
"Creates a mask showing which parts of each patch are observed, allowing reliability estimates to distinguish visible evidence from missing or occluded regions.",
"Creates normalized position coordinates while preserving caller order, so downstream stages can reason about where evidence belongs.",
"Combines segmentation confidence with visible fraction to estimate how dependable each patch is as sensory evidence.",
"Creates separate storage channels for visual, orthographic, lexical, phonological, syntactic, semantic and contextual information instead of collapsing them into one value.",
"Checks which adapters are connected and whether a configured N3 model is fitted before attempting to use it.",
"Fingerprints connected model resources before inference so unintended changes to read-only models can be detected.",
"Records the requested task and whether learning has been explicitly authorized. Recognition and learning remain separate decisions.",
"Copies supplied context into a run-specific snapshot so this run uses a stable view of the caller's context.",
"Defines how visual, orthographic, attention, lexical, syntax, semantic, discourse, feedback, decision and learning stages depend on each other.",
"Measures basic brightness statistics for image patches when pixels are supplied. Without image data, this operation is marked not applicable.",
"Retrieves N1 ON/OFF center-surround responses when N1 image processing was run. These channels emphasize local brightness increases and decreases.",
"Retrieves normalized N1 feature maps when available so local visual responses can be compared on a controlled scale.",
"Retrieves orientation-sensitive N1 energy maps when available. These represent evidence for edges, bars and other oriented patterns.",
"Retrieves N2 multiscale representations when present, preserving visual structure at different spatial scales.",
"Retrieves N2 contour evidence when available, which can help describe boundaries and junction-like structures.",
"Builds or copies a compact descriptor for each patch from available N2 or caller-provided features. Missing feature sources receive an explicit neutral placeholder.",
"Stores the selected descriptor source for each patch and records whether it came from N2, caller features or probability-only input.",
"Keeps caller-localized patch boundaries as explicit hypotheses. It records that automatic segmentation was not performed.",
"Computes pairwise spatial grouping strengths from patch coordinates. This is a simple grouping proxy, not a full figure-ground or stroke-segmentation system.",
"Associates each localized patch with its feature availability and region index so later candidate evidence remains aligned to its source.",
"Marks which evidence channels are available for the next stage, including raw observations, N1 visual outputs, N2 structure and caller-supplied grapheme probabilities.",
"Builds the active symbol vocabulary from supplied probability labels or a fitted N3 model. If neither exists, it creates an explicit unknown-only fallback.",
"Creates or retrieves a probability distribution over grapheme classes for every patch. Rows remain aligned with the original patch sequence.",
"Measures probability assigned to an explicit unknown class, when that class exists in the vocabulary.",
"Creates an identity-like position matrix indicating each patch's absolute location in the current sequence.",
"Computes pairwise relative horizontal offsets so the system can represent how patches are positioned with respect to one another.",
"Normalizes patch descriptors and compares them pairwise to estimate feature overlap or similarity.",
"Keeps grapheme probabilities and position codes as separate channels so location does not overwrite symbol identity.",
"Uses the declared reading direction while preserving caller-provided logical patch order; it does not reorder patches without explicit evidence.",
"Checks whether a fitted N3 transition model is available and records its fingerprint when present. No transition memory is invented when absent.",
"Expands strong grapheme alternatives into a bounded beam of candidate sequences, limiting paths to the configured beam width.",
"Uses caller-provided word spans when available and adds single-patch spans for uncovered positions so every patch remains represented.",
"Packages visual probabilities, position codes and transition-model availability as distinct evidence sources for later inference.",
"Calculates distribution entropy at each position. A flatter distribution means the system is less certain about which grapheme is present.",
"Uses N3 classwise prediction error when available; otherwise it uses grapheme uncertainty as a conservative error proxy.",
"Combines normalized uncertainty, prediction error and supplied task relevance to estimate which positions deserve more processing.",
"Adds bottom-up salience and task relevance into position-wise scores before competitive attention is allocated.",
"Applies softmax across positions so attention is distributed competitively rather than assigning every position the same unconstrained weight.",
"Blends attention with a uniform component and smooths neighboring positions, preventing a single location from receiving all resources.",
"Uses observation reliability and allocated attention to estimate how strongly sensory evidence should influence later belief updates.",
"Estimates the strength of prior predictions. When N4 is available its output is used; otherwise N9 computes a bounded fallback.",
"Compares sensory precision with prior precision to show which evidence channel should have greater influence at each position.",
"Builds class-selective gates from prediction errors, scaling mismatch signals by the relevant precision estimates.",
"Packages attention and precision values for downstream adapters without changing the preserved visual probability channel.",
"Stores attention state needed for later feedback analysis. This implementation prepares metadata; it does not run an eye-movement controller.",
"Starts sequence inference from attended grapheme probabilities, normalizing each position into a probability distribution.",
"Uses learned forward transition statistics to predict the next symbol from the preceding position when a transition model is available.",
"Uses learned reverse transition statistics to predict a symbol from the following position when a transition model is available.",
"Combines forward and backward expectations into a contextual sequence prior and reconciles it with visual evidence using precision-weighted scores.",
"Calls the configured N6 lexical adapter with grapheme probabilities, patch IDs, word spans and context. Without it, N9 uses a surface-form fallback.",
"Reads stem, affix or other morphology alternatives returned by N6. Missing adapter evidence is reported as not configured.",
"Reads pronunciation or phonological-form alternatives returned by N6 when available; it does not invent pronunciations.",
"Keeps the visual route and any adapter-provided lexical route identifiable while packaging word, morphology and phonology candidates.",
"Sorts lexical hypotheses by their supplied scores so the strongest supported candidate is considered first.",
"Compares the best lexical score with the configured confidence threshold and marks the result for abstention when evidence is too weak.",
"Retains the word-span alignment used for lexical resolution so candidate words remain tied to their source patches.",
"Marks the lexical interface output as ready, including lexical alternatives and the status of optional morphology and phonology fields.",
"Reads grammatical-category and morphosyntactic evidence from the lexical adapter when such evidence has been supplied.",
"Prepares candidate grammatical structures from lexical hypotheses and any part-of-speech features.",
"Loads only grammar rules explicitly supplied in context. It does not assume an undocumented grammar.",
"Calls the configured syntax parser with lexical candidates, grammatical features, grammar constraints and context.",
"Reads agent, patient and other thematic-role assignments only when the syntax output provides them.",
"Reads phrase-level structures from the parser so words can be grouped into larger grammatical units.",
"Reads scope information for negation, modality and quantifiers when the parser supplies it; missing evidence remains unavailable.",
"Reads explicit connective and discourse-relation outputs rather than guessing relations that were not supplied.",
"Accepts coreference links only from configured parser output, avoiding unsupported assumptions about which entities refer to one another.",
"Reads propositions from the semantic adapter or lexical interface when available, preserving the distinction between sentence form and meaning.",
"Obtains a semantic graph from the configured adapter and checks its node and edge structure against the expected contract.",
"Records adapter-provided completeness and ambiguity estimates. Without semantic composition, completeness stays low and ambiguity stays high.",
"Selects supplied prior-turn, document, domain and topic context fields for this run.",
"Reads the semantic-memory snapshot supplied by the caller. This step does not update the memory.",
"Reads explicitly supplied episodic or document context and records whether it is available.",
"Calls the configured N7 context adapter to obtain intent, domain, predictions and situation-model information.",
"Reads compatibility scores between propositions and current context when the adapter provides them.",
"Reads context-conditioned predictions from N7. The controller does not fabricate likely next words when the adapter is absent.",
"Collects temporal and causal links supplied by the context adapter.",
"Stores the situation-model representation returned by N7 when one is available.",
"Records the context adapter's discourse-coherence score, or a neutral default when no adapter is connected.",
"Collects explicit knowledge-conflict reports from the context adapter; it does not independently verify world facts.",
"Clips the context adapter's uncertainty estimate to a valid range and records it for confidence calibration.",
"Packages selected context, intent, predictions and situation information for feedback and final integration.",
"Collects lexical predictions from context and semantic adapters to form higher-level feedback signals.",
"Maps adapter-provided semantic or lexical predictions onto the known grapheme vocabulary where symbols match.",
"Projects lexical word-form hypotheses back onto individual grapheme positions, weighted by each candidate's score.",
"Reads predicted visual-feature evidence from the semantic adapter when supplied; otherwise this feedback channel remains unavailable.",
"Measures differences between visual probabilities and the current proposal, and separately tracks lexical-feedback mismatch.",
"Scales visual and lexical mismatch signals by their respective precision estimates.",
"Computes a free-energy-inspired objective from visual fit and divergence from the bidirectional prior. It is a computational proxy, not a measured brain quantity.",
"Repeatedly updates posterior probabilities using visual, lexical-feedback and contextual terms, with damping and a maximum iteration count.",
"Measures disagreement between preserved visual probabilities and the revised posterior using Jensen–Shannon divergence.",
"Identifies positions with high uncertainty or error for possible reinspection. The current implementation records indices; it does not automatically rerun image processing.",
"Checks whether successive posterior updates became sufficiently small before the iteration limit was reached.",
"Stores visual, orthographic, posterior and contextual-prior distributions separately so feedback cannot erase the original evidence.",
"Combines available sequence, lexical, syntax, semantic and context structures into one integrated representation.",
"Builds joint candidates by attaching available semantic and contextual support to each leading lexical hypothesis.",
"Combines syntax, semantic-completeness and discourse-coherence evidence into a cross-level consistency estimate.",
"Collects support from visual, attended, lexical, semantic and contextual channels for the final decision.",
"Combines posterior confidence, lexical confidence and contextual uncertainty into a bounded confidence estimate and candidate margin.",
"Chooses a candidate only when confidence and ambiguity gates permit it; otherwise the result is explicitly AMBIGUOUS or ABSTAIN.",
"Builds the machine-readable result containing the decision and ranked grapheme alternatives for each source patch.",
"Creates task-facing output fields, such as recognized text for reading or transcription tasks, while retaining decision status.",
"Collects evidence sources, adapter usage, warnings and limitations that explain how the decision was reached.",
"Checks posterior integrity and, under strict mode, restores preserved visual probabilities and abstains if the revised state is invalid.",
"Records run-size indicators such as patch count, class count, feature count, feedback iterations and elapsed time.",
"Verifies finite probabilities, normalized rows and normalized attention before allowing the result to be returned.",
"Hashes the inference result before considering learning, creating a boundary between recognition and model modification.",
"Identifies uncertain, unknown or conflicting positions that could be reviewed as possible learning examples.",
"Checks the request's explicit learning-authorization flag. No update is permitted merely because an input was recognized.",
"When authorized and N5 exists, deep-copies the adapter and proposes an update on that detached copy. The parent model is checked for mutation.",
"Counts adjacent symbol pairs in candidate paths as reviewable statistics without directly changing the active transition model.",
"Finds repeated transition patterns that may merit minimum-description-length review; promotion still requires review.",
"Records regularization and stability conditions for a proposed learning result; it does not claim to reproduce human biological homeostasis.",
"Attaches parent and child model fingerprints and provenance metadata to any learning proposal.",
"Records components to isolate or ablate in future experiments to test whether each module contributes useful evidence.",
"Records the need for real-corpus, held-out and independent validation rather than treating synthetic software tests as proof of recognition quality.",
"Keeps software correctness, task performance and biological plausibility as separate acceptance criteria.",
"Packages the final N9 result, diagnostics, operation trace, decision and governance information for the caller.",
)

if N9_ALGORITHM_STEPS and len(N9_ALGORITHM_STEPS) != len(_N9_STEP_DESCRIPTIONS):
    raise RuntimeError("N9 narration descriptions do not match the operation registry.")
N9_OPERATION_EXPLANATIONS = dict(zip(N9_ALGORITHM_STEPS, _N9_STEP_DESCRIPTIONS))


def _plain_name(value: Any) -> str:
    text = str(value or "operation").strip()
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return text.replace("_", " ").replace(".", " ")


def _safe_detail(value: Any, depth: int = 0) -> Any:
    """Bound event details; never expand full image or array payloads."""
    if depth >= 5:
        return "..."
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:240]
    if isinstance(value, Mapping):
        return {str(key)[:80]: _safe_detail(item, depth + 1)
                for key, item in list(value.items())[:30]}
    if isinstance(value, (list, tuple)):
        items = [_safe_detail(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            items.append(f"... {len(value) - 12} more value(s)")
        return items
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {"array_shape": list(shape), "dtype": str(dtype or "unknown")}
    return repr(value)[:240]


def _phase_for_step(step: Any) -> tuple[str, str]:
    try:
        number = int(step)
    except (TypeError, ValueError):
        return ("Component operation", "A component-authored diagnostic step.")
    for start, end, title, description in N9_PHASES:
        if start <= number <= end:
            return title, description
    return ("Component operation", "A component-authored diagnostic step.")


def explain_operation(operation: Any, step: Any = None, status: Any = "ok") -> dict[str, str]:
    name = str(operation or "unnamed_operation")
    phase, phase_description = _phase_for_step(step)
    explanation = N9_OPERATION_EXPLANATIONS.get(
        name, f"ORE is carrying out the component step: {_plain_name(name)}."
    )
    status_text = str(status or "ok").lower()
    status_explanations = {
        "ok": "Completed.",
        "returned": "Completed and the result was packaged.",
        "not_applicable": "This input did not contain the evidence needed for this operation, so the step was correctly marked not applicable.",
        "not_configured": "The required optional adapter or model is not configured. ORE preserves that absence instead of inventing an output.",
        "fallback": "The configured component was unavailable, so ORE used the explicit fallback described in the operation details.",
        "abstention_fallback": "The required recognition evidence was unavailable. ORE used an unknown-only fallback and avoids claiming a recognized symbol.",
        "abstain": "The evidence did not meet the configured confidence requirement, so ORE abstains.",
        "ambiguous": "More than one interpretation remains too close to select confidently.",
        "not_authorized": "Learning was not authorized; the inference result remains separate from model updates.",
        "skipped_without_explicit_authorization": "Skipped because explicit learning authorization was not provided.",
        "authorized_but_N5_unavailable": "Authorization was present, but the N5 learning adapter was not configured.",
        "error": "The operation failed. Inspect the exception and traceback for the cause.",
    }
    return {
        "title": _plain_name(name), "phase": phase,
        "phase_explanation": phase_description, "explanation": explanation,
        "status_explanation": status_explanations.get(status_text, f"Recorded status: {status_text}."),
    }


class OREExecutionLogger:
    """Thread-safe JSONL and Markdown logger for one ORE run."""

    def __init__(self, output_dir: str | Path, run_id: str, component: str,
                 callable_name: str, started_at: str,
                 on_event: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.run_id = run_id
        self.component = component
        self.callable_name = callable_name
        self.started_at = started_at
        self.on_event = on_event
        self._lock = threading.RLock()
        self._clock = time.perf_counter()
        self._sequence = 0
        self._event_count = 0
        self._operation_keys: set[tuple[str, str]] = set()
        self._closed = False
        self._errors: list[str] = []
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.run_dir = Path(output_dir).expanduser() / day
        self.run_dir.mkdir(parents=True, exist_ok=True)
        stem = f"ore-{run_id}"
        self.jsonl_path = self.run_dir / f"{stem}.jsonl"
        self.markdown_path = self.run_dir / f"{stem}.md"
        self._json = self.jsonl_path.open("a", encoding="utf-8")
        self._markdown = self.markdown_path.open("a", encoding="utf-8")
        self._markdown.write(
            f"# ORE execution narrative\n\n- Run ID: {run_id}\n"
            f"- Component: {component}\n- Callable: {callable_name}\n"
            f"- Started (UTC): {started_at}\n\n"
            "Raw prompts, image pixels and full input arrays are not written to this log.\n\n"
            "## Event timeline\n"
        )
        self._markdown.flush()
        self._emit({
            "kind": "run_started",
            "explanation": f"Started {callable_name} in component {component}.",
            "details": {"input_values_logged": False},
        })

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "jsonl_path": str(self.jsonl_path),
            "markdown_path": str(self.markdown_path),
            "event_count": self._event_count,
            "logger_errors": list(self._errors),
        }

    def _emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            if self._closed:
                return
            self._sequence += 1
            record = {
                "sequence": self._sequence, "run_id": self.run_id,
                "component": self.component, "callable": self.callable_name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": round((time.perf_counter() - self._clock) * 1000, 3),
                **event,
            }
            record["details"] = _safe_detail(record.get("details", {}))
            try:
                self._json.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._json.flush()
                self._write_markdown(record)
                self._event_count += 1
            except Exception as exc:
                self._errors.append(f"{type(exc).__name__}: {exc}")
            if self.on_event is not None:
                try:
                    self.on_event(dict(record))
                except Exception:
                    pass

    def _write_markdown(self, event: Mapping[str, Any]) -> None:
        kind = str(event.get("kind", "event"))
        if kind == "operation":
            heading = f'Step {int(event.get("step", 0)):03d} · {event.get("phase", "ORE")} · {event.get("title", event.get("operation", "operation"))}'
        elif kind == "component_trace":
            heading = f'Component trace · {event.get("title", event.get("operation", "step"))}'
        else:
            heading = f'{kind.replace("_", " ").title()} · {event.get("function", event.get("callable", self.callable_name))}'
        self._markdown.write(f"\n### {heading}\n\n")
        self._markdown.write(f'- Time: {event.get("elapsed_ms", 0)} ms\n')
        if event.get("status") is not None:
            self._markdown.write(f'- Status: {event["status"]}\n')
        if event.get("explanation"):
            self._markdown.write(f'\n{event["explanation"]}\n')
        if event.get("status_explanation"):
            self._markdown.write(f'\n{event["status_explanation"]}\n')
        if event.get("phase_explanation"):
            self._markdown.write(f'\n_{event["phase_explanation"]}_\n')
        if event.get("details"):
            self._markdown.write("\nRecorded details\n\nJSON:\n")
            self._markdown.write(json.dumps(event["details"], indent=2, ensure_ascii=False, default=str))
            self._markdown.write("\n")
        if event.get("file") or event.get("line"):
            self._markdown.write(f'\nSource: {event.get("file", "")}:{event.get("line", "")}\n')
        self._markdown.flush()

    def log_python_event(self, event: Mapping[str, Any]) -> None:
        row = dict(event)
        kind = row.get("kind", "python_event")
        if not row.get("explanation"):
            label = _plain_name(row.get("short_name") or row.get("function"))
            if kind == "call":
                row["explanation"] = f"Python entered {label}."
            elif kind == "return":
                row["explanation"] = f"Python finished {label} and returned {row.get('return_type', 'a value')}."
            elif kind == "exception":
                row["explanation"] = f"{row.get('exception_type', 'An exception')} was raised while {label} was running."
            elif kind == "line":
                row["explanation"] = f"Python is executing source line {row.get('line')} in {label}."
        self._emit(row)

    def log_operation(self, operation_event: Mapping[str, Any]) -> None:
        row = dict(operation_event)
        name = str(row.get("operation", row.get("name", "unnamed_operation")))
        step = row.get("step", row.get("number"))
        status = row.get("status", "ok")
        details = {k: v for k, v in row.items()
                   if k not in {"kind", "step", "number", "operation", "name", "status"}}
        translated = explain_operation(name, step, status)
        self._operation_keys.add((str(step), name))
        self._emit({
            "kind": "operation", "step": step, "operation": name,
            "title": translated["title"], "phase": translated["phase"],
            "phase_explanation": translated["phase_explanation"],
            "explanation": translated["explanation"], "status": status,
            "status_explanation": translated["status_explanation"],
            "details": details, "source": row.get("source", "live_operation_callback"),
        })

    def log_component_traces(self, rows: list[Mapping[str, Any]]) -> None:
        """Log returned traces from components that do not stream operation events."""
        for index, source_row in enumerate(rows):
            row = dict(source_row)
            name = str(row.get("operation", row.get("name", row.get("stage", row.get("step_name", f"trace_{index + 1}")))))
            step = row.get("step", row.get("index", index + 1))
            if (str(step), name) in self._operation_keys:
                continue
            status = row.get("status", "recorded")
            translated = explain_operation(name, step, status)
            self._emit({
                "kind": "component_trace", "step": step, "operation": name,
                "title": translated["title"], "phase": translated["phase"],
                "phase_explanation": translated["phase_explanation"],
                "explanation": translated["explanation"], "status": status,
                "status_explanation": translated["status_explanation"],
                "details": row, "source": "returned_component_trace",
            })

    def finish(self, status: str, elapsed_ms: float | None,
               output_type: str | None = None, error: str | None = None) -> dict[str, Any]:
        with self._lock:
            if not self._closed:
                self._emit({
                    "kind": "run_finished", "status": status,
                    "explanation": (
                        f"Execution {status}. Output type: {output_type or 'none'}."
                        if not error else f"Execution failed: {error[:500]}"
                    ),
                    "details": {"elapsed_ms": elapsed_ms, "output_type": output_type},
                })
                self._closed = True
                self._json.close()
                self._markdown.write("\n\n---\n\nEnd of execution log.\n")
                self._markdown.flush()
                self._markdown.close()
        return self.metadata
