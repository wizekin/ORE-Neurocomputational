"""Integrated 15-operation functional and biologically motivated encoding pipeline."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import time
import uuid
from typing import Any, Iterable

import numpy as np

from .biology import CA1Integrator, CA3AutoassociativeNetwork, DentateGyrus
from .contracts import EncodingConfig, EncodingResult, EventInput
from .functional import (
    apply_attention_gain,
    assess_initial_trace_quality,
    build_context_representation,
    build_item_representations,
    build_relational_event_index,
    build_temporal_representation,
    converge_entorhinal_streams,
    detect_event_boundary,
    estimate_novelty,
    extract_parallel_features,
    recurrent_cortical_update,
)
from .registry import OPERATION_REGISTRY


class HippocampalEncodingProcessor:
    """Process observed events through cortical feature and hippocampal circuit abstractions.

    The functional pathway constructs item, context, temporal, and event-index representations.
    The biological pathway models sparse DG recruitment, recurrent CA3 association, CA1 input
    integration, and a simplified calcium-threshold plasticity rule. Neither pathway is a literal
    brain simulation; intermediate variables remain inspectable for experiments and testing.
    """

    def __init__(
        self,
        config: EncodingConfig | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize model state and optionally enable the repository SQLite annotation logger."""
        self.config = config or EncodingConfig()
        self.database_path = Path(database_path).expanduser().resolve() if database_path else None
        self.repository_root = (
            Path(repository_root).expanduser().resolve()
            if repository_root else Path(__file__).resolve().parents[1]
        )
        self.dentate: DentateGyrus | None = None
        self.ca3: CA3AutoassociativeNetwork | None = None
        self.ca1: CA1Integrator | None = None
        self._memory_codes: list[np.ndarray] = []
        self._memory_event_ids: list[str] = []
        self._previous_code: np.ndarray | None = None
        self._previous_context: np.ndarray | None = None
        self._cortical_state: np.ndarray | None = None
        self._last_timestamp_ms: float | None = None

    @staticmethod
    def _fixed_width(values: np.ndarray, width: int = 32) -> np.ndarray:
        """Resample a variable-length feature vector into a fixed model interface width."""
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size == 0:
            return np.zeros(width, dtype=np.float64)
        if vector.size == width:
            return vector.copy()
        old_axis = np.linspace(0.0, 1.0, vector.size)
        new_axis = np.linspace(0.0, 1.0, width)
        return np.interp(new_axis, old_axis, vector).astype(np.float64)

    def _initialize_biological_network(self, entorhinal_size: int) -> None:
        """Allocate fixed-dimension DG, CA3, and CA1 populations on the first event."""
        self.dentate = DentateGyrus(self.config, entorhinal_size)
        self.ca3 = CA3AutoassociativeNetwork(self.config, entorhinal_size, self.config.dentate_units)
        self.ca1 = CA1Integrator(self.config, entorhinal_size)

    def _run_sqlite_logger(self, operation_trace: list[dict[str, Any]], elapsed_ms: float) -> dict[str, Any]:
        """Persist function/class annotations and this event's 15 operation records in SQLite."""
        if self.database_path is None:
            return {"enabled": False, "reason": "database_path_not_configured"}
        from ore_logger.recorder import SQLiteRunRecorder

        run_id = uuid.uuid4().hex
        recorder = SQLiteRunRecorder(
            database_path=self.database_path,
            repository_root=self.repository_root,
            run_id=run_id,
            component="ore_hippocampal_encoding",
            callable_name="HippocampalEncodingProcessor.process",
            started_at_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            entry_kind="neurocomputational_component",
        )
        for index, (spec, trace) in enumerate(zip(OPERATION_REGISTRY, operation_trace), start=1):
            recorder.log_operation({
                "step": index,
                "operation": spec.operation_id,
                "name": spec.name,
                "status": "completed",
                "phase": "rapid_episodic_encoding",
                "explanation": spec.purpose,
                "mathematical_form": spec.mathematical_form,
                "biological_reference": spec.biological_reference,
                "evidence_level": spec.evidence_level,
                "inputs": list(spec.inputs),
                "outputs": list(spec.outputs),
                "functional_output": trace.get("functional_output", {}),
                "biological_output": trace.get("biological_output", {}),
            })
        metadata = recorder.finish("completed", elapsed_ms, output_type="EncodingResult")
        return {"enabled": True, **metadata}

    def process(self, event: EventInput) -> EncodingResult:
        """Encode one observed episode and return all functional, biological, and diagnostic states."""
        started = time.perf_counter()
        operation_trace: list[dict[str, Any]] = []

        def record(
            operation_index: int,
            functional_output: dict[str, Any],
            biological_output: dict[str, Any] | None = None,
        ) -> None:
            """Append an inspectable trace entry aligned with the canonical operation registry."""
            spec = OPERATION_REGISTRY[operation_index]
            operation_trace.append({
                "operation_id": spec.operation_id,
                "operation": spec.name,
                "evidence_level": spec.evidence_level,
                "functional_output": functional_output,
                "biological_output": biological_output or {},
            })

        # HE01: segment an event using current feature and context changes.
        sensory_features = extract_parallel_features(event.sensory_signals)
        raw_item, component_codes = build_item_representations(event, sensory_features, {
            name: 1.0 / len(sensory_features) for name in sensory_features
        })
        raw_context = build_context_representation(event, sensory_features)
        current_code = np.concatenate((
            self._fixed_width(raw_item), self._fixed_width(raw_context),
            self._fixed_width(build_temporal_representation(
                event.timestamp_ms, len(event.components), self.config.temporal_frequencies
            )),
        ))
        goal_relevance = float(np.mean(list(event.goals.values()))) if event.goals else 0.0
        boundary = detect_event_boundary(
            current_code,
            self._previous_code,
            raw_context,
            self._previous_context,
            goal_relevance,
            self.config.event_boundary_threshold,
        )
        boundary["previous_context_retained"] = self._previous_context is not None
        boundary["prior_event_available"] = bool(self._memory_event_ids)
        record(0, boundary)

        # HE02: keep observed channels separate while extracting population-like feature vectors.
        record(1, {
            "channels": list(sensory_features),
            "feature_dimensions": {name: int(vector.size) for name, vector in sensory_features.items()},
            "normalized_features": {name: vector.tolist() for name, vector in sensory_features.items()},
        })

        # HE03: modulate channel gain according to caller-supplied goals, salience, and uncertainty.
        attention_weights, attended_features = apply_attention_gain(
            sensory_features, event.goals, event.salience, event.uncertainty,
            self.config.attention_temperature,
        )
        record(2, {"attention_weights": attention_weights,
                   "attended_features": {name: vector.tolist() for name, vector in attended_features.items()}})

        # HE04-HE06: produce item, contextual, and temporal feature codes.
        item_code, component_codes = build_item_representations(event, attended_features, attention_weights)
        item_code = self._fixed_width(item_code)
        record(3, {"item_representation": item_code.tolist(),
                   "component_ids": list(component_codes)})

        context_code = self._fixed_width(build_context_representation(event, attended_features))
        record(4, {"context_representation": context_code.tolist(),
                   "context_source": "explicit_context_and_observed_scene_channels"})

        temporal_code = self._fixed_width(build_temporal_representation(
            event.timestamp_ms, len(event.components), self.config.temporal_frequencies
        ))
        record(5, {"temporal_representation": temporal_code.tolist(),
                   "timestamp_ms": event.timestamp_ms,
                   "component_order_preserved": [str(item.get("id", index))
                                                  for index, item in enumerate(event.components)]})

        # HE07: converge three streams while retaining their intermediate representations.
        entorhinal_code = converge_entorhinal_streams(item_code, context_code, temporal_code)
        record(6, {"entorhinal_representation": entorhinal_code.tolist(),
                   "stream_lengths": {"item": int(item_code.size), "context": int(context_code.size),
                                      "temporal": int(temporal_code.size)}})

        # HE08: estimate novelty relative to stored initial event codes, not a save/no-save flag.
        novelty = estimate_novelty(entorhinal_code, self._memory_codes)
        if novelty.get("nearest_index") is not None:
            nearest_index = int(novelty["nearest_index"])
            novelty["nearest_episode_id"] = self._memory_event_ids[nearest_index]
        record(7, novelty)

        # Lazily create biologically motivated networks after the fixed EC code dimension is known.
        if self.dentate is None or self.ca3 is None or self.ca1 is None:
            self._initialize_biological_network(entorhinal_code.size)
        assert self.dentate is not None and self.ca3 is not None and self.ca1 is not None

        # HE09: DG sparse competition / pattern separation.
        dentate_activity, dentate_report = self.dentate.encode(entorhinal_code, float(novelty["score"]))
        record(8, {"dentate_activity": dentate_activity.tolist()}, dentate_report)

        # HE10: CA3 associative coding from DG and direct entorhinal afferents.
        ca3_activity, ca3_report = self.ca3.encode(dentate_activity, entorhinal_code)
        record(9, {"ca3_activity": ca3_activity.tolist()}, ca3_report)

        # HE11: CA1 integrates CA3-mediated and direct entorhinal input.
        ca1_activity, prediction_error, ca1_report = self.ca1.integrate(ca3_activity, entorhinal_code)
        record(10, {"ca1_activity": ca1_activity.tolist(),
                    "prediction_error": prediction_error}, ca1_report)

        # HE12: plasticity at recurrent CA3 connections under a modulatory/calcium proxy.
        modulatory_level = float(np.mean(list(event.modulatory_state.values()))) if event.modulatory_state else 0.5
        calcium_proxy = float(np.clip(
            np.mean(ca3_activity) * (0.55 + 0.45 * float(novelty["score"]))
            * (0.65 + 0.70 * modulatory_level), 0.0, 1.0
        ))
        plasticity = self.ca3.update_associations(ca3_activity, calcium_proxy)
        record(11, {"calcium_proxy": calcium_proxy, "plasticity": plasticity}, plasticity)

        # HE13: link index, items, context, time, and only caller-observed relational edges.
        event_index = build_relational_event_index(
            event, ca1_activity, component_codes, context_code, temporal_code
        )
        record(12, {"node_count": len(event_index["nodes"]),
                    "edge_count": len(event_index["edges"]),
                    "event_id": event.event_id})

        # HE14: return a recurrent hippocampal influence to a compact cortical-state surrogate.
        cortical_feedback = recurrent_cortical_update(
            self._cortical_state, ca1_activity, feedback_gain=0.35
        )
        record(13, {"cortical_feedback": cortical_feedback.tolist(),
                    "recurrent_state_updated": True})

        # HE15: estimate initial trace quality without implying permanence or systems consolidation.
        trace_quality = assess_initial_trace_quality(
            float(novelty["score"]), event_index,
            float(plasticity["mean_absolute_delta"]), ca1_activity,
        )
        record(14, {"trace_quality": trace_quality,
                    "consolidation_status": "NOT_EVALUATED_BY_POINT_01"})

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        logger_metadata = self._run_sqlite_logger(operation_trace, elapsed_ms)
        self._memory_codes.append(entorhinal_code.copy())
        self._memory_event_ids.append(event.event_id)
        if len(self._memory_codes) > self.config.max_memory_records:
            self._memory_codes.pop(0)
            self._memory_event_ids.pop(0)
        self._previous_code = current_code.copy()
        self._previous_context = raw_context.copy()
        self._cortical_state = cortical_feedback.copy()
        self._last_timestamp_ms = event.timestamp_ms

        return EncodingResult(
            event_id=event.event_id,
            event_boundary=boundary,
            sensory_features={name: vector.tolist() for name, vector in sensory_features.items()},
            attention_weights=attention_weights,
            item_representation=item_code.tolist(),
            context_representation=context_code.tolist(),
            temporal_representation=temporal_code.tolist(),
            entorhinal_representation=entorhinal_code.tolist(),
            novelty=novelty,
            dentate_activity=dentate_activity.tolist(),
            ca3_activity=ca3_activity.tolist(),
            ca1_activity=ca1_activity.tolist(),
            ca1_prediction_error=prediction_error,
            plasticity=plasticity,
            event_index=event_index,
            cortical_feedback=cortical_feedback.tolist(),
            trace_quality=trace_quality,
            operation_trace=operation_trace,
            biological={
                "dentate_gyrus": dentate_report,
                "ca3": ca3_report,
                "ca1": ca1_report,
                "plasticity": plasticity,
                "status": "biologically_motivated_surrogate_not_physiological_validation",
                "sleep_replay_and_slow_systems_consolidation": "out_of_scope_for_point_01",
            },
            metadata={
                "schema": "ore.rapid_hippocampal_encoding.1.0",
                "operation_count": len(operation_trace),
                "configuration": self.config.to_dict(),
                "timestamp_semantics": "caller-supplied model milliseconds",
                "sqlite_logger": logger_metadata,
            },
        )

    def process_stream(self, events: Iterable[EventInput]) -> list[EncodingResult]:
        """Process a sequence of events in order while retaining recurrent state between events."""
        return [self.process(event) for event in events]
