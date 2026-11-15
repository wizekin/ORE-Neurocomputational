"""Master orchestration of the nine-stage biological and functional ORE tracks."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from ore_logger.catalog import sync_repository
from ore_logger.database import LoggerDatabase
from ore_n9 import N9Adapters, N9Processor

from .biology import AssociativeMemory, NeuralSimulationEngine, phase_coherence, project_drive
from .config import BNBEAConfig
from .contracts import BNBEARequest, BNBEAResult
from .math_ops import (
    bayesian_posterior,
    information_gain,
    normalize_probabilities,
    probability_entropy,
    precision_weighted_prediction_error,
    stable_softmax,
    word_surprisal,
)
from .operation_registry import (
    OPERATION_SPECS,
    STAGES,
    OperationSpec,
    operation_symbol_id,
    register_operation_annotations,
)
from .validation import validate_brain_data


def _json_safe(value: Any) -> Any:
    """Convert numpy and nested values to JSON-compatible, finite Python values."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _compact_event(value: Any, depth: int = 0) -> Any:
    """Keep event metadata small and descriptive instead of storing raw signals."""
    if depth > 4:
        return "…"
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, Mapping):
        return {
            str(key): _compact_event(item, depth + 1)
            for key, item in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple)):
        items = [_compact_event(item, depth + 1) for item in value[:12]]
        if len(value) > 12:
            items.append(f"... {len(value) - 12} omitted")
        return items
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return _json_safe(value)


def _dynamic_symbol_id(source_path: str, kind: str, qualified_name: str) -> str:
    """Generate the stable identifier used for dynamically discovered operations."""
    raw = f"{source_path}|{kind}|{qualified_name}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


class BNBEAProcessor:
    """Execute the nine-stage BNBEA master algorithm and persist symbol annotations.

    The functional track delegates visual, orthographic, attention, lexical,
    syntax, discourse, and semantic processing to the repository's N9
    coordinator. The biological track independently simulates LIF populations,
    recurrent conductance, plasticity, and memory. Outputs are compared but
    never conflated.
    """

    def __init__(
        self,
        config: BNBEAConfig | None = None,
        adapters: N9Adapters | None = None,
        database_path: str | Path | None = None,
        repository_root: str | Path | None = None,
    ) -> None:
        """Initialize reusable controllers, associative memory, and SQLite catalogue."""
        self.config = config or BNBEAConfig()
        self.adapters = adapters or N9Adapters()
        self.repository_root = Path(
            repository_root or Path(__file__).resolve().parents[1]
        ).expanduser().resolve()
        self.database_path = Path(
            database_path or self.repository_root / "ore_logs" / "ore_logger.sqlite3"
        ).expanduser().resolve()
        self.database = LoggerDatabase(self.database_path)
        self.catalog_report = sync_repository(self.database, self.repository_root)
        self.operation_annotation_count = register_operation_annotations(self.database)
        self.memory = AssociativeMemory()
        self.last_run_id: str | None = None

    def _record_event(
        self,
        run_id: str,
        sequence: int,
        event_kind: str,
        symbol_id: str | None,
        source_path: str,
        elapsed_ms: float,
        metadata: Mapping[str, Any] | None = None,
        exception_type: str | None = None,
        exception_message: str | None = None,
    ) -> None:
        """Write a compact, annotation-linked event into the shared SQLite database."""
        self.database.record_event(
            run_id=run_id,
            sequence_no=sequence,
            event_kind=event_kind,
            symbol_id=symbol_id,
            source_path=source_path,
            elapsed_ms=elapsed_ms,
            exception_type=exception_type,
            exception_message=exception_message,
            metadata=_compact_event(dict(metadata or {})),
        )

    def _record_master_stage(
        self,
        run_id: str,
        sequence: int,
        started: float,
        stage_index: int,
        statuses: Mapping[str, str],
        details: Mapping[str, Any] | None = None,
        callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> tuple[int, list[dict[str, Any]]]:
        """Persist all detailed operations for one stage and return their trace rows."""
        stage = STAGES[stage_index - 1]
        rows: list[dict[str, Any]] = []
        specs = [spec for spec in OPERATION_SPECS if spec.stage_index == stage_index]
        for local_index, spec in enumerate(specs, start=1):
            sequence += 1
            status = statuses.get(spec.name, "executed")
            summary = dict(details or {})
            row = {
                "step": local_index,
                "operation_id": spec.operation_id,
                "operation": spec.name,
                "stage": stage.index,
                "stage_code": stage.code,
                "stage_name": stage.name,
                "status": status,
                "explanation": spec.explanation,
                "details": _json_safe(summary),
            }
            rows.append(row)
            self._record_event(
                run_id,
                sequence,
                "bnbea_operation",
                operation_symbol_id(spec.name),
                "ore_bnbea/operation_registry.py",
                (time.perf_counter() - started) * 1000.0,
                {
                    "operation_id": spec.operation_id,
                    "operation": spec.name,
                    "stage_index": stage.index,
                    "status": status,
                    **summary,
                },
            )
            if callback is not None:
                try:
                    callback(dict(row))
                except Exception:
                    pass
        return sequence, rows

    def _record_n9_operation(
        self,
        run_id: str,
        sequence: int,
        started: float,
        event: Mapping[str, Any],
        callback: Callable[[dict[str, Any]], None] | None,
    ) -> int:
        """Persist N9's lower-level functional trace using generated SQLite annotations."""
        row = dict(event)
        name = str(row.get("operation", "unnamed_n9_operation"))
        qualified_name = f"N9.{name}"
        source_path = "ore_n9/pipeline.py"
        symbol_id = _dynamic_symbol_id(source_path, "operation", qualified_name)
        annotation = (
            f"N9 operation {name.replace('_', ' ')}. This is part of the functional "
            "ORE integration track; its detailed implementation and source docstrings "
            "are indexed by the repository AST annotation catalogue."
        )
        self.database.register_annotation(
            symbol_id=symbol_id,
            module_name="ore_n9.pipeline",
            qualified_name=qualified_name,
            symbol_kind="operation",
            source_path=source_path,
            annotation=annotation,
            annotation_source="operation_catalog",
            source_hash=hashlib.sha256(annotation.encode("utf-8")).hexdigest(),
        )
        sequence += 1
        self._record_event(
            run_id,
            sequence,
            "n9_operation",
            symbol_id,
            source_path,
            (time.perf_counter() - started) * 1000.0,
            {
                "step": row.get("step"),
                "operation": name,
                "status": row.get("status", "ok"),
                "details": row,
            },
        )
        if callback is not None:
            try:
                callback(row)
            except Exception:
                pass
        return sequence

    @staticmethod
    def _contextual_prediction(
        class_names: tuple[str, ...],
        likelihood: np.ndarray,
        supplied_priors: Mapping[str, float],
        epsilon: float,
    ) -> dict[str, Any]:
        """Fuse context priors with N9 likelihoods and quantify prediction errors."""
        labels = tuple(class_names)
        evidence = normalize_probabilities(likelihood, axis=1, epsilon=epsilon)
        if supplied_priors:
            prior = np.asarray([float(supplied_priors.get(label, 0.0)) for label in labels])
            if prior.sum() <= epsilon:
                prior = np.ones(len(labels), dtype=np.float64)
        else:
            prior = np.ones(len(labels), dtype=np.float64)
        prior = normalize_probabilities(prior, axis=0, epsilon=epsilon)
        contextual = bayesian_posterior(evidence, prior, epsilon=epsilon)
        predicted = np.broadcast_to(prior[None, :], evidence.shape)
        variance = np.maximum(contextual * (1.0 - contextual), epsilon)
        error = precision_weighted_prediction_error(
            evidence, predicted, variance, epsilon=epsilon
        )
        gains = information_gain(predicted, contextual, epsilon=epsilon)
        entropies = probability_entropy(contextual, axis=1, epsilon=epsilon)
        best_probabilities = np.max(contextual, axis=1)
        best_indices = np.argmax(contextual, axis=1)
        surprisal = word_surprisal(best_probabilities, epsilon=epsilon)
        return {
            "class_names": list(labels),
            "prior_probabilities": prior.astype(np.float32).tolist(),
            "sensory_likelihoods": evidence.astype(np.float32).tolist(),
            "contextual_posteriors": contextual.astype(np.float32).tolist(),
            "predicted_prior_per_position": predicted.astype(np.float32).tolist(),
            "prediction_error_mean": float(np.mean(error["error"])),
            "precision_weighted_error_energy": float(error["weighted_error_energy"]),
            "mean_precision": float(np.mean(error["precision"])),
            "position_entropy_nats": entropies.astype(np.float32).tolist(),
            "position_surprisal_nats": surprisal.astype(np.float32).tolist(),
            "information_gain_nats": gains.astype(np.float32).tolist(),
            "ranked_labels": [labels[index] for index in best_indices],
            "uncertainty_mean_nats": float(np.mean(entropies)),
        }

    def process(
        self,
        request: BNBEARequest,
        on_operation: Callable[[dict[str, Any]], None] | None = None,
    ) -> BNBEAResult:
        """Run stages 1–9, record every operation, and return separated model tracks."""
        if not isinstance(request, BNBEARequest):
            raise TypeError("request must be a BNBEARequest")
        started = time.perf_counter()
        sequence = 0
        trace: list[dict[str, Any]] = []
        run_id = self.database.create_run(
            script_path="ore_bnbea/pipeline.py",
            module_name="ore_bnbea.pipeline",
            entry_kind="bnbea_master",
            working_directory=os.getcwd(),
            process_id=os.getpid(),
            python_version=sys.version.split()[0],
        )
        self.last_run_id = run_id

        def record_stage(
            stage_index: int, statuses: Mapping[str, str] | None = None,
            **details: Any,
        ) -> None:
            nonlocal sequence, trace
            sequence, rows = self._record_master_stage(
                run_id, sequence, started, stage_index, statuses or {},
                details, on_operation,
            )
            trace.extend(rows)

        try:
            # Stage 1: initialize the shared neural substrate and advance its clock.
            engine = NeuralSimulationEngine.build_default_network(self.config)
            engine.advance({}, steps=1)
            record_stage(
                1,
                population_count=len(engine.populations),
                neuron_count=sum(pop.size for pop in engine.populations.values()),
                membrane_time_constant_seconds=self.config.membrane_time_constant_seconds,
                initialization_step=1,
            )

            # Stage 2–4: N9 performs the functional visual-to-orthographic pass.
            n9_events: list[dict[str, Any]] = []

            def n9_callback(event: dict[str, Any]) -> None:
                nonlocal sequence
                sequence = self._record_n9_operation(
                    run_id, sequence, started, event, None
                )
                n9_events.append(_compact_event(event))
                if on_operation is not None:
                    try:
                        on_operation({"track": "functional_n9", **event})
                    except Exception:
                        pass

            functional = N9Processor(adapters=self.adapters).process(
                request.n9_request, on_operation=n9_callback
            )
            image_count = sum(patch.image is not None for patch in request.n9_request.patches)
            supplied_n2 = sum(patch.n2_result is not None for patch in request.n9_request.patches)
            supplied_features = sum(patch.features is not None for patch in request.n9_request.patches)
            stage2_status = (
                "executed" if image_count else
                "upstream_provided" if supplied_n2 else
                "not_applicable"
            )
            record_stage(
                2,
                {name: stage2_status for name in STAGES[1].operations},
                image_patch_count=image_count,
                precomputed_n2_count=supplied_n2,
                caller_feature_count=supplied_features,
                n9_functional_operation_count=len(functional.operation_trace),
            )
            stage3_status = (
                "executed" if image_count else
                "upstream_provided" if supplied_n2 else
                "not_applicable"
            )
            record_stage(
                3,
                {name: stage3_status for name in STAGES[2].operations},
                structured_visual_patch_count=image_count + supplied_n2,
                caller_feature_count=supplied_features,
            )
            record_stage(
                4,
                {name: "executed" for name in STAGES[3].operations},
                grapheme_class_count=len(functional.class_names),
                patch_count=int(functional.posterior_beliefs.shape[0]),
                lexical_hypothesis_count=len(functional.lexical_hypotheses),
                semantic_node_count=len(functional.semantic_graph.get("nodes", [])),
            )

            # Stage 5: drive the recurrent biological network from functional outputs.
            visual_drive = np.mean(functional.visual_probabilities, axis=0)
            orthographic_drive = np.mean(functional.posterior_beliefs, axis=0)
            attention_drive = np.mean(functional.attention_allocation, axis=0)
            predictive_seed = np.mean(functional.posterior_beliefs, axis=0)
            drives = {
                "sensory_retina_lgn_v1": project_drive(
                    visual_drive, self.config.neurons_per_population,
                    self.config.external_current_gain_a,
                ),
                "visual_hierarchy": project_drive(
                    visual_drive, self.config.neurons_per_population,
                    self.config.external_current_gain_a,
                ),
                "orthographic_vwfa": project_drive(
                    orthographic_drive, self.config.neurons_per_population,
                    self.config.external_current_gain_a,
                ),
                "language_association": project_drive(
                    orthographic_drive, self.config.neurons_per_population,
                    self.config.external_current_gain_a,
                ),
                "associative_memory": project_drive(
                    orthographic_drive, self.config.neurons_per_population,
                    self.config.external_current_gain_a * 0.50,
                ),
                "predictive_context": project_drive(
                    predictive_seed, self.config.neurons_per_population,
                    self.config.external_current_gain_a * 0.60,
                ),
                "executive_control": project_drive(
                    attention_drive, self.config.neurons_per_population,
                    self.config.external_current_gain_a * 0.40,
                ),
                "validation_observer": np.zeros(self.config.neurons_per_population),
                "ore_integrator": project_drive(
                    orthographic_drive, self.config.neurons_per_population,
                    self.config.external_current_gain_a * 0.50,
                ),
            }
            spike_history = engine.advance(drives)
            population_rates = np.asarray([
                engine.populations[name].firing_rate_hz for name in engine.POPULATION_NAMES
            ], dtype=np.float64)
            competition = stable_softmax(population_rates, axis=0)
            phases: list[float] = []
            for activity in spike_history.values():
                mean_activity = np.mean(activity, axis=1)
                if mean_activity.size > 2 and np.any(mean_activity):
                    spectrum = np.fft.rfft(mean_activity - np.mean(mean_activity))
                    if spectrum.size > 1 and abs(spectrum[1]) > self.config.epsilon:
                        phases.append(float(np.angle(spectrum[1])))
            coherence = phase_coherence(np.asarray(phases)) if len(phases) >= 2 else None
            network_state = engine.state_summary()
            network_state["competitive_population_allocation"] = {
                name: float(competition[index])
                for index, name in enumerate(engine.POPULATION_NAMES)
            }
            network_state["population_phase_coherence_proxy"] = coherence
            network_state["spike_raster_shapes"] = {
                name: list(values.shape) for name, values in spike_history.items()
            }
            record_stage(
                5,
                recurrent_steps=self.config.simulation_steps,
                recurrent_dt_seconds=self.config.dt_seconds,
                population_count=len(engine.populations),
                projection_count=len(engine.projections),
                phase_coherence_proxy=coherence,
                mean_population_rate_hz=float(np.mean(population_rates)),
            )

            # Stage 6: apply local plasticity and store a fast/slow experience trace.
            learning_metrics = engine.apply_learning()
            experience_pattern = np.mean(functional.posterior_beliefs, axis=0)
            memory_key = request.n9_request.sequence_id
            self.memory.encode(memory_key, experience_pattern)
            retrieved = self.memory.retrieve(experience_pattern, top_k=3)
            replay = self.memory.replay(memory_key)
            replay_count = int(replay.size) if replay is not None else 0
            consolidated = self.memory.consolidate(memory_key)
            self.memory.decay_fast_memory()
            memory_state = self.memory.summary()
            record_stage(
                6,
                coactivation_metric=float(np.mean(np.square(experience_pattern))),
                synaptic_learning=learning_metrics,
                associative_memory=memory_state,
                retrieval_candidate_count=len(retrieved),
                replay_vector_size=replay_count,
                slow_memory_consolidated=consolidated,
                persistent_n5_learning_authorized=bool(request.n9_request.learning_authorized),
                n5_adapter_configured=self.adapters.n5 is not None,
            )

            # Stage 7: compute a separate prior/likelihood update and prediction-error report.
            prediction = self._contextual_prediction(
                tuple(functional.class_names),
                functional.posterior_beliefs,
                request.context_priors,
                self.config.epsilon,
            )
            record_stage(
                7,
                mean_uncertainty_nats=prediction["uncertainty_mean_nats"],
                precision_weighted_error_energy=prediction["precision_weighted_error_energy"],
                mean_information_gain_nats=float(np.mean(prediction["information_gain_nats"])),
                context_prior_supplied=bool(request.context_priors),
            )

            # Stage 8: score only real, caller-supplied experimental observations.
            validation = validate_brain_data(request.brain_data)
            validation_statuses = validation["operation_statuses"]
            record_stage(
                8,
                validation_statuses=validation_statuses,
                validation_status=validation["status"],
                evidence_status=validation["evidence_status"],
                measured_metric_count=len(validation["metrics"]),
            )

            # Stage 9: assemble the two tracks and audit state/provenance integrity.
            if request.lesion_population is not None:
                engine.lesion(request.lesion_population, request.lesion_strength)
                network_state = engine.state_summary()
            adapter_summary = {
                name: ("configured" if getattr(self.adapters, name) is not None else "not_configured")
                for name in ("n3", "n4", "n5", "n6", "n7", "n8", "syntax")
            }
            stage9_statuses = {name: "executed" for name in STAGES[8].operations}
            if self.adapters.n6 is None:
                stage9_statuses["activate_language_network_interfaces"] = "partially_available"
            if not request.n9_request.learning_authorized:
                stage9_statuses["govern_controlled_model_updates"] = "not_authorized"
            elif self.adapters.n5 is None:
                stage9_statuses["govern_controlled_model_updates"] = "not_configured"
            if validation["status"] == "not_tested":
                stage9_statuses["compare_integrated_outputs_with_experimental_evidence"] = "not_tested"
            record_stage(
                9,
                stage9_statuses,
                functional_operation_count=len(functional.operation_trace),
                master_operation_count=len(trace) + len(STAGES[8].operations),
                adapter_availability=adapter_summary,
                database_path=str(self.database_path),
            )

            duration_ms = (time.perf_counter() - started) * 1000.0
            stage_counts = {
                stage.code: sum(1 for row in trace if row["stage_code"] == stage.code)
                for stage in STAGES
            }
            diagnostics = {
                "valid": bool(np.isfinite(functional.posterior_beliefs).all()),
                "stage_count": len(STAGES),
                "master_operation_count": len(trace),
                "expected_master_operation_count": len(OPERATION_SPECS),
                "functional_n9_operation_count": len(functional.operation_trace),
                "stage_operation_counts": stage_counts,
                "simulation_steps": engine.step_count,
                "simulation_duration_seconds": engine.time_seconds,
                "validation_status": validation["status"],
                "biological_acceptance": "not_established",
                "model_scope": (
                    "Biologically motivated point-neuron and population approximations. "
                    "Software execution and synthetic tests do not establish neural validity."
                ),
                "duration_ms": duration_ms,
            }
            self.database.finish_run(run_id, "completed", duration_ms)
            summary = self.database.summary()
            run_summary = self.database.get_run(run_id) or {}
            annotation_count_by_kind: dict[str, int] = {}
            with self.database.connection() as connection:
                for row in connection.execute(
                    "SELECT symbol_kind, COUNT(*) AS n FROM annotations GROUP BY symbol_kind"
                ):
                    annotation_count_by_kind[str(row["symbol_kind"])] = int(row["n"])
            logger_metadata = {
                "database_path": str(self.database.path),
                "run_id": run_id,
                "event_count": int(run_summary.get("event_count", 0)),
                "database_summary": summary,
                "annotation_catalog": self.catalog_report,
                "master_operation_annotation_count": self.operation_annotation_count,
                "annotation_count_by_kind": annotation_count_by_kind,
            }
            biological_state = {
                "network": network_state,
                "plasticity": learning_metrics,
                "memory": memory_state,
            }
            return BNBEAResult(
                sequence_id=request.n9_request.sequence_id,
                functional_result=functional,
                biological_state=_json_safe(biological_state),
                predictive_state=_json_safe(prediction),
                validation=_json_safe(validation),
                operation_trace=trace,
                diagnostics=diagnostics,
                logger_metadata=logger_metadata,
            )
        except Exception as exc:
            duration_ms = (time.perf_counter() - started) * 1000.0
            try:
                sequence += 1
                self._record_event(
                    run_id, sequence, "bnbea_exception", None,
                    "ore_bnbea/pipeline.py", duration_ms,
                    {"failed": True}, type(exc).__name__, str(exc),
                )
                self.database.finish_run(
                    run_id, "failed", duration_ms, f"{type(exc).__name__}: {exc}"
                )
            except Exception:
                pass
            raise


__all__ = ["BNBEAProcessor"]
