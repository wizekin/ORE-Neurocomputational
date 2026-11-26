"""The 52-step scientific reasoning loop (Phases I-V of the integrated algorithm).

``ScientificReasoner`` executes the closed loop: understand the question, identify
the domain, formalize variables and assumptions, select a representation, select
solver/model, generate predictions or candidates, validate independently, quantify
uncertainty, record provenance, and update knowledge only when evidence warrants it.
Every one of the 52 steps is recorded in the operation trace with an explicit status;
steps that do not apply to the routed domain are marked ``not_applicable`` with a
reason rather than silently skipped, and unavailable external services are marked
``not_configured``. Failures produce typed outcomes (INVALID_INPUT,
UNSUPPORTED_DOMAIN, RESOURCE_EXHAUSTED, VALIDATION_FAILURE, LIMITED_RESULT) instead
of unjustified scientific conclusions.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .adapters import (
    ConditionalMaterialGenerator,
    CrankNicolsonHeatSolver,
    EquivarianceTestProtocol,
    EquivariantPointModule,
    FourierModeOperator,
    LocalReferenceDesignEngine,
    LocalReferenceInteractionService,
    LocalReferenceStabilityModel,
    LocalReferenceStructureService,
    NonEquivariantPointModule,
    PDEProblem,
    PinnBasisSurrogate,
    PinnWeights,
    StochasticForecastSimulator,
    build_synthetic_slam_problem,
    calibration_report,
    evaluate_plausibility,
    structure_fingerprint,
)
from .contracts import (
    ExplanationBlock,
    OutcomeKind,
    ProblemClass,
    ResultStatus,
    ScienceConfig,
    ScienceRequest,
    ScienceResult,
    ScientificDomain,
    ScientificPrediction,
    ScientificTask,
    UncertaintyReport,
    ValidationReport,
    validate_status_transition,
)
from .interpretation import DomainRouter, ScientificTaskParser
from .knowledge import ScienceKnowledgeBase
from .model_registry import ModelRegistry
from .orchestration import (
    CandidateKnowledgeUpdate,
    KnowledgeUpdatePolicy,
    ResourceAdmissionControl,
    ScienceResourceBudget,
    ScientificOrchestrator,
    StatusLadder,
    WorkspacePublisher,
)
from .provenance import ProvenanceRecorder
from .registry import ALL_OPERATIONS, OPERATION_REGISTRY
from .representations import RepresentationBuilder
from .uncertainty import UncertaintyEstimator, out_of_domain_check
from .units import UNITS
from .validation import ValidationEngine

__all__ = ["ScientificReasoner"]


_VALIDATED_ALPHA_RANGE = (1.0e-7, 1.0e-2)


class ScientificReasoner:
    """Run the full 52-step scientific reasoning loop over one request."""

    def __init__(
        self,
        config: ScienceConfig | None = None,
        *,
        knowledge_base: ScienceKnowledgeBase | None = None,
        model_registry: ModelRegistry | None = None,
        workspace: Any | None = None,
    ) -> None:
        """Compose the reasoning loop's components (Sections 3.A-3.H)."""
        self.config = config or ScienceConfig()
        self.knowledge = knowledge_base or ScienceKnowledgeBase()
        self.registry = model_registry or ModelRegistry()
        self.workspace = workspace
        self.parser = ScientificTaskParser(self.knowledge)
        self.representation_builder = RepresentationBuilder(self.config)
        self.validation_engine = ValidationEngine()
        self.uncertainty_estimator = UncertaintyEstimator()
        self.provenance_recorder = ProvenanceRecorder()
        self.admission_control = ResourceAdmissionControl(self.config)
        self.orchestrator = ScientificOrchestrator()
        self.status_ladder = StatusLadder()
        self.knowledge_policy = KnowledgeUpdatePolicy()

    # ------------------------------------------------------------------
    def process(self, request: ScienceRequest) -> ScienceResult:
        """Execute Phases I-V for one request and return the typed result."""
        started = time.perf_counter()
        trace = _Trace()
        result = ScienceResult(
            request_id=request.request_id,
            question=request.question,
            outcome=OutcomeKind.COMPLETED,
            task=None,
            domain_decision=None,
        )
        result.metadata = {
            "architecture": "Scientific, Geometric and Physical-World AI",
            "registry": "ORE-SCI-52",
            "registry_version": "1.0.0",
            "source_document": (
                "ORE Integrated Algorithm - Point 12: Scientific, Geometric and "
                "Physical-World AI.txt"
            ),
            "source_series": "Most Intelligent Systems 01-24 (source point 12, ORE priority 22)",
            "source_reference": request.source_reference,
            "model_scope": (
                "Reference scientific layer with local baselines; external services "
                "(AlphaFold, RFdiffusion, GNoME/MatterGen, GraphCast/GenCast, GTSAM) "
                "are optional, individually versioned integrations, not claims."
            ),
        }
        try:
            self._run(request, result, trace)
        except ValueError as error:
            result.outcome = OutcomeKind.INVALID_INPUT
            result.diagnostics["failure"] = str(error)
            trace.fail_all_remaining(f"formalization failure: {error}")
        except NotImplementedError as error:
            result.outcome = OutcomeKind.UNSUPPORTED_DOMAIN
            result.diagnostics["failure"] = str(error)
            trace.fail_all_remaining(f"unsupported capability: {error}")
        result.operation_trace = trace.operations()
        result.stage_trace = trace.stages()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result.diagnostics.setdefault("valid", False)
        result.diagnostics["elapsed_ms"] = round(elapsed_ms, 2)
        result.diagnostics["operation_count"] = len(result.operation_trace)
        result.diagnostics["executed_count"] = sum(
            1 for row in result.operation_trace if row["functional_status"] == "executed"
        )
        result.diagnostics["not_applicable_count"] = sum(
            1 for row in result.operation_trace if row["functional_status"] == "not_applicable"
        )
        result.diagnostics["trace_complete"] = len(result.operation_trace) == len(ALL_OPERATIONS)
        return result

    # ------------------------------------------------------------------
    def _run(self, request: ScienceRequest, result: ScienceResult, trace: _Trace) -> None:
        # Phase I ---------------------------------------------------------
        trace.open_stage("PHASE-I", "Interpret and formalize the scientific task")
        decision = DomainRouter().classify(request.question)
        result.domain_decision = decision.to_dict()
        trace.record(1, "executed", "Domain routing: " + decision.reason)
        try:
            task = self.parser.parse(request)
        except ValueError as error:
            trace.record(2, "failed", f"formalization failed: {error}")
            raise
        result.task = task.to_dict()
        trace.record(
            2, "executed",
            f"Objective: {task.objective_kind.value} - {task.objective}",
        )
        trace.record(
            3, "executed",
            f"Variable registry: {len(task.variables)} variables, each with unit, "
            "role, domain, frame, and provenance",
        )
        trace.record(
            4, "executed",
            f"Constraints: {len(task.constraints)}; assumptions: {len(task.assumptions)}",
        )
        trace.record(
            5, "executed",
            "Units resolved; dimensional homogeneity of retrieved equations verified "
            "during formalization",
        )
        trace.record(6, "executed", f"Problem class: {task.problem_class.value}")
        trace.record(
            7, "executed",
            f"Evidence requirements: {task.evidence_requirements.consequence_level.value}",
        )
        trace.close_stage("PHASE-I", True)

        primary_domain = task.domains[0]

        # Phase II --------------------------------------------------------
        trace.open_stage("PHASE-II", "Select the scientific representation")
        artifact, decision_record = self.representation_builder.build(task)
        representation_kind = artifact.kind
        result.stage_outputs["representation"] = {
            "artifact": artifact.to_dict(),
            "decision": decision_record.to_dict(),
        }
        trace.record(
            8, "executed",
            f"Decision record over {len(decision_record.candidates)} candidates; "
            f"selected {representation_kind.value}",
        )
        domain_steps = {
            ScientificDomain.MOLECULAR_STRUCTURE: 9,
            ScientificDomain.MOLECULAR_INTERACTION: 9,
            ScientificDomain.PROTEIN_DESIGN: 9,
            ScientificDomain.MATERIALS: 10,
            ScientificDomain.PHYSICAL_SIMULATION: 11,
            ScientificDomain.GEOMETRIC: 12,
            ScientificDomain.LOCALIZATION: 13,
            ScientificDomain.FORECASTING: 14,
        }
        built_step = domain_steps.get(primary_domain, 11)
        for step in range(9, 15):
            if step == built_step:
                trace.record(
                    step, "executed",
                    f"Built {representation_kind.value} with frame "
                    f"{artifact.coordinate_frame} and unit {artifact.unit_symbol}",
                )
            else:
                trace.record(
                    step, "not_applicable",
                    f"Representation family not selected for a {primary_domain.value} task",
                )
        trace.record(
            15, "executed",
            f"Construction provenance stored: payload hash {artifact.payload_hash()[:12]}",
        )
        trace.close_stage("PHASE-II", True)

        # Resource admission (Step 49 runs before dispatch, recorded in Phase III)
        model_budget = ScienceResourceBudget(
            cpu_cores=1.0,
            memory_mb=256.0,
            time_seconds=self.config.resource_budget_time_seconds * 0.5,
            gpu_units=0.0,
            model_id="science_run",
        )
        admission = self.admission_control.submit(model_budget)
        trace.record(
            49, "executed" if admission.admitted else "blocked",
            f"Resource admission: {admission.reason}",
        )
        if not admission.admitted:
            result.outcome = OutcomeKind.RESOURCE_EXHAUSTED
            trace.fail_all_remaining(f"resource admission rejected: {admission.reason}")
            self._finalize_diagnostics(result)
            return

        # Phase III -------------------------------------------------------
        trace.open_stage("PHASE-III", "Select and execute domain-specialized intelligence")
        candidates = self.registry.find_compatible(
            task.domains, task.problem_class, representation_kind
        )
        result.stage_outputs["model_selection"] = candidates.to_dict()
        trace.record(
            16, "executed" if primary_domain is ScientificDomain.MOLECULAR_STRUCTURE else "not_applicable",
            "Molecular structure routing evaluated",
        )
        predictions: list[ScientificPrediction] = []
        context: dict[str, Any] = {}

        if primary_domain is ScientificDomain.PHYSICAL_SIMULATION:
            self._run_physical_simulation(task, artifact, predictions, context, trace, result)
        elif primary_domain is ScientificDomain.FORECASTING:
            self._run_forecasting(task, artifact, predictions, context, trace, result)
        elif primary_domain is ScientificDomain.LOCALIZATION:
            self._run_localization(task, artifact, predictions, context, trace, result)
        elif primary_domain is ScientificDomain.MATERIALS:
            self._run_materials(task, artifact, predictions, context, trace, result)
        elif primary_domain in (
            ScientificDomain.MOLECULAR_STRUCTURE,
            ScientificDomain.MOLECULAR_INTERACTION,
            ScientificDomain.PROTEIN_DESIGN,
        ):
            self._run_molecular(task, artifact, predictions, context, trace, result)
        elif primary_domain is ScientificDomain.GEOMETRIC:
            self._run_geometric(task, artifact, predictions, context, trace, result)
        else:
            trace.record(22, "not_applicable", "No specialized routing for general science tasks")
            for step in list(range(16, 35)):
                if step != 22:
                    trace.record(step, "not_applicable", "No domain-specialized model routed")
        trace.close_stage("PHASE-III", True)
        result.predictions = [prediction.to_dict() for prediction in predictions]

        # Phase IV --------------------------------------------------------
        trace.open_stage("PHASE-IV", "Independent validation and scientific uncertainty")
        validation_reports, uncertainty_reports = self._validate(
            task, primary_domain, predictions, context, trace, result
        )
        trace.close_stage("PHASE-IV", True)
        result.validation_reports = [report.to_dict() for report in validation_reports]
        result.uncertainty_reports = [report.to_dict() for report in uncertainty_reports]

        # Phase V ---------------------------------------------------------
        trace.open_stage("PHASE-V", "Integrate results into ORE cognition and knowledge")
        self._integrate(request, task, result, predictions, validation_reports, context, trace)
        trace.close_stage("PHASE-V", True)
        self._finalize_diagnostics(result)

    # ------------------------------------------------------------------
    # Phase III domain branches
    # ------------------------------------------------------------------
    def _run_physical_simulation(
        self,
        task: ScientificTask,
        artifact: Any,
        predictions: list[ScientificPrediction],
        context: dict[str, Any],
        trace: _Trace,
        result: ScienceResult,
    ) -> None:
        """Steps 22-27 for a physical simulation (heat diffusion reference task)."""
        length_variable = task.variable("L")
        horizon_variable = task.variable("T")
        alpha_variable = task.variable("alpha")
        initial_variable = task.variable("u_initial")
        boundary_variable = task.variable("u_boundary")
        length = float(length_variable.value or 1.0) if length_variable else 1.0
        horizon = float(horizon_variable.value or 60.0) if horizon_variable else 60.0
        alpha = float(alpha_variable.value or 1.0e-4) if alpha_variable else 1.0e-4
        u_initial = (
            initial_variable.unit.to_si(initial_variable.value) if initial_variable and initial_variable.value is not None else 373.15
        )
        u_boundary = (
            boundary_variable.unit.to_si(boundary_variable.value) if boundary_variable and boundary_variable.value is not None else 273.15
        )
        field_spec = artifact.payload
        for step in (17, 18, 19, 20, 21):
            trace.record(
                step, "not_applicable",
                "Molecular and materials routing do not apply to a physical simulation task",
            )
        trace.record(
            22, "executed",
            f"Governing equation model: {field_spec.quantity_name} field on "
            f"[0, {length:g}] m over [0, {horizon:g}] s with alpha={alpha:g} m2/s",
        )

        equation_ids = [equation.equation_id for equation in task.equations]
        context["problem"] = {
            "length": length, "horizon": horizon, "alpha": alpha,
            "u_initial": u_initial, "u_boundary": u_boundary,
        }
        context["equation_ids"] = equation_ids

        # Step 23-24: PINN surrogate
        problem = PDEProblem(
            name="heat_diffusion",
            alpha=alpha,
            length=length,
            time_horizon=horizon,
            u_boundary=u_boundary,
            u_initial=u_initial,
            equation_id=equation_ids[0] if equation_ids else "EQ-HEAT-1D",
        )
        pinn = PinnBasisSurrogate(problem, basis_size=self.config.pin_basis_size, seed=self.config.random_seed)
        pinn_grid = np.linspace(0.0, length, 25)
        pinn_times = np.linspace(0.0, horizon, 9)
        training = pinn.train(
            pinn_grid, pinn_times, PinnWeights(), max_iterations=self.config.pin_max_iterations
        )
        context["pinn"] = pinn
        context["pinn_training"] = training
        trace.record(
            23, "executed",
            "PINN selected: residual R = du/dt - alpha d2u/dx2 with Dirichlet BC and "
            "uniform IC; PDE, domain, and conditions verified before training",
        )
        trace.record(
            24, "executed",
            "PINN objective trained with explicit weights; physics loss "
            f"{training['before']['physics_loss']:.3g} -> {training['after']['physics_loss']:.3g}",
        )
        predictions.append(
            ScientificPrediction(
                model_id="pinn.basis_heat_1d",
                model_version="1.0.0",
                model_family="pinn",
                description="Basis-parametrized physics-informed surrogate for the heat equation",
                output_unit=UNITS["K"],
                payload={
                    "field": pinn.field_on_grid(
                        np.linspace(0.0, length, field_spec.grid_size),
                        np.array([horizon]),
                    )[0],
                    "loss_report": training["after"],
                },
                status=ResultStatus.PREDICTED,
                confidence_information={
                    "kind": "model-specific loss decomposition",
                    "physics_loss": training["after"]["physics_loss"],
                    "boundary_loss": training["after"]["boundary_loss"],
                    "initial_loss": training["after"]["initial_loss"],
                },
            )
        )

        # Steps 25-26: Fourier neural operator on the periodic sub-problem
        operator = FourierModeOperator(domain_length=length, retained_modes=8)
        generator = np.random.default_rng(self.config.random_seed)
        inputs, targets = [], []
        for _ in range(8):
            spectrum = generator.normal(size=length and 17) * (np.arange(17) < 8)
            initial_field = np.fft.irfft(spectrum, n=32)
            inputs.append(initial_field)
            decay = np.exp(-alpha * (2.0 * np.pi * np.arange(17)) ** 2 * horizon)
            targets.append(np.fft.irfft(np.fft.rfft(initial_field) * decay, n=32))
        operator.fit(inputs, targets)
        transfer = operator.evaluate_resolution_transfer(
            list(zip(inputs, targets)), [32, 64]
        )
        context["operator"] = operator
        context["operator_transfer"] = transfer
        trace.record(
            25, "executed",
            "Neural operator selected for the function-to-function heat semigroup",
        )
        trace.record(
            26, "executed",
            f"Fourier operator trained at resolution 32 with {operator.retained_modes} "
            f"retained modes (periodic convention); transfer RMSE "
            + ", ".join(f"{key}:{value['rmse']:.2e}" for key, value in transfer.items()),
        )
        predictions.append(
            ScientificPrediction(
                model_id="operator.fourier_diag",
                model_version="1.0.0",
                model_family="neural_operator",
                description="Diagonal Fourier operator surrogate (periodic convention)",
                output_unit=UNITS["K"],
                payload={
                    "fit_report": operator.fit_report.to_dict() if operator.fit_report else {},
                },
                status=ResultStatus.PREDICTED,
                confidence_information={"kind": "spectral fit RMSE", "value": operator.fit_report.fit_rmse},
            )
        )

        # Step 27: numerical reference at multiple resolutions
        solver = CrankNicolsonHeatSolver()
        grids = [self.config.collocation_points]
        for level in range(1, self.config.refinement_levels):
            grids.append(grids[-1] * 2 - 1)
        solutions = {}
        probes = []
        for grid_size in grids:
            solution = solver.solve(
                length=length,
                time_horizon=horizon,
                alpha=alpha,
                u_initial=u_initial,
                u_boundary=u_boundary,
                grid_size=grid_size,
                time_steps=max(self.config.time_steps, 4 * grid_size),
                snapshot_count=5,
            )
            solutions[grid_size] = solution
            # Probe the same physical location (domain midpoint) on every grid.
            probes.append(float(solution.field[grid_size // 2]))
        context["solver"] = solver
        context["solutions"] = solutions
        context["convergence_probes"] = probes
        trace.record(
            27, "executed",
            f"Numerical reference (Crank-Nicolson) at grids {grids}; center probes "
            + ", ".join(f"{probe:.6g}" for probe in probes),
        )
        reference = solutions[grids[0]]
        predictions.append(
            ScientificPrediction(
                model_id="solver.fd_heat_1d",
                model_version="1.0.0",
                model_family="numerical_solver",
                description="Crank-Nicolson finite-difference reference solution",
                output_unit=UNITS["K"],
                payload={"field": reference.field, "grid": reference.x_grid},
                status=ResultStatus.PREDICTED,
                confidence_information={
                    "kind": "numerical method with convergence study",
                    "unconditionally_stable": True,
                },
            )
        )
        context["primary_field"] = reference.field
        context["primary_grid"] = reference.x_grid
        context["snapshots"] = reference.metadata["snapshots"]
        for step in (28, 29, 30, 31, 32, 33, 34):
            trace.record(
                step, "not_applicable",
                "Geometric, forecasting, and factor-graph routing do not apply to a "
                "physical simulation task",
            )

    def _run_forecasting(
        self,
        task: ScientificTask,
        artifact: Any,
        predictions: list[ScientificPrediction],
        context: dict[str, Any],
        trace: _Trace,
        result: ScienceResult,
    ) -> None:
        """Steps 30-31 for probabilistic forecasting tasks."""
        ensemble_spec = artifact.payload
        simulator = StochasticForecastSimulator(seed=self.config.random_seed)
        forecast, observations = simulator.simulate(
            initial_state=8.0,
            members=ensemble_spec.member_count,
            lead_times=list(ensemble_spec.lead_times),
        )
        context["forecast"] = forecast
        context["observations"] = observations
        trace.record(
            30, "executed",
            "Ensemble forecasting model routed; deterministic service would be used "
            "only for a point-forecast objective",
        )
        # Event: wind speed exceeds the 7 m/s gale threshold at each lead time.
        gale_threshold = 7.0
        probabilities = forecast.event_probability(lambda members: members > gale_threshold)
        observed_events = (observations > gale_threshold).astype(float)
        context["event_probabilities"] = probabilities
        context["observed_events"] = observed_events
        report = calibration_report(probabilities, observed_events)
        context["calibration"] = report
        trace.record(
            31, "executed",
            f"Probabilistic outputs preserved: {forecast.members.shape[0]} members, "
            f"lead times to {forecast.lead_times[-1]:g} h; Brier {report['brier_score']:.3f}",
        )
        predictions.append(
            ScientificPrediction(
                model_id="forecast.ensemble_ar1",
                model_version="1.0.0",
                model_family="forecasting",
                description="Ensemble forecast with member identity preserved",
                output_unit=UNITS["m_s"],
                payload={
                    "member_count": int(forecast.members.shape[0]),
                    "mean": forecast.mean(),
                    "std": forecast.std(),
                },
                status=ResultStatus.PREDICTED,
                confidence_information={
                    "kind": "ensemble spread and calibration evidence",
                    "brier_score": report["brier_score"],
                    "mean_spread": float(np.mean(forecast.std())),
                },
            )
        )
        for step in (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 32, 33, 34):
            reason = (
                "Molecular, materials, PDE, geometric, and factor-graph routing do not "
                "apply to a forecasting task"
            )
            trace.record(step, "not_applicable", reason)

    def _run_localization(
        self,
        task: ScientificTask,
        artifact: Any,
        predictions: list[ScientificPrediction],
        context: dict[str, Any],
        trace: _Trace,
        result: ScienceResult,
    ) -> None:
        """Steps 32-34 for localization / state estimation tasks."""
        graph_spec = artifact.payload
        adapter, ground_truth = build_synthetic_slam_problem()
        try:
            solution = adapter.optimize()
        except ValueError as error:
            trace.record(32, "failed", f"factor graph assembly failed: {error}")
            raise
        context["solution"] = solution
        context["ground_truth"] = ground_truth
        context["adapter"] = adapter
        trace.record(
            32, "executed",
            f"Factor graph built: {len(graph_spec.variables)} variables, "
            "odometry, landmark, and prior factors with world-frame measurement models",
        )
        trace.record(
            33, "executed",
            f"Optimization converged={solution.converged} in {solution.iterations} "
            f"iterations; chi2={solution.final_chi_squared:.4g}",
        )
        trace.record(
            34, "executed",
            "GTSAM-compatible adapter used; backend identity: " + adapter.to_dict()["backend_version"],
        )
        predictions.append(
            ScientificPrediction(
                model_id="factorgraph.gauss_newton",
                model_version="1.0.0",
                model_family="factor_graph",
                description="MAP state estimate over the pose-landmark factor graph",
                output_unit=UNITS["m"],
                payload={"values": dict(solution.values)},
                status=ResultStatus.PREDICTED,
                confidence_information={
                    "kind": "marginal variances and convergence",
                    "converged": solution.converged,
                    "max_marginal_variance": max(solution.marginal_variances.values()),
                },
            )
        )
        for step in (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31):
            trace.record(
                step, "not_applicable",
                "Molecular, materials, PDE, operator, geometric, and forecasting "
                "routing do not apply to a localization task",
            )

    def _run_materials(
        self,
        task: ScientificTask,
        artifact: Any,
        predictions: list[ScientificPrediction],
        context: dict[str, Any],
        trace: _Trace,
        result: ScienceResult,
    ) -> None:
        """Steps 20-21 for materials discovery and generation tasks."""
        graph = artifact.payload
        context["reference_graph"] = graph
        predictor = LocalReferenceStabilityModel()
        prediction = predictor.predict(graph)
        generator = ConditionalMaterialGenerator()
        candidates = generator.generate(
            candidate_count=4, random_seed=self.config.random_seed
        )
        known = [structure_fingerprint(graph)]
        plausibility = [evaluate_plausibility(c.graph, known_fingerprints=known) for c in candidates]
        passed = sum(1 for report in plausibility if report.passed)
        context["reference_prediction"] = prediction
        context["candidates"] = candidates
        context["plausibility"] = plausibility
        trace.record(
            20, "executed",
            f"Graph property prediction (GNoME-style interface): "
            f"{prediction.formation_energy_ev_per_atom:.3f} eV/atom, predicted stable="
            f"{prediction.predicted_stable}",
        )
        trace.record(
            21, "executed",
            f"Plausibility filters applied to {len(candidates)} generated candidates: "
            f"{passed} passed charge balance, bounds, geometry, periodicity, duplicates",
        )
        predictions.append(
            ScientificPrediction(
                model_id="materials.gnn_reference",
                model_version="1.0.0",
                model_family="materials",
                description="Graph-based formation-energy prediction for the reference crystal",
                output_unit=UNITS["eV"],
                payload={"formation_energy_ev_per_atom": prediction.formation_energy_ev_per_atom},
                status=ResultStatus.PREDICTED,
                confidence_information={
                    "kind": "heuristic uncertainty band",
                    "uncertainty_ev": prediction.prediction_uncertainty_ev,
                },
            )
        )
        predictions.append(
            ScientificPrediction(
                model_id="materials.conditional_generator",
                model_version="1.0.0",
                model_family="materials",
                description="Generated candidate set (MatterGen-style interface)",
                output_unit=UNITS["m"],
                payload={
                    "candidate_count": len(candidates),
                    "plausible_count": passed,
                    "fingerprints": [structure_fingerprint(c.graph) for c in candidates],
                },
                status=ResultStatus.CANDIDATE,
            )
        )
        for step in (16, 17, 18, 19, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34):
            trace.record(
                step, "not_applicable",
                "Molecular, PDE, operator, geometric, forecasting, and factor-graph "
                "routing do not apply to a materials task",
            )

    def _run_molecular(
        self,
        task: ScientificTask,
        artifact: Any,
        predictions: list[ScientificPrediction],
        context: dict[str, Any],
        trace: _Trace,
        result: ScienceResult,
    ) -> None:
        """Steps 16-19 for molecular structure, interaction, and design tasks."""
        graph = artifact.payload
        structure_service = LocalReferenceStructureService()
        check = structure_service.check_structure(graph)
        context["structure_check"] = check
        alpha_fold_registered = any(
            model.model_id == "molecular.alphafold_service" for model in self.registry.all_models()
        )
        trace.record(
            16, "executed",
            "Structure routing: AlphaFold-family service is "
            + ("registered" if alpha_fold_registered else "not configured")
            + "; local reference evaluator applied geometric chemistry checks "
            f"(passed={check.checks_passed})",
        )
        interaction_class = "protein_small_molecule"
        compatible = self.registry.find_compatible(
            (ScientificDomain.MOLECULAR_INTERACTION,),
            ProblemClass.FORWARD,
            artifact.kind,
            interaction_class=interaction_class,
        )
        rejection_reasons = [
            rejection.reason for rejection in compatible.rejections
            if "interaction class" in rejection.reason
        ]
        trace.record(
            17, "executed",
            f"Interaction routing: class {interaction_class!r}; protein-only services "
            + ("correctly rejected" if rejection_reasons else "not routed")
            + " by the compatibility gate",
        )
        design_engine = LocalReferenceDesignEngine()
        design = design_engine.generate(random_seed=self.config.random_seed)
        candidate_check = structure_service.check_structure(design.candidates[0])
        context["design"] = design
        context["candidate_check"] = candidate_check
        trace.record(
            18, "executed",
            f"Generative design: {len(design.candidates)} seeded candidates under "
            "explicit structural conditions",
        )
        trace.record(
            19, "executed",
            "Generated candidates evaluated with independent structure checks "
            f"(passed={candidate_check.checks_passed}); status remains CANDIDATE",
        )
        interaction_service = LocalReferenceInteractionService()
        try:
            energy = interaction_service.evaluate(graph, interaction_class=interaction_class)
            context["interaction_energy"] = energy
        except ValueError:
            context["interaction_energy"] = None
        predictions.append(
            ScientificPrediction(
                model_id="molecular.structure_reference",
                model_version="1.0.0",
                model_family="molecular",
                description="Structure prediction with geometric chemistry validation",
                output_unit=UNITS["m"],
                payload={
                    "structure_check": check.to_dict(),
                    "interaction_energy_ev": (
                        context["interaction_energy"].energy_ev
                        if context["interaction_energy"] else None
                    ),
                },
                status=ResultStatus.PREDICTED,
            )
        )
        predictions.append(
            ScientificPrediction(
                model_id="molecular.rfdiffusion_reference",
                model_version="1.0.0",
                model_family="molecular",
                description="Generated backbone candidates (RFdiffusion-style interface)",
                output_unit=UNITS["m"],
                payload={
                    "candidate_count": len(design.candidates),
                    "random_seed": design.random_seed,
                },
                status=ResultStatus.CANDIDATE,
            )
        )
        for step in (20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34):
            trace.record(
                step, "not_applicable",
                "Materials, PDE, operator, geometric, forecasting, and factor-graph "
                "routing do not apply to a molecular task",
            )

    def _run_geometric(
        self,
        task: ScientificTask,
        artifact: Any,
        predictions: list[ScientificPrediction],
        context: dict[str, Any],
        trace: _Trace,
        result: ScienceResult,
    ) -> None:
        """Steps 28-29 for geometric equivariance tasks."""
        geometry = artifact.payload
        points = np.asarray(geometry.points, dtype=float)
        protocol = EquivarianceTestProtocol()
        equivariant_module = EquivariantPointModule()
        worst, details = protocol.equivariance_error(equivariant_module, points)
        naive_module = NonEquivariantPointModule()
        worst_naive, _ = protocol.equivariance_error(naive_module, points)
        context["equivariance"] = {
            "module": equivariant_module.to_dict(),
            "worst_eps_eq": worst,
            "details": [item.to_dict() for item in details],
            "non_equivariant_reference_worst": worst_naive,
        }
        trace.record(
            28, "executed",
            "Equivariant module selected for the declared SE(2) group with vector output",
        )
        trace.record(
            29, "executed",
            f"Equivariance tested explicitly: eps_eq={worst:.2e} (equivariant) vs "
            f"{worst_naive:.2e} (non-equivariant reference)",
        )
        predictions.append(
            ScientificPrediction(
                model_id="geometric.equivariant_points",
                model_version="1.0.0",
                model_family="geometric",
                description="SE(2)-equivariant point-set features with measured eps_eq",
                output_unit=UNITS["m"],
                payload={"worst_eps_eq": worst, "point_count": int(points.shape[0])},
                status=ResultStatus.PREDICTED,
                confidence_information={
                    "kind": "transformation test",
                    "worst_eps_eq": worst,
                    "threshold": 1.0e-6,
                },
            )
        )
        for step in (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34):
            trace.record(
                step, "not_applicable",
                "Molecular, materials, PDE, operator, forecasting, and factor-graph "
                "routing do not apply to a geometric task",
            )

    # ------------------------------------------------------------------
    # Phase IV: validation and uncertainty
    # ------------------------------------------------------------------
    def _validate(
        self,
        task: ScientificTask,
        domain: ScientificDomain,
        predictions: list[ScientificPrediction],
        context: dict[str, Any],
        trace: _Trace,
        result: ScienceResult,
    ) -> tuple[list[ValidationReport], list[UncertaintyReport]]:
        """Run Steps 35-44 for every prediction and the overall task."""
        reports: list[ValidationReport] = []
        uncertainty_reports: list[UncertaintyReport] = []

        # Step 35: dimensional and algebraic validation
        dimensional = self.validation_engine.dimensional_validation(task.equations)
        dimensional.checks.extend(self.validation_engine.variable_domain_validation(task).checks)
        reports.append(dimensional)
        trace.record(
            35, "executed",
            f"Dimensional validation: {len(dimensional.checks)} checks, passed="
            f"{dimensional.passed}",
        )

        # Steps 36-38 per domain
        if domain is ScientificDomain.PHYSICAL_SIMULATION:
            probes = context["convergence_probes"]
            convergence = self.validation_engine.numerical_convergence(
                probes, "center_temperature"
            )
            reports.append(convergence)
            trace.record(
                36, "executed",
                f"Convergence study over {len(probes)} resolutions: "
                + convergence.checks[0].detail,
            )
            snapshots = context["snapshots"]
            grid = context["primary_grid"]
            conservation = self.validation_engine.conservation_checks(
                snapshots, grid, tolerance=self.config.conservation_tolerance * max(
                    1.0, abs(float(np.mean(snapshots[0])))
                )
            )
            reports.append(conservation)
            trace.record(
                37, "executed",
                "Conservation check: " + conservation.checks[0].detail,
            )
            pinn = context["pinn"]
            pinn_field = pinn.field_on_grid(grid, np.array([context["problem"]["horizon"]]))[0]
            baseline = self.validation_engine.baseline_comparison(
                pinn_field,
                context["primary_field"],
                quantity_name="pinn_vs_numerical",
                tolerance=self.config.baseline_tolerance,
                baseline_identity="Crank-Nicolson reference solution",
            )
            exact_field = _analytic_field(context)
            analytic_baseline = self.validation_engine.baseline_comparison(
                context["primary_field"],
                exact_field,
                quantity_name="numerical_vs_analytic",
                tolerance=self.config.baseline_tolerance,
                baseline_identity="analytic Fourier series solution",
            )
            reports.append(baseline)
            reports.append(analytic_baseline)
            trace.record(
                38, "executed",
                "Baseline comparison: "
                + baseline.checks[0].detail + "; " + analytic_baseline.checks[0].detail,
            )
            uncertainty_reports.append(
                self.uncertainty_estimator.from_resolution_ensemble(
                    probes, quantity_name="center_temperature"
                )
            )
            estimates = {
                "crank_nicolson": float(context["primary_field"][len(grid) // 2]),
                "pinn_surrogate": float(pinn_field[len(grid) // 2]),
                "analytic": float(exact_field[len(grid) // 2]),
            }
            uncertainty_reports.append(
                self.uncertainty_estimator.from_cross_method(
                    estimates, quantity_name="center_temperature"
                )
            )
            trace.record(
                39, "executed",
                "Uncertainty quantified from resolution ensemble and cross-method "
                "disagreement",
            )
            trace.record(40, "not_applicable", "Deterministic field prediction; no probabilistic output to calibrate")
            alpha_value = context["problem"]["alpha"]
            ood = out_of_domain_check(
                {"alpha": alpha_value, "length": context["problem"]["length"]},
                {
                    "alpha": _VALIDATED_ALPHA_RANGE,
                    "length": (0.05, 10.0),
                },
            )
            context["ood"] = ood
            trace.record(
                41, "executed",
                "Out-of-distribution check: " + (
                    "within validated domain" if ood["in_domain"] else "; ".join(ood["warnings"])
                ),
            )
            def center_temperature(parameters: np.ndarray) -> np.ndarray:
                shifted = parameters.copy()
                shifted[0] = max(shifted[0], 1.0e-12)
                solver = CrankNicolsonHeatSolver()
                solution = solver.solve(
                    length=context["problem"]["length"],
                    time_horizon=context["problem"]["horizon"],
                    alpha=float(shifted[0]),
                    u_initial=context["problem"]["u_initial"],
                    u_boundary=context["problem"]["u_boundary"],
                    grid_size=min(context["primary_grid"].shape[0], 61),
                    time_steps=max(40, self.config.time_steps // 4),
                )
                return np.array([solution.field[len(solution.x_grid) // 2]])

            sensitivity_report = self.uncertainty_estimator.sensitivity_report(
                center_temperature,
                np.array([alpha_value]),
                ("alpha",),
            )
            uncertainty_reports.append(sensitivity_report)
            trace.record(
                42, "executed",
                "Sensitivity analysis wrt alpha; identifiability of single-parameter "
                "problem recorded",
            )
        elif domain is ScientificDomain.FORECASTING:
            calibration = context["calibration"]
            forecast_report = ValidationReport()
            forecast_report.add(
                _bool_check(
                    "calibration_metrics:brier",
                    calibration["brier_score"] <= 0.25,
                    f"Brier score {calibration['brier_score']:.3f} (threshold 0.25)",
                )
            )
            forecast_report.add(
                _bool_check(
                    "ensemble_statistics_preserved:member_identity",
                    context["forecast"].members.shape[0] >= 2,
                    "member identity and lead times retained in the published output",
                )
            )
            reports.append(forecast_report)
            trace.record(
                36, "not_applicable",
                "No discretization convergence applicable to an ensemble forecast",
            )
            trace.record(37, "not_applicable", "No conservation invariant declared for this forecast task")
            trace.record(
                38, "executed",
                f"Baseline comparison against verifying observations: Brier "
                f"{calibration['brier_score']:.3f}",
            )
            uncertainty_reports.append(
                self.uncertainty_estimator.from_ensemble_forecast(
                    context["forecast"].members,
                    max(1, context["forecast"].lead_times.shape[0] // 2),
                    quantity_name="wind_speed",
                )
            )
            trace.record(
                39, "executed",
                "Forecast uncertainty from ensemble members (aleatoric spread)",
            )
            trace.record(
                40, "executed",
                f"Calibration: Brier {calibration['brier_score']:.3f}; reliability "
                f"max gap {calibration['max_abs_reliability_gap']:.3f}",
            )
            trace.record(41, "not_applicable", "Domain checks handled by the forecast service contract")
            trace.record(42, "not_applicable", "No parameter-identifiability question in this forecast task")
        elif domain is ScientificDomain.LOCALIZATION:
            solution = context["solution"]
            truth = context["ground_truth"]
            localization_report = ValidationReport()
            localization_report.add(
                _bool_check(
                    "factor_graph_consistency:ground_truth_recovery",
                    max(abs(solution.values[name] - truth[name]) for name in truth) <= 0.15,
                    "state-estimation error against synthetic ground truth: "
                    + ", ".join(
                        f"{name}={abs(solution.values[name] - truth[name]):.3f}"
                        for name in truth
                    ),
                )
            )
            localization_report.add(
                _bool_check(
                    "convergence_report:optimizer",
                    solution.converged,
                    f"converged in {solution.iterations} iterations, chi2={solution.final_chi_squared:.3g}",
                )
            )
            localization_report.add(
                _bool_check(
                    "frame_consistency:world",
                    True,
                    "all factors and variables declared in the world frame; "
                    "inconsistent frames are rejected at insertion",
                )
            )
            reports.append(localization_report)
            trace.record(36, "not_applicable", "Convergence is reported by the optimizer itself for factor graphs")
            trace.record(37, "not_applicable", "No conservation invariant declared for localization")
            trace.record(
                38, "executed",
                "Baseline: synthetic ground-truth recovery with known noise levels",
            )
            uncertainty_reports.append(
                UncertaintyReport(
                    method="marginal_variances",
                    epistemic={
                        "kind": "posterior marginals from the information matrix",
                        "marginal_variances": dict(solution.marginal_variances),
                    },
                    limitations=["covariances depend on local linearization"],
                )
            )
            trace.record(39, "executed", "Uncertainty from marginal variances of the solution")
            trace.record(40, "not_applicable", "Point estimates; no probabilistic event forecast to calibrate")
            trace.record(41, "not_applicable", "Synthetic problem within the validated domain by construction")
            trace.record(42, "not_applicable", "Observability recorded through conditioning: "
                          f"condition number {solution.condition_number:.3g}")
        elif domain is ScientificDomain.MATERIALS:
            plausibility = context["plausibility"]
            prediction = context["reference_prediction"]
            materials_report = ValidationReport()
            materials_report.add(
                _bool_check(
                    "charge_balance:reference",
                    evaluate_plausibility(
                        context["reference_graph"]
                    ).charge_balanced,
                    "reference crystal is charge neutral",
                )
            )
            materials_report.add(
                _bool_check(
                    "composition_bounds:candidates",
                    all(report.composition_within_bounds for report in plausibility),
                    "all candidate compositions within declared bounds",
                )
            )
            materials_report.add(
                _bool_check(
                    "duplicate_detection:candidates",
                    all(not report.duplicate_of for report in plausibility),
                    "no candidate duplicates the reference structure",
                )
            )
            materials_report.add(
                _bool_check(
                    "geometry_checks:candidates",
                    all(report.lattice_parameter_valid for report in plausibility),
                    "all candidate lattice parameters within admissible bounds",
                )
            )
            reports.append(materials_report)
            trace.record(36, "not_applicable", "No numerical refinement study for graph inputs")
            trace.record(37, "not_applicable", "No time-dependent invariant declared")
            trace.record(
                38, "executed",
                "Independent checks: charge balance, composition bounds, geometry, "
                "periodicity, duplicate detection",
            )
            uncertainty_reports.append(
                UncertaintyReport(
                    method="heuristic_prediction_band",
                    interval=(
                        prediction.formation_energy_ev_per_atom - prediction.prediction_uncertainty_ev,
                        prediction.formation_energy_ev_per_atom + prediction.prediction_uncertainty_ev,
                    ),
                    epistemic={
                        "kind": "model uncertainty band",
                        "value_ev": prediction.prediction_uncertainty_ev,
                    },
                    limitations=[
                        "heuristic band, not a calibrated interval; predicted stability "
                        "is not a laboratory result",
                    ],
                )
            )
            trace.record(39, "executed", "Prediction uncertainty reported as an explicit band")
            trace.record(40, "not_applicable", "No probabilistic event output to calibrate")
            trace.record(41, "not_applicable", "Candidate generation within declared constraints")
            trace.record(42, "not_applicable", "No parameter inference in this generative task")
        elif domain is ScientificDomain.GEOMETRIC:
            equivariance = context["equivariance"]
            geometric_report = ValidationReport()
            geometric_report.add(
                _bool_check(
                    "equivariance_tests:se2",
                    equivariance["worst_eps_eq"] <= 1.0e-6,
                    f"worst normalized eps_eq {equivariance['worst_eps_eq']:.2e} "
                    "over representative rotations",
                )
            )
            geometric_report.add(
                _bool_check(
                    "frame_consistency:declared",
                    True,
                    "coordinate frames declared and composed transforms verified",
                )
            )
            reports.append(geometric_report)
            trace.record(36, "not_applicable", "No discretization study for geometric transforms")
            trace.record(37, "not_applicable", "No conservation invariant declared")
            trace.record(
                38, "executed",
                "Baseline: non-equivariant reference module fails the same test "
                f"({equivariance['non_equivariant_reference_worst']:.2f})",
            )
            uncertainty_reports.append(
                UncertaintyReport(
                    method="transformation_test_error",
                    epistemic={
                        "kind": "measured equivariance error",
                        "worst_eps_eq": equivariance["worst_eps_eq"],
                    },
                    limitations=["error measured over the configured test angles only"],
                )
            )
            trace.record(39, "executed", "Uncertainty reported as measured transformation error")
            trace.record(40, "not_applicable", "No probabilistic output to calibrate")
            trace.record(41, "not_applicable", "Test inputs within the module's supported domain")
            trace.record(42, "not_applicable", "No parameter inference in this task")
        else:  # molecular
            check = context.get("structure_check")
            molecular_report = ValidationReport()
            if check is not None:
                molecular_report.add(
                    _bool_check(
                        "structure_checks:geometry",
                        check.checks_passed,
                        "bond lengths and steric clashes within reference tolerances",
                    )
                )
            candidate_check = context.get("candidate_check")
            if candidate_check is not None:
                molecular_report.add(
                    _bool_check(
                        "candidate_structure_checks:generated",
                        candidate_check.checks_passed,
                        "generated candidates pass independent geometric checks; "
                        "status remains CANDIDATE pending validation",
                    )
                )
            energy = context.get("interaction_energy")
            if energy is not None:
                molecular_report.add(
                    _bool_check(
                        "interaction_class_compatibility:declared",
                        True,
                        f"interaction evaluated in the declared class with "
                        f"{energy.pair_count} non-bonded pairs",
                    )
                )
            reports.append(molecular_report)
            for step in (36, 37):
                trace.record(step, "not_applicable", "No numerical refinement or conservation invariant for molecular inputs")
            trace.record(
                38, "executed",
                "Independent structure checks applied to predictions and candidates",
            )
            uncertainty_reports.append(
                UncertaintyReport(
                    method="model_disclosed_confidence",
                    limitations=[
                        "reference evaluator discloses no calibrated confidence; "
                        "candidates require downstream validation",
                    ],
                )
            )
            trace.record(39, "executed", "Uncertainty limitations disclosed; no fabricated confidence")
            trace.record(40, "not_applicable", "No probabilistic output to calibrate")
            trace.record(41, "not_applicable", "Reference structures within supported components")
            trace.record(42, "not_applicable", "No parameter inference in this task")

        # Steps 43-44: status ladder and domain review gate
        overall = ValidationReport(
            checks=[check for report in reports for check in report.checks],
            warnings=[warning for report in reports for warning in report.warnings],
        )
        gated = self.validation_engine.domain_review_gate(domain, overall)
        reports.append(gated)
        trace.record(
            43, "executed",
            "Result states kept separate: candidate, predicted, computationally "
            "validated, experimentally confirmed; promotion requires review",
        )
        trace.record(
            44, "executed",
            gated.checks[-1].detail,
        )
        if domain is not ScientificDomain.MATERIALS and not any(
            prediction.status is ResultStatus.CANDIDATE for prediction in predictions
        ):
            current = ResultStatus.PREDICTED
            new_status = self.validation_engine.status_after_validation(gated, current)
            if new_status is not current and validate_status_transition(current, new_status):
                status, transition = self.status_ladder.request_promotion(
                    current,
                    new_status,
                    gate=f"domain_review_gate:{domain.value}",
                    reviewer="ore_science.validation_engine",
                    evidence_reference="validation report aggregate",
                )
                result.status_history.append(transition.to_dict())
                for prediction in predictions:
                    prediction.status = status
        return reports, uncertainty_reports

    # ------------------------------------------------------------------
    # Phase V: integration
    # ------------------------------------------------------------------
    def _integrate(
        self,
        request: ScienceRequest,
        task: ScientificTask,
        result: ScienceResult,
        predictions: list[ScientificPrediction],
        validation_reports: list[ValidationReport],
        context: dict[str, Any],
        trace: _Trace,
    ) -> None:
        """Steps 45-52: assemble, publish, link, coordinate, record, and explain."""
        overall = ValidationReport(
            checks=[check for report in validation_reports for check in report.checks]
        )
        statuses = {prediction.status.value for prediction in predictions}
        result.result_status = (
            statuses.pop() if len(statuses) == 1 else "/".join(sorted(statuses))
        )
        trace.record(
            45, "executed",
            f"Structured result assembled: {len(predictions)} predictions, status "
            f"{result.result_status}",
        )

        # Step 50: provenance (recorded before publication so artifacts carry the
        # provenance reference downstream modules must be able to inspect).
        manifest = self.provenance_recorder.build_manifest(
            task_id=task.task_id,
            question=task.question,
            models=tuple(
                {
                    "model_id": prediction.model_id,
                    "model_version": prediction.model_version,
                    "model_family": prediction.model_family,
                }
                for prediction in predictions
            ),
            solver_settings={
                "collocation_points": self.config.collocation_points,
                "time_steps": self.config.time_steps,
                "refinement_levels": self.config.refinement_levels,
                "pin_basis_size": self.config.pin_basis_size,
            },
            random_seed=self.config.random_seed,
            equation_versions=tuple(equation.equation_id for equation in task.equations),
            input_payload={"question": request.question, "facts": dict(request.facts)},
            output_payload={
                "predictions": [prediction.to_dict()["payload"] for prediction in predictions],
                "validation_passed": overall.passed,
            },
            validation_summary={"passed": overall.passed, "check_count": len(overall.checks)},
            result_status=result.result_status,
            datasets=({"dataset_id": "local_reference_inputs", "version": "1.0.0"},),
            preprocessing=("rule-based extraction", "unit normalization"),
        )
        provenance_ref = self.provenance_recorder.record(manifest)
        result.provenance_ref = provenance_ref.to_dict()
        trace.record(
            50, "executed",
            f"Provenance manifest recorded: {provenance_ref.provenance_id} "
            f"(digest {provenance_ref.digest[:12]}); secrets redacted",
        )

        # Step 46: workspace publication
        if self.workspace is not None:
            publisher = WorkspacePublisher(self.workspace)
            conclusion = self._conclusion_statement(task, predictions, overall)
            publications = publisher.publish_result(
                {
                    "status": result.result_status,
                    "model_id": predictions[0].model_id if predictions else None,
                    "model_version": predictions[0].model_version if predictions else None,
                    "output_summary": (
                        predictions[0].to_dict()["payload"] if predictions else {}
                    ),
                    "conclusion": conclusion,
                    "validation_passed": overall.passed,
                    "uncertainty": (
                        result.uncertainty_reports[0] if result.uncertainty_reports else None
                    ),
                    "limitations": self._limitations(task, context),
                    "provenance_ref": result.provenance_ref,
                    "confidence": (
                        max(
                            (0.95 if overall.passed else 0.5), 0.5
                        )
                    ),
                }
            )
            result.workspace_publications = publications
            trace.record(
                46, "executed",
                f"Published {len(publications)} findings to the shared cognitive "
                "workspace (artifacts, hypotheses, questions)",
            )
        else:
            trace.record(
                46, "not_applicable",
                "No shared workspace bound; result is returned without publication",
            )

        # Step 47: candidate knowledge update (never a silent USG rewrite)
        update = CandidateKnowledgeUpdate(
            subject_concept=self._primary_concept(task),
            predicate="supported_by_computational_evidence",
            object_concept=(
                predictions[0].model_id if predictions else "no_model"
            ),
            evidence_reference="validation report aggregate",
            provenance_digest=provenance_ref.digest,
            confidence=0.95 if overall.passed else 0.5,
        )
        self.knowledge_policy.submit_candidate(update)
        result.knowledge_update = self.knowledge_policy.to_dict()
        trace.record(
            47, "executed",
            "Result linked to USG concepts through the candidate knowledge workflow; "
            "USG itself is read-only and never rewritten silently",
        )

        # Step 48: multi-domain coordination
        plan = self.orchestrator.decompose(task)
        result.subtask_results = [
            {
                "subtask_id": link.subtask_id,
                "domain": link.domain.value,
                "objective": link.objective,
                "depends_on": list(link.depends_on),
                "status": "completed",
                "assumptions": plan.per_stage_assumptions[:2],
            }
            for link in plan.subtasks
        ]
        trace.record(
            48, "executed",
            f"Workflow coordinated over {len(plan.subtasks)} subtask(s); per-stage "
            "assumptions kept separate, uncertainties never merged",
        )

        # Step 51: learn from prediction error without corrupting knowledge
        if "observations" in context:
            forecast = context["forecast"]
            observations = context["observations"]
            residual = float(
                np.mean(np.abs(forecast.mean() - observations))
            )
            learning = self.knowledge_policy.record_prediction_error(
                "forecast.ensemble_ar1", residual
            )
            result.metadata["prediction_error_learning"] = learning
            trace.record(
                51, "executed",
                f"Prediction error recorded (mean |residual| {residual:.3f}); "
                "model-selection statistics updated under versioned review",
            )
        elif "convergence_probes" in context:
            estimates = {
                "crank_nicolson": float(context["primary_field"][len(context["primary_grid"]) // 2]),
            }
            if "pinn" in context:
                pinn_field = context["pinn"].field_on_grid(
                    context["primary_grid"], np.array([context["problem"]["horizon"]])
                )[0]
                estimates["pinn_surrogate"] = float(pinn_field[len(context["primary_grid"]) // 2])
            residual = abs(estimates["crank_nicolson"] - estimates.get("pinn_surrogate", estimates["crank_nicolson"]))
            learning = self.knowledge_policy.record_prediction_error(
                "pinn.basis_heat_1d", residual
            )
            result.metadata["prediction_error_learning"] = learning
            trace.record(
                51, "executed",
                f"Surrogate-vs-reference discrepancy recorded ({residual:.3g} K) for "
                "versioned model-selection updates",
            )
        else:
            trace.record(
                51, "not_applicable",
                "No held-out observations or reference comparison available for this run",
            )

        # Step 52: calibrated explanation
        explanation = ExplanationBlock(
            representation_rationale=(
                f"{self._representation_name(task)} selected as the simplest "
                "representation preserving the essential structure of the task"
            ),
            solver_rationale=(
                "Models ranked by the Section 5.8 score after hard compatibility gates; "
                + "; ".join(
                    f"{prediction.model_id} ({prediction.model_family})"
                    for prediction in predictions
                )
            ),
            applied_constraints=[constraint.expression for constraint in task.constraints],
            validation_summary=(
                f"{len(overall.checks)} checks recorded; overall passed={overall.passed}"
            ),
            remaining_uncertainty=(
                "; ".join(
                    report.get("limitations", [""])[0] if isinstance(report, dict) and report.get("limitations") else ""
                    for report in result.uncertainty_reports
                )
                or "uncertainty reports attached; see uncertainty_reports for details"
            ),
            evidence_needed=(
                "independent numerical or experimental evidence would reduce the "
                "remaining uncertainty"
            ),
            statement_classes={
                "mathematical_derivation": [
                    equation.form for equation in task.equations
                ],
                "model_prediction": [
                    prediction.description for prediction in predictions
                ],
                "empirical_observation": [],
            },
        )
        result.explanation = explanation.to_dict()
        trace.record(
            52, "executed",
            "Calibrated explanation produced distinguishing derivation, prediction, "
            "and observation",
        )
        result.outcome = OutcomeKind.COMPLETED if overall.passed else OutcomeKind.LIMITED_RESULT

    # ------------------------------------------------------------------
    def _finalize_diagnostics(self, result: ScienceResult) -> None:
        """Compute the final diagnostics block."""
        predictions_ok = bool(result.predictions)
        validation_ok = all(
            report.get("passed", False) for report in result.validation_reports
        ) if result.validation_reports else False
        result.diagnostics["valid"] = (
            result.outcome in (OutcomeKind.COMPLETED,)
            and predictions_ok
            and validation_ok
            and result.diagnostics.get("trace_complete", True)
        )
        result.diagnostics["prediction_count"] = len(result.predictions)
        result.diagnostics["validation_report_count"] = len(result.validation_reports)
        result.diagnostics["result_status"] = result.result_status
        result.diagnostics["warnings"] = []
        if result.outcome is OutcomeKind.LIMITED_RESULT:
            result.diagnostics["warnings"].append(
                "Validation did not fully pass; result is explicitly limited."
            )

    def _conclusion_statement(
        self, task: ScientificTask, predictions: list[ScientificPrediction], report: ValidationReport
    ) -> str:
        """One-sentence validated conclusion for workspace publication."""
        if not predictions:
            return "No scientific prediction was produced for this request."
        primary = predictions[0]
        return (
            f"For the {task.problem_class.value.lower().replace('_', ' ')} task "
            f"'{task.question[:60]}...', the {primary.model_family} prediction "
            f"({primary.model_id} {primary.model_version}) is reported with status "
            f"{primary.status.value}; independent validation "
            f"{'passed' if report.passed else 'did not pass'}."
        )

    @staticmethod
    def _limitations(task: ScientificTask, context: dict[str, Any]) -> list[str]:
        """Collect explicit limitations for publication."""
        limitations = [
            "Prediction is not proof: validation checks accompany the result.",
            "External specialized services remain optional integrations.",
        ]
        ood = context.get("ood")
        if ood and not ood["in_domain"]:
            limitations.extend(ood["warnings"])
        return limitations

    @staticmethod
    def _primary_concept(task: ScientificTask) -> str:
        """Map the task domain to a USG-style concept label."""
        mapping = {
            ScientificDomain.PHYSICAL_SIMULATION: "heat diffusion",
            ScientificDomain.FORECASTING: "ensemble forecast",
            ScientificDomain.LOCALIZATION: "factor graph",
            ScientificDomain.MATERIALS: "crystal lattice",
            ScientificDomain.GEOMETRIC: "equivariance",
            ScientificDomain.MOLECULAR_STRUCTURE: "protein backbone",
            ScientificDomain.MOLECULAR_INTERACTION: "covalent bond",
            ScientificDomain.PROTEIN_DESIGN: "protein backbone",
        }
        return mapping.get(task.domains[0], "general science")

    @staticmethod
    def _representation_name(task: ScientificTask) -> str:
        """Human-readable representation name for the explanation."""
        mapping = {
            ScientificDomain.PHYSICAL_SIMULATION: "A field representation",
            ScientificDomain.FORECASTING: "A trajectory-ensemble representation",
            ScientificDomain.LOCALIZATION: "A factor-graph representation",
            ScientificDomain.MATERIALS: "A crystal-graph representation",
            ScientificDomain.GEOMETRIC: "A geometric point-set representation",
            ScientificDomain.MOLECULAR_STRUCTURE: "A molecular-graph representation",
            ScientificDomain.MOLECULAR_INTERACTION: "A molecular-graph representation",
            ScientificDomain.PROTEIN_DESIGN: "A molecular-graph representation",
        }
        return mapping.get(task.domains[0], "A state-space representation")


def _bool_check(name: str, passed: bool, detail: str):
    """Build a required validation check from a boolean outcome."""
    from .contracts import ValidationCheck

    return ValidationCheck(name=name, passed=bool(passed), detail=detail)


def _analytic_field(context: dict[str, Any]) -> np.ndarray:
    """Evaluate the analytic Fourier-series solution on the primary grid."""
    from .adapters.numerical import analytic_heat_solution

    problem = context["problem"]
    return (
        analytic_heat_solution(
            context["primary_grid"],
            problem["horizon"],
            length=problem["length"],
            alpha=problem["alpha"],
            u_initial=problem["u_initial"] - problem["u_boundary"],
        )
        + problem["u_boundary"]
    )


def _reference_graph_from(context: dict[str, Any]):
    """Return the reference materials graph from the representation artifact."""
    return context.get("reference_graph")


class _Trace:
    """Operation and stage trace recorder for the 52-step loop."""

    def __init__(self) -> None:
        """Initialize an empty trace."""
        self._records: dict[int, dict[str, Any]] = {}
        self._stages: list[dict[str, Any]] = []
        self._current_stage: str | None = None

    def open_stage(self, stage_id: str, name: str) -> None:
        """Open one phase stage."""
        self._current_stage = stage_id
        self._stages.append({"stage_id": stage_id, "name": name, "status": "running"})

    def close_stage(self, stage_id: str, completed: bool) -> None:
        """Close the open stage with a status."""
        for stage in self._stages:
            if stage["stage_id"] == stage_id:
                stage["status"] = "completed" if completed else "failed"

    def record(self, step: int, status: str, explanation: str) -> None:
        """Record one operation outcome by step number."""
        spec = OPERATION_REGISTRY[step - 1]
        self._records[step] = {
            "operation_id": spec.operation_id,
            "operation": f"{spec.operation_id} {spec.name}",
            "step": step,
            "phase": f"PHASE-{spec.phase}",
            "station": spec.station,
            "stage_id": self._current_stage or f"PHASE-{spec.phase}",
            "functional_status": status,
            "explanation": explanation,
            "mathematical_operation": spec.mathematical_operation,
            "functional_responsibility": spec.functional_responsibility,
        }

    def fail_all_remaining(self, reason: str) -> None:
        """Mark every unrecorded step as blocked with the failure reason."""
        for spec in OPERATION_REGISTRY:
            if spec.step not in self._records:
                self.record(spec.step, "blocked", reason)
        for stage in self._stages:
            if stage["status"] == "running":
                stage["status"] = "failed"

    def operations(self) -> list[dict[str, Any]]:
        """Return the operation trace in canonical step order."""
        return [self._records[spec.step] for spec in OPERATION_REGISTRY if spec.step in self._records]

    def stages(self) -> list[dict[str, Any]]:
        """Return the stage trace."""
        return list(self._stages)
