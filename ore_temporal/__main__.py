"""Synthetic demonstration of the Point 13 temporal and event-driven intelligence layer.

Run with ``python -m ore_temporal`` from the repository root.  The demo builds an
irregularly sampled synthetic stream with a level shift, runs admission, ingestion,
baseline, CfC and SSM candidates, event detection, conformal uncertainty, portable-graph
validation, and prints the typed result envelope summary.
"""
from __future__ import annotations

import argparse

import numpy as np

from ore_temporal import (
    CalibrationData,
    ObservationChannel,
    TemporalEngine,
    TemporalObservation,
    TemporalRequest,
    TimestampBasis,
)


def make_request() -> TemporalRequest:
    """Build the demo request with a declared channel schema and budgets."""
    return TemporalRequest(
        task_id="demo-task-001",
        source_id="demo-sensor-01",
        objective="Track a noisy level signal and detect significant changes.",
        observation_schema=(
            ObservationChannel(name="level", unit="V", lower_bound=-10.0, upper_bound=10.0,
                               description="normalized sensor level"),
        ),
        timestamp_semantics=TimestampBasis.WALL_CLOCK,
        horizon=1.0,
        latency_budget=1.0,
        resource_budget=1.0,
    )


def make_observations(n: int = 64, seed: int = 7) -> list[TemporalObservation]:
    """Build an irregularly sampled stream with a mid-stream level shift."""
    rng = np.random.default_rng(seed)
    timestamps = np.cumsum(rng.uniform(0.05, 0.35, size=n))
    observations: list[TemporalObservation] = []
    for index, timestamp in enumerate(timestamps):
        level = 0.5 + 0.2 * np.sin(0.7 * timestamp) + 0.05 * rng.standard_normal()
        if index >= n // 2:
            level += 1.2  # injected level shift for change detection
        observations.append(TemporalObservation(
            source_id="demo-sensor-01",
            event_id=f"obs-{index:03d}",
            timestamp=round(float(timestamp), 6),
            time_basis=TimestampBasis.WALL_CLOCK,
            values=(round(float(level), 6),),
            units=("V",),
            provenance={"generator": "ore_temporal.__main__"},
        ))
    return observations


def make_calibration(observations: list[TemporalObservation]) -> CalibrationData:
    """Build held-out chronological calibration pairs from the pre-shift segment."""
    pre = observations[: len(observations) // 2]
    values = tuple((obs.values[0],) for obs in pre[1:])
    predictions = tuple((float(np.mean([o.values[0] for o in pre[: i + 1]])),)
                        for i in range(1, len(pre)))
    return CalibrationData(values=values, predictions=predictions,
                           source="chronological pre-shift segment")


def main(argv: list[str] | None = None) -> int:
    """Run the Point 13 demonstration end to end."""
    parser = argparse.ArgumentParser(description="Point 13 TEDIL demonstration")
    parser.add_argument("--observations", type=int, default=64,
                        help="number of synthetic observations")
    parser.add_argument("--seed", type=int, default=7, help="synthetic stream seed")
    args = parser.parse_args(argv)

    request = make_request()
    observations = make_observations(n=args.observations, seed=args.seed)
    calibration = make_calibration(observations)
    engine = TemporalEngine()
    result = engine.run(request, observations, calibration=calibration)

    print("=== ORE Point 13 -- Temporal and Event-Driven Intelligence Layer ===")
    print(f"task:                {result.task_id}")
    print(f"problem class:       {result.problem_class.value}")
    print(f"validation status:   {result.validation_status.value} "
          f"({result.diagnostics['task_status']})")
    family = (result.model_identity.family.value if result.model_identity
              else "baseline")
    print(f"selected family:     {family}")
    print(f"window:              [{result.interval[0]:.3f}, {result.interval[1]:.3f}] s, "
          f"{len(observations)} observations")
    print(f"operation steps:     {len(result.operation_trace)} "
          f"(completed: {sum(1 for s in result.operation_trace if s['status'] == 'completed')})")
    print(f"typed estimates:     {len(result.estimates)}")
    for estimate in result.estimates:
        interval = ("" if estimate.confidence_interval is None
                    else f" CI=({estimate.confidence_interval[0]:.3f}, "
                         f"{estimate.confidence_interval[1]:.3f})")
        print(f"  - {estimate.channel}: {estimate.value:.4f} {estimate.unit}"
              f" [{estimate.epistemic_status.value}]{interval}")
    print(f"typed events:        {len(result.events)}")
    for event in result.events[:6]:
        print(f"  - {event.event_type} @ t={event.timestamp:.3f} "
              f"magnitude={event.magnitude:.3f} severity={event.severity.value} "
              f"hypothesis={event.cause_hypothesis}")
    if result.uncertainty is not None:
        print(f"uncertainty:         method={result.uncertainty.method} "
              f"calibration={result.uncertainty.calibration_version} "
              f"ood={result.uncertainty.out_of_distribution}")
    if result.baseline_comparison is not None:
        comparison = result.baseline_comparison
        print(f"baseline comparison: model {comparison.model_metric:.4f} vs baseline "
              f"{comparison.baseline_metric:.4f} {comparison.metric_name.upper()} "
              f"(beats baseline: {comparison.beats_baseline})")
    print(f"graph validation:    {result.diagnostics['graph_validation']}")
    print(f"drift status:        {result.diagnostics['drift']['status']} "
          f"(PSI={result.diagnostics['drift']['psi']:.3f})")
    print(f"promotion:           {result.diagnostics['promotion']['promoted']}")
    envelope = result.build_envelope()
    print("--- result envelope ---")
    for key, value in envelope.to_dict().items():
        if key != "model_identity":
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
