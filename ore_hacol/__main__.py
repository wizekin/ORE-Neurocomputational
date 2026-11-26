"""Command-line demo for the HACOL Point 21 module.

Runs a small end-to-end demonstration of the 48-step integrated algorithm
across the supported workload kinds and prints the resulting envelopes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .contracts import (
    CorrectnessContract,
    ConsequenceLevel,
    OptimizationRequest,
)
from .pipeline import HardwareAwareOptimizer
from .profiling import (
    attention_fixture,
    least_squares_fixture,
    low_rank_matrix,
    qubo_assignment_fixture,
)


def _request(kind: str, shape, operation_id: str, **kwargs) -> OptimizationRequest:
    return OptimizationRequest(
        task_id=f"demo-task-{kind}",
        operation_id=operation_id,
        operation_kind=kind,
        input_shape=shape,
        calling_module="ore_hacol.__main__",
        calling_function="demo",
        correctness_contract=CorrectnessContract(
            abs_tolerance=kwargs.pop("abs_tolerance", 1e-6),
            rel_tolerance=kwargs.pop("rel_tolerance", 1e-3),
            consequence_level=ConsequenceLevel.ADVISORY,
        ),
        **kwargs,
    )


def demo_attention(optimizer: HardwareAwareOptimizer) -> dict[str, Any]:
    fixture = attention_fixture(sequence=128, dim=24, mask_kind="causal", seed=21)
    request = _request(
        "attention", (128, 128, 24), "demo-attention",
        approximation_allowed=False, call_frequency=50.0,
    )
    result = optimizer.run(request, fixture)
    return result.to_envelope()


def demo_factorization(optimizer: HardwareAwareOptimizer) -> dict[str, Any]:
    fixture = low_rank_matrix(rows=200, columns=120, rank=6, noise=1e-4, seed=21)
    request = _request(
        "factorization", (200, 120), "demo-randomized-svd",
        approximation_allowed=True, call_frequency=200.0,
    )
    result = optimizer.run(request, {"A": fixture["A"]})
    return result.to_envelope()


def demo_least_squares(optimizer: HardwareAwareOptimizer) -> dict[str, Any]:
    fixture = least_squares_fixture(rows=300, columns=24, condition=1e3, seed=21)
    request = _request(
        "least_squares", (300, 24), "demo-sketched-lstsq",
        approximation_allowed=True, call_frequency=10.0,
    )
    result = optimizer.run(request, {"A": fixture["A"], "b": fixture["b"]})
    return result.to_envelope()


def demo_matmul(optimizer: HardwareAwareOptimizer) -> dict[str, Any]:
    import numpy as np

    rng = np.random.default_rng(21)
    inputs = {
        "A": rng.uniform(0.5, 1.5, size=(96, 64)),
        "B": rng.uniform(0.5, 1.5, size=(64, 48)),
    }
    request = _request(
        "matrix_multiply", (96, 64, 48), "demo-mixed-precision",
        approximation_allowed=False, call_frequency=20.0,
    )
    result = optimizer.run(request, inputs)
    return result.to_envelope()


def demo_optimization(optimizer: HardwareAwareOptimizer) -> dict[str, Any]:
    fixture = qubo_assignment_fixture(tasks=4, resources=5, seed=21)
    request = _request(
        "optimization", (4, 5), "demo-assignment-qubo",
        approximation_allowed=False, call_frequency=1.0,
    )
    result = optimizer.run(request, {"cost": fixture["cost"]})
    return result.to_envelope()


DEMOS = {
    "attention": demo_attention,
    "linalg": demo_factorization,
    "least_squares": demo_least_squares,
    "matmul": demo_matmul,
    "optimization": demo_optimization,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ORE HACOL Point 21 demo")
    parser.add_argument(
        "--demo",
        default="all",
        choices=["all", *DEMOS],
        help="which demo to run (default: all)",
    )
    parser.add_argument("--seed", type=int, default=21, help="global fixture seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optional JSON path for the result envelopes",
    )
    args = parser.parse_args(argv)

    optimizer = HardwareAwareOptimizer(calibrate_hardware=True)
    hardware = optimizer.discover_capabilities()
    print("HACOL hardware profile:", hardware.hardware_identity())
    print("  cores:", hardware.core_count, "| bandwidth GB/s:",
          round(hardware.bandwidth_bytes_per_second / 1e9, 2),
          "| GEMM GFLOP/s:", round(hardware.compute_rate_flops_per_second / 1e9, 3))
    print("  accelerators:", list(hardware.accelerator_types) or "none (CPU reference/optimized only)")

    demos = list(DEMOS) if args.demo == "all" else [args.demo]
    envelopes: dict[str, Any] = {}
    for name in demos:
        print(f"\n=== demo: {name} ===")
        envelope = DEMOS[name](optimizer)
        envelopes[name] = envelope
        print(json.dumps(envelope, indent=2, sort_keys=True, default=str))

    if args.output is not None:
        payload = {
            "hardware": hardware.to_dict(),
            "envelopes": envelopes,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
        print(f"\nenvelopes written to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
