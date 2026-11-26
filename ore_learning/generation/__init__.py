"""GAN/diffusion interfaces and synthetic-data validation (Section 5)."""
from .diffusion import (  # noqa: F401
    DiffusionBenchmark,
    benchmark_diffusion_vs_autoregressive,
    corrupt_sequences,
)
from .gan import (  # noqa: F401
    GANObjectiveError,
    GANTrainingReport,
    GenerationObjective,
    admit_generation_objective,
    train_gan,
)
from .gating import GeneratedCandidate, GeneratedOutputGate, generate_id  # noqa: F401
from .synthetic import (  # noqa: F401
    SyntheticDataReport,
    approve_synthetic_for_training,
    evaluate_synthetic,
    memorization_flags,
    utility_comparison,
)

__all__ = (
    "DiffusionBenchmark",
    "GANObjectiveError",
    "GANTrainingReport",
    "GeneratedCandidate",
    "GeneratedOutputGate",
    "GenerationObjective",
    "SyntheticDataReport",
    "admit_generation_objective",
    "approve_synthetic_for_training",
    "benchmark_diffusion_vs_autoregressive",
    "corrupt_sequences",
    "evaluate_synthetic",
    "generate_id",
    "memorization_flags",
    "train_gan",
    "utility_comparison",
)
