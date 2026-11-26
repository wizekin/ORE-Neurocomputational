"""GAN laboratory: bounded generative tooling (Point 14, Step 23).

A GAN is admitted only for a defined generation objective (measurable scarcity,
simulation, or augmentation need), never as an unconstrained generator.  The
min-max objective

    min_G max_D E_x~data log D(x) + E_z~pz log(1 - D(G(z)))

is trained here on toy data with deterministic seeded numpy networks; the generated
samples are always treated as unvalidated candidates (Step 26 gates them).
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np

from ..config import LearningConfig
from ..functional import gan_discriminator_objective, gan_generator_objective
from ..nn import TinyDiscriminator, TinyGenerator, TinyMLP

__all__ = ("GenerationObjective", "GANObjectiveError", "GANTrainingReport",
           "admit_generation_objective", "train_gan")


class GANObjectiveError(ValueError):
    """Raised when a generative objective is undefined or unjustified (Step 23)."""


@dataclass(frozen=True, slots=True)
class GenerationObjective:
    """A declared, measurable generation objective (Step 23 gate)."""

    objective_kind: str  # scarcity | simulation | augmentation
    description: str
    measurable_need: str
    evaluation_plan: str
    license_ok: bool = True

    def __post_init__(self) -> None:
        """Validate the objective declaration."""
        if self.objective_kind not in ("scarcity", "simulation", "augmentation"):
            raise GANObjectiveError(
                f"unsupported objective kind {self.objective_kind!r}")
        for name, value in (("description", self.description),
                            ("measurable_need", self.measurable_need),
                            ("evaluation_plan", self.evaluation_plan)):
            if not isinstance(value, str) or not value.strip():
                raise GANObjectiveError(f"{name} is mandatory for a defined objective")
        if not self.license_ok:
            raise GANObjectiveError("the source data license forbids derived generation")


def admit_generation_objective(objective: GenerationObjective) -> dict[str, Any]:
    """Step 23 -- admit the GAN only for the defined objective; record the gate."""
    return {"admitted": True, "objective_kind": objective.objective_kind,
            "measurable_need": objective.measurable_need,
            "evaluation_plan": objective.evaluation_plan,
            "note": ("generated samples are not assumed to be representative or "
                     "correct; they remain untrusted candidates")}


@dataclass(slots=True)
class GANTrainingReport:
    """Measured outcome of one bounded GAN training run."""

    steps: int
    final_discriminator_loss: float
    final_generator_loss: float
    discriminator_accuracy: float
    elapsed_seconds: float
    generator_digest: str
    discriminator_digest: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe view."""
        return {"steps": self.steps,
                "final_discriminator_loss": round(self.final_discriminator_loss, 6),
                "final_generator_loss": round(self.final_generator_loss, 6),
                "discriminator_accuracy": round(self.discriminator_accuracy, 6),
                "elapsed_seconds": round(self.elapsed_seconds, 3),
                "generator_digest": self.generator_digest,
                "discriminator_digest": self.discriminator_digest,
                "note": "training happens only in the generative data laboratory"}


def train_gan(real_data: np.ndarray, *, steps: int | None = None,
              seed: int = 17, noise_dim: int = 4, hidden: int = 8,
              batch_size: int = 32, learning_rate: float = 0.05,
              objective: GenerationObjective | None = None,
              config: LearningConfig | None = None) -> tuple[TinyGenerator,
                                                              TinyDiscriminator,
                                                              GANTrainingReport]:
    """Train a tiny GAN under a declared objective (Step 23).

    Alternating full-batch gradient steps implement the min-max objective through
    the shared functional losses; everything is seeded and deterministic.  The
    caller must pass a :class:`GenerationObjective` -- an undefined objective raises
    :class:`GANObjectiveError`.
    """
    if objective is None:
        raise GANObjectiveError("a GAN requires a defined generation objective (Step 23)")
    admit_generation_objective(objective)
    effective_config = config or LearningConfig()
    effective_steps = int(steps if steps is not None
                          else effective_config.gan_training_steps)
    data = np.asarray(real_data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 4:
        raise ValueError("real data must be a (N, d) batch with N >= 4")
    rng = np.random.default_rng(seed)
    generator = TinyGenerator(noise_dim=noise_dim, data_dim=data.shape[1],
                              hidden=hidden, seed=seed + 1)
    discriminator = TinyDiscriminator(data_dim=data.shape[1], hidden=hidden,
                                      seed=seed + 2)
    started = time.perf_counter()
    d_loss = g_loss = 0.0
    accuracy = 0.5
    for _ in range(effective_steps):
        batch = data[rng.integers(0, data.shape[0], size=min(batch_size, data.shape[0]))]
        noise = rng.normal(size=(batch.shape[0], generator.noise_dim))
        fake = generator.generate(noise)

        # ---- discriminator step: max_D E log D(x) + E log(1 - D(G(z)))
        real_output, real_cache = discriminator.network.forward(batch)
        fake_output, fake_cache = discriminator.network.forward(fake)
        p_real = np.clip(real_output[:, 0], 1e-7, 1 - 1e-7)
        p_fake = np.clip(fake_output[:, 0], 1e-7, 1 - 1e-7)
        d_loss = gan_discriminator_objective(p_real, p_fake)
        grad_real = (-1.0 / p_real)[:, None] / batch.shape[0]
        grad_fake = (1.0 / (1.0 - p_fake))[:, None] / batch.shape[0]
        grads = _merge_gradients(
            discriminator.network.backward(real_cache, grad_real),
            discriminator.network.backward(fake_cache, grad_fake))
        discriminator.network.sgd_step(grads, learning_rate)

        # ---- generator step: min_G -E log D(G(z))
        noise = rng.normal(size=(batch.shape[0], generator.noise_dim))
        fake = generator.generate(noise)
        fake_output, fake_cache = discriminator.network.forward(fake)
        p_fake_g = np.clip(fake_output[:, 0], 1e-7, 1 - 1e-7)
        g_loss = gan_generator_objective(p_fake_g)
        grad_p = (-1.0 / p_fake_g)[:, None] / batch.shape[0]
        grad_input = discriminator.network.input_gradient(fake_cache, grad_p)
        generated_output, generator_cache = generator.network.forward(noise)
        generator.network.sgd_step(
            generator.network.backward(generator_cache, grad_input), learning_rate)
        accuracy = float((np.mean(p_real > 0.5) + np.mean(p_fake < 0.5)) / 2.0)
    report = GANTrainingReport(
        steps=effective_steps, final_discriminator_loss=float(d_loss),
        final_generator_loss=float(g_loss), discriminator_accuracy=accuracy,
        elapsed_seconds=time.perf_counter() - started,
        generator_digest=generator.network.digest(),
        discriminator_digest=discriminator.network.digest())
    return generator, discriminator, report


def _merge_gradients(left: dict[str, np.ndarray],
                     right: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Sum two gradient dictionaries (real and fake batches)."""
    return {name: left[name] + right[name] for name in left}
